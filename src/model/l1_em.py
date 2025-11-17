# src/model/l1_em.py
from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Optional, Any

from numpy import dtype, float64, floating, ndarray
from scipy.linalg import solveh_banded
from scipy.optimize import minimize

# ---------------------------------------------------------
# Banded system builders for A = diag(obs_w) + D' diag(prec_beta) D
# ---------------------------------------------------------

def _build_bands_order1_w(prec_beta: np.ndarray, obs_w: np.ndarray) -> np.ndarray:
    """
    First difference: p = n-1, tri-diagonal SPD.
    A = diag(obs_w) + D' diag(prec_beta) D
    Returns ab with shape (2, n) for solveh_banded(lower=False).
    """
    q = np.asarray(prec_beta, dtype=float)  # length n-1
    w = np.asarray(obs_w, dtype=float)      # length n
    n = q.size + 1
    if w.size != n:
        raise ValueError("obs_w length mismatch")
    d0 = w.copy()
    d0[:n-1] += q
    d0[1:]   += q
    d1 = -q
    ab = np.zeros((2, n))
    ab[0, 1:] = d1
    ab[1, :]  = d0
    return ab


def _build_bands_order2_w(prec_beta: np.ndarray, obs_w: np.ndarray) -> np.ndarray:
    """
    Second difference: p = n-2, penta-diagonal SPD.
    A = diag(obs_w) + D' diag(prec_beta) D
    Returns ab with shape (3, n) for solveh_banded(lower=False).
    """
    q = np.asarray(prec_beta, dtype=float)  # length n-2
    w = np.asarray(obs_w, dtype=float)      # length n
    n = q.size + 2
    if w.size != n:
        raise ValueError("obs_w length mismatch")

    d0 = w.copy()
    d0[:n-2]  += q
    d0[1:n-1] += 4.0 * q
    d0[2:n]   += q

    d1 = np.zeros(n-1)
    d1[:n-2]  += -2.0 * q
    d1[1:n-1] += -2.0 * q

    d2 = np.zeros(n-2)
    d2[:n-2] += q

    ab = np.zeros((3, n))
    ab[0, 2:] = d2
    ab[1, 1:] = d1
    ab[2, :]  = d0
    return ab


def _solve_alpha_banded_safe(
    prec_beta: np.ndarray,
    y: np.ndarray,
    order: int,
    *,
    obs_w: Optional[np.ndarray] = None,
    initial_jitter: float = 0.0,
    max_tries: int = 5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Robust solve of (diag(obs_w) + D' diag(prec_beta) D) alpha = diag(obs_w) y.
    Adds diagonal jitter on failure; last resort dense fallback.
    Returns (alpha, ab).
    """
    y = np.asarray(y, float).ravel()
    n = y.size
    w = np.ones(n, float) if obs_w is None else np.asarray(obs_w, float).ravel()
    rhs = w * y

    jitter = float(initial_jitter)

    for _ in range(max_tries):
        # assemble banded A
        if order == 1:
            ab = _build_bands_order1_w(prec_beta, w)
        elif order == 2:
            ab = _build_bands_order2_w(prec_beta, w)
        else:
            raise ValueError("order must be 1 or 2")

        if jitter > 0.0:
            ab[-1, :] += jitter

        if not np.isfinite(ab).all():
            jitter = 1e-10 if jitter == 0.0 else min(jitter * 10.0, 1e-1)
            continue

        try:
            alpha = solveh_banded(
                ab, rhs, lower=False, overwrite_ab=False, overwrite_b=False, check_finite=False
            )
            return alpha, ab
        except np.linalg.LinAlgError:
            jitter = 1e-10 if jitter == 0.0 else min(jitter * 10.0, 1e-1)

    # Dense fallback
    p = prec_beta.size
    # Build dense D
    D = np.zeros((n - order, n))
    coeffs = np.array([(-1)**k * math.comb(order, k) for k in range(order + 1)], float)
    for i in range(n - order):
        D[i, i:i+order+1] = coeffs
    A = np.diag(w) + D.T @ (prec_beta[:, None] * D)
    A[np.diag_indices(n)] += max(1e-8, jitter)
    alpha = np.linalg.solve(A, rhs)
    return alpha, None


# ---------------------------------------------------------
# diag(X'X) for variance diag approx (pure TF basis)
# ---------------------------------------------------------

def _diag_xtx(order: int, n: int) -> np.ndarray:
    if order == 1:
        m = np.arange(n - 1, 0, -1, dtype=float)
        return m
    elif order == 2:
        m = np.arange(n - 2, 0, -1, dtype=float)
        return m * (m + 1.0) * (2.0 * m + 1.0) / 6.0
    else:
        raise ValueError("order must be 1 or 2")


# ---------------------------------------------------------
# Optional numeric objective for τ² (Laplace prior)
# ---------------------------------------------------------

def _tau_objective_l1(x_log_tau2: np.ndarray, EbW: float, sigma2: float, p: int) -> float:
    tau2 = float(np.exp(x_log_tau2[0]))
    f = (p / 2.0) * np.log(tau2) + (EbW / (2.0 * sigma2 * tau2))
    f += np.log1p(tau2)  # weak global prior
    return float(f)


# ---------------------------------------------------------
# Main algorithm: EM for Laplace (L1) trend filtering
# ---------------------------------------------------------

def tf_l1_em_banded_pure(y: np.ndarray,
                         order: int = 2,
                         tau2: float = 1e-3,
                         lam: float = 1.0,
                         sigma2: float | None = None,
                         conv_tol: float = 1e-6,
                         max_iter: int = 1000,
                         update_tau2: bool = False,
                         damp_tau: float = 1.0,
                         min_scale: float = 1e-12,
                         verbose: bool = True,
                         do_threshold: bool = True,
                         obs_w: Optional[np.ndarray] = None
                         ) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    """
    Laplace (L1) prior + EM (MAP) for trend filtering using banded solves.

    A = diag(obs_w) + D' diag(prec_beta) D,   rhs = diag(obs_w) y
    With obs_w=None defaults to ones (standard fitting). For CV, pass
    obs_w = 0 on validation indices to hold them out.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if n - order <= 0:
        raise ValueError("y must have length > order")
    p = n - order

    # initialise sigma2
    if sigma2 is None:
        r = np.diff(y, n=order)
        mad = np.median(np.abs(r - np.median(r)))
        sigma2 = float(mad * mad) if mad > 0 else float(np.var(y))
    else:
        sigma2 = float(sigma2)
    tau2 = float(tau2)
    lam = float(lam)

    # clamps
    W_MIN, W_MAX = 1e-12, 1e12
    PREC_MAX = 1e12
    EPS_BETA = 1e-18

    # start from α=y
    alpha = y.copy()
    beta = np.diff(alpha, n=order)

    # precompute diagXtX for optional diagnostics (not used in L1 var)
    # keep for parity with HS-EM if needed later
    # diag_xtx = _diag_xtx(order, n)

    it = 0
    while it < max_iter:
        it += 1
        beta_old = beta.copy()

        # E: weights for Laplace mixture
        absb = np.abs(beta)
        w_raw = (lam * math.sqrt(max(sigma2, min_scale))) / np.maximum(absb, EPS_BETA)
        w = np.clip(w_raw, W_MIN, W_MAX)

        # Prior precision on β: w/τ² (clamped)
        prec_beta = np.clip(w / max(tau2, min_scale), min_scale, PREC_MAX)

        # α update with observation weights
        alpha, _ = _solve_alpha_banded_safe(
            prec_beta, y, order, obs_w=obs_w, initial_jitter=0.0, max_tries=5
        )
        beta = np.diff(alpha, n=order)

        # σ² ECME-like update
        resid = (np.ones(n) if obs_w is None else obs_w) * (y - alpha)
        RSS = float((resid @ (y - alpha)))  # equivalently y^T W (I - S) y style
        EbW = float(np.sum((beta * beta) * w))
        sigma2_new = (RSS + (EbW / max(tau2, min_scale))) / (n + p)
        sigma2 = 0.5 * sigma2 + 0.5 * sigma2_new

        # τ² update (optional)
        if update_tau2:
            obj = lambda x: _tau_objective_l1(x, EbW, sigma2, p)
            res = minimize(obj, np.array([math.log(max(tau2, 1e-12))]),
                           method="BFGS", options={"disp": False})
            tau2_new = float(np.exp(res.x[0]))
            tau2 = float(damp_tau * tau2 + (1.0 - damp_tau) * tau2_new)

        # convergence
        eps = 1e-12
        rel_rms = np.linalg.norm(beta - beta_old) / (eps + np.linalg.norm(beta))
        rel_inf = np.max(np.abs(beta - beta_old)) / (eps + np.max(np.abs(beta)))

        if verbose:
            print(f"iter={it:4d}  rel_rms={rel_rms:.3e}  rel_inf={rel_inf:.3e}  "
                  f"tau2={tau2:.3e}  sig2={sigma2:.3e}")

        if (rel_rms < conv_tol) and (rel_inf < conv_tol):
            break

    if do_threshold:
        thr = 1.0 / (5.0 * math.sqrt(n))
        beta = beta.copy()
        beta[np.abs(beta) < thr] = 0.0

    return alpha, beta, float(tau2), float(sigma2), it


def to_banded_lower(
        A: np.ndarray,
        order=2,
        n_points=100
):
    ab = np.zeros((order + 1, n_points), dtype=A.dtype)
    ab[0] = np.diag(A)
    for diagonal in range(1, order + 1):
        ab[diagonal, :n_points - diagonal] = np.diag(A, k=-diagonal)
    return ab


def build_spline_matrix(order: int, n_points: int) -> np.ndarray:
    i, j = np.indices((n_points, n_points))
    k = order - 1
    base = np.maximum(i - j + 1, 0).astype(float)
    return np.where(i >= j, base ** k, 0.0)


def build_difference_matrix(order: int, n_points: int) -> np.ndarray:
    coefficients = np.array([(-1) ** k * math.comb(order, k) for k in range(order + 1)], dtype=float)
    D = np.zeros((n_points, n_points), dtype=float)
    for i in range(n_points - order):
        D[i:i + order + 1, i] = coefficients
    for i in range(1, order + 1):
        D[n_points - i:n_points, n_points - i] = coefficients[0:i]
    return D

def solve_spd(
        A: np.ndarray = None,
        b: np.ndarray = None,
        use_cholesky=False,
        use_scipy=False,
        n:int=512
):
    if use_cholesky:
        if use_scipy:
            ab = to_banded_lower(A, 2, n)
            return solveh_banded(ab, b, lower=True)
        else:
            L = np.linalg.cholesky(A)
            z = np.linalg.solve(L, b)
            return np.linalg.solve(L.T, z)
    else:
        return np.linalg.inv(A) @ b

def _build_bands_order1(prec_beta: np.ndarray) -> np.ndarray:
    """
    D is first-difference (size p = n-1); A = I + D' diag(prec_beta) D is tridiagonal SPD.
    Returns ab with shape (2, n) for solveh_banded(lower=False).
    """
    q = np.asarray(prec_beta, dtype=float)  # length n-1
    n = q.size + 1
    d0 = np.ones(n)
    d0[:n-1] += q         # diag contributions from left neighbor
    d0[1:]   += q         # diag contributions from right neighbor
    d1 = -q               # super-diagonal (and sub-diagonal)
    ab = np.zeros((2, n))
    ab[0, 1:] = d1
    ab[1, :]  = d0
    return ab


def _build_bands_order2(prec_beta: np.ndarray) -> np.ndarray:
    """
    D is second-difference (size p = n-2); A = I + D' diag(prec_beta) D is penta-diagonal SPD.
    Returns ab with shape (3, n) for solveh_banded(lower=False).
    """
    q = np.asarray(prec_beta, dtype=float)  # length n-2
    n = q.size + 2
    d0 = np.ones(n)
    # main diagonal contributions
    d0[:n-2]  += q               # i
    d0[1:n-1] += 4.0 * q         # i+1
    d0[2:n]   += q               # i+2
    # first super-diagonal
    d1 = np.zeros(n-1)
    d1[:n-2]  += -2.0 * q        # (i, i+1)
    d1[1:n-1] += -2.0 * q        # (i+1, i+2)
    # second super-diagonal
    d2 = np.zeros(n-2)
    d2[:n-2]  += q               # (i, i+2)

    ab = np.zeros((3, n))
    ab[0, 2:] = d2
    ab[1, 1:] = d1
    ab[2, :]  = d0
    return ab

def _solve_alpha_banded(prec_beta: np.ndarray, y: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve (I + D' diag(prec_beta) D) alpha = y via banded Cholesky.
    Returns alpha and the banded matrix 'ab' used.
    """
    if order == 1:
        ab = _build_bands_order1(prec_beta)
    elif order == 2:
        ab = _build_bands_order2(prec_beta)
    else:
        raise ValueError("order must be 1 or 2")
    alpha = solveh_banded(ab, y, lower=False, overwrite_ab=False, overwrite_b=False, check_finite=False)
    return alpha, ab

# ---------------------------------------------------------
# Previous algorithm: EM for Laplace (L1) trend filtering
# ---------------------------------------------------------
# Try the older way and see if that performs better
def tf_l1_em_legacy(
    y: np.ndarray,
    order: int = 2,
    tau2: float = 1e-3,
    lam: float = 1.0,
    sigma2: float | None = None,
    conv_tol: float = 1e-6,
    max_iter: int = 1000,
    update_tau2: bool = False,
    damp_tau: float = 1.0,
    min_scale: float = 1e-12,
    verbose: bool = True,
    do_threshold: bool = True,
    obs_w: Optional[np.ndarray] = None,
) -> tuple[
    ndarray[tuple[int], Any] | ndarray[tuple[Any, ...], dtype[Any]], ndarray[tuple[Any, ...], dtype[float64]] | ndarray[
        tuple[Any, ...], dtype[Any]], float, float, int, floating[Any], Any, float | floating[Any]]:
    """
    L1 trend filtering via EM in the older 'dense solve' style.

    Solves   (W + D^T Q D) alpha = W y
    with Q = diag(w / tau^2),  w_i = lam * sqrt(sigma2) / max(|(D alpha)_i|, eps)
    Returns: alpha (signal), beta = diff(alpha, order), tau2, sigma2, #iters
    """

    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if n - order <= 0:
        raise ValueError("y must have length > order")

    p = n - order
    # Initial scales: large lambda^2 and tau^2
    # lambda2 = np.ones(p) * 1e6
    # tau2 = 1e6

    # Noise variance init via MAD of order-differences
    if sigma2 is None:
        r = np.diff(y, n=order)
        mad = np.median(np.abs(r - np.median(r)))
        sigma2 = float(mad * mad) if mad > 0 else float(np.var(y))
    else:
        sigma2 = float(sigma2)

    # Diagonal var approx denominator term
    diag_xtx = _diag_xtx(order, n)

    # EM loop
    it = 0
    alpha = y.copy()
    beta = np.ones(p) * 1e6
    beta_old = None

    while it < max_iter:
        it += 1
        beta_old = beta.copy()

        # E-step: lambda from prior precision on lambda
        beta = np.diff(alpha, n=order)
        Elambda2inv = np.minimum(
            1e10,
            np.sqrt(
                tau2 * sigma2 / (beta ** 2)
            )
        )
        prec_beta = Elambda2inv/tau2

        # M-step (calculate alpha and update sigma2)
        alpha, _ = _solve_alpha_banded(prec_beta, y, order)
        beta = np.diff(alpha, n=order)

        if it % 3 == 0:
            resid = y - alpha
            RSS = float(resid @ resid)
            prior_term = float(np.sum(beta**2 / np.maximum(Elambda2inv, min_scale)) / np.maximum(tau2, min_scale))
            sigma2_new = (RSS + prior_term) / (n + p)
            sigma2 = 0.5 * sigma2 + 0.5 * sigma2_new

        eps = 1e-12
        rel_rms = np.linalg.norm(beta - beta_old) / (eps + np.linalg.norm(beta))
        rel_inf = np.max(np.abs(beta - beta_old)) / (eps + np.max(np.abs(beta)))
        thr_active = 1.0 / (50.0 * math.sqrt(n))
        S = np.union1d(np.where(np.abs(beta_old) > thr_active)[0],
                       np.where(np.abs(beta) > thr_active)[0])
        rel_rms_active = 0.0 if S.size == 0 else np.linalg.norm((beta - beta_old)[S]) / (eps + np.linalg.norm(beta[S]))

        if verbose:
            print(f"iter={it:4d}  rel_rms={rel_rms:.3e}  rel_inf={rel_inf:.3e}  rel_rms_act={rel_rms_active:.3e}  "
                  f"tau2={tau2:.3e}  sig2={sigma2:.3e}")

        if (rel_rms < conv_tol) and (rel_inf < conv_tol) and (rel_rms_active < conv_tol):
            break

    # Optional hard-thresholding for sparsity reporting
    if do_threshold:
        thr = 1.0 / (5.0 * math.sqrt(n))
        beta = beta.copy()
        beta[np.abs(beta) < thr] = 0.0

    return alpha, beta, float(tau2), float(sigma2), it