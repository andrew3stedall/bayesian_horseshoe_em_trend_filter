# src/model/hs_em.py
from __future__ import annotations
import math
import numpy as np
from typing import Tuple
from scipy.linalg import solveh_banded
from scipy.optimize import minimize


# ----------------------------
# Helpers: banded system builders for (I + D' diag(prec_beta) D)
# ----------------------------

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


# ----------------------------
# Helpers: diag(X'X) for variance diag approx (pure TF basis)
# ----------------------------

def _diag_xtx(order: int, n: int) -> np.ndarray:
    """
    diag(X'X) used in the diagonal variance approximation for beta = D^{(order)} alpha.
    order=1: length p=n-1 with values [n-1, n-2, ..., 1]
    order=2: length p=n-2 with values m(m+1)(2m+1)/6 for m=[n-2,...,1]
    """
    if order == 1:
        m = np.arange(n - 1, 0, -1, dtype=float)
        return m
    elif order == 2:
        m = np.arange(n - 2, 0, -1, dtype=float)
        return m * (m + 1.0) * (2.0 * m + 1.0) / 6.0
    else:
        raise ValueError("order must be 1 or 2")


# ----------------------------
# Objective for numeric τ² update (optimize over log τ²)
# ----------------------------

def _tau_objective(x_log_tau2: np.ndarray, Eb2: np.ndarray, sigma2: float, p: int) -> float:
    """
    Expected negative log posterior profile in τ² (up to constants), using
    the closed-form λ²(τ²) update for horseshoe. Prior on τ² here uses a simple
    exponential (adds '+ tau2'); if you prefer half-Cauchy, replace with log(1 + tau2).
    """
    tau2 = float(np.exp(x_log_tau2[0]))
    # closed-form λ² via horseshoe EM update:
    W = Eb2 / (2.0 * sigma2 * tau2)
    lambda2 = 0.25 * (np.sqrt(1.0 + 6.0 * W + W**2) + W - 1.0)
    lambda2 = np.maximum(lambda2, 1e-10)

    f = np.sum(np.log(lambda2)) \
        + (1.0 / (2.0 * sigma2 * tau2)) * np.sum(Eb2 / lambda2) \
        + (p / 2.0) * np.log(tau2)
    # horseshoe local prior term (integrated form)
    f += np.sum(np.log1p(lambda2))
    # global prior on tau2: exponential (can switch to np.log1p(tau2) for half-Cauchy)
    f += np.log1p(tau2)
    return float(f)


# ----------------------------
# Main algorithm
# ----------------------------

def tf_hs_em_banded_pure(y: np.ndarray,
                         order: int = 2,
                         sigma2: float | None = None,
                         conv_tol: float = 1e-6,
                         max_iter: int = 1000,
                         damp_tau: float = 1.0,
                         min_scale: float = 1e-12,
                         verbose: bool = True,
                         do_threshold: bool = True,
                         verbose_search: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    """
    Horseshoe EM (MAP) for trend filtering using banded solves on the dual system.

    Model (equally spaced, centered target):
        y = alpha + eps,    eps ~ N(0, sigma^2 I)
        beta = D^{(order)} alpha
        beta_j | tau^2, lambda_j^2, sigma^2 ~ N(0, sigma^2 * tau^2 * lambda_j^2)
        tau ~ half-Cauchy(0,1) [implemented as exponential prior on tau^2 in optimizer; see _tau_objective]
        lambda_j ~ half-Cauchy(0,1)

    Updates:
        alpha: solve (I + D' diag(1/(tau^2 lambda^2)) D) alpha = y  (via banded Cholesky)
        E[b_j^2] ≈ (D alpha)_j^2 + sigma^2 / (diag(X'X)_j + 1/(tau^2 lambda_j^2))
        lambda^2: closed-form horseshoe EM update
        tau^2: 1-D numeric search on log(tau^2)
        sigma^2: ECME-like refresh every 3 iterations

    Returns:
        alpha_hat, beta_hat, Eb2, lambda2, tau2, sigma2, iters
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if n - order <= 0:
        raise ValueError("y must have length > order")

    p = n - order
    # Initial scales: large lambda^2 and tau^2
    lambda2 = np.ones(p) * 1e6
    tau2 = 1e6

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
    alpha = np.zeros_like(y)
    beta = np.ones(p) * 1e6
    beta_old = None

    while it < max_iter:
        it += 1
        beta_old = beta.copy()

        # E-step: alpha from banded SPD system with prior precision on beta
        prec_beta = 1.0 / np.maximum(lambda2 * tau2, min_scale)  # diag prior precision on beta
        alpha, _ = _solve_alpha_banded(prec_beta, y, order)
        beta = np.diff(alpha, n=order)

        # Diagonal variance approximation for beta
        var_beta_diag = sigma2 * (1.0 / (diag_xtx + prec_beta))
        Eb2 = beta**2 + var_beta_diag

        # M-step (τ² via 1-D numeric search over log τ²)
        obj = lambda x: _tau_objective(x, Eb2, sigma2, p)
        res = minimize(obj, np.array([math.log(tau2)]), method="BFGS", options={"disp": verbose_search})
        tau2_new = float(np.exp(res.x[0]))
        # optional damping
        tau2 = float(damp_tau * tau2 + (1.0 - damp_tau) * tau2_new)

        # M-step (λ² closed-form given τ²)
        W = Eb2 / (2.0 * sigma2 * tau2)
        lambda2 = 0.25 * (np.sqrt(1.0 + 6.0 * W + W**2) + W - 1.0)
        lambda2 = np.maximum(lambda2, 1e-10)

        # ECME-like σ² refresh (every 3 iters; mild averaging for stability)
        if it % 3 == 0:
            resid = y - alpha
            RSS = float(resid @ resid)
            prior_term = float(np.sum(Eb2 / np.maximum(lambda2, min_scale)) / np.maximum(tau2, min_scale))
            sigma2_new = (RSS + prior_term) / (n + p)
            sigma2 = 0.5 * sigma2 + 0.5 * sigma2_new

        # Convergence checks on beta (including active set)
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

    # if verbose:
    #     print(lambda2)

    return alpha, beta, Eb2, lambda2, float(tau2), float(sigma2), it
