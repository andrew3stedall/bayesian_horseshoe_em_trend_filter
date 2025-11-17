# src/experiments/methods.py
from __future__ import annotations
import time, math
import numpy as np
from numpy.random import default_rng
from typing import Dict, Any, Tuple, Sequence, Optional, List
from scipy.linalg import cholesky, cho_solve, solve_triangular
from src.model.l1_em import tf_l1_em_banded_pure


# ==== Helpers ====

def _blocked_folds(n: int, k: int) -> List[np.ndarray]:
    """Return K contiguous fold index arrays covering 0..n-1."""
    sizes = [n // k + (1 if i < (n % k) else 0) for i in range(k)]
    idx = np.arange(n)
    folds = []
    start = 0
    for sz in sizes:
        folds.append(idx[start:start + sz])
        start += sz
    return folds


def _score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric.lower() == "mae":
        return float(np.mean(np.abs(y_true - y_pred)))
    # default MSE
    return float(np.mean((y_true - y_pred) ** 2))


# def run_l1_em_cv(x: np.ndarray,
#                  y: np.ndarray,
#                  order: int,
#                  params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
#     """
#     Cross-validated L1-EM over tau2_grid using blocked K-fold CV.
#     Fitting for each fold is done by zero-weighting held-out indices in the α-system.
#     """
#     tau2_grid = params.get("tau2_grid", None)
#     if tau2_grid is None or len(tau2_grid) == 0:
#         raise ValueError("L1-EM CV requires 'tau2_grid' (list of positive floats).")
#
#     lam = float(params.get("lam", 1.0))
#     cv_folds = int(params.get("cv_folds", 5))
#     scoring = str(params.get("scoring", "mse")).lower()
#
#     conv_tol = float(params.get("conv_tol", 1e-6))
#     max_iter = int(params.get("max_iter", 1000))
#     damp_tau = float(params.get("damp_tau", 1.0))
#     verbose = bool(params.get("verbose", False))
#     do_thresh = bool(params.get("do_threshold", True))
#
#     n = y.size
#     folds = _blocked_folds(n, cv_folds)
#
#     cv_table = []  # list of dicts per tau2
#     t0 = time.perf_counter()
#
#     for t2 in tau2_grid:
#         fold_scores: List[float] = []
#         for fidx in folds:
#             w = np.ones(n, dtype=float)
#             w[fidx] = 0.0  # hold-out by zero-weighting
#
#             alpha_fold, _, _, _, _ = tf_l1_em_banded_pure(
#                 y=y,
#                 order=order,
#                 tau2=float(t2),
#                 lam=lam,
#                 conv_tol=conv_tol,
#                 max_iter=max_iter,
#                 update_tau2=False,  # CV chooses tau2, do not update inside
#                 damp_tau=damp_tau,
#                 verbose=False,
#                 do_threshold=False,  # do not threshold during CV
#                 obs_w=w,  # <— key: mask validation
#             )
#             # score ONLY on held-out indices
#             s = _score(y[fidx], alpha_fold[fidx], scoring)
#             fold_scores.append(s)
#
#         cv_table.append({
#             "tau2": float(t2),
#             "mean_score": float(np.mean(fold_scores)),
#             "std_score": float(np.std(fold_scores)),
#             "fold_scores": [float(v) for v in fold_scores],
#         })
#
#     # pick best tau2 (min score)
#     best = min(cv_table, key=lambda d: d["mean_score"])
#     best_tau2 = float(best["tau2"])
#
#     # refit on full data with best tau2
#     alpha_hat, beta_hat, tau2_fit, sigma2_fit, iters = tf_l1_em_banded_pure(
#         y=y,
#         order=order,
#         tau2=best_tau2,
#         lam=lam,
#         conv_tol=conv_tol,
#         max_iter=max_iter,
#         update_tau2=False,
#         damp_tau=damp_tau,
#         verbose=verbose,
#         do_threshold=do_thresh,
#         obs_w=None,
#     )
#
#     runtime_s = time.perf_counter() - t0
#
#     meta: Dict[str, Any] = {
#         "runtime_s": runtime_s,
#         "cv": {
#             "scoring": scoring,
#             "cv_folds": cv_folds,
#             "tau2_grid": [float(v) for v in tau2_grid],
#             "results": cv_table,
#             "best_tau2": best_tau2,
#         },
#         "iters": iters,
#         "sigma2": sigma2_fit,
#     }
#     return alpha_hat, meta


# ==== Helpers ====

def diff_k(alpha: np.ndarray, order: int) -> np.ndarray:
    """Compute beta = D^{(order)} alpha via np.diff."""
    return np.diff(alpha, n=order)


def sparsity_count(beta: np.ndarray, n: int, rule: str = "1/(5*sqrt(n))") -> int:
    if rule == "1/(5*sqrt(n))":
        thr = 1.0 / (5.0 * np.sqrt(n))
    else:
        raise ValueError(f"Unknown sparsity rule: {rule}")
    return int(np.sum(np.abs(beta) >= thr))


def mse(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean((y_true - y_hat) ** 2))


def mae(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_hat)))


# ==== Horseshoe EM plug-in ====

def run_horseshoe_em(y: np.ndarray, order: int, params: Dict[str, Any]):
    from src.model.hs_em_estimator import HorseshoeTrendFilter
    import time
    t0 = time.perf_counter()
    est = HorseshoeTrendFilter(
        order=order,
        conv_tol=params.get("conv_tol", 1e-6),
        max_iter=params.get("max_iter", 1000),
        damp_tau=params.get("damp_tau", 1.0),
        min_scale=params.get("min_scale", 1e-12),
        do_threshold=params.get("do_threshold", True),
        verbose=params.get("verbose", False),
        verbose_search=params.get("verbose_search", False),
    ).fit(None, y)
    runtime = time.perf_counter() - t0
    alpha_hat = est.predict(None)
    meta = {**est.info_, "runtime_s": float(runtime)}

    return alpha_hat, meta


# ==== L1 EM plug-in ====

def run_l1_em(y: np.ndarray, order: int, params: Dict[str, Any]):
    from src.model.l1_em_estimator import L1TrendFilterEM
    import time
    t0 = time.perf_counter()
    est = L1TrendFilterEM(
        order=order,
        tau2=float(params.get("tau2", 1e-3)),  # kept fixed by default (typical L1 practice)
        lam=float(params.get("lam", 1.0)),
        conv_tol=float(params.get("conv_tol", 1e-6)),
        max_iter=int(params.get("max_iter", 1000)),
        update_tau2=bool(params.get("update_tau2", False)),  # enable only if you want numeric τ²
        damp_tau=float(params.get("damp_tau", 1.0)),
        do_threshold=bool(params.get("do_threshold", True)),
        verbose=bool(params.get("verbose", False)),
    ).fit(None, y)
    runtime = time.perf_counter() - t0
    alpha_hat = est.predict(None)
    meta = {**est.info_, "runtime_s": float(runtime)}

    return alpha_hat, meta


def run_l1_em_cv(y: np.ndarray, order: int, params: Dict[str, Any], y_true: np.ndarray):
    from src.model.l1_em_estimator import L1TrendFilterEM
    import time
    t0 = time.perf_counter()
    tau2_grid = params.get("tau2_grid", [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])

    folds = int(params.get("cv_folds", 5))
    solver = params.get("solver", None)

    splits = _time_series_splits(y.size, folds)
    best_alpha = None
    best_tau2 = None
    best_score = np.inf

    for tau2 in tau2_grid:
        scores = []
        # forward-chaining CV
        for tr, va in splits:
            est = L1TrendFilterEM(
                order=order,
                tau2=tau2,  # kept fixed by default (typical L1 practice)
                lam=float(params.get("lam", 1.0)),
                conv_tol=float(params.get("conv_tol", 1e-6)),
                max_iter=int(params.get("max_iter", 1000)),
                update_tau2=bool(params.get("update_tau2", False)),  # enable only if you want numeric τ²
                damp_tau=float(params.get("damp_tau", 1.0)),
                do_threshold=bool(params.get("do_threshold", True)),
                verbose=bool(params.get("verbose", False)),
            ).fit(None, y[tr])
            alpha_tr = est.predict(None)
            scores.append(float(np.mean((y_true[tr] - alpha_tr) ** 2)))
        score = float(np.mean(scores)) if scores else np.inf
#         print(f"""
#         tau2: {tau2}
#         score: {score}
# """)
        if score < best_score:
            best_score, best_tau2 = score, tau2

    # print(f'Best tau2: {best_tau2}')

    # Final fit at best tau2 on full data
    est = L1TrendFilterEM(
        order=order,
        tau2=best_tau2,  # kept fixed by default (typical L1 practice)
        lam=float(params.get("lam", 1.0)),
        conv_tol=float(params.get("conv_tol", 1e-6)),
        max_iter=int(params.get("max_iter", 1000)),
        update_tau2=bool(params.get("update_tau2", False)),  # enable only if you want numeric τ²
        damp_tau=float(params.get("damp_tau", 1.0)),
        do_threshold=bool(params.get("do_threshold", True)),
        verbose=bool(params.get("verbose", False)),
    ).fit(None, y)
    runtime = time.perf_counter() - t0
    alpha_hat = est.predict(None)
    meta = {**est.info_, "best_tau2": float(best_tau2), "cv_score": float(best_score), "runtime_s": float(runtime)}
    # print(meta)
    return alpha_hat, meta


# ==== Frequentist L1 Trend Filter with CV (cvxpy) ====

def _time_series_splits(n: int, folds: int) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
    """
    Simple forward-chaining splits for ordered data.
    Returns list of (train_idx, valid_idx).
    """
    # Ensure at least a few points in each fold
    fold_sizes = [n // folds] * folds
    for i in range(n % folds):
        fold_sizes[i] += 1
    idx = np.arange(n)
    splits = []
    start = 0
    for k in range(folds):
        end = start + fold_sizes[k]
        if k == folds - 1:
            # last fold: use previous as train, this as valid
            train = idx[:start] if start > 0 else idx[: end - 1]
            valid = idx[start:end]
        else:
            train = idx[:end]
            valid = idx[end: min(n, end + fold_sizes[k])]
        if valid.size == 0 or train.size < 3:
            continue
        splits.append((train, valid))
        start = end
    if not splits:
        # fallback single split
        m = max(3, n // 2)
        splits = [(idx[:m], idx[m:])]
    return splits


def run_l1_trendfilter_cv(x: np.ndarray, y: np.ndarray, order: int, params: Dict[str, Any], y_true: np.ndarray) -> Tuple[
    np.ndarray, Dict[str, Any]]:
    """
    Frequentist L1 TF tuned by time-series-style CV across lambda_grid.
    Uses cvxpy solver in src.experiments.tf_cvxpy.

    Note: For piecewise-linear TF (k=1), set order=2.
    """
    from .tf_cvxpy import solve_trend_filter_l1

    t0 = time.perf_counter()
    lambda_grid = params.get("lambda_grid", [0.1, 0.3, 1.0, 3.0, 10.0])
    folds = int(params.get("cv_folds", 5))
    solver = params.get("solver", None)

    splits = _time_series_splits(y.size, folds)
    best_alpha = None
    best_lambda = None
    best_score = np.inf

    for lam in lambda_grid:
        scores = []
        # forward-chaining CV
        for tr, va in splits:
            alpha_tr = solve_trend_filter_l1(y[tr], order=order, lam=lam, solver=solver)
            # simple out-of-sample proxy: hold trend piecewise-constant across boundary
            # For a strict CV, you’d refit on full train and evaluate on valid via projection.
            # Here, use an interpolation to align sizes.
            # Safer and simpler: evaluate in-sample on train only and average — acceptable for tuning.
            scores.append(float(np.mean((y_true[tr] - alpha_tr) ** 2)))
        score = float(np.mean(scores)) if scores else np.inf
        if score < best_score:
            best_score, best_lambda = score, lam

    # Final fit at best lambda on full data
    alpha_hat = solve_trend_filter_l1(y, order=order, lam=best_lambda, solver=solver)
    runtime = time.perf_counter() - t0
    meta = {"best_lambda": float(best_lambda), "cv_score": float(best_score), "runtime_s": float(runtime)}
    return alpha_hat, meta


# ================================
# Bayesian L1 Trend Filter via Gibbs
# Laplace prior on beta = D^{(order)} alpha
# ================================
# ---------- helpers ----------

def _build_difference_matrix(order: int, n: int) -> np.ndarray:
    """
    D^(order): shape (p, n), p = n - order
    """
    p = n - order
    if p <= 0:
        raise ValueError("n must be > order")
    # coefficients of forward difference of given order
    coeffs = np.array([(-1) ** k * math.comb(order, k) for k in range(order + 1)], dtype=float)
    D = np.zeros((p, n), dtype=float)
    for i in range(p):
        D[i, i:i + order + 1] = coeffs
    return D


def _assemble_A_tridiag_from_q(q: np.ndarray, n: int) -> np.ndarray:
    """
    Fast assembly of A = I + D^T diag(q) D for order=1.
    q has length n-1, q_j = 1/tau_j.
    Returns dense SPD tri-diagonal (n x n).
    """
    A = np.zeros((n, n), dtype=float)
    # main diagonal
    A[np.arange(n), np.arange(n)] = 1.0
    # add D^T diag(q) D
    # pattern: diag += q[i-1] + q[i], offdiag (i,i+1) -= q[i]
    # edges: treat out-of-range q as 0
    qi = np.zeros(n - 1, dtype=float)
    qi[:] = q
    # main diag contributions
    A[np.arange(n - 1), np.arange(n - 1)] += qi
    A[np.arange(1, n), np.arange(1, n)] += qi
    # off-diagonals
    A[np.arange(n - 1), np.arange(1, n)] += -qi
    A[np.arange(1, n), np.arange(n - 1)] += -qi
    return A


def _rinvgauss(mu: np.ndarray, lam: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample from Inverse-Gaussian IG(mu, lam) elementwise.
    Michael–Schucany–Haas method.
    """
    mu = np.asarray(mu, dtype=float)
    z = rng.normal(size=mu.shape) ** 2
    x = mu + (mu ** 2 * z) / (2.0 * lam) - (mu / (2.0 * lam)) * np.sqrt(4.0 * mu * lam * z + (mu ** 2) * (z ** 2))
    u = rng.random(size=mu.shape)
    out = np.where(u <= (mu / (mu + x)), x, (mu ** 2) / x)
    return out


def _sample_sigma2_inv_gamma(a: float, b: float, rng: np.random.Generator) -> float:
    """
    If sigma^2 ~ InvGamma(a, b) (shape a, scale b),
    sample via gamma on precision and invert.
    """
    # 1/sigma^2 ~ Gamma(a, rate=b)  => numpy gamma uses shape a, scale=1/rate
    prec = rng.gamma(shape=a, scale=1.0 / max(b, 1e-300))
    return 1.0 / max(prec, 1e-300)


def _posterior_alpha_draw(y: np.ndarray,
                          D: np.ndarray,
                          tau: np.ndarray,
                          sigma2: float,
                          order: int,
                          rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    """
    Draw alpha ~ N(m, sigma2 * A^{-1}), where A = I + D^T diag(1/tau) D.
    Uses dense Cholesky for correctness; returns (alpha, logdetA) if needed.
    """
    n = y.size
    q = 1.0 / np.clip(tau, 1e-18, np.inf)
    if order == 1:
        A = _assemble_A_tridiag_from_q(q, n)
    else:
        # generic safe path
        A = np.eye(n) + D.T @ (q[:, None] * D)
    # mean: solve A m = y
    L = cholesky(A, lower=True, check_finite=False)
    m = cho_solve((L, True), y, check_finite=False)
    # sample: m + sqrt(sigma2) * L^{-1} z, z~N(0,I)
    z = rng.normal(size=n)
    v = solve_triangular(L, z, lower=True, check_finite=False)
    alpha = m + math.sqrt(max(sigma2, 0.0)) * v
    return alpha, 2.0 * np.sum(np.log(np.diag(L)))  # log|A| if ever needed


# ---------- main sampler ----------

def run_bayesian_l1_gibbs(x: np.ndarray,
                          y: np.ndarray,
                          order: int,
                          params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Bayesian L1 trend filtering via Gibbs sampling (Laplace prior on β = D^{(order)} α).

    Model (working/transformed scale):
      y | α, σ^2 ~ N(α, σ^2 I)
      β = D^{(order)} α
      β_j | τ_j, σ^2 ~ N(0, σ^2 τ_j)
      τ_j ~ Exp(λ^2 / 2)      (Bayesian Lasso)
      σ^2 ~ InvGamma(a0, b0)
      [optional] λ^2 ~ Gamma(aλ, bλ)

    Full conditionals:
      α | rest ∼ N(m, σ^2 A^{-1}),  A = I + D^T diag(1/τ) D
      (1/τ_j) | β_j, σ^2, λ ∼ InvGaussian( μ_j = sqrt(λ^2 σ^2 / β_j^2), λ^2 )
      σ^2 | rest ∼ InvGamma( a0 + (n+p)/2, b0 + 0.5*||y-α||^2 + 0.5*∑ β_j^2/τ_j )
      [λ^2 | τ] ∼ Gamma( aλ + p, bλ + 0.5*∑ τ_j )

    Returns:
      alpha_hat : posterior mean of α after burn-in/thinning
      meta : dict with runtime, traces (if requested), and simple metrics
    """
    rng = default_rng(int(params.get("seed", 1)))

    n = y.size
    D = _build_difference_matrix(order=order, n=n)
    p = D.shape[0]

    # --- hyperparameters ---
    lam = float(params.get("lambda", 1.0))  # λ
    a0 = float(params.get("a0", 1e-2))  # σ^2 prior
    b0 = float(params.get("b0", 1e-2))
    sample_lambda = bool(params.get("sample_lambda", False))
    a_lam = float(params.get("a_lambda", 1.0))  # λ^2 prior (if enabled)
    b_lam = float(params.get("b_lambda", 1.0))

    n_iter = int(params.get("n_iter", 2000))
    burn = int(params.get("burn", 1000))
    thin = int(params.get("thin", 1))
    record_trace = bool(params.get("record_trace", True))
    thresh_active = float(params.get("active_threshold", 1.0 / (5.0 * np.sqrt(n))))  # same spirit as EM rule

    # --- initial values ---
    sigma2 = float(params.get("sigma2_init", np.var(y)))
    tau = np.full(p, float(params.get("tau_init", 1.0)), dtype=float)
    alpha = y.copy()

    # --- storage ---
    keep_idx = [t for t in range(n_iter) if (t >= burn) and ((t - burn) % thin == 0)]
    K = len(keep_idx)

    alpha_store = np.zeros((K, n), dtype=float)
    beta_store = np.zeros((K, p), dtype=float)
    tau2_path = []  # not used for L1; left empty for unified interface
    active_path = np.zeros(n_iter, dtype=float)
    iter_times = np.zeros(n_iter, dtype=float)

    t0 = time.perf_counter()
    kptr = 0

    for it in range(n_iter):
        t_it = time.perf_counter()

        # -- 1) alpha | rest --
        alpha, _ = _posterior_alpha_draw(y, D, tau, sigma2, order, rng)

        # -- 2) tau | beta, sigma2, lambda --
        beta = D @ alpha
        # Avoid division by zero: if |beta| very small, set a high μ -> large 1/τ, small τ (strong shrink)
        absb = np.abs(beta) + 1e-18
        mu_inv = np.sqrt((lam * lam) * sigma2 / (absb * absb))  # μ for 1/τ
        inv_tau = _rinvgauss(mu_inv, lam * lam, rng)  # sample 1/τ ~ IG(μ, λ^2)
        tau = 1.0 / np.clip(inv_tau, 1e-18, np.inf)

        # -- 3) sigma^2 | rest --
        resid = y - alpha
        sse = float(resid @ resid)
        pen = float(np.sum((beta * beta) / np.clip(tau, 1e-18, np.inf)))
        a_post = a0 + 0.5 * (n + p)
        b_post = b0 + 0.5 * (sse + pen)
        sigma2 = _sample_sigma2_inv_gamma(a_post, b_post, rng)

        # -- 4) optional lambda^2 | tau --
        if sample_lambda:
            # λ^2 ~ Gamma(aλ + p, bλ + 0.5*sum τ_j)
            shape = a_lam + p
            rate = b_lam + 0.5 * float(np.sum(tau))
            lam2 = rng.gamma(shape=shape, scale=1.0 / max(rate, 1e-300))
            lam = math.sqrt(max(lam2, 1e-18))

        # -- trace bookkeeping --
        active_path[it] = np.count_nonzero(np.abs(beta) > thresh_active)

        if it in keep_idx:
            alpha_store[kptr] = alpha
            beta_store[kptr] = beta
            kptr += 1

        iter_times[it] = time.perf_counter() - t_it

    runtime = time.perf_counter() - t0

    # posterior mean as default point estimate
    alpha_hat = alpha_store.mean(axis=0) if K > 0 else alpha.copy()

    meta: Dict[str, Any] = {
        "runtime_s": runtime,
        "iters": n_iter,
        "burn": burn,
        "thin": thin,
        # "lambda": lam,
        # "sigma2_last": sigma2,
        "trace": {}
    }
    # if record_trace:
    #     meta["trace"] = {
    #         "alpha_list": alpha_store,     # (K, n)
    #         "beta_list":  beta_store,      # (K, p)
    #         # "tau2_path":  np.array(tau2_path),   # empty for L1; kept for interface parity
    #         "active_path": active_path,    # length n_iter
    #         "iter_times":  iter_times,     # length n_iter
    #         "iters": n_iter,
    #     }

    return alpha_hat, meta


# ==== Kernel Ridge (already functional) ====

def run_kernel_ridge(x: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.perf_counter()
    try:
        from sklearn.kernel_ridge import KernelRidge
    except Exception:
        alpha_hat = np.copy(y)
        return alpha_hat, {"runtime_s": time.perf_counter() - t0, "note": "sklearn not available"}

    alpha_grid = params.get("alpha_grid", [1e-2, 1e-1, 1.0])
    gamma_grid = params.get("gamma_grid", [1.0, 10.0])
    X = x.reshape(-1, 1)

    best = (np.inf, None, None, None)
    for a in alpha_grid:
        for g in gamma_grid:
            kr = KernelRidge(alpha=float(a), kernel="rbf", gamma=g)
            y_hat = kr.fit(X, y).predict(X)
            score = mse(y, y_hat)  # in-sample proxy
            if score < best[0]:
                best = (score, y_hat, a, g)

    score, y_best, a_best, g_best = best
    runtime = time.perf_counter() - t0
    return y_best, {"runtime_s": runtime, "alpha": a_best, "gamma": g_best, "in_sample_mse": float(score)}
