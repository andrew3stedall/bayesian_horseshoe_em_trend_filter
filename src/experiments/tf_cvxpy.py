# src/experiments/tf_cvxpy.py
from __future__ import annotations
import numpy as np
import math

def build_difference_matrix(order: int, n: int) -> np.ndarray:
    """
    Forward-difference operator D^{(order)} of shape (n-order, n).
    order=1 -> first differences, order=2 -> second differences, etc.
    """
    if order < 1 or order >= n:
        raise ValueError("order must be in {1,2,...,n-1}")
    coeffs = np.array([(-1)**k * int(math.comb(order, k)) for k in range(order + 1)], dtype=float)
    D = np.zeros((n - order, n), dtype=float)
    for i in range(n - order):
        D[i, i:i + order + 1] = coeffs
    return D

def solve_trend_filter_l1(y: np.ndarray,
                          order: int,
                          lam: float,
                          solver: str | None = None) -> np.ndarray:
    """
    Solve:  min_alpha  0.5*||y - alpha||_2^2 + lam * || D^{(order)} alpha ||_1
    Returns alpha_hat (np.ndarray, shape (n,)).

    Note: For piecewise linear fits (k=1 TF), use order=2.
    """
    import cvxpy as cp  # imported lazily so package is optional

    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    D = build_difference_matrix(order, n)

    alpha = cp.Variable(n)
    obj = 0.5 * cp.sum_squares(y - alpha) + lam * cp.norm1(D @ alpha)
    prob = cp.Problem(cp.Minimize(obj))

    # Good defaults: OSQP or ECOS
    chosen = solver or ("OSQP" if "OSQP" in cp.installed_solvers() else
                        "ECOS" if "ECOS" in cp.installed_solvers() else None)
    prob.solve(solver=chosen, verbose=False)
    if alpha.value is None:
        # Retry with a different solver if available
        for cand in ("ECOS", "OSQP", "SCS"):
            if cand == chosen or cand not in cp.installed_solvers():
                continue
            prob.solve(solver=cand, verbose=False)
            if alpha.value is not None:
                break
    if alpha.value is None:
        raise RuntimeError("cvxpy failed to solve trend filtering problem.")
    return np.asarray(alpha.value, dtype=float)
