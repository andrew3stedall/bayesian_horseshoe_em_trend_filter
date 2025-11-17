from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, Sequence, Union, List
from .methods import mse, mae, diff_k, sparsity_count

ArrayLike = Union[np.ndarray, Sequence[float], List[float]]

def compute_metrics(y: np.ndarray,
                    alpha_hat: np.ndarray,
                    order: int,
                    sparsity_rule: str,
                    runtime_s: float) -> Dict[str, Any]:
    y_hat = alpha_hat
    n = y.size
    beta_hat = diff_k(alpha_hat, order)
    return {
        "mse": mse(y, y_hat),
        "mae": mae(y, y_hat),
        "sparsity_count": sparsity_count(beta_hat, n, sparsity_rule),
        "runtime_s": float(runtime_s),
    }


def compute_basic_errors(
    y: ArrayLike,
    y_hat: ArrayLike,
    y_true: Optional[ArrayLike] = None
) -> Dict[str, float]:
    y = np.asarray(y).ravel()
    y_pred = np.asarray(y_hat).ravel()
    out = {"mse": mse(y, y_pred), "mae": mae(y, y_pred)}
    if y_true is not None:
        f = np.asarray(y_true).ravel()
        out["mse_true"] = mse(f, y_pred)
        out["mae_true"] = mae(f, y_pred)
    return out

def detect_knots(
    beta: ArrayLike,
    threshold: float,
    x: Optional[ArrayLike] = None
) -> Dict[str, Any]:
    b = np.asarray(beta).ravel()
    idx = np.where(np.abs(b) > threshold)[0]
    out: Dict[str, Any] = {
        "knots_idx": [int(i) for i in idx],
        "knots_count": int(idx.size),
    }
    if x is not None:
        x = np.asarray(x).ravel()
        out["knots_x"] = [float(x[min(j + 1, x.size - 1)]) for j in idx]
    return out

def summarise_beta(beta: ArrayLike, threshold: float) -> Dict[str, Any]:
    b = np.asarray(beta).ravel()
    active_mask = np.abs(b) > threshold
    active_idx = np.where(active_mask)[0]
    return {
        "beta_active": int(np.count_nonzero(active_mask)),
        "beta_nonzero_idx": [int(i) for i in active_idx],
        "beta_nonzero_val": [float(b[i]) for i in active_idx],
        "beta_abs_median": float(np.median(np.abs(b))) if b.size else 0.0,
        "beta_abs_max": float(np.max(np.abs(b))) if b.size else 0.0,
    }

def summarise_trace(
    tau2_path: Optional[ArrayLike] = None,
    iter_times: Optional[ArrayLike] = None,
    active_path: Optional[ArrayLike] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if tau2_path is not None:
        t2 = np.asarray(tau2_path).ravel()
        if t2.size:
            out.update({
                "tau2_final": float(t2[-1]),
                "tau2_init": float(t2[0]),
                "tau2_median": float(np.median(t2)),
                "iter_count": int(t2.size),
            })
    if iter_times is not None:
        it = np.asarray(iter_times).ravel()
        if it.size:
            out.update({
                "iter_time_median_sec": float(np.median(it)),
                "iter_time_p90_sec": float(np.percentile(it, 90.0)),
                "iter_time_sum_sec": float(np.sum(it)),
            })
    if active_path is not None:
        ap = np.asarray(active_path).ravel()
        if ap.size:
            out.update({
                "active_final": int(ap[-1]),
                "active_init": int(ap[0]),
            })
    return out
