from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any

def center_series(y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    y = np.asarray(y, dtype=float).ravel()
    mu = float(np.mean(y))
    return y - mu, {"removed_mean": mu}

def ensure_equally_spaced(x: np.ndarray, rtol: float = 1e-6) -> None:
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        return
    diffs = np.diff(x)
    if not np.allclose(diffs, diffs[0], rtol=rtol, atol=0.0):
        raise ValueError("x is not equally spaced within tolerance.")

def make_x_grid(n: int, start: float = 0.0, stop: float = 1.0) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be at least 2.")
    return np.linspace(start, stop, num=n, dtype=float)

def center_and_validate(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    ensure_equally_spaced(x)
    y_centered, meta = center_series(y)
    return x, y_centered, meta
