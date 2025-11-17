from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any

from .base import Transform

class NoOp(Transform):
    name = "noop"
    def fit(self, x, y, meta): return self
    def transform(self, x, y, meta): return x, y, meta
    def inverse_target(self, y): return y

class EnsureFloat64(Transform):
    name = "ensure_float64"
    def fit(self, x, y, meta): return self
    def transform(self, x, y, meta):
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), meta
    def inverse_target(self, y): return np.asarray(y, dtype=np.float64)

class EnforceEqualSpacing(Transform):
    """Raises if x is not equally spaced within tolerance."""
    name = "enforce_equal_spacing"
    def __init__(self, rtol: float = 1e-6) -> None:
        self.rtol = rtol
    def fit(self, x, y, meta): return self
    def transform(self, x, y, meta):
        x = np.asarray(x, float).ravel()
        if x.size >= 2:
            d = np.diff(x)
            if not np.allclose(d, d[0], rtol=self.rtol, atol=0.0):
                raise ValueError("x is not equally spaced within tolerance.")
        return x, y, meta
    def inverse_target(self, y): return y

class ScaleX01(Transform):
    """Scales x to [0,1]. Safe if already scaled; stored in meta."""
    name = "scale_x_01"
    def fit(self, x, y, meta):
        self.xmin = float(np.min(x)) if x.size else 0.0
        self.xmax = float(np.max(x)) if x.size else 1.0
        self.span = max(self.xmax - self.xmin, 1e-12)
        return self
    def transform(self, x, y, meta):
        x = (x - self.xmin) / self.span
        meta = {**meta, "x_scaling": {"xmin": self.xmin, "xmax": self.xmax}}
        return x, y, meta
    def inverse_target(self, y): return y  # only x is scaled

class StandardizeTarget(Transform):
    """Z-score standardisation on y; invertible."""
    name = "standardize_target"
    def fit(self, x, y, meta):
        self.mu = float(np.mean(y)) if y.size else 0.0
        self.sd = float(np.std(y, ddof=0)) if y.size else 1.0
        if self.sd < 1e-12: self.sd = 1.0
        return self
    def transform(self, x, y, meta):
        y_std = (y - self.mu) / self.sd
        meta = {**meta, "y_standardization": {"mu": self.mu, "sd": self.sd}}
        return x, y_std, meta
    def inverse_target(self, y):
        return y * self.sd + self.mu

class ClipYQuantiles(Transform):
    """Optional robust clipping to reduce extreme outliers."""
    name = "clip_y_quantiles"
    def __init__(self, lo: float = 0.01, hi: float = 0.99) -> None:
        assert 0.0 <= lo < hi <= 1.0
        self.lo, self.hi = lo, hi
    def fit(self, x, y, meta):
        self.a = float(np.quantile(y, self.lo)) if y.size else 0.0
        self.b = float(np.quantile(y, self.hi)) if y.size else 0.0
        return self
    def transform(self, x, y, meta):
        y2 = np.clip(y, self.a, self.b)
        meta = {**meta, "y_clip": {"lo": self.lo, "hi": self.hi, "a": self.a, "b": self.b}}
        return x, y2, meta
    def inverse_target(self, y): return y

class DetrendLinear(Transform):
    """Optional: remove best linear fit vs x; invertible (adds back)."""
    name = "detrend_linear"
    def fit(self, x, y, meta):
        if x.size >= 2:
            X = np.vstack([np.ones_like(x), x]).T
            # least squares fit
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.c = float(beta[0]); self.m = float(beta[1])
        else:
            self.c = 0.0; self.m = 0.0
        return self
    def transform(self, x, y, meta):
        y2 = y - (self.c + self.m * x)
        meta = {**meta, "detrend_linear": {"c": self.c, "m": self.m}}
        return x, y2, meta
    def inverse_target(self, y):
        # Note: inverse needs x to add back; handled at loader stage.
        return y  # Loader composes a closure with x to add back after all steps.
