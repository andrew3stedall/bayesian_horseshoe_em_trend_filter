# src/model/hs_em_estimator.py
from __future__ import annotations
from dataclasses import asdict
from typing import Optional, Dict, Any
import numpy as np
import time

# Optional sklearn integration (with safe fallbacks)
try:
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.metrics import r2_score as _sk_r2
    from sklearn.utils.validation import check_is_fitted as _sk_check
except Exception:
    class BaseEstimator:  # minimal no-op
        def get_params(self, deep: bool = True) -> Dict[str, Any]:
            return {k: v for k, v in self.__dict__.items() if not k.endswith("_")}
        def set_params(self, **params):
            for k, v in params.items():
                setattr(self, k, v)
            return self
    class RegressorMixin: pass
    def _sk_r2(y_true, y_pred):
        y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        mean_y = float(np.mean(y_true))
        ss_tot = float(np.sum((y_true - mean_y) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    def _sk_check(estimator, attributes=None):
        attrs = attributes or ["alpha_"]
        for a in attrs:
            if not hasattr(estimator, a):
                raise AttributeError("Estimator not fitted; missing attribute: " + a)

# Import your implementation
from src.model.hs_em import tf_hs_em_banded_pure

class HorseshoeTrendFilter(BaseEstimator, RegressorMixin):
    """
    Horseshoe prior + EM (MAP) trend filter with banded dual solves.
    Scikit-learn style API for easy comparison.

    Parameters
    ----------
    order : int, default=2
        Difference order penalised on alpha. For piecewise-linear TF (k=1), use order=2.
    conv_tol : float, default=1e-6
        Convergence tolerance used inside EM.
    max_iter : int, default=1000
        Maximum EM iterations.
    damp_tau : float, default=1.0
        Damping factor for tau^2 update (1.0 means no damping).
    min_scale : float, default=1e-12
        Numerical floor to keep precisions/scales positive.
    do_threshold : bool, default=True
        Apply fast sparsification rule |beta_j| < (5*sqrt(n))^{-1} -> 0.
    verbose : bool, default=False
        Print EM progress (internal).
    verbose_search : bool, default=False
        Print objective details for tau^2 line search (if used).

    Attributes (after fit)
    ----------------------
    alpha_ : np.ndarray, shape (n,)
        Fitted mean trend on the working (transformed) scale.
    beta_ : np.ndarray, shape (n-order,)
        Discrete differences D^{(order)} alpha_.
    Eb2_ : np.ndarray
        E[b_j^2] used in EM updates.
    lambda2_ : np.ndarray
        Local shrinkage scales squared.
    tau2_ : float
        Global shrinkage scale squared.
    sigma2_ : float
        Noise variance estimate.
    n_iter_ : int
        Number of EM iterations used.
    runtime_s_ : float
        Wall-clock runtime for fit.
    n_features_in_ : int
        Training sample size (n). Stored for sklearn compatibility.
    """

    def __init__(self,
                 order: int = 2,
                 conv_tol: float = 1e-6,
                 max_iter: int = 1000,
                 damp_tau: float = 1.0,
                 min_scale: float = 1e-12,
                 do_threshold: bool = True,
                 verbose: bool = False,
                 verbose_search: bool = False):
        self.order = order
        self.conv_tol = float(conv_tol)
        self.max_iter = max_iter
        self.damp_tau = damp_tau
        self.min_scale = min_scale
        self.do_threshold = do_threshold
        self.verbose = verbose
        self.verbose_search = verbose_search

        # set during fit
        self.alpha_: Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None
        self.Eb2_: Optional[np.ndarray] = None
        self.lambda2_: Optional[np.ndarray] = None
        self.tau2_: Optional[float] = None
        self.sigma2_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        self.runtime_s_: Optional[float] = None
        self.n_features_in_: Optional[int] = None

    # sklearn-style signature; X is optional (ignored), y is required
    def fit(self, X, y=None):
        """
        Fit the trend filter. For convenience, either call:
            fit(None, y)        # preferred
        or:
            fit(X, y)           # X is ignored (kept for sklearn compatibility)
        """
        if y is None:
            # Allow fit(y) style if someone calls fit(y) by mistake
            # Detect if X is the actual y vector
            if X is None:
                raise ValueError("y must be provided.")
            y = np.asarray(X, dtype=float).ravel()
        else:
            y = np.asarray(y, dtype=float).ravel()

        n = y.size
        if n < max(3, self.order + 1):
            raise ValueError(f"Insufficient samples (n={n}) for order={self.order}")

        t0 = time.perf_counter()
        alpha, beta, Eb2, lambda2, tau2, sigma2, iters = tf_hs_em_banded_pure(
            y,
            order=self.order,
            conv_tol=self.conv_tol,
            max_iter=self.max_iter,
            damp_tau=self.damp_tau,
            min_scale=self.min_scale,
            verbose=self.verbose,
            do_threshold=self.do_threshold,
            verbose_search=self.verbose_search,
        )
        self.runtime_s_ = time.perf_counter() - t0

        # Store fitted attributes
        self.alpha_ = np.asarray(alpha, dtype=float)
        self.beta_ = np.asarray(beta, dtype=float)
        self.Eb2_ = np.asarray(Eb2, dtype=float)
        self.lambda2_ = np.asarray(lambda2, dtype=float)
        self.tau2_ = float(tau2) if tau2 is not None else None
        self.sigma2_ = float(sigma2) if sigma2 is not None else None
        self.n_iter_ = int(iters) if iters is not None else None
        self.n_features_in_ = int(n)
        return self

    def predict(self, X=None) -> np.ndarray:
        """
        Return fitted mean trend alpha_. If X is provided, only a length check is performed
        (trend filtering is fitted on the training grid). New-X prediction is not supported.
        """
        _sk_check(self, ["alpha_"])
        if X is not None:
            # If a vector is passed, just check length matches training length
            if np.ndim(X) == 0:
                pass
            else:
                m = np.asarray(X).shape[0]
                if self.n_features_in_ is not None and m != self.n_features_in_:
                    raise ValueError("Predict called with a different number of points than fit; "
                                     "trend filtering does not currently extrapolate/interpolate.")
        return self.alpha_

    def score(self, X, y) -> float:
        """R^2 score to align with sklearn regressors."""
        y = np.asarray(y, dtype=float).ravel()
        yhat = self.predict(X)
        return _sk_r2(y, yhat)

    # Convenience accessors
    @property
    def beta(self) -> np.ndarray:
        _sk_check(self, ["beta_"])
        return self.beta_

    @property
    def info_(self) -> Dict[str, Any]:
        """Small dict summarising key learned scales and runtime."""
        _sk_check(self, ["alpha_"])
        return {
            "tau2": self.tau2_,
            "sigma2": self.sigma2_,
            "n_iter": self.n_iter_,
            "runtime_s": self.runtime_s_,
            "lambda2": self.lambda2_.tolist(),
            "beta": self.beta_.tolist(),
        }
