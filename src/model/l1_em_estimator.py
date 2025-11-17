# src/model/l1_em_estimator.py
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import time

# Optional sklearn integration (safe fallbacks if sklearn not installed)
try:
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.metrics import r2_score as _sk_r2
    from sklearn.utils.validation import check_is_fitted as _sk_check
except Exception:
    class BaseEstimator:
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

from src.model.l1_em import tf_l1_em_banded_pure, tf_l1_em_legacy

class L1TrendFilterEM(BaseEstimator, RegressorMixin):
    """
    Laplace (L1) prior + EM (MAP) trend filter with banded dual solves.
    Scikit-learn style API for fair comparison against HS-EM.

    Parameters
    ----------
    order : int, default=2
        Difference order in beta = D^{(order)} alpha. For piecewise-linear TF (k=1), use order=2.
    tau2 : float, default=1e-3
        Global scale squared for Laplace mixture (typically tuned via CV for L1).
    lam : float, default=1.0
        Laplace rate λ controlling sparsity in the E[1/ω] weights.
    conv_tol : float, default=1e-6
        Convergence tolerance on beta.
    max_iter : int, default=1000
        Maximum EM iterations.
    update_tau2 : bool, default=False
        If True, numerically updates τ² each iteration (kept False to reflect typical L1 practice).
    damp_tau : float, default=1.0
        Damping for τ² updates when enabled.
    do_threshold : bool, default=True
        Apply the fast sparsification rule |β_j| < (5*sqrt(n))^{-1} -> 0 for reporting.
    verbose : bool, default=False
        Print EM progress.

    Attributes (after fit)
    ----------------------
    alpha_ : (n,)
    beta_  : (n-order,)
    tau2_  : float
    sigma2_: float
    n_iter_: int
    runtime_s_: float
    n_features_in_: int
    """

    def __init__(self,
                 order: int = 2,
                 tau2: float = 1e-3,
                 lam: float = 1.0,
                 conv_tol: float = 1e-6,
                 max_iter: int = 1000,
                 update_tau2: bool = False,
                 damp_tau: float = 1.0,
                 do_threshold: bool = True,
                 verbose: bool = False):
        self.order = int(order)
        self.tau2 = float(tau2)
        self.lam = float(lam)
        self.conv_tol = float(conv_tol)
        self.max_iter = int(max_iter)
        self.update_tau2 = bool(update_tau2)
        self.damp_tau = float(damp_tau)
        self.do_threshold = bool(do_threshold)
        self.verbose = bool(verbose)

        # set during fit
        self.alpha_: Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None
        self.tau2_: Optional[float] = None
        self.sigma2_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        self.runtime_s_: Optional[float] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X, y=None):
        """
        Fit L1 EM trend filter. Preferred call: fit(None, y).
        If X is supplied, it is ignored (kept for sklearn compatibility).
        """
        if y is None:
            if X is None:
                raise ValueError("y must be provided.")
            y = np.asarray(X, dtype=float).ravel()
        else:
            y = np.asarray(y, dtype=float).ravel()

        n = y.size
        if n < max(3, self.order + 1):
            raise ValueError(f"Insufficient samples (n={n}) for order={self.order}")

        t0 = time.perf_counter()
        alpha, beta, tau2, sigma2, iters = tf_l1_em_legacy(
        # alpha, beta, tau2, sigma2, iters = tf_l1_em_banded_pure(
            y=y,
            order=self.order,
            tau2=self.tau2,
            lam=self.lam,
            conv_tol=self.conv_tol,
            max_iter=self.max_iter,
            update_tau2=self.update_tau2,
            damp_tau=self.damp_tau,
            verbose=self.verbose,
            do_threshold=self.do_threshold,
        )
        self.runtime_s_ = time.perf_counter() - t0

        self.alpha_ = np.asarray(alpha, dtype=float)
        self.beta_  = np.asarray(beta, dtype=float)
        self.tau2_  = float(tau2)
        self.sigma2_ = float(sigma2)
        self.n_iter_ = int(iters)
        self.n_features_in_ = int(n)
        return self

    def predict(self, X=None) -> np.ndarray:
        _sk_check(self, ["alpha_"])
        if X is not None:
            if np.ndim(X) != 0:
                m = np.asarray(X).shape[0]
                if self.n_features_in_ is not None and m != self.n_features_in_:
                    raise ValueError("Predict called with different number of points than fit.")
        return self.alpha_

    def score(self, X, y) -> float:
        y = np.asarray(y, dtype=float).ravel()
        yhat = self.predict(X)
        return _sk_r2(y, yhat)

    @property
    def info_(self) -> Dict[str, Any]:
        _sk_check(self, ["alpha_"])
        return {
            "tau2": self.tau2_,
            "sigma2": self.sigma2_,
            "n_iter": self.n_iter_,
            "runtime_s": self.runtime_s_,
            # "lambda2": self.lambda2_.tolist(),
            "beta": self.beta_.tolist(),
        }
