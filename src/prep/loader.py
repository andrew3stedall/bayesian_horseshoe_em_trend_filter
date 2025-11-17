from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable
import numpy as np

from src.registry import load_registry_from_yaml, load_custom_registry_from_yaml
from src.data.base import DataBundle
from .base import PreparedData, Pipeline, Transform
from .transforms import EnsureFloat64, EnforceEqualSpacing

class StandardLoader:
    """
    Loads a dataset by name via the registry and applies a preprocessing pipeline.
    Produces a PreparedData object with an inverse_target that undoes y-transforms.
    """
    def __init__(self, registry_yaml: str, transforms: Optional[List[Transform]] = None) -> None:
        self.registry_yaml = registry_yaml
        # Always enforce float64 + equal spacing first
        base_prefix: List[Transform] = [EnsureFloat64(), EnforceEqualSpacing()]
        self.pipeline = Pipeline(steps=base_prefix + (transforms or []))

    def load_with_params(self, name: str, **params: Any) -> PreparedData:
        reg = load_registry_from_yaml(self.registry_yaml,1)
        bundle: DataBundle = reg.load_with_params(name, **params)
        x = np.asarray(bundle.x, dtype=np.float64).ravel()
        y = np.asarray(bundle.y, dtype=np.float64).ravel()
        y_true = np.asarray(bundle.y_true, dtype=np.float64).ravel()
        meta: Dict[str, Any] = dict(bundle.meta) if bundle.meta else {}

        # Fit/transform
        x_t, y_t, meta_t = self.pipeline.fit_transform(x, y, meta)
        x_t, y_true, meta_t = self.pipeline.fit_transform(x, y_true, meta)

        # Compose inverse target; special-case transforms needing x context (e.g., DetrendLinear)
        inv = self.pipeline.inverse_target

        # If DetrendLinear used, its inverse needs x; make a closure to add back c+m*x.
        det = meta_t.get("detrend_linear")
        if det is not None:
            c = float(det["c"]);
            m = float(det["m"])

            def inv_with_trend(yhat: np.ndarray) -> np.ndarray:
                y0 = inv(yhat)
                return y0 + (c + m * x_t)

            inverse_fn: Callable[[np.ndarray], np.ndarray] = inv_with_trend
        else:
            inverse_fn = inv

        return PreparedData(
            name=bundle.name,
            x=x_t,
            y=y_t,
            y_true=y_true,
            meta=meta_t,
            inverse_target_fn=inverse_fn,
        )

    def load(self, name: str, seed: Optional[int]) -> PreparedData:
        reg = load_registry_from_yaml(self.registry_yaml, seed)
        bundle: DataBundle = reg.load(name)
        x = np.asarray(bundle.x, dtype=np.float64).ravel()
        y = np.asarray(bundle.y, dtype=np.float64).ravel()
        y_true = np.asarray(bundle.y_true, dtype=np.float64).ravel()
        meta: Dict[str, Any] = dict(bundle.meta) if bundle.meta else {}

        # Fit/transform
        x_t, y_t, meta_t = self.pipeline.fit_transform(x, y, meta)

        # Compose inverse target; special-case transforms needing x context (e.g., DetrendLinear)
        inv = self.pipeline.inverse_target

        # If DetrendLinear used, its inverse needs x; make a closure to add back c+m*x.
        det = meta_t.get("detrend_linear")
        if det is not None:
            c = float(det["c"]); m = float(det["m"])
            def inv_with_trend(yhat: np.ndarray) -> np.ndarray:
                y0 = inv(yhat)
                return y0 + (c + m * x_t)
            inverse_fn: Callable[[np.ndarray], np.ndarray] = inv_with_trend
        else:
            inverse_fn = inv

        return PreparedData(
            name=bundle.name,
            x=x_t,
            y=y_t,
            y_true=y_true,
            meta=meta_t,
            inverse_target_fn=inverse_fn,
        )