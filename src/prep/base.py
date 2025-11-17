from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Tuple, Dict, Any, List, Optional
import numpy as np

@dataclass
class PreparedData:
    """Final, model-ready view of a dataset (after preprocessing)."""
    name: str
    x: np.ndarray          # float64, 1D, typically scaled to [0,1]
    y: np.ndarray          # float64, 1D, possibly standardised
    y_true: Optional[np.ndarray]  # if available
    meta: Dict[str, Any]   # includes provenance + transform metadata
    # Inverse function to map model predictions back to original y scale:
    inverse_target_fn: Any = field(repr=False, default=lambda y: y)

    def inverse_target(self, y_hat: np.ndarray) -> np.ndarray:
        return self.inverse_target_fn(y_hat)

class Transform(Protocol):
    """
    Preprocessing step that may be stateful.
    Must accept and return (x, y, meta). Should be dtype/shape-safe and idempotent where possible.
    """
    name: str

    def fit(self, x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]) -> "Transform": ...
    def transform(self, x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: ...
    def inverse_target(self, y: np.ndarray) -> np.ndarray: ...

class Pipeline:
    """Sequentially applies a list of Transforms and composes inverse_target transforms."""
    def __init__(self, steps: List[Transform]) -> None:
        self.steps = steps[:]

    def fit_transform(self, x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        for t in self.steps:
            t.fit(x, y, meta)
            x, y, meta = t.transform(x, y, meta)
        return x, y, meta

    def inverse_target(self, y: np.ndarray) -> np.ndarray:
        # apply inverse in reverse order
        for t in reversed(self.steps):
            y = t.inverse_target(y)
        return y
