from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Protocol
import numpy as np

@dataclass
class DataBundle:
    """Uniform data container returned by all data sources."""
    name: str
    x: np.ndarray
    y: np.ndarray
    y_true: Optional[np.ndarray]
    meta: Dict[str, Any]

class DataSource(Protocol):
    """Interface for all data sources.""" 
    def load(self) -> DataBundle: ...

    def add_params(self, **params: Any) -> DataSource:
        for k,v in params.items():
            setattr(self, k, v)
        # print(f'DataSource:{self}')
        return self
