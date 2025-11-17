from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any, Tuple
from .base import DataSource, DataBundle
from .utils import center_and_validate, make_x_grid

class SilsoMonthly(DataSource):
    """SILSO monthly sunspot numbers (v2.0).
    Standard file: YYYY MM DECIMAL_DATE MONTHLY_MEAN SMOOTHED ...
    We take MONTHLY_MEAN (4th column).
    """
    def __init__(self, path: str, year_range: Optional[Tuple[int, int]] = None, name: str = "silso_monthly", seed:int=42) -> None:
        self.path = path
        self.year_range = year_range
        self.name = name

    def load(self) -> DataBundle:
        arr = np.loadtxt(self.path, comments="#")
        years = arr[:, 0].astype(int)
        monthly_mean = arr[:, 3].astype(float)
        if self.year_range is not None:
            lo, hi = self.year_range
            keep = (years >= lo) & (years <= hi)
            years = years[keep]; monthly_mean = monthly_mean[keep]
        n = monthly_mean.size; x = make_x_grid(n, 0.0, 1.0)
        x, y, meta_c = center_and_validate(x, monthly_mean)
        meta: Dict[str, Any] = {
            "source_path": self.path,
            "years_span": (int(years.min()), int(years.max())) if n > 0 else None,
            "center_meta": meta_c,
        }
        return DataBundle(name=self.name, x=x, y=y, y_true=None, meta=meta)
