# src/data/gistemp.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from .base import DataSource, DataBundle
from .utils import center_and_validate, make_x_grid


@dataclass
class GISTempAnnual(DataSource):
    """
    Loader for NASA GISTEMP Land–Ocean Temperature Index (annual anomalies, °C).

    Expected text format (like nasa_temp.txt):
        Land-Ocean Temperature Index (C)
        --------------------------------
        Year No_Smoothing  Lowess(5)
        ----------------------------
        1880    -0.17   -0.10
        1881    -0.09   -0.13
        ...
    Non-numeric header/separator lines are ignored.

    Parameters
    ----------
    path : str
        Local path to the nasa_temp.txt file.
    use_column : {"no_smoothing", "lowess"}
        Which series to load (default: "no_smoothing").
    year_range : Optional[tuple[int, int]]
        Inclusive filter (e.g., (1880, 2024)).
    name : str
        Dataset name.
    """

    path: str
    use_column: str = "no_smoothing"  # or "lowess"
    year_range: Optional[Tuple[int, int]] = None
    name: str = "gistemp_annual"
    seed: int = 42

    def _parse_file(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        years: List[int] = []
        no_smooth: List[float] = []
        lowess: List[float] = []

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                # Skip typical header/separator lines
                if s[0].isalpha() or s.startswith("-"):
                    continue
                parts = s.split()
                # Expect at least 3 columns: year, no_smoothing, lowess
                try:
                    yr = int(parts[0])
                    val_no = float(parts[1])
                    val_lo = float(parts[2])
                except Exception:
                    # Not a data row
                    continue
                years.append(yr)
                no_smooth.append(val_no)
                lowess.append(val_lo)

        if len(years) == 0:
            raise ValueError(f"No data rows found in {self.path}")

        return (
            np.asarray(years, dtype=int),
            np.asarray(no_smooth, dtype=float),
            np.asarray(lowess, dtype=float),
        )

    def load(self) -> DataBundle:
        years, no_smooth, lowess = self._parse_file()

        # Optional year filtering
        if self.year_range is not None:
            lo, hi = self.year_range
            m = (years >= lo) & (years <= hi)
            years = years[m]
            no_smooth = no_smooth[m]
            lowess = lowess[m]

        # Choose series
        if self.use_column.lower() in ("no_smoothing", "no", "nosmooth", "no_smooth"):
            series = no_smooth
            col_used = "No_Smoothing"
        elif self.use_column.lower() in ("lowess", "smooth", "loess"):
            series = lowess
            col_used = "Lowess(5)"
        else:
            raise ValueError("use_column must be one of {'no_smoothing','lowess'}")

        # Map years to an equally spaced grid [0,1] for the model assumptions
        n = series.size
        x = make_x_grid(n, 0.0, 1.0)

        # Center y for the model; record removed mean in meta
        x, y, meta_c = center_and_validate(x, series)

        meta: Dict[str, Any] = {
            "source_path": self.path,
            "years_span": (int(years.min()), int(years.max())),
            "column_used": col_used,
            "units": "deg C (anomaly)",
            "center_meta": meta_c,
        }
        return DataBundle(name=self.name, x=x, y=y, y_true=None, meta=meta)
