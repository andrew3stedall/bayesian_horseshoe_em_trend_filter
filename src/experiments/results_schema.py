from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class RunSpec:
    dataset: str             # e.g. "gistemp_no_smooth", "silso", "synth_spikes_n512_s005"
    method: str              # e.g. "HS-EM-TF", "L1-TF-CV", "KRR", "BayesL1-TF"
    seed: int
    n: int
    k: int                   # trend-filtering order (here 1)
    sigma: Optional[float] = None        # synthetic noise level, if applicable
    threshold_c: Optional[float] = None  # c used in c/sqrt(n)
    notes: Optional[str] = None

@dataclass
class TimingInfo:
    wall_clock_sec: float
    cv_time_sec: Optional[float] = None  # for CV-based methods
    iters: Optional[int] = None
