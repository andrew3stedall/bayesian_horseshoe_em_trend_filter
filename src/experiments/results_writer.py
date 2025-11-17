from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union, List
import json, gzip, time, numpy as np

from .results_schema import RunSpec, TimingInfo
from .metrics import (
    compute_basic_errors,
    detect_knots,
    summarise_beta,
    summarise_trace,
)

ArrayLike = Union[np.ndarray, Sequence[float], List[float]]

def _to_list(x: ArrayLike) -> List[float]:
    a = np.asarray(x).ravel()
    return [float(v) for v in a]

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _open_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode="at", encoding="utf-8")
    return open(path, mode="a", encoding="utf-8")

def write_run_entry(
    path_jsonl: Union[str, Path],
    spec: RunSpec,
    y_pred: ArrayLike,             # alpha_hat
    y_obs: ArrayLike,              # observed y
    threshold_abs: float,          # absolute threshold (e.g., c/sqrt(n))
    *,
    beta: Optional[ArrayLike] = None,       # D^{(2)} alpha_hat (length n-2 for k=1)
    x: Optional[ArrayLike] = None,
    y_true: Optional[ArrayLike] = None,     # synthetic ground truth, if available
    timing: Optional[TimingInfo] = None,
    lambda2_summary: Optional[Mapping[str, float]] = None,
    tau2: Optional[float] = None,
    tau2_path: Optional[ArrayLike] = None,  # optional per-iter tau2
    iter_times: Optional[ArrayLike] = None, # optional per-iter seconds
    active_path: Optional[ArrayLike] = None,# optional per-iter active counts
    extra_metrics: Optional[Mapping[str, float]] = None,
    include_beta_full: bool = False,
) -> None:
    """
    Append one JSON object per run to results/runs.jsonl(.gz).
    """
    path = Path(path_jsonl)
    path.parent.mkdir(parents=True, exist_ok=True)

    metrics = compute_basic_errors(y_obs, y_pred, y_true)

    payload: Dict[str, Any] = {
        "timestamp": _now_iso(),
        "spec": asdict(spec),
        "dataset": spec.dataset,
        "method": spec.method,
        "n": int(np.asarray(y_obs).size),
        "k": int(spec.k),
        "threshold_abs": float(threshold_abs),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "y_pred": _to_list(y_pred),
        "y_obs": _to_list(y_obs),
    }

    if x is not None:
        payload["x"] = _to_list(x)
    if y_true is not None:
        payload["y_true"] = _to_list(y_true)

    if beta is not None:
        payload["beta_summary"] = summarise_beta(beta, threshold_abs)
        payload["knots"] = detect_knots(beta, threshold_abs, x=x)
        if include_beta_full:
            payload["beta_full"] = _to_list(beta)

    if timing is not None:
        payload["timing"] = {k: (float(v) if isinstance(v, (int, float)) else v)
                             for k, v in asdict(timing).items() if v is not None}

    trace = summarise_trace(
        tau2_path=tau2_path,
        iter_times=iter_times,
        active_path=active_path,
    )
    if tau2 is not None:
        trace["tau2_final"] = float(tau2)
    if trace:
        payload["trace"] = trace

    if lambda2_summary is not None:
        payload["lambda2_summary"] = {k: float(v) for k, v in lambda2_summary.items()}

    if extra_metrics:
        payload["metrics"].update({k: float(v) for k, v in extra_metrics.items()})

    with _open_maybe_gzip(path) as f:
        f.write(json.dumps(payload, separators=(",", ":")) + "\n")
