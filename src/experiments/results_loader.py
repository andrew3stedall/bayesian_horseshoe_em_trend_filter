from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import json, gzip, pandas as pd

def load_runs_jsonl(path: str | Path) -> pd.DataFrame:
    """
    Load results/runs.jsonl(.gz) and return a flattened DataFrame for analysis.
    """
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    recs: List[Dict[str, Any]] = []
    for r in rows:
        base: Dict[str, Any] = {
            "dataset": r.get("dataset_key"),
            "method": r.get("method_key"),
            "n": r.get("n"),
            "threshold_abs": r.get("threshold_abs"),
            "timestamp": r.get("timestamp"),
        }
        if "spec" in r and isinstance(r["spec"], dict):
            for k, v in r["spec"].items():
                base[f"spec_{k}"] = v
        if "metrics" in r and isinstance(r["metrics"], dict):
            for k, v in r["metrics"].items():
                base[f"metric_{k}"] = v
        if "timing" in r and isinstance(r["timing"], dict):
            for k, v in r["timing"].items():
                base[f"time_{k}"] = v
        if "trace" in r and isinstance(r["trace"], dict):
            for k, v in r["trace"].items():
                base[f"trace_{k}"] = v
        if "beta_summary" in r and isinstance(r["beta_summary"], dict):
            base["beta_active"] = r["beta_summary"].get("beta_active", None)
        recs.append(base)

    return pd.DataFrame(recs)
