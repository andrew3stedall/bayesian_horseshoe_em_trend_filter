#!/usr/bin/env python3
# summarize_results.py
# Usage:
#   python summarize_results.py \
#       --timing ../report/timing.csv \
#       --metrics ../results/metrics.csv \
#       --write-csv  # optional: writes helper CSVs to ./summaries/
#
# Paste the JSON block it prints back to ChatGPT so I can build your LaTeX tables.

import argparse, json, math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def iqr(series: pd.Series) -> float:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return float(q3 - q1)


def safe_read_csv(path: Path, kind: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{kind} file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{kind} file is empty: {path}")
    return df


def detect_id_cols(df: pd.DataFrame) -> List[str]:
    """Heuristic: likely identifiers/provenance columns not to aggregate."""
    candidates = {
        "dataset", "data", "series", "name", "label",
        "method", "model", "algo", "algorithm", "order",
        "seed", "fold", "cv_fold", "rep", "replication",
        "n", "size", "num_points",
        "spikes", "num_spikes",
        "converged",
    }
    cols = []
    for c in df.columns:
        cl = c.lower()
        if any(tok in cl for tok in candidates):
            cols.append(c)
    return cols


def numeric_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in nums if c not in exclude]


def uniq_vals(df: pd.DataFrame, names: List[str]) -> Dict[str, List]:
    out = {}
    for c in names:
        if c in df.columns:
            vals = df[c].dropna().unique().tolist()
            out[c] = sorted(vals, key=lambda x: (str(type(x)), x))
    return out


def group_summaries(df: pd.DataFrame, group_by: List[str], value_cols: List[str]) -> pd.DataFrame:
    if not group_by or not value_cols:
        return pd.DataFrame()
    agg_funcs = {
        "mean": np.mean,
        "std": np.std,
        "median": np.median,
        "iqr": iqr,
        "count": "count",
    }
    out = []
    gb = df.groupby(group_by, dropna=False)
    for keys, sub in gb:
        row: Dict[str, object] = {}
        if not isinstance(keys, tuple):
            keys = (keys,)
        for kname, kval in zip(group_by, keys):
            row[kname] = kval
        for col in value_cols:
            s = pd.to_numeric(sub[col], errors="coerce").dropna()
            if s.empty:
                row[f"{col}__mean"] = None
                row[f"{col}__std"] = None
                row[f"{col}__median"] = None
                row[f"{col}__iqr"] = None
                row[f"{col}__count"] = 0
            else:
                row[f"{col}__mean"] = float(np.mean(s))
                row[f"{col}__std"] = float(np.std(s, ddof=0))
                row[f"{col}__median"] = float(np.median(s))
                row[f"{col}__iqr"] = iqr(s)
                row[f"{col}__count"] = int(s.shape[0])
        out.append(row)
    return pd.DataFrame(out)


def maybe_compute_time_per_k(timing: pd.DataFrame) -> pd.DataFrame:
    """Add time_per_1k if not present and n + runtime-like column exist."""
    df = timing.copy()
    lower_cols = {c.lower(): c for c in df.columns}
    n_col = None
    for cand in ["n", "size", "num_points"]:
        if cand in lower_cols:
            n_col = lower_cols[cand]
            break
    runtime_col = None
    for cand in ["runtime_s", "time_s", "wall_time_s", "elapsed_s"]:
        if cand in lower_cols:
            runtime_col = lower_cols[cand]
            break
    if n_col and runtime_col and "time_per_1k_s" not in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = df[n_col].astype(float) / 1000.0
            denom = denom.replace(0, np.nan)
            df["time_per_1k_s"] = df[runtime_col].astype(float) / denom
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", type=Path, default=Path("../report/figs/timing.csv"))
    ap.add_argument("--metrics", type=Path, default=Path("../results/metrics.csv"))
    ap.add_argument("--write-csv", action="store_true", help="Write helper CSVs to ./summaries/")
    args = ap.parse_args()

    timing = safe_read_csv(args.timing, "Timing")
    metrics = safe_read_csv(args.metrics, "Metrics")

    # Enrich timing with time_per_1k if possible
    timing = maybe_compute_time_per_k(timing)

    # Heuristic ID columns and numeric columns
    timing_id = detect_id_cols(timing)
    metrics_id = detect_id_cols(metrics)

    timing_num = numeric_cols(timing, exclude=timing_id)
    metrics_num = numeric_cols(metrics, exclude=metrics_id)

    # Common categorical keys to try for grouping
    timing_keys_pref = [k for k in ["dataset", "method", "n", "num_spikes", "spikes", "converged"] if k in timing.columns.str.lower()]
    # Map back to actual case of columns
    lower_to_real = {c.lower(): c for c in timing.columns}
    timing_keys = [lower_to_real[k] for k in timing_keys_pref if k in lower_to_real]

    metrics_keys_pref = [k for k in ["dataset", "method"] if k in metrics.columns.str.lower()]
    lower_to_real_m = {c.lower(): c for c in metrics.columns}
    metrics_keys = [lower_to_real_m[k] for k in metrics_keys_pref if k in lower_to_real_m]

    # Unique values for quick inspection
    timing_uniques = uniq_vals(timing, timing_id)
    metrics_uniques = uniq_vals(metrics, metrics_id)

    # Grouped summaries
    timing_by_n = group_summaries(timing, [c for c in timing.columns if c.lower() in {"n"}], timing_num)
    timing_by_n_spikes = group_summaries(
        timing,
        [c for c in timing.columns if c.lower() in {"n", "spikes", "num_spikes", "converged"}],
        timing_num,
    )

    metrics_by_ds_method = group_summaries(metrics, metrics_keys or [], metrics_num)
    metrics_by_method = group_summaries(metrics, [c for c in metrics.columns if c.lower() in {"method"}], metrics_num)
    metrics_by_dataset = group_summaries(metrics, [c for c in metrics.columns if c.lower() in {"dataset"}], metrics_num)

    # Build JSON report
    report = {
        "schema": {
            "timing_columns": timing.columns.tolist(),
            "metrics_columns": metrics.columns.tolist(),
        },
        "identifiers_detected": {
            "timing_id_cols": timing_id,
            "metrics_id_cols": metrics_id,
        },
        "numeric_detected": {
            "timing_numeric_cols": timing_num,
            "metrics_numeric_cols": metrics_num,
        },
        "unique_values": {
            "timing": timing_uniques,
            "metrics": metrics_uniques,
        },
        "summaries": {
            "timing_by_n": timing_by_n.to_dict(orient="records"),
            "timing_by_n_spikes_converged": timing_by_n_spikes.to_dict(orient="records"),
            "metrics_by_dataset_method": metrics_by_ds_method.to_dict(orient="records"),
            "metrics_by_method": metrics_by_method.to_dict(orient="records"),
            "metrics_by_dataset": metrics_by_dataset.to_dict(orient="records"),
        },
        "notes": {
            "time_per_1k_added": "time_per_1k_s" in timing.columns,
            "instructions": "Paste this entire JSON back to ChatGPT so I can generate LaTeX tables with exact values.",
        },
    }

    print("=== PASTE_THIS_JSON_TO_CHATGPT_START ===")
    print(json.dumps(report, indent=2))
    print("=== PASTE_THIS_JSON_TO_CHATGPT_END ===")

    if args.write_csv:
        outdir = Path("summaries"); outdir.mkdir(parents=True, exist_ok=True)
        timing_by_n.to_csv(outdir / "timing_by_n.csv", index=False)
        timing_by_n_spikes.to_csv(outdir / "timing_by_n_spikes_converged.csv", index=False)
        metrics_by_ds_method.to_csv(outdir / "metrics_by_dataset_method.csv", index=False)
        metrics_by_method.to_csv(outdir / "metrics_by_method.csv", index=False)
        metrics_by_dataset.to_csv(outdir / "metrics_by_dataset.csv", index=False)
        print(f"Wrote helper CSVs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
