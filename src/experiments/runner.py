from __future__ import annotations
from dataclasses import dataclass
import os, json, csv, time, shutil
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np

from src.prep.loader import StandardLoader
from src.prep.transforms import ScaleX01, StandardizeTarget
from .matrix import ExperimentMatrix, RunSpec
from .methods import (
    run_horseshoe_em,
    run_l1_em,
    run_l1_em_cv,
    run_l1_trendfilter_cv,
    run_bayesian_l1_gibbs,
    run_kernel_ridge,
)
from .metrics import compute_metrics
from .methods import diff_k
from .viz import save_fit_plot, save_fit_plot_multi

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _unique_path(base: Path) -> Path:
    """If base exists, add suffix -1, -2, ... to avoid overwrite."""
    if not base.exists():
        return base
    stem, suf = base.stem, base.suffix
    parent = base.parent
    k = 1
    while True:
        cand = parent / f"{stem}-{k}{suf}"
        if not cand.exists():
            return cand
        k += 1

def _save_trace(traces_dir: Path,
                basename: str,
                arrays: Dict[str, np.ndarray],
                side_meta: Dict[str, Any],
                allow_overwrite: bool) -> Tuple[Path, Path]:
    _ensure_dir(traces_dir)
    npz_path = traces_dir / f"{basename}.npz"
    json_path = traces_dir / f"{basename}.json"

    if not allow_overwrite:
        npz_path = _unique_path(npz_path)
        json_path = npz_path.with_suffix(".json")

    # Only numpy arrays go into npz
    np.savez_compressed(npz_path, **arrays)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(side_meta, f, ensure_ascii=False)

    return npz_path, json_path

def _method_switch(spec: RunSpec, x: np.ndarray, y: np.ndarray, y_true: np.ndarray):
    k = spec.method_key
    if k == "hs_em":
        alpha_hat, meta = run_horseshoe_em(y, spec.order, spec.params)
    elif k == "l1_em":
        alpha_hat, meta = run_l1_em(y, spec.order, spec.params)
    elif k == "l1_em_cv":
        alpha_hat, meta = run_l1_em_cv(y, spec.order, spec.params, y_true)
    elif k == "l1_tf_cv":
        alpha_hat, meta = run_l1_trendfilter_cv(x, y, spec.order, spec.params, y_true)
    elif k == "bayes_l1_gibbs":
        alpha_hat, meta = run_bayesian_l1_gibbs(x, y, spec.order, spec.params)
    elif k == "krr":
        y_hat, meta = run_kernel_ridge(x, y, spec.params)
        alpha_hat = y_hat  # align naming: alpha == fitted mean
    else:
        raise ValueError(f"Unknown method key: {k}")
    return alpha_hat, meta

# --- small helper: k-th finite difference ---
def _diff_k(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return arr.copy()
    return np.diff(arr, n=k, axis=-1)

@dataclass
class _SingleSpec:
    dataset_key: str
    dataset_label: str
    method_key: str
    method_label: str
    repetition: int
    seed: int
    order: int
    params: Dict[str, Any]

def run_all(matrix_yaml: str,
            datasets_yaml: str,
            output_dir: str,
            sparsity_rule: str,
            write_preds: bool,
            overwrite: bool,
            save_traces: bool,
            traces_dir: str,
            trace_scale: str,
            save_figs: bool = True,
            figs_dir: str = "results/figs",
            fig_scale: str = "original",
            fig_dpi: int = 150
            ):
    """
    trace_scale: 'original' | 'transformed' | 'both'
    fig_scale: 'original' | 'transformed' | 'both'
    overwrite controls metrics/runs and whether traces can replace same-named files.
    """
    mat = ExperimentMatrix.from_yaml(matrix_yaml)
    out_dir = Path(output_dir).resolve()
    _ensure_dir(out_dir)

    metrics_path = out_dir / "metrics2.csv"
    runs_path = out_dir / "runs2.jsonl"

    # Overwrite handling
    if overwrite:
        if metrics_path.exists():
            metrics_path.unlink()
        if runs_path.exists():
            runs_path.unlink()
        # Optionally clean traces dir if you want a fresh slate
        if save_traces:
            tdir = Path(traces_dir)
            if tdir.exists():
                # comment this out if you DON'T want to clean old traces
                shutil.rmtree(tdir)

    # Prepare CSV header if new
    if not metrics_path.exists():
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "dataset_key","dataset_label","method_key","method_label","repetition",
                "n","order","mse","mae","mse_true","mae_true","sparsity_count","runtime_s"
            ])

    # Loader with standard transforms (optional: scale X, standardize y)
    loader = StandardLoader(
        registry_yaml=datasets_yaml,
        transforms=[ScaleX01(), StandardizeTarget()]
    )

    print(f'Total iterations needed: {len(mat.runs)}')

    # Iterate runs
    for i, spec in enumerate(mat.runs, start=1):

        t_run = time.perf_counter()
        data = loader.load(spec.dataset_key, spec.dataset_seed)
        x, y, y_true = data.x, data.y, data.y_true  # transformed/working scale

        print(f"""
        {i}/{len(mat.runs)}
        Working on {spec.dataset_label}-{spec.method_label}:
            n:              {y.size}
            iteration:      {i}
        """)

        alpha_hat, meta_m = _method_switch(spec, x, y, y_true)

        rt = float(meta_m.get("runtime_s", (time.perf_counter() - t_run)))
        m = compute_metrics(y, alpha_hat, spec.order, sparsity_rule, rt)
        m_true = compute_metrics(y_true, alpha_hat, spec.order, sparsity_rule, rt)
        y_y_true = compute_metrics(y, y_true, spec.order, sparsity_rule, rt)

        # Save CSV row
        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                spec.dataset_key,
                spec.dataset_label,
                spec.method_key,
                spec.method_label,
                spec.repetition,
                y.size,
                spec.order,
                f"{m['mse']:.6g}",
                f"{m['mae']:.6g}",
                f"{m_true['mse']:.6g}",
                f"{m_true['mae']:.6g}",
                int(m["sparsity_count"]),
                f"{m['runtime_s']:.6g}",
            ])

        # Save JSONL (rich meta)
        run_rec: Dict[str, Any] = {
            "dataset_key": spec.dataset_key,
            "dataset_label": spec.dataset_label,
            "method_key": spec.method_key,
            "method_label": spec.method_label,
            "repetition": spec.repetition,
            "n": int(y.size),
            "order": int(spec.order),
            "metrics": m,
            "method_meta": meta_m,
            "trace_paths": None,
        }

        # Optional traces: x, y, y_hat (+ original scale and beta_hat)
        if save_traces:
            traces_root = Path(traces_dir).resolve()
            basename = f"{spec.dataset_key}__{spec.method_key}__r{spec.repetition}"

            # transformed-scale arrays
            beta_hat = diff_k(alpha_hat, spec.order)
            arrays_trans = {
                "x": x.astype(np.float64, copy=False),
                "y": y.astype(np.float64, copy=False),
                "y_hat": alpha_hat.astype(np.float64, copy=False),
                "beta_hat": beta_hat.astype(np.float64, copy=False),
            }

            saved = {}

            if trace_scale in ("transformed", "both"):
                npz_t, json_t = _save_trace(
                    traces_root,
                    basename + "__trans",
                    arrays_trans,
                    side_meta={
                        "dataset_key": spec.dataset_key,
                        "method_key": spec.method_key,
                        "repetition": spec.repetition,
                        "scale": "transformed",
                    },
                    allow_overwrite=overwrite,
                )
                saved["transformed"] = {"npz": str(npz_t), "json": str(json_t)}

            if trace_scale in ("original", "both"):
                y_orig = data.inverse_target(y)
                y_hat_orig = data.inverse_target(alpha_hat)
                arrays_orig = {
                    "x": x.astype(np.float64, copy=False),  # x may already be [0,1]; keep as-is
                    "y": y_orig.astype(np.float64, copy=False),
                    "y_hat": y_hat_orig.astype(np.float64, copy=False),
                    "beta_hat": beta_hat.astype(np.float64, copy=False),  # beta w.r.t alpha on transformed scale
                }
                npz_o, json_o = _save_trace(
                    traces_root,
                    basename + "__orig",
                    arrays_orig,
                    side_meta={
                        "dataset_key": spec.dataset_key,
                        "method_key": spec.method_key,
                        "repetition": spec.repetition,
                        "scale": "original",
                        "inverse_notes": "Reconstructed via PreparedData.inverse_target",
                    },
                    allow_overwrite=overwrite,
                )
                saved["original"] = {"npz": str(npz_o), "json": str(json_o)}

            run_rec["trace_paths"] = saved
            # print(run_rec)

        saved_figs = {}
        if save_figs:
            figs_root = Path(figs_dir).resolve()
            basename = f"{spec.dataset_key}__{spec.method_key}__r{spec.repetition}"
            # transformed
            if fig_scale in ("transformed", "both"):
                png_t = figs_root / (basename + "__trans.png")
                if not overwrite:
                    png_t = _unique_path(png_t)
                title = f"{spec.dataset_label} • {spec.method_label} • transformed"
                save_fit_plot(x, y, alpha_hat, png_t, y_true=y_true, title=title, dpi=fig_dpi)
                saved_figs["transformed"] = str(png_t)
            # original
            if fig_scale in ("original", "both"):
                png_o = figs_root / (basename + "__orig.png")
                # png_o = figs_root / (basename + "__orig.png")
                if not overwrite:
                    png_o = _unique_path(png_o)
                y_orig = data.inverse_target(y)
                y_hat_orig = data.inverse_target(alpha_hat)
                title = f"{spec.dataset_label} • {spec.method_label} • original"
                save_fit_plot(x, y_orig, y_hat_orig, png_o, y_true=y_true, title=title, dpi=fig_dpi)
                saved_figs["original"] = str(png_o)



        if save_figs:
            run_rec["fig_paths"] = saved_figs

        with runs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(run_rec) + "\n")

        print(f"[{i}/{len(mat.runs)}] {spec.dataset_label} | {spec.method_label} | r={spec.repetition}"
              f" -> mse={m['mse']:.4g}  spars={m['sparsity_count']}  time={m['runtime_s']:.3f}s")
