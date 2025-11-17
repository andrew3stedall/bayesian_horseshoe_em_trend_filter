#!/usr/bin/env python3
# file: scripts/make_figs_from_yaml.py
from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np

# Local imports (aligned to your tree)
from src.experiments.results_loader import load_runs_jsonl

# --- Matplotlib (no seaborn) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yaml  # PyYAML
except ImportError as e:
    raise SystemExit("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from e


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _select_runs(df, dataset: str, methods: List[str], seed: Optional[int]) -> Dict[str, Dict[str, Any]]:
    """
    Return a dict method -> latest row (optionally filtered by seed).
    """
    out = {}
    for m in methods:
        dff = df[(df["dataset"] == dataset) & (df["method"] == m)]
        # if seed is not None:
        #     dff = dff[dff.get("spec_seed", dff.get("seed", None)) == seed]
        if dff.empty:
            print(f"[WARN] No run for dataset={dataset}, method={m}, seed={seed}")
            continue
        # take latest by timestamp
        row = dff.sort_values("timestamp").iloc[-1].to_dict()
        out[m] = row
    return out


def _plot_overlay_fits(cfg: Dict[str, Any], df, figs_dir: Path):
    title = cfg.get("title", "")
    dataset = cfg["dataset"]
    methods = cfg["methods"]
    seed = cfg.get("seed", None)
    out_path = figs_dir / cfg["output"]

    runs = _select_runs(df, dataset, methods, seed)
    if not runs:
        print(f"[SKIP] overlay_fits: no runs for dataset={dataset}")
        return

    # Expect y_obs / y_pred / x to be stored in JSONL lines (lists)
    # Plot y scatter and multiple y_pred lines
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    first = next(iter(runs.values()))
    x = np.asarray(first.get("x", np.arange(first["n"])))
    y = np.asarray(first.get("y", []))
    if y.size == 0:
        print("[WARN] overlay_fits: y_obs missing; plotting fits only")
    else:
        ax.scatter(x, y, s=12, alpha=0.30, label="observations")

    for m, r in runs.items():
        yp = np.asarray(r.get("y_pred", []))
        if yp.size == 0:
            print(f"[WARN] overlay_fits: missing y_pred for {m}; skipping")
            continue
        # Sort by x for clean polylines (in case x not monotone)
        idx = np.argsort(x)
        ax.plot(x[idx], yp[idx], linewidth=1.4, label=m)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", frameon=False)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] overlay_fits → {out_path}")


def _plot_runtime_bars(cfg: Dict[str, Any], df, figs_dir: Path):
    title = cfg.get("title", "")
    dataset = cfg["dataset"]
    out_path = figs_dir / cfg["output"]
    show_cv = bool(cfg.get("show_cv", True))

    dff = df[df["dataset"] == dataset]
    if dff.empty:
        print(f"[SKIP] runtime_bars: no runs for dataset={dataset}")
        return

    # Latest per method
    dff = dff.sort_values("timestamp").groupby("method", as_index=False).tail(1)

    methods = dff["method"].tolist()
    total = dff.get("time_wall_clock_sec", None)
    cv = dff.get("time_cv_time_sec", None)

    if total is None:
        print("[WARN] runtime_bars: time_wall_clock_sec missing; skipping")
        return

    total = total.to_numpy(dtype=float)
    cv_vals = cv.to_numpy(dtype=float) if show_cv and cv is not None else None

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    xpos = np.arange(len(methods))
    ax.bar(xpos, total, label="total")

    if cv_vals is not None:
        # hatched overlay for CV portion where available (NaN treated as 0)
        cv_plot = np.nan_to_num(cv_vals, nan=0.0)
        ax.bar(xpos, cv_plot, label="CV", hatch="///")

    ax.set_xticks(xpos, methods, rotation=20)
    ax.set_title(title)
    ax.set_ylabel("seconds")
    ax.legend(frameon=False)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] runtime_bars → {out_path}")


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _plot_beta_histograms(cfg: Dict[str, Any], df, figs_dir: Path):
    title = cfg.get("title", "")
    out_path = figs_dir / cfg["output"]
    bins = int(cfg.get("bins", 60))

    # locate run & trace npz
    npz_path = Path(cfg["trace_npz"])
    if not npz_path.exists():
        print(f"[SKIP] beta_histograms: npz not found: {npz_path}")
        return

    arrays = _load_npz(npz_path)
    # Flexible keys: support either 2D (iters x p) 'beta_list' or per-iter names
    iters = cfg["iters"]
    fig, axes = plt.subplots(1, len(iters), figsize=(4.0*len(iters), 3.1), sharey=True)
    if len(iters) == 1:
        axes = [axes]

    # Discover available iteration indexing
    # Preferred: a 2D array 'beta_list' with shape (T, p)
    if "beta_list" in arrays and arrays["beta_list"].ndim == 2:
        B = arrays["beta_list"]
        T = B.shape[0]
        def get_beta_t(ti: int) -> np.ndarray:
            return B[ti]
        final_idx = T - 1
    else:
        # Fall back: look for keys like 'beta_t0', 'beta_t1', ...
        keys = [k for k in arrays.keys() if re.match(r"beta_t\d+", k)]
        if not keys:
            print("[SKIP] beta_histograms: no per-iteration beta found in npz")
            return
        keys_sorted = sorted(keys, key=lambda k: int(k.split("beta_t")[-1]))
        key_to_arr = {k: arrays[k] for k in keys_sorted}
        def get_beta_t(ti: int) -> np.ndarray:
            key = f"beta_t{ti}"
            return key_to_arr[key]
        final_idx = int(keys_sorted[-1].split("beta_t")[-1])

    for ax, it in zip(axes, iters):
        ti = final_idx if (isinstance(it, str) and it.lower() == "final") else int(it)
        beta_t = get_beta_t(ti).ravel()
        ax.hist(beta_t, bins=bins)
        ax.set_title(f"iter {ti}")
        ax.set_xlabel(r"$\beta$")
        ax.grid(False)
    axes[0].set_ylabel("count")
    fig.suptitle(title)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] beta_histograms → {out_path}")


def _plot_bubble_frontier(cfg: Dict[str, Any], df, figs_dir: Path):
    title = cfg.get("title", "")
    dataset = cfg["dataset"]
    metric = cfg.get("metric", "mse")
    out_path = figs_dir / cfg["output"]

    dff = df[df["dataset"] == dataset].sort_values("timestamp").groupby("method", as_index=False).tail(1)
    if dff.empty:
        print(f"[SKIP] bubble_frontier: no runs for dataset={dataset}")
        return

    # Compose bubble size from sparsity proportion = beta_active / (n - (k+1))
    n = dff["n"].to_numpy()
    denom = (n).clip(min=1)
    beta_active = dff.get("beta_active", None)
    if beta_active is None:
        print("[WARN] bubble_frontier: beta_active missing; defaulting to size=constant")
        size = np.full(len(dff), 120.0)
    else:
        size = 400.0 * (beta_active.to_numpy() / denom)  # scale factor tweak as needed

    x = dff.get(f"metric_{metric}", None)
    if x is None:
        print(f"[SKIP] bubble_frontier: metric {metric} not found")
        return
    x = x.to_numpy(dtype=float)

    y = dff.get("time_wall_clock_sec", None)
    if y is None:
        print("[SKIP] bubble_frontier: time_wall_clock_sec not found")
        return
    y = y.to_numpy(dtype=float)

    labels = dff["method"].tolist()

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    sc = ax.scatter(x, y, s=size, alpha=0.65)
    for xi, yi, lab in zip(x, y, labels):
        ax.text(xi, yi, lab, fontsize=8, ha="left", va="bottom")
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("seconds (total)")
    ax.set_title(title)
    ax.grid(False)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] bubble_frontier → {out_path}")


def _plot_scaling(cfg: Dict[str, Any], df, figs_dir: Path):
    title = cfg.get("title", "")
    method = cfg["method"]
    dataset_regex = cfg["dataset_regex"]
    out_path = figs_dir / cfg["output"]
    loglog = bool(cfg.get("loglog", True))

    pat = re.compile(dataset_regex)
    dff = df[(df["method"] == method) & (df["dataset"].apply(lambda s: bool(pat.search(str(s)))))]

    if dff.empty:
        print(f"[SKIP] scaling: no runs for method={method} matching {dataset_regex}")
        return

    # Latest per dataset
    dff = dff.sort_values("timestamp").groupby("dataset", as_index=False).tail(1)
    n = dff["n"].to_numpy(dtype=float)
    y = dff.get("trace_iter_time_median_sec", None)
    if y is None:
        print("[SKIP] scaling: iter_time_median not found in runs.jsonl (trace_iter_time_median_sec)")
        return
    y = y.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(n, y, marker="o", linestyle="-")
    ax.set_xlabel("n")
    ax.set_ylabel("per-iteration median time (s)")
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(False)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] scaling → {out_path}")


def _plot_elbow(cfg: Dict[str, Any], df, figs_dir: Path):
    title = cfg.get("title", "")
    out_path = figs_dir / cfg["output"]

    npz_path = Path(cfg["trace_npz"])
    if not npz_path.exists():
        print(f"[SKIP] elbow: npz not found: {npz_path}")
        return
    arrays = _load_npz(npz_path)

    # Optional alpha per iteration
    alpha_iter = None
    if "alpha_list" in arrays and arrays["alpha_list"].ndim == 2:
        alpha_iter = arrays["alpha_list"]  # shape (T, n)

    # Optional per-iteration tau2, active
    tau2_path = arrays.get("tau2_path", None)
    active_path = arrays.get("active_path", None)  # if you saved it

    # Need y_obs for MSE(y, alpha_t); fetch from latest matching run
    dataset = cfg["dataset"]; method = cfg["method"]; seed = cfg.get("seed", None)
    runs = _select_runs(df, dataset, [method], seed)
    if not runs:
        print(f"[SKIP] elbow: no matching run for dataset={dataset}, method={method}")
        return
    run = list(runs.values())[0]
    y = np.asarray(run.get("y_obs", []))

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    plotted = False
    # 1) MSE vs iteration using alpha_list if available
    if alpha_iter is not None and y.size:
        T = alpha_iter.shape[0]
        mse = np.array([np.mean((y - alpha_iter[t])**2) for t in range(T)])
        ax.plot(np.arange(T), mse, label="MSE(y, α_t)")
        ax.set_xlabel("iteration")
        ax.set_ylabel("train MSE")
        plotted = True

    # 2) Secondary axes for tau2 and/or active set if present
    twin = None
    lines2 = []
    labels2 = []
    if tau2_path is not None:
        twin = twin or ax.twinx()
        t = np.arange(len(tau2_path))
        l2, = twin.plot(t, tau2_path, linestyle="--", label=r"$\tau^2$", alpha=0.7)
        lines2.append(l2); labels2.append(r"$\tau^2$")
    if active_path is not None:
        twin = twin or ax.twinx()
        t = np.arange(len(active_path))
        l2b, = twin.plot(t, active_path, linestyle=":", label="active set", alpha=0.7)
        lines2.append(l2b); labels2.append("active set")

    if twin is not None:
        twin.set_ylabel("auxiliary scale")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = lines2, labels2
        ax.legend(h1+h2, l1+l2, frameon=False, loc="best")
    else:
        if plotted:
            ax.legend(frameon=False, loc="best")

    if not plotted:
        ax.text(0.5, 0.5, "No per-iteration α found in trace.\nAdd 'alpha_list' to traces to enable MSE elbow.",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_title(title)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] elbow → {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Make figures from YAML spec.")
    ap.add_argument("--config", type=str, default="../configs/figures.yaml")
    args = ap.parse_args()

    config = _load_yaml(Path(args.config))
    io = config.get("io", {})
    runs_path = Path(io.get("runs_path", "results/runs.jsonl.gz"))
    figs_dir = Path(io.get("figs_dir", "results/figs"))

    df = load_runs_jsonl(runs_path)

    # Execute sections if present
    if "overlay_fits" in config:
        for block in config["overlay_fits"]:
            _plot_overlay_fits(block, df, figs_dir)

    if "runtime_bars" in config:
        for block in config["runtime_bars"]:
            _plot_runtime_bars(block, df, figs_dir)

    if "beta_histograms" in config:
        for block in config["beta_histograms"]:
            _plot_beta_histograms(block, df, figs_dir)

    if "bubble_frontiers" in config:
        for block in config["bubble_frontiers"]:
            _plot_bubble_frontier(block, df, figs_dir)

    if "scaling" in config:
        for block in config["scaling"]:
            _plot_scaling(block, df, figs_dir)

    if "elbows" in config:
        for block in config["elbows"]:
            _plot_elbow(block, df, figs_dir)


if __name__ == "__main__":
    main()
