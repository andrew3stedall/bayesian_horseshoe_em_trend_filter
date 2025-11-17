from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .backend import FigureContext, save_figure
# from .runtime import ExperimentExecutor, synth_slug_from_n_sigma

# ------------------------------
# 1) Overlay fits
# ------------------------------
@dataclass
class OverlayFits:
    title: str
    methods: List[str]
    dataset: str
    seed: int = 1
    output: str = "overlay.png"

    def render(self, ctx: FigureContext):
        ex = ExperimentExecutor()
        fig, ax = plt.subplots(figsize=(8.6, 3.8))
        x = y = None
        for i, mslug in enumerate(self.methods):
            method_name = map_method(ctx.method_map, mslug)
            rr = ex.run(self.dataset, method_name, seed=self.seed, record_trace=False)
            if x is None:
                x = rr.x; y = rr.y
                ax.scatter(x, y, s=12, alpha=0.30, label="observations")
            colour = ctx.colours[i % len(ctx.colours)]
            idx = np.argsort(x)
            ax.plot(x[idx], rr.y_pred[idx], linewidth=1.6, label=method_name, color=colour)
        ax.set_title(self.title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(frameon=False, loc="best")
        save_figure(ctx, self.output, fig)

# ------------------------------
# 2) Beta histograms across iterations
# ------------------------------
@dataclass
class BetaHistograms:
    title: str
    method: str
    dataset: str
    seed: int = 1
    iterations: List[object] = field(default_factory=lambda: [1,2,5,10,"final"])
    bins: int = 60
    output: str = "beta_hist.png"

    def render(self, ctx: FigureContext):
        ex = ExperimentExecutor()
        method_name = map_method(ctx.method_map, self.method)
        rr = ex.run(self.dataset, method_name, seed=self.seed, record_trace=True)
        # Prefer per-iteration β; fallback to final β only
        if rr.beta_list is not None and rr.beta_list.ndim == 2:
            T = rr.beta_list.shape[0]
            def get_beta_t(ti: int) -> np.ndarray: return rr.beta_list[ti]
            final_idx = T - 1
        elif rr.beta_final is not None:
            def get_beta_t(_: int) -> np.ndarray: return rr.beta_final
            final_idx = 0
        else:
            print("[SKIP] BetaHistograms: no beta_list or beta_final returned by runner.")
            return

        iters = []
        for it in self.iterations:
            iters.append(final_idx if (isinstance(it, str) and it.lower() == "final") else int(it))

        fig, axes = plt.subplots(1, len(iters), figsize=(4.2*len(iters), 3.1), sharey=True)
        if len(iters) == 1: axes = [axes]
        for ax, ti in zip(axes, iters):
            beta_t = np.ravel(get_beta_t(ti))
            ax.hist(beta_t, bins=int(self.bins))
            ax.set_title(f"iter {ti}")
            ax.set_xlabel(r"$\beta$")
        axes[0].set_ylabel("count")
        fig.suptitle(self.title)
        save_figure(ctx, self.output, fig)

# ------------------------------
# 3) Scaling (median per-iteration time vs n)
# ------------------------------
@dataclass
class Scaling:
    title: str
    method: str
    n: List[int]
    sigma: float
    seed: int = 1
    output: str = "scaling.png"
    loglog: bool = True

    def render(self, ctx: FigureContext):
        ex = ExperimentExecutor()
        method_name = map_method(ctx.method_map, self.method)
        rows = []
        for nv in self.n:
            ds = synth_slug_from_n_sigma(nv, self.sigma)
            rr = ex.run(ds, method_name, seed=self.seed, record_trace=True, n_override=nv)
            # prefer per-iteration timings if present
            if rr.iter_times is not None and rr.iter_times.size:
                val = float(np.median(rr.iter_times))
            else:
                # rough fallback: total_time / iters
                val = float(rr.total_time_sec) / max(1, int(rr.iters or 1)) if rr.total_time_sec is not None else np.nan
            rows.append((nv, val))
        rows.sort()
        xs = np.array([r[0] for r in rows], float)
        ys = np.array([r[1] for r in rows], float)

        fig, ax = plt.subplots(figsize=(6.6, 4.0))
        ax.plot(xs, ys, marker="o", linestyle="-")
        ax.set_xlabel("n"); ax.set_ylabel("per-iteration median time (s)")
        if self.loglog:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(self.title)
        save_figure(ctx, self.output, fig)

# ------------------------------
# 4) Elbow (MSE vs iteration) + τ² / active overlay if present
# ------------------------------
@dataclass
class Elbow:
    title: str
    method: str
    dataset: str
    seed: int = 1
    output: str = "elbow.png"
    show: Dict[str, bool] = None  # {"mse_train": True, "tau2": True, "active_set": True}

    def render(self, ctx: FigureContext):
        ex = ExperimentExecutor()
        method_name = map_method(ctx.method_map, self.method)
        rr = ex.run(self.dataset, method_name, seed=self.seed, record_trace=True)

        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        plotted = False
        if (self.show or {}).get("mse_train", True) and rr.alpha_list is not None and rr.y is not None:
            T = rr.alpha_list.shape[0]
            mse = np.array([np.mean((rr.y - rr.alpha_list[t])**2) for t in range(T)])
            ax.plot(np.arange(T), mse, label="MSE(y, α_t)")
            ax.set_xlabel("iteration"); ax.set_ylabel("train MSE")
            plotted = True

        twin = None; lines2=[]; labels2=[]
        if (self.show or {}).get("tau2", False) and rr.tau2_path is not None:
            twin = twin or ax.twinx()
            t = np.arange(len(rr.tau2_path))
            l, = twin.plot(t, rr.tau2_path, linestyle="--", label=r"$\tau^2$", alpha=0.7)
            lines2.append(l); labels2.append(r"$\tau^2$")
        if (self.show or {}).get("active_set", False) and rr.active_path is not None:
            twin = twin or ax.twinx()
            t = np.arange(len(rr.active_path))
            l, = twin.plot(t, rr.active_path, linestyle=":", label="active set", alpha=0.7)
            lines2.append(l); labels2.append("active set")

        if twin is not None:
            twin.set_ylabel("auxiliary scale")
            h1, l1 = ax.get_legend_handles_labels()
            ax.legend(h1+lines2, l1+labels2, frameon=False, loc="best")
        elif plotted:
            ax.legend(frameon=False, loc="best")

        ax.set_title(self.title)
        save_figure(ctx, self.output, fig)

# ------------------------------
# 5) Bubble frontier (metric vs total time; size = sparsity proportion)
# ------------------------------
@dataclass
class BubbleFrontier:
    title: str
    dataset: str
    methods: List[str]
    metric: str = "mse"
    seed: int = 1
    output: str = "bubble.png"

    def render(self, ctx: FigureContext):
        ex = ExperimentExecutor()
        rows = []
        for mslug in self.methods:
            method_name = map_method(ctx.method_map, mslug)
            rr = ex.run(self.dataset, method_name, seed=self.seed, record_trace=True)
            # metric
            metric_val = None
            if rr.metrics and self.metric in rr.metrics:
                metric_val = float(rr.metrics[self.metric])
            else:
                # fallback: compute vs y
                metric_val = float(np.mean((rr.y - rr.y_pred)**2)) if self.metric == "mse" else float(np.mean(np.abs(rr.y - rr.y_pred)))
            # time
            tsec = float(rr.total_time_sec) if rr.total_time_sec is not None else (
                float(np.sum(rr.iter_times)) if rr.iter_times is not None else np.nan
            )
            # sparsity proportion ~ |{j: β_j != 0}| / p ; if beta_final missing and β-list exists, take final
            if rr.beta_final is not None:
                bfin = rr.beta_final
            elif rr.beta_list is not None:
                bfin = rr.beta_list[-1]
            else:
                bfin = None
            if bfin is not None and bfin.size:
                p = bfin.size
                active = int(np.count_nonzero(np.abs(bfin) > 0))
                sparsity_prop = active / max(1, p)
            else:
                sparsity_prop = 0.0

            rows.append({"method": method_name, "metric": metric_val, "time": tsec, "size": 400.0 * sparsity_prop})

        xs = np.array([r["metric"] for r in rows], float)
        ys = np.array([r["time"] for r in rows], float)
        sizes = np.array([r["size"] for r in rows], float)
        labels = [r["method"] for r in rows]

        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.scatter(xs, ys, s=sizes, alpha=0.65)
        for xi, yi, lab in zip(xs, ys, labels):
            ax.text(xi, yi, lab, fontsize=8, ha="left", va="bottom")
        ax.set_xlabel(self.metric.upper())
        ax.set_ylabel("seconds (total)")
        ax.set_title(self.title)
        save_figure(ctx, self.output, fig)
