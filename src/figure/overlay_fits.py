# src/reporting/visualisations/overlay_fits.py
from __future__ import annotations
from ..experiments.metrics import compute_metrics
from typing import Dict, Any, List
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .base import FigureBase
from .utils import (
    make_dataset_from_ref, cartesian_grid, get_method_runner,
    save_png
)


class OverlayFits(FigureBase):
    """
    Overlay multiple method fits (optionally across multiple seeds) on the same synthetic dataset.
    Expected params:
      - methods: [method_keys]
      - dataset: name in ctx.datasets_cfg OR provide dataset_cls + params
      - output: filename
      - seeds (optional): [1,3,5] to override/augment dataset params grid
      - order (optional): int (default 2 for piecewise-linear)
    """

    def prepare(self) -> Dict[str, Any]:
        p = self.spec.params

        overwrite = bool(p.get("overwrite", False))
        order = int(p.get("order", 2))
        out_name = p["output"]
        sigma = p.get("sigma")
        num_spikes = p.get("num_spikes")
        # Resolve dataset spec
        ds_name = p.get("dataset")
        ds_spec = self.ctx.datasets_cfg.get(ds_name)
        if ds_spec is None:
            # allow inline dataset reference inside figure params: {dataset_cls, ...}
            ds_cls = p.get("dataset_cls")
            ds_params = {k: v for k, v in p.items() if
                         k not in ("methods", "output", "title", "order", "dataset", "dataset_cls")}
            ds_spec = {"cls": ds_cls, "params": ds_params}
        # Expand seeds grid if provided in figure params
        fig_seeds = p.get("seeds")

        base = ds_spec.get("params", {})
        base['sigma'] = sigma
        if num_spikes is not None:
            base['num_spikes'] = num_spikes
        if fig_seeds:
            combos = cartesian_grid({**base, "seed": fig_seeds})
        else:
            combos = cartesian_grid(base) if base else [{}]

        ds_refs = [{"cls": ds_spec["cls"], "params": c} for c in combos]

        methods = p.get("methods", [])
        runners = get_method_runner()

        groups: Dict[str, Dict[str, Any]] = {}
        output_dir = '../report'
        out_dir = Path(output_dir).resolve()
        metrics_path = out_dir / "overlay_metrics.csv"

        if overwrite:
            if metrics_path.exists():
                metrics_path.unlink()

        if not metrics_path.exists():
            with metrics_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "dataset_label", "method_key", "seed", "sigma", "mse_true", "mae_true", "sparsity_count", "runtime_s", "spikes"
                ])

        for di, ds_ref in enumerate(ds_refs):
            print(f'ds_ref:{ds_ref}')
            ds_params = ds_ref.get("params", {})

            data = make_dataset_from_ref(ds_ref)  # {"x","y",...}
            x = data.x
            y = data.y
            y_true = data.y_true

            # Extract seed value for grouping; default to "noseed" if not present
            seed_val = ds_ref.get("params", {}).get("seed", None)
            seed_key = str(seed_val) if seed_val is not None else "noseed"

            if seed_key not in groups:
                groups[seed_key] = {"x": x, "y": y, "y_true": y_true, "series": []}

            for mi, mk in enumerate(methods):
                run = runners[mk]
                y_hat, meta = run(x, y, order, self.ctx.methods_cfg[mk].get("params", {}), y_true)
                label = f"{self.ctx.methods_cfg[mk].get('label', mk)}"

                m = compute_metrics(y, y_hat, 2, "1/(5*sqrt(n))", 0)
                m_true = compute_metrics(y_true, y_hat, 2, "1/(5*sqrt(n))", 0)

                groups[seed_key]["series"].append({
                    "y_hat": y_hat,
                    "label": label,
                    "method_key": mk,
                    "meta": meta
                })


                # Save CSV row
                with metrics_path.open("a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        ds_params.get('name'),
                        mk,
                        ds_params.get('seed'),
                        ds_params.get('sigma'),
                        f"{m_true['mse']:.6g}",
                        f"{m_true['mae']:.6g}",
                        int(m["sparsity_count"]),
                        f"{meta['runtime_s']:.6g}",
                        ds_params.get('num_spikes'),
                    ])
        return {
            "title": self.spec.title,
            "groups": groups,
            "output_base": out_name,
            "order": order,
            "sigma": sigma,
            "num_spikes": num_spikes,
        }

    def render(self, prepared: Dict[str, Any]) -> List[Path]:
        title = prepared["title"]
        groups = prepared["groups"]
        num_spikes = prepared["num_spikes"]
        sigma = prepared["sigma"]
        out_base = prepared["output_base"]
        colours = self.ctx.colours

        saved_paths: List[Path] = []

        ncols = min(3, len(groups))
        # ncols = min(2, len(groups))
        nrows = int(np.ceil(len(groups) / ncols))

        print(f'rows:{nrows}, cols:{ncols}')

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=True)

        if num_spikes is not None:
            fig.suptitle(f"Model Smoothing Comparison - (Synthetic {num_spikes} spikes - " + r"$\sigma^2=$" + f"{sigma})")
        else:
            fig.suptitle(f"Model Smoothing Comparison - ({title} - " + r"$\sigma^2=$" + f"{sigma})")

        plt.subplots_adjust(
            left=0.05,  # the left side of the subplots of the figure
            right=0.95,  # the right side of the subplots of the figure
            bottom=0.05,  # the bottom of the subplots of the figure
            top=0.9,  # the top of the subplots of the figure
            wspace=0.01,  # the amount of width reserved for blank space between subplots
            hspace=0.01  # the amount of height reserved for white space between subplots
        )

        # One figure per seed group
        for gi, (seed_key, g) in enumerate(groups.items()):
            x = g["x"]
            y = g["y"]
            y_true = g["y_true"]
            series = g["series"]

            print(f'gi // ncols:{gi // ncols} - gi % ncols: {gi % ncols}')
            fig_solo = plt.figure(figsize=(6, 6))
            ax = axes[gi % ncols]
            # ax = axes[gi // ncols][gi % ncols]
            ax_solo = fig_solo.add_subplot(111)

            # Lines for each method on this seed
            for i, s in enumerate(series):
                # Scatter once per seed
                ax.plot(x, s["y_hat"] - i * 1.5, lw=1.6, alpha=1, color=colours[i % len(colours)], label=rf'{s["label"]}',
                        zorder=3)
                ax.plot(x, y_true - i * 1.5, lw=1, alpha=1, color='grey', zorder=2)
                ax.scatter(x, y - i * 1.5, s=2, alpha=0.1, color='grey', zorder=1)
                # ax.scatter(x, y - i * 1.5, s=3, alpha=0.3, color=colours[i % len(colours)], zorder=1)

                # ax.scatter(x, y - i * 1.5, s=3, alpha=0.2, lw=0.1, color=colours[i % len(colours)])
                ax_solo.plot(x, s["y_hat"] - i * 1.5, lw=1.5, color=colours[i % len(colours)], label=s["label"])
                ax_solo.scatter(x, y - i * 1.5, s=3, alpha=0.2, lw=0.1, color=colours[i % len(colours)])

            # Title / labels
            seed_suffix = "" if seed_key == "noseed" else f" (seed={seed_key})"
            ax_solo.set_title(f"{seed_suffix}")
            # ax.set_title(f"{seed_suffix}")
            # ax.set_xlabel("x")
            ax_solo.set_xlabel("x")
            ax.yaxis.set_visible(False)
            ax.xaxis.set_visible(False)

            if gi // ncols == nrows - 1:
                ax.set_xlabel(r"$x$")

            ax_solo.legend(loc="best", frameon=False, fontsize=8)
            if gi == 1:
                ax.legend(loc="best", frameon=False, fontsize=8)

            plt.tight_layout()

            # Save with seed_ prefix
            out_name = f"{seed_key}_{out_base}" if seed_key != "noseed" else out_base
            out_path = self.ctx.output_dir / out_name
            saved_paths.append(save_png(fig_solo, out_path, dpi=self.ctx.dpi, overwrite=self.ctx.overwrite))

        if num_spikes is not None:
            out_name = f"{out_base}_sigma{int(sigma * 100)}_spikes{num_spikes}.png"
        else:
            out_name = f"{out_base}_sigma{int(sigma * 100)}.png"
        out_path = self.ctx.output_dir / out_name
        saved_paths.append(save_png(fig, out_path, dpi=self.ctx.dpi, overwrite=self.ctx.overwrite))

        return saved_paths
