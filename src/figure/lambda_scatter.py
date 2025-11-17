# src/reporting/visualisations/beta_histograms.py
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, LogNorm

from .base import FigureBase
from .utils import make_dataset_from_ref, cartesian_grid, get_method_runner, diff_k, save_png
from ..prep import StandardLoader
from ..prep.transforms import ScaleX01, StandardizeTarget


class LambdaScatter(FigureBase):
    """
    Histograms of β across EM iterations for HS–EM.
    Params:
      - method: "hs_em"
      - dataset: name in ctx.datasets_cfg OR provide dataset_cls + params
      - iterations: [1,2,5,10,20,"final"]
      - bins: int
      - output: filename
      - order (optional): default 2
    Strategy: refit HS–EM with max_iter set to each 'k' and record beta.
    """
    def prepare(self) -> Dict[str, Any]:
        datasets_yaml = "../configs/datasets.yaml"
        loader = StandardLoader(
            registry_yaml=datasets_yaml,
            transforms=[ScaleX01(), StandardizeTarget()]
        )

        p = self.spec.params
        use_csv = bool(p.get("use_csv", True))
        output = p.get("output", "scaling.png")

        method_key = p.get("method", "hs_em")
        ds_key = p.get("dataset", "synth_spikes_s005")
        sigma = float(p.get("sigma", 0.05))

        em_iterations_list = [s for s in p["em_iterations_list"]]
        num_spikes = int(p.get("num_spikes", 3))
        bins = int(p.get("bins", 60))
        n = int(p.get("n", 1024))
        seed = int(p.get("seed", 42))

        runners = get_method_runner()
        run = runners[method_key]

        # run
        extra_params = {'n': n, 'num_spikes': num_spikes, 'seed': seed}
        print(f'Currently working on: {extra_params}')
        data = loader.load_with_params(ds_key, **extra_params)
        x = data.x
        y = data.y
        y_true = data.y_true
        order = 2

        lambdas: List[np.ndarray] = []
        labels: List[str] = []
        for k in em_iterations_list:
            params_base = dict(self.ctx.methods_cfg[method_key].get("params", {}))

            if k == "final":
                y_hat, meta = run(x, y, order, params_base)
            else:
                params_base["max_iter"] = int(k)
                y_hat, meta = run(x, y, order, params_base)
            lambdas.append(meta.get('lambda2'))
            labels.append(str(k))

        return {"title": self.spec.title, "lambdas": lambdas, "labels": labels, "bins": bins, "output": output}

    def render(self, prepared: Dict[str, Any]) -> Path:
        lambdas = prepared["lambdas"]
        labels = prepared["labels"]
        cols = self.ctx.colours

        nodes = [
            0,
            0.05,
            0.1,
            0.5,
            0.7,
            1.0,
        ]
        colors = [
            cols[2],
            cols[5],
            cols[4],
            cols[3],
            cols[0],
            cols[1],
        ]
        custom_cmap = LinearSegmentedColormap.from_list("lambda_values", list(zip(nodes, colors)))
        vmin = float(1e-10)
        vmax = float(1e1)
        norm = LogNorm(vmin=vmin, vmax=vmax)

        ncols = min(5, len(labels))
        nrows = int(np.ceil(len(labels)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3), squeeze=True)

        for i, (lam, lab) in enumerate(zip(lambdas, labels)):
            ax = axes[i // ncols][i % ncols]
            im = ax.scatter(np.arange(len(lam)), lam, c=lam, cmap=custom_cmap, s=np.minimum(np.maximum(lam,0.005) * 1000,50.0), alpha=0.5, norm=norm)
            ax.set_title(f"iter={lab}")
            ax.set_ylim([1e-10, 1e4])
            if i // ncols == 0:
                ax.set_xlabel(r"index")
            if i % ncols == 0:
                ax.set_ylabel(r"$\lambda$")
            ax.set_yscale("log")

        fig.suptitle(prepared["title"])

        plt.tight_layout()

        # tidy unused axes
        for j in range(len(labels), nrows*ncols):
            axes[j // ncols][j % ncols].axis("off")
        return save_png(fig, self.ctx.output_dir / prepared["output"], dpi=self.ctx.dpi, overwrite=self.ctx.overwrite)
