# src/reporting/visualisations/beta_histograms.py
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .base import FigureBase
from .utils import make_dataset_from_ref, cartesian_grid, get_method_runner, diff_k, save_png

class BetaHistograms(FigureBase):
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
        p = self.spec.params
        order = int(p.get("order", 2))
        iters = p.get("iterations", [1, 2, 5, 10, 20, "final"])
        bins = int(p.get("bins", 60))
        out_name = p["output"]

        # Resolve dataset
        ds_name = p.get("dataset")
        ds_spec = self.ctx.datasets_cfg.get(ds_name)
        if ds_spec is None:
            ds_cls = p.get("dataset_cls"); ds_params = {k: v for k, v in p.items()
                if k not in ("method","dataset","dataset_cls","iterations","bins","output","title","order")}
            ds_spec = {"cls": ds_cls, "params": ds_params}
        data = make_dataset_from_ref(ds_spec)
        x = data.x
        y = data.y

        # Run HS–EM with different iteration caps
        runners = get_method_runner()
        hs_runner = runners[p.get("method","hs_em")]
        betas: List[np.ndarray] = []
        labels: List[str] = []
        for k in iters:
            params_base = dict(self.ctx.methods_cfg["hs_em"].get("params", {}))
            if k == "final":
                y_hat, meta = hs_runner(x, y, order, params_base)
            else:
                params_base["max_iter"] = int(k)
                y_hat, meta = hs_runner(x, y, order, params_base)
            beta = meta.get('beta')
            betas.append(beta)
            labels.append(str(k))

        return {"title": self.spec.title, "betas": betas, "labels": labels, "bins": bins, "output": out_name}

    def render(self, prepared: Dict[str, Any]) -> Path:
        betas = prepared["betas"]
        labels = prepared["labels"]
        bins = prepared["bins"]
        cols = self.ctx.colours
        ncols = min(3, len(labels)); nrows = int(np.ceil(len(labels)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), squeeze=True)
        for i, (beta, lab) in enumerate(zip(betas, labels)):
            ax = axes[i // ncols][i % ncols]
            active_beta = beta
            # active_beta = beta[np.abs(beta)>=float(1e-4)]
            print(len(active_beta))
            ax.hist(active_beta, bins=bins, alpha=0.8, edgecolor="none")
            ax.set_title(f"iter={lab}")
            ax.set_xlabel(r"$\beta$")
            ax.set_ylabel("count")
        fig.suptitle(prepared["title"])
        # tidy unused axes
        for j in range(len(labels), nrows*ncols):
            axes[j // ncols][j % ncols].axis("off")
        return save_png(fig, self.ctx.output_dir / prepared["output"], dpi=self.ctx.dpi, overwrite=self.ctx.overwrite)
