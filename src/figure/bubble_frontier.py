# src/reporting/visualisations/bubble_frontier.py
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .base import FigureBase
from .utils import make_dataset_from_ref, get_method_runner, mse, diff_k, sparsity_count, save_png

class BubbleFrontier(FigureBase):
    """
    Accuracy vs runtime vs sparsity bubble chart.
    Params:
      - dataset: name (or provide dataset_cls + params)
      - methods: ["hs_em","l1_tf_cv","krr",...]
      - metric: "mse" or "mae" (we'll do mse here)
      - output: filename
      - order (optional): default 2
    """
    def prepare(self) -> Dict[str, Any]:
        p = self.spec.params
        metric = p.get("metric", "mse")
        order = int(p.get("order", 2))
        out_name = p["output"]

        # dataset resolve
        ds_name = p.get("dataset")
        ds_spec = self.ctx.datasets_cfg.get(ds_name)
        if ds_spec is None:
            ds_cls = p.get("dataset_cls"); ds_params = {k: v for k, v in p.items()
                       if k not in ("methods","metric","dataset","dataset_cls","output","title","order")}
            ds_spec = {"cls": ds_cls, "params": ds_params}
        data = make_dataset_from_ref(ds_spec)
        x, y = data["x"], data["y"]

        # methods
        methods = p.get("methods", [])
        runners = get_method_runner()
        recs = []
        for mk in methods:
            runner = runners[mk]
            y_hat, meta = runner(x, y, order, self.ctx.methods_cfg[mk].get("params", {}))
            beta = diff_k(y_hat, order)
            m = mse(y, y_hat) if metric == "mse" else float("nan")
            rt = float(meta.get("runtime_s", np.nan))
            spars = sparsity_count(beta, y.size)
            recs.append({"label": self.ctx.methods_cfg[mk].get("label", mk),
                         "mse": m, "runtime": rt, "spars": spars})
        return {"recs": recs, "metric": metric, "title": self.spec.title, "output": out_name}

    def render(self, prepared: Dict[str, Any]) -> Path:
        recs = prepared["recs"]
        fig = plt.figure(figsize=(6.4, 3.8))
        ax = fig.add_subplot(111)
        for i, r in enumerate(recs):
            ax.scatter(r["runtime"], r["mse"], s=20 + 6*r["spars"], alpha=0.6,
                       label=r["label"], edgecolor="none")
            ax.annotate(r["label"], (r["runtime"], r["mse"]), xytext=(4, 4),
                        textcoords="offset points", fontsize=8)
        ax.set_xlabel("runtime (s)"); ax.set_ylabel("MSE"); ax.set_title(prepared["title"])
        ax.grid(True, alpha=0.2)
        return save_png(fig, self.ctx.output_dir / prepared["output"], dpi=self.ctx.dpi, overwrite=self.ctx.overwrite)
