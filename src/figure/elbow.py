# src/reporting/visualisations/elbow.py
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .base import FigureBase
from .utils import make_dataset_from_ref, get_method_runner, mse, diff_k, save_png

class Elbow(FigureBase):
    """
    EM elbow: metric vs iteration. Params:
      - method: "hs_em"
      - dataset: name (or provide dataset_cls + params)
      - show: {mse_train: bool, tau2: bool, active_set: bool}
      - iters (optional): [1,2,3,5,10,20,50,100]
      - output: filename
      - order (optional): default 2
    Strategy: refit with different max_iter and record metrics.
    """
    def prepare(self) -> Dict[str, Any]:
        p = self.spec.params
        order = int(p.get("order", 2))
        out_name = p["output"]
        show = p.get("show", {"mse_train": True, "tau2": True, "active_set": True})
        iters = p.get("iters", [1,2,3,5,10,20,50,100])

        # dataset
        ds_name = p.get("dataset")
        ds_spec = self.ctx.datasets_cfg.get(ds_name)
        if ds_spec is None:
            ds_cls = p.get("dataset_cls"); ds_params = {k: v for k, v in p.items()
                       if k not in ("method","dataset","dataset_cls","iters","show","output","title","order")}
            ds_spec = {"cls": ds_cls, "params": ds_params}
        data = make_dataset_from_ref(ds_spec)
        x, y = data["x"], data["y"]

        runners = get_method_runner()
        runner = runners[p.get("method","hs_em")]

        mse_train, tau2_list, active_list = [], [], []
        for k in iters:
            params = dict(self.ctx.methods_cfg["hs_em"].get("params", {}))
            params["max_iter"] = int(k)
            y_hat, meta = runner(x, y, order, params)
            mse_train.append(mse(y, y_hat))
            tau2_list.append(float(meta.get("tau2", np.nan)))
            beta = diff_k(y_hat, order)
            active_list.append(int(np.sum(np.abs(beta) > 1.0/(5*np.sqrt(y.size)))))

        return {"iters": iters, "mse": mse_train, "tau2": tau2_list, "active": active_list,
                "show": show, "title": self.spec.title, "output": out_name}

    def render(self, prepared: Dict[str, Any]) -> Path:
        iters = prepared["iters"]; show = prepared["show"]
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        if show.get("mse_train", True):
            ax.plot(iters, prepared["mse"], label="MSE (train)", lw=2)
        if show.get("tau2", True):
            ax.plot(iters, prepared["tau2"], label=r"$\tau^2$", lw=2)
        if show.get("active_set", True):
            ax.plot(iters, prepared["active"], label="Active set size", lw=2)
        ax.set_xlabel("EM iterations"); ax.set_title(prepared["title"])
        ax.legend(loc="best", frameon=False)
        return save_png(fig, self.ctx.output_dir / prepared["output"], dpi=self.ctx.dpi, overwrite=self.ctx.overwrite)
