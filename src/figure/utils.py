# src/reporting/visualisations/utils.py
from __future__ import annotations
import importlib, itertools, json, time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable, Optional
import numpy as np
import yaml

# Headless-safe Matplotlib
import matplotlib
try:
    if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
        matplotlib.use("Agg", force=True)
except Exception:
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# --- YAML loader ---

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# --- Dynamic import ---

def import_object(dotted_path: str):
    mod, _, name = dotted_path.rpartition(".")
    if not mod:
        raise ValueError(f"Invalid dotted path: {dotted_path}")
    module = importlib.import_module(mod)
    return getattr(module, name)

# --- Dataset maker ---

def make_dataset_from_ref(ds_ref: Dict[str, Any]) -> Dict[str, Any]:
    """
    ds_ref = {"cls": "src.data.synthetic.SyntheticSpikes", "params": {...}}
    The class should expose either .generate(**params) or .load(**params).
    Returns dict with at least "x" and "y".
    """
    cls_path = ds_ref["cls"]
    params = ds_ref.get("params", {}) or {}
    DataClass = import_object(cls_path)
    obj = DataClass(**params) if callable(DataClass) else DataClass  # allow class or module-ns factory
    if hasattr(obj, "generate"):
        return obj.generate()
    if hasattr(obj, "load"):
        return obj.load()
    raise TypeError(f"{cls_path} must expose .generate(**params) or .load(**params)")

# --- Param grid expansion ---

def cartesian_grid(param_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    keys = list(param_dict.keys())
    vals = [v if isinstance(v, list) else [v] for v in param_dict.values()]
    combos = []
    for tpl in itertools.product(*vals):
        combos.append({k: tpl[i] for i, k in enumerate(keys)})
    return combos

# --- Method runners (wrap your existing experiment methods) ---

def get_method_runner():
    """
    Returns a dict mapping method keys -> runner func(X, y, order, params) -> (y_hat, meta)
    """
    from src.experiments.methods import (
        run_horseshoe_em, run_l1_em, run_l1_em_cv, run_l1_trendfilter_cv, run_bayesian_l1_gibbs, run_kernel_ridge
    )
    def hs(y, order, params, y_true:Optional):
        y_hat, meta = run_horseshoe_em(y, order, params)
        return y_hat, meta
    def l1_em(y, order, params, y_true:Optional):
        y_hat, meta = run_l1_em(y, order, params)
        return y_hat, meta
    def l1_em_cv(y, order, params, y_true:Optional):
        y_hat, meta = run_l1_em_cv(y, order, params, y_true)
        return y_hat, meta
    def l1(x, y, order, params, y_true:Optional):
        y_hat, meta = run_l1_trendfilter_cv(x, y, order, params, y_true)
        return y_hat, meta
    def gibbs(x, y, order, params, y_true:Optional):
        y_hat, meta = run_bayesian_l1_gibbs(x, y, order, params)
        return y_hat, meta
    def krr(x, y, params, y_true:Optional):
        y_hat, meta = run_kernel_ridge(x, y, params)
        return y_hat, meta

    return {
        "hs_em": lambda x, y, order, params, y_true: hs(y, order, params, y_true),
        "l1_em": lambda x, y, order, params, y_true: l1_em(y, order, params, y_true),
        "l1_em_cv": lambda x, y, order, params, y_true: l1_em_cv(y, order, params, y_true),
        "l1_tf_cv": lambda x, y, order, params, y_true: l1(x, y, order, params, y_true),
        "bayes_l1_gibbs": lambda x, y, order, params, y_true: gibbs(x, y, order, params, y_true),
        "krr": lambda x, y, order, params, y_true: krr(x, y, params, y_true),
    }

# --- Plot helpers ---

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_png(fig: plt.Figure, path: Path, dpi: int = 150, overwrite: bool = True) -> Path:
    ensure_dir(path.parent)
    if not overwrite:
        k = 1
        base = path
        while path.exists():
            path = base.with_name(f"{base.stem}-{k}{base.suffix}")
            k += 1
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path

# --- Metrics & beta utilities ---

def mse(y, yhat) -> float:
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(np.mean((y - yhat) ** 2))

def sparsity_count(beta: np.ndarray, n: int, rule: str = "1/(5*sqrt(n))") -> int:
    thr = 1.0 / (5.0 * np.sqrt(n)) if rule == "1/(5*sqrt(n))" else 0.0
    return int(np.sum(np.abs(beta) >= thr))

def diff_k(alpha: np.ndarray, order: int) -> np.ndarray:
    return np.diff(alpha, n=order)
