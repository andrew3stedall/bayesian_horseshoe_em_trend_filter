# scripts/make_figures.py
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.figure.base import FigureSpec, Context
from src.figure.utils import load_yaml, import_object
from src.figure import __all__ as _  # ensure classes are importable

def main(cfg_path: str = "../configs/figures.yaml"):
    cfg = load_yaml(cfg_path)

    colours = cfg.get("defaults", {}).get("colours", ["red","blue","green","purple","orange"])
    output_cfg = cfg.get("output", {}) or {}
    out_dir = ROOT / output_cfg.get("figs_dir", "report/figs")
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = Context(
        root=ROOT,
        colours=colours,
        methods_cfg=cfg.get("methods", {}),
        datasets_cfg=cfg.get("datasets", {}),
        output_dir=out_dir,
        overwrite=bool(output_cfg.get("overwrite", True)),
        dpi=int(output_cfg.get("fig_dpi", 150)),
    )

    figures: Dict[str, Any] = cfg.get("figures", {})
    for key, node in figures.items():
        # print(node)
        cls_ref = node["cls"]
        title = node.get("title", key)
        method_title = node.get("method_title", key)
        params = node.get("params", {}) or {}
        spec = FigureSpec(key=key, title=title, cls_ref=cls_ref, params=params, method_title=method_title)

        Cls = import_object(cls_ref)
        fig_obj = Cls(spec, ctx)
        out_path = fig_obj.run()
        print(f"[OK] {key}: {out_path}")

if __name__ == "__main__":
    main()
