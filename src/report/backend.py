from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

@dataclass
class FigureContext:
    figs_dir: Path
    overwrite: bool
    # method_map: Dict[str, str]
    colours: List[str]

def map_method(method_map: Dict[str, str], key: str) -> str:
    return method_map.get(str(key), str(key))

def save_figure(ctx: FigureContext, filename: str, fig):
    out = ctx.figs_dir / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not ctx.overwrite:
        print(f"[SKIP] exists: {out}")
        return
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out}")
