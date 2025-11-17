# src/reporting/visualisations/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

@dataclass
class FigureSpec:
    key: str
    title: str
    method_title: str
    cls_ref: str
    params: Dict[str, Any]

@dataclass
class Context:
    root: Path
    colours: List[str]
    methods_cfg: Dict[str, Dict[str, Any]]
    datasets_cfg: Dict[str, Dict[str, Any]]
    output_dir: Path
    overwrite: bool
    dpi: int

class FigureBase:
    """Abstract base for all figure types."""
    def __init__(self, spec: FigureSpec, ctx: Context):
        self.spec = spec
        self.ctx = ctx

    def prepare(self) -> Dict[str, Any]:
        """Collect data needed to render the figure (run methods, assemble series)."""
        raise NotImplementedError

    def render(self, prepared: Dict[str, Any]) -> Path:
        """Render the figure to a PNG and return the path."""
        raise NotImplementedError

    def run(self) -> Path:
        prepared = self.prepare()
        return self.render(prepared)
