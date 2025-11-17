# hs_tf_project/src/registry/data_registry.py
from __future__ import annotations
import importlib
import os
from dataclasses import dataclass
from typing import Callable, Dict, Any, Mapping, Optional
from pathlib import Path

import yaml
from src.data.base import DataSource, DataBundle

Factory = Callable[[], DataSource]


@dataclass
class _Entry:
    name: str
    factory: Factory


class DataRegistry:
    """Simple registry mapping dataset names to factories."""

    def __init__(self) -> None:
        self._entries: Dict[str, _Entry] = {}

    def register(self, name: str, factory: Factory) -> None:
        if name in self._entries:
            raise KeyError(f"Dataset '{name}' is already registered")
        self._entries[name] = _Entry(name, factory)

    def register_cls(self, class_name: str, cls: type, **params: Any) -> None:
        def _factory() -> DataSource:
            return cls(**params)  # type: ignore

        self.register(class_name, _factory)

    def load(self, name: str) -> DataBundle:
        if name not in self._entries:
            raise KeyError(f"Dataset '{name}' is not registered")
        ds = self._entries[name].factory()
        return ds.load()

    def load_with_params(self, name: str, **params) -> DataBundle:
        ds = self.add_params(name, **params)
        # print(f'ds:{ds}')
        # print(self._entries[name])
        return ds.load()

    def add_params(self, name:str, **params) -> DataSource:
        if name not in self._entries:
            raise KeyError(f"Dataset '{name}' is not registered")
        ds = self._entries[name].factory()
        return ds.add_params(**params)


    def list(self) -> Dict[str, Dict[str, Any]]:
        return {k: {"name": v.name} for k, v in self._entries.items()}


def _import_from_string(path: str) -> type:
    mod, obj = path.rsplit(".", 1)
    module = importlib.import_module(mod)
    return getattr(module, obj)


def _resolve_paths_in_params(params: Any, base_dir: Path, keys=("path",)) -> Any:
    """
    Recursively resolve any dict item whose key is in `keys` as a filesystem path:
    - expandvars / expanduser
    - make absolute relative to YAML's directory
    """
    if isinstance(params, Mapping):
        out = {}
        for k, v in params.items():
            if k in keys and isinstance(v, str):
                s = os.path.expandvars(os.path.expanduser(v))
                p = Path(s)
                if not p.is_absolute():
                    p = (base_dir / p).resolve()
                out[k] = str(p)
            else:
                out[k] = _resolve_paths_in_params(v, base_dir, keys)
        return out
    elif isinstance(params, list):
        return [_resolve_paths_in_params(x, base_dir, keys) for x in params]
    else:
        return params


def load_registry_from_yaml(yaml_path: str | os.PathLike[str], seed: Optional[int]) -> DataRegistry:
    yaml_path = Path(yaml_path).resolve()
    base_dir = yaml_path.parent
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}

    reg = DataRegistry()
    datasets = spec.get("datasets", {}) or {}
    for name, node in datasets.items():
        cls_path = node["cls"]
        raw_params = node.get("params", {}) or {}
        params = _resolve_paths_in_params(raw_params, base_dir, keys=("path", "paths"))
        params['seed'] = seed
        # print(f'params:{params}')
        # print(f'type(params):{type(params)}')
        cls = _import_from_string(cls_path)
        reg.register_cls(name, cls, **params)
    return reg


def load_custom_registry_from_yaml(yaml_path: str | os.PathLike[str], **params) -> DataRegistry:
    yaml_path = Path(yaml_path).resolve()
    base_dir = yaml_path.parent
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}

    reg = DataRegistry()
    datasets = spec.get("datasets", {}) or {}
    for name, node in datasets.items():
        cls_path = node["cls"]
        raw_params = node.get("params", {}) or {}
        params = _resolve_paths_in_params(raw_params, base_dir, keys=("path", "paths"))
        custom_params = _resolve_paths_in_params(params, base_dir, keys=("path", "paths"))
        combined_params = {**params, **custom_params}
        # print(f'params:{params}')
        # print(f'type(params):{type(params)}')
        # print(f'custom_params:{custom_params}')
        # print(f'type(custom_params):{type(custom_params)}')
        # print(f'combined_params:{combined_params}')
        # print(f'type(combined_params):{type(combined_params)}')
        cls = _import_from_string(cls_path)
        reg.register_cls(name, cls, **combined_params)
    return reg
