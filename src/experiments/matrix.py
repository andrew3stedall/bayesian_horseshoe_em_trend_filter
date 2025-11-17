from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Iterable
import yaml
import random

@dataclass
class RunSpec:
    dataset_key: str
    dataset_label: str
    dataset_seed: int
    method_key: str
    method_label: str
    order: int
    params: Dict[str, Any]
    repetition: int
#
# @dataclass
# class DatasetSpec:
#     dataset_key: str
#     dataset_label: str
#     method_key: str
#     method_label: str
#     order: int
#     params: Dict[str, Any]
#     repetition: int

@dataclass
class ExperimentMatrix:
    runs: List[RunSpec]

    @staticmethod
    def from_yaml(path: str) -> "ExperimentMatrix":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        defaults = cfg.get("defaults", {})
        order = int(defaults.get("order", 1))
        reps_default = int(defaults.get("repetitions", 1))

        runs: List[RunSpec] = []
        for ds in cfg.get("datasets", []):
            key = ds["name"]
            label = ds.get("label", key)
            reps = int(ds.get("repetitions", reps_default))
            for r in range(1, reps + 1):
                seed = random.randint(1, 10000)
                for m in cfg.get("methods", []):
                    mkey = m["key"]
                    mlabel = m.get("label", mkey)
                    params = m.get("params", {}) or {}
                    runs.append(RunSpec(
                        dataset_key=key,
                        dataset_label=label,
                        dataset_seed=seed,
                        method_key=mkey,
                        method_label=mlabel,
                        order=order,
                        params=params,
                        repetition=r,
                    ))
        return ExperimentMatrix(runs=runs)
