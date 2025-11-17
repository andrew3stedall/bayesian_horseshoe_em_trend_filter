import os, sys
from pathlib import Path

from src.experiments.runner import run_all
import yaml

def main():
    matrix_yaml = '../configs/experiments.yaml'
    datasets_yaml = "../configs/datasets.yaml"

    with open(matrix_yaml, "r", encoding="utf-8") as f:
        mat_cfg = yaml.safe_load(f) or {}

    defaults = mat_cfg.get("defaults", {})
    out_cfg = mat_cfg.get("output", {})

    sparsity_rule = defaults.get("sparsity_threshold_rule", "1/(5*sqrt(n))")
    write_preds  = bool(out_cfg.get("write_preds", False))         # legacy
    overwrite    = bool(out_cfg.get("overwrite", False))
    save_traces  = bool(out_cfg.get("save_traces", True))
    traces_dir   = str(out_cfg.get("traces_dir", "results/traces"))
    trace_scale  = str(out_cfg.get("trace_scale", "original"))

    save_figs = bool(out_cfg.get("save_figs", True))
    figs_dir = str(out_cfg.get("figs_dir", "results/figs"))
    fig_scale = str(out_cfg.get("fig_scale", "original"))
    fig_dpi = int(out_cfg.get("fig_dpi", 150))

    run_all(
        matrix_yaml=str(matrix_yaml),
        datasets_yaml=str(datasets_yaml),
        output_dir=str(out_cfg.get("dir", "results")),
        sparsity_rule=sparsity_rule,
        write_preds=write_preds,
        overwrite=overwrite,
        save_traces=save_traces,
        traces_dir=traces_dir,
        trace_scale=trace_scale,
        save_figs=save_figs,
        figs_dir=figs_dir,
        fig_scale=fig_scale,
        fig_dpi=fig_dpi,
    )

if __name__ == "__main__":
    main()
