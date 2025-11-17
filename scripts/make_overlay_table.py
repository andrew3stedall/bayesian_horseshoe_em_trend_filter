import math
from typing import Tuple, Dict

import numpy as np
import pandas as pd


def build_spikes_model_table(
    csv_path: str,
    dataset_label: str = "synth_spikes",
    sigma_value: float = None,
    spikes: int = None,
    sig: int = 3,
    target_range=(1.0, 100.0),
    model_map: Dict[str, str] | None = None,
    add_thin_group_rules: bool = True,
) -> Tuple[str, dict]:

    # ----------------------------- helpers -----------------------------------
    def _normalize_label_scalar(s: str) -> str:
        s = "" if s is None else str(s)
        return "".join(ch for ch in s.lower() if ch.isalnum())

    def _normalize_label_series(s: pd.Series) -> pd.Series:
        return s.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)

    def _to_camel(s: str) -> str:
        """Turn 'foo_bar_baz' -> 'FooBarBaz'."""
        parts = [p for p in str(s).replace("-", "_").split("_") if p]
        return "".join(p[:1].upper() + p[1:] for p in parts) if parts else str(s)

    def _se(x: pd.Series) -> float:
        n = int(x.shape[0])
        if n <= 1:
            return float("nan")
        sd = float(x.std(ddof=1))
        return sd / (n ** 0.5)

    def _sd(x: pd.Series) -> float:
        if x.shape[0] <= 1:
            return float("nan")
        return float(x.std(ddof=1))

    def _choose_scale(values: np.ndarray, target=(1.0, 1000.0)) -> int:
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            return 0
        gmean = float(np.exp(np.mean(np.log(vals))))
        p = 0
        while gmean < target[0]:
            gmean *= 10.0
            p += 1
        while gmean >= target[1]:
            gmean /= 10.0
            p -= 1
        return p

    def _fmt(x: float, _sig: int = 3) -> str:
        if not (isinstance(x, (int, float)) and math.isfinite(x)):
            return "--"
        return f"{x:.{_sig}f}"

    # ----------------------- model display names ------------------------------
    # Your requested names:
    #   hs_em                -> Bayesian Horseshoe Trend Filter (BHS-TF)
    #   bayes_l1_em|l1_em   -> Bayesian \ell_1 Trend Filter (BL1-TF)
    #   l1_tf_cv|tf_l1      -> Frequentist \ell_1 Trend Filter (FL1-TF)
    #   krr                 -> Kernel Regression (KRR)
    default_map = {
        "hs_em": r"BHS-TF",
        "bayes_l1_em": r"B$\ell_1$-TF",
        "l1_em": r"B$\ell_1$-TF",
        "l1_em_cv": r"B$\ell_1$-TF",
        "l1_tf_cv": r"F$\ell_1$-TF",
        "krr": r"KRR",
    }
    MODEL_MAP = (model_map or {}) | default_map  # allow user overrides

    # --------------------------- load & filter --------------------------------
    df = pd.read_csv(csv_path)

    # Robust dataset_label filter; caption uses CamelCase display
    want = _normalize_label_scalar(dataset_label)
    df = df[_normalize_label_series(df["dataset_label"]) == want]

    # sigma filter
    if sigma_value is not None:
        df = df[np.isclose(df["sigma"].astype(float), float(sigma_value))]

    if spikes is not None:
        df = df[np.isclose(df["spikes"].astype(int), int(spikes))]

    if df.empty:
        raise ValueError(
            f"No rows after filtering: dataset_label={dataset_label!r}, sigma={sigma_value}"
        )

    # ------------------------- aggregate over seeds ---------------------------
    if sigma_value is not None:
        grouped = (
            df.groupby(["spikes", "method_key"])
            .agg(
                mse_mean=("mse_true", "mean"),
                mse_se=("mse_true", _se),
                spars_mean=("sparsity_count", "mean"),
                spars_sd=("sparsity_count", _sd),
                time_mean=("runtime_s", "mean"),
                time_sd=("runtime_s", _sd),
            )
            .reset_index()
            .sort_values(["spikes", "method_key"])
        )
    elif spikes is not None:
        grouped = (
            df.groupby(["sigma", "method_key"])
            .agg(
                mse_mean=("mse_true", "mean"),
                mse_se=("mse_true", _se),
                spars_mean=("sparsity_count", "mean"),
                spars_sd=("sparsity_count", _sd),
                time_mean=("runtime_s", "mean"),
                time_sd=("runtime_s", _sd),
            )
            .reset_index()
            .sort_values(["sigma", "method_key"])
        )
    else:
        grouped = (
            df.groupby(["sigma", "method_key"])
            .agg(
                mse_mean=("mse_true", "mean"),
                mse_se=("mse_true", _se),
                spars_mean=("sparsity_count", "mean"),
                spars_sd=("sparsity_count", _sd),
                time_mean=("runtime_s", "mean"),
                time_sd=("runtime_s", _sd),
            )
            .reset_index()
            .sort_values(["sigma", "method_key"])
        )

    # Decide scaling for MSE (×10^p)
    mse_scale_p = _choose_scale(grouped["mse_mean"].values, target=target_range)

    # Time unit: s or ms
    with np.errstate(invalid="ignore"):
        tmax = np.nanmax(grouped["time_mean"].values)
        tsdmax = np.nanmax(grouped["time_sd"].values)
    time_unit, time_factor = ("ms", 1000.0) if (
        np.isfinite(tmax) and tmax < 10.0 and np.isfinite(tsdmax) and tsdmax < 1.0
    ) else ("s", 1.0)

    # ------------------------ build table body rows ---------------------------
    lines = []
    if sigma_value is not None:
        groups = list(grouped.groupby("spikes", sort=True))

    elif spikes is not None:
        groups = list(grouped.groupby("sigma", sort=True))

    else:
        groups = list(grouped.groupby("sigma", sort=True))

    n_groups = len(groups)

    for gi, (grp, sub) in enumerate(groups):
        sub = sub.reset_index(drop=True)
        nrows = sub.shape[0]
        for j, row in sub.iterrows():
            if sigma_value is not None:
                grp_cell = rf"\multirow{{{nrows}}}{{*}}{{{int(grp)}}}" if j == 0 else ""
            elif spikes is not None:
                grp_cell = rf"\multirow{{{nrows}}}{{*}}{{{float(grp)}}}" if j == 0 else ""
            else:
                grp_cell = rf"\multirow{{{nrows}}}{{*}}{{{float(grp)}}}" if j == 0 else ""


            # Model display name
            key = str(row["method_key"])
            model_disp = MODEL_MAP.get(key, _to_camel(key))

            # MSE Mean | SE  (with scaling)
            mse_mean_scaled = row["mse_mean"] * (10 ** mse_scale_p)
            mse_se_scaled = row["mse_se"] * (10 ** mse_scale_p) if math.isfinite(row["mse_se"]) else float("nan")
            mse_mean_cell = _fmt(mse_mean_scaled, sig)
            mse_se_cell = _fmt(mse_se_scaled, sig)

            # Sparsity Mean | SD
            spars_mean_cell = _fmt(row["spars_mean"], sig)
            spars_sd_cell = _fmt(row["spars_sd"], sig)

            # Time Mean | SD (with unit)
            tmean = row["time_mean"] * time_factor
            tsd = row["time_sd"] * time_factor
            time_mean_cell = _fmt(tmean, sig)
            time_sd_cell = _fmt(tsd, sig)

            lines.append(
                " & ".join([
                    grp_cell,
                    model_disp,
                    mse_mean_cell, mse_se_cell,
                    spars_mean_cell, spars_sd_cell,
                    time_mean_cell, time_sd_cell
                ]) + r" \\"
            )

        # thin ruled line between parent groups (across all 8 columns)
        if add_thin_group_rules and gi < n_groups - 1:
            # small vertical breathing room + thin rule
            lines.append(r"\addlinespace[2pt]")
            lines.append(r"\cmidrule(lr){1-8}")

    # ----------------------------- LaTeX header -------------------------------
    # Top row with grouped headers and subcolumns aligned
    mse_head = r"MSE (SE)" + (rf" $\times 10^{{{mse_scale_p}}}$" if mse_scale_p != 0 else "")
    dataset_camel = _to_camel(dataset_label)

    header = (
        r"\begin{table}[H]" "\n"
        r"\centering" "\n"
        r"\small" "\n"
        r"\begin{tabular}{l l rr rr rr}" "\n"
        r"\toprule" "\n"
        r"\# Spikes & Model & \multicolumn{2}{c}{" + mse_head + r"} & "
        r"\multicolumn{2}{c}{Sparsity (SD)} & "
        r"\multicolumn{2}{c}{Time (" + time_unit + r") (SD)} \\" "\n"
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}" "\n"
        r" &  & Mean & SE & Mean & SD & Mean & SD \\" "\n"
        r"\midrule"
    )

    body = "\n".join(lines)

    footer = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        rf"\caption{{Contrast of models by {'spike count' if sigma_value is not None else rf'noise level'}" 
        rf" for synthetic data at fixed {rf'noise level $\sigma^2={sigma_value}$' if sigma_value is not None else rf'spike count={spikes}'}"
        r"MSE columns report mean and standard error over seeds; Sparsity and Time report mean and sample standard deviation.}" "\n"
        rf"\label{{tab:{'spikes' if sigma_value is not None else 'noise'}-model-contrast}}" "\n"
        r"\end{table}"
    )

    latex = "\n".join([header, body, footer])
    meta = {"mse_scale_power": int(mse_scale_p), "time_unit": time_unit}
    return latex, meta

def build_spikes_compare_table(
    csv_path: str,
    dataset_label: str = "synth_spikes",
    sig: int = 3,
    target_range=(1.0, 100.0),
    model_map: Dict[str, str] | None = None,
    add_thin_group_rules: bool = True,
) -> Tuple[str, dict]:

    # ----------------------------- helpers -----------------------------------
    def _normalize_label_scalar(s: str) -> str:
        s = "" if s is None else str(s)
        return "".join(ch for ch in s.lower() if ch.isalnum())

    def _normalize_label_series(s: pd.Series) -> pd.Series:
        return s.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)

    def _to_camel(s: str) -> str:
        """Turn 'foo_bar_baz' -> 'FooBarBaz'."""
        parts = [p for p in str(s).replace("-", "_").split("_") if p]
        return "".join(p[:1].upper() + p[1:] for p in parts) if parts else str(s)

    def _se(x: pd.Series) -> float:
        n = int(x.shape[0])
        if n <= 1:
            return float("nan")
        sd = float(x.std(ddof=1))
        return sd / (n ** 0.5)

    def _sd(x: pd.Series) -> float:
        if x.shape[0] <= 1:
            return float("nan")
        return float(x.std(ddof=1))

    def _choose_scale(values: np.ndarray, target=(1.0, 1000.0)) -> int:
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            return 0
        gmean = float(np.exp(np.mean(np.log(vals))))
        p = 0
        while gmean < target[0]:
            gmean *= 10.0
            p += 1
        while gmean >= target[1]:
            gmean /= 10.0
            p -= 1
        return p

    def _fmt(x: float, _sig: int = 3) -> str:
        if not (isinstance(x, (int, float)) and math.isfinite(x)):
            return "--"
        return f"{x:.{_sig}f}"

    # ----------------------- model display names ------------------------------
    # Your requested names:
    #   hs_em                -> Bayesian Horseshoe Trend Filter (BHS-TF)
    #   bayes_l1_em|l1_em   -> Bayesian \ell_1 Trend Filter (BL1-TF)
    #   l1_tf_cv|tf_l1      -> Frequentist \ell_1 Trend Filter (FL1-TF)
    #   krr                 -> Kernel Regression (KRR)
    default_map = {
        "hs_em": r"BHS-TF",
        "bayes_l1_em": r"B$\ell_1$-TF",
        "l1_em": r"B$\ell_1$-TF",
        "l1_em_cv": r"B$\ell_1$-TF",
        "l1_tf_cv": r"F$\ell_1$-TF",
        "krr": r"KRR",
    }
    MODEL_MAP = (model_map or {}) | default_map  # allow user overrides

    # --------------------------- load & filter --------------------------------
    df = pd.read_csv(csv_path)



    # ------------------------- aggregate over seeds ---------------------------
    grouped = (
        df.groupby(["method", "n"])
        .agg(
            mse_mean=("mse_true", "mean"),
            mse_se=("mse_true", _se),
            spars_mean=("sparsity_count", "mean"),
            spars_sd=("sparsity_count", _sd),
            iter_mean=("iterations", "mean"),
            iter_sd=("iterations", _sd),
            time_mean=("run_time_s", "mean"),
            time_sd=("run_time_s", _sd),
        )
        .reset_index()
        .sort_values(["method", "n"])
    )

    # Decide scaling for MSE (×10^p)
    mse_scale_p = _choose_scale(grouped["mse_mean"].values, target=target_range)

    # Time unit: s or ms
    with np.errstate(invalid="ignore"):
        tmax = np.nanmax(grouped["time_mean"].values)
        tsdmax = np.nanmax(grouped["time_sd"].values)
    time_unit, time_factor = ("ms", 1000.0) if (
        np.isfinite(tmax) and tmax < 10.0 and np.isfinite(tsdmax) and tsdmax < 1.0
    ) else ("s", 1.0)

    # ------------------------ build table body rows ---------------------------
    lines = []
    groups = list(grouped.groupby("method", sort=True))

    n_groups = len(groups)

    for gi, (grp, sub) in enumerate(groups):
        sub = sub.reset_index(drop=True)
        nrows = sub.shape[0]
        key = str(grp)
        model_disp = MODEL_MAP.get(key, _to_camel(key))
        for j, row in sub.iterrows():
            grp_cell = rf"\multirow{{{nrows}}}{{*}}{{{str(model_disp)}}}" if j == 0 else ""

            # Model display name
            n=str(row['n'])

            # MSE Mean | SE  (with scaling)
            mse_mean_scaled = row["mse_mean"] * (10 ** mse_scale_p)
            mse_se_scaled = row["mse_se"] * (10 ** mse_scale_p) if math.isfinite(row["mse_se"]) else float("nan")
            mse_mean_cell = _fmt(mse_mean_scaled, sig)
            mse_se_cell = _fmt(mse_se_scaled, sig)

            # Sparsity Mean | SD
            spars_mean_cell = _fmt(row["spars_mean"], sig)
            spars_sd_cell = _fmt(row["spars_sd"], sig)

            # Sparsity Mean | SD
            iter_mean_cell = _fmt(row["iter_mean"], sig)
            iter_sd_cell = _fmt(row["iter_sd"], sig)

            # Time Mean | SD (with unit)
            tmean = row["time_mean"] * time_factor
            tsd = row["time_sd"] * time_factor
            time_mean_cell = _fmt(tmean, sig)
            time_sd_cell = _fmt(tsd, sig)

            lines.append(
                " & ".join([
                    grp_cell,
                    n,
                    mse_mean_cell, mse_se_cell,
                    spars_mean_cell, spars_sd_cell,
                    iter_mean_cell, iter_sd_cell,
                    time_mean_cell, time_sd_cell
                ]) + r" \\"
            )

        # thin ruled line between parent groups (across all 8 columns)
        if add_thin_group_rules and gi < n_groups - 1:
            # small vertical breathing room + thin rule
            lines.append(r"\addlinespace[2pt]")
            lines.append(r"\cmidrule(lr){1-10}")

    # ----------------------------- LaTeX header -------------------------------
    # Top row with grouped headers and subcolumns aligned
    mse_head = r"MSE (SE)" + (rf" $\times 10^{{{mse_scale_p}}}$" if mse_scale_p != 0 else "")
    dataset_camel = _to_camel(dataset_label)

    header = (
        r"\begin{table}[H]" "\n"
        r"\centering" "\n"
        r"\small" "\n"
        r"\begin{tabular}{l l rr rr rr rr}" "\n"
        r"\toprule" "\n"
        r"Model & n & \multicolumn{2}{c}{" + mse_head + r"} & "
        r"\multicolumn{2}{c}{Sparsity (SD)} & "
        r"\multicolumn{2}{c}{Iterations (SD)} & "
        r"\multicolumn{2}{c}{Time (" + time_unit + r") (SD)} \\" "\n"
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}" "\n"
        r" &  & Mean & SE & Mean & SD & Mean & SD & Mean & SD \\" "\n"
        r"\midrule"
    )

    body = "\n".join(lines)

    footer = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        rf"\caption{{ASD}}" "\n"
        rf"\label{{tab:scale-model-contrast}}" "\n"
        r"\end{table}"
    )

    latex = "\n".join([header, body, footer])
    meta = {"mse_scale_power": int(mse_scale_p), "time_unit": time_unit}
    return latex, meta

# ---------------------------------------------------------------------------
# Example usage inside your own main:
#
# if __name__ == "__main__":
#     latex, meta = build_spikes_model_table(
#         csv_path="../report/overlay_metrics.csv",
#         dataset_label="synth_spikes",   # CamelCase appears in caption
#         sigma_value=0.2,
#         sig=3,                          # significant digits
#         add_thin_group_rules=True,
#     )
#     print(latex)  # or write to a .tex file
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def main():


    latex, meta = build_spikes_model_table(
        csv_path="../report/overlay_metrics.csv",
        dataset_label="synth_inhom",  # CamelCase appears in caption
        # dataset_label="synth_spikes",  # CamelCase appears in caption
        # spikes=4,
        # sigma_value=0.2,
        sig=2,                          # significant digits
        add_thin_group_rules=True,
    )
    print(latex)  # or write to a .tex file

def main_scale():


    latex, meta = build_spikes_compare_table(
        csv_path="../report/figs/scaling_compare_timing.csv",
        sig=2,                          # significant digits
        add_thin_group_rules=True,
    )
    print(latex)  # or write to a .tex file

if __name__ == "__main__":
    main_scale()