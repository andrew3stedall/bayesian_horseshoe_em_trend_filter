# src/reporting/visualisations/scaling.py
from __future__ import annotations

from email.policy import default
from typing import Dict, Any, List
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

import pandas as pd

from .base import FigureBase
from .utils import get_method_runner, save_png, make_dataset_from_ref
from src.data.synthetic import SyntheticSpikes  # adjust if different path
from src.prep.loader import StandardLoader
from src.prep.transforms import ScaleX01, StandardizeTarget
from ..experiments.metrics import compute_metrics
from ..registry import DataRegistry


class Scaling(FigureBase):
    """
    HS–EM scaling: median runtime vs n. Params:
      - method: "hs_em"
      - dataset_cls: "src.data.synthetic.SyntheticSpikes"
      - n: [512, 1024, ...]
      - sigma: 0.05
      - seeds (optional): [1,2,3] default [1,2,3]
      - output: filename
      - order (optional): default 2
      - loglog (optional): bool
    """
    def prepare(self) -> Dict[str, Any]:
        p = self.spec.params
        methods = p.get("methods", None)
        inputs = p.get("inputs", None)

        use_csv = bool(p.get("use_csv", True))
        output = p.get("output", "scaling.png")

        if methods is not None:
            return self.prepare_compare()

        if use_csv:
            out_name = f"{output}_timing.csv"
            out_path = self.ctx.output_dir / out_name
            results = pd.read_csv(out_path)
        else:
            # Loader with standard transforms (optional: scale X, standardize y)
            datasets_yaml = "../configs/datasets.yaml"
            loader = StandardLoader(
                registry_yaml=datasets_yaml,
                transforms=[ScaleX01(), StandardizeTarget()]
            )
            method_key = p.get("method", "hs_em")
            ds_key = p.get("dataset", "synth_spikes_s005")
            sigma = float(p.get("sigma", 0.2))
            num_spikes_list = [int(s) for s in p["spikes"]]
            iterations = int(p.get("iterations", 10))
            n_list = [int(n) for n in p["n"]]
            runners = get_method_runner()
            run = runners[method_key]

            # Loop through each combination of seed and n to create a dataset and fit model to get timing
            results = []
            for scale_iter in range(iterations):
                seed = random.randint(1, 100000)
                for num_spikes in num_spikes_list:
                    for n in n_list:
                        extra_params = {'n': n, 'num_spikes': num_spikes, 'seed': seed, 'sigma': sigma}
                        print(f'Currently working on: {extra_params}')
                        data = loader.load_with_params(ds_key, **extra_params)
                        t_run = time.perf_counter()
                        x = data.x
                        y = data.y
                        y_true = data.y_true
                        order = 2
                        y_hat, meta = run(x, y, order, self.ctx.methods_cfg[method_key].get("params", {}), y_true)

                        rt = time.perf_counter() - t_run
                        m = compute_metrics(y, y_hat, 2, "1/(5*sqrt(n))", rt)
                        m_true = compute_metrics(y_true, y_hat, 2, "1/(5*sqrt(n))", rt)

                        iterations = meta.get("n_iter", 1)

                        results.append({
                            'mse': m['mse'],
                            'mae': m['mae'],
                            'mse_true': m_true['mse'],
                            'mae_true': m_true['mae'],
                            'sparsity_count': int(m["sparsity_count"]),
                            'run': scale_iter,
                            'run_time_s': rt,
                            'n': n,
                            'seed': seed,
                            'spikes': num_spikes,
                            'iterations': iterations,
                        })
        return {
            'title': self.spec.title,
            'output': output,
            'order': 2,
            'results': pd.DataFrame(results)
        }

    def prepare_compare(self) -> Dict[str, Any]:
        p = self.spec.params
        methods = p.get("methods", None)
        inputs = p.get("inputs", None)

        use_csv = bool(p.get("use_csv", True))
        output = p.get("output", "scaling.png")


        consolidated_results = []
        # Loop through the methods and inputs to get the data we need
        for i, (method, input) in enumerate(zip(methods, inputs)):
            out_name = f"{input}_timing.csv"
            out_path = self.ctx.output_dir / out_name
            results = pd.read_csv(out_path)
            method_label = f"{self.ctx.methods_cfg[method].get('label', method)}"
            results['method'] = method
            results['method_label'] = method_label
            consolidated_results.append(results)

        results = pd.concat(consolidated_results, ignore_index=True, sort=False)

        out_name = f"{output}_timing.csv"
        out_path = self.ctx.output_dir / out_name
        results.to_csv(out_path, index=False)

        return {
            'title': self.spec.title,
            'output': output,
            'order': 2,
            'results': pd.DataFrame(results),
            'consolidated': True
        }


    def render(self, prepared: Dict[str, Any]) -> Path:
        title = prepared["title"]
        results = prepared["results"]
        output = prepared["output"]
        consolidated = prepared.get("consolidated", False)
        colours = self.ctx.colours

        p = self.spec.params
        use_csv = bool(p.get("use_csv", True))

        if not use_csv:
            # save to csv in case we want to just render without running next time
            out_name = f"{output}_timing.csv"
            out_path = self.ctx.output_dir / out_name
            results.to_csv(out_path, index=False)

        df = prepared['results']
        df['run_time_per_thousand_n'] = df['run_time_s'] / df['n'] * 1e3
        # df['converged'] = df['iterations'] < 1000

        # # Calculate Q1, Q3, and IQR for the 'Value' column
        # Q1 = df['run_time_per_million_n'].quantile(0.05)
        # Q3 = df['run_time_per_million_n'].quantile(0.95)
        # IQR = Q3 - Q1
        #
        # # Define outlier bounds
        # lower_bound = Q1 - 1.5 * IQR
        # upper_bound = Q3 + 1.5 * IQR
        # df = df[(df['run_time_per_million_n'] >= lower_bound) & (df['run_time_per_million_n'] <= upper_bound)]


        # Filter the DataFrame to exclude outliers
        if consolidated:
            cat, val = ["n","method"], "run_time_per_thousand_n"
        else:
            cat, val = "n", "run_time_per_thousand_n"

        k = 1.5  # Tukey multiplier

        q1 = df.groupby(cat)[val].transform(lambda s: s.quantile(0.25))
        q3 = df.groupby(cat)[val].transform(lambda s: s.quantile(0.75))
        iqr = q3 - q1
        lo = q1 - k * iqr
        hi = q3 + k * iqr

        df = df[(df[val] >= lo) & (df[val] <= hi)].copy()

        fig = plt.figure(figsize=(6,8))
        ax = fig.add_subplot(111)
        # ax.set_yscale('log', base=2)
        # sns.swarmplot(
        #     data=df,
        #     # data=df[df['spikes'] <= 2],
        #     y="run_time_s",
        #     x="n",
        #     # hue="spikes",
        #     # palette=colours,
        #     ax=ax,
        #     orient="v",
        #     size=2,
        #     alpha=0.3,
        #     # dodge=True,
        #     # log_scale=2
        # )
        if consolidated:
            # Overlay stream (line + filled band)
            ycol = "run_time_s"
            order = sorted(df["n"].unique())
            pos = np.arange(len(order))
            g = (
                df.dropna(subset=[ycol])
                .groupby(cat)[ycol]
                .agg(mean="mean", sd=lambda x: x.std(ddof=1), count="count")
                .reset_index()
            )

            methods = g[cat].unique()
            idx = pd.MultiIndex.from_product([order, methods], names=cat)
            g = g.set_index(["n", cat]).reindex(idx)

            k = 1.0  # 1 SD band (set 2.0 for ±2 SD)
            mean = g["mean"].to_numpy()
            low = mean - k * g["sd"].to_numpy()
            high = mean + k * g["sd"].to_numpy()
            label_line = "Mean"
            label_band = f"±{k} SD band"

            ax.plot(pos, mean, lw=2, zorder=10, label=label_line)
            ax.fill_between(pos, low, high, alpha=0.22, zorder=9, label=label_band, color="grey")

            # Save with seed_ prefix
            out_name = f"{output}.png"
            out_path = self.ctx.output_dir / out_name
            return save_png(fig, out_path, dpi=self.ctx.dpi, overwrite=self.ctx.overwrite)


        sns.violinplot(
            data=df,
            # data=df[df['spikes'] <= 2],
            y="run_time_s",
            x="n",
            # hue="spikes",
            # palette=colours,
            ax=ax,
            bw_adjust=0.5,
            linewidth=0.5,
            alpha=0.3,
            # split=True,
            # fill=False,
            gap=0.05,
            width=0.9,
            density_norm="area",
            # inner="quart",
            legend=False,
            cut=0.0
        )



        # Overlay stream (line + filled band)
        ycol = "run_time_s"
        order = sorted(df["n"].unique())
        pos = np.arange(len(order))
        g = (df.dropna(subset=[ycol])
             .groupby("n")[ycol]
             .agg(mean="mean", sd="std", count="count")
             .reindex(order))

        # Choose one of these two:
        use_ci = False  # set False to draw an SD band instead of a CI

        if use_ci:
            z = 1.96  # 95% z-interval (you can change)
            se = g["sd"].to_numpy() / np.sqrt(g["count"].to_numpy())
            mean = g["mean"].to_numpy()
            low = mean - z * se
            high = mean + z * se
            label_line = "Mean"
            label_band = "95% CI (z)"
        else:
            k = 1.0  # 1 SD band (set 2.0 for ±2 SD)
            mean = g["mean"].to_numpy()
            low = mean - k * g["sd"].to_numpy()
            high = mean + k * g["sd"].to_numpy()
            label_line = "Mean"
            label_band = f"±{k} SD band"

        ax.plot(pos, mean, lw=2, zorder=10, label=label_line)
        ax.fill_between(pos, low, high, alpha=0.22, zorder=9, label=label_band, color="grey")

        ax.set_title(rf"{title}")
        ax.set_ylabel("Total time taken (s)")
        ax.set_xlabel("n")
        # ax.set_yscale('log', base=2)
        ax.set_ylim([0, 8])
        # ax.legend(loc="best", frameon=False, fontsize=8, title="# Spikes")

        # Save with seed_ prefix
        out_name = f"{output}.png"
        out_path = self.ctx.output_dir / out_name
        return save_png(fig, out_path, dpi=self.ctx.dpi, overwrite=self.ctx.overwrite)


