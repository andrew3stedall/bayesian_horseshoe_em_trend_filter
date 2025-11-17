# src/reporting/visualisations/beta_histograms.py
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, LogNorm, SymLogNorm

from .base import FigureBase
from .utils import make_dataset_from_ref, cartesian_grid, get_method_runner, diff_k, save_png
from ..prep import StandardLoader
from ..prep.transforms import ScaleX01, StandardizeTarget


class BetaScatter(FigureBase):
    """
    Histograms of β across EM iterations for HS–EM.
    Params:
      - method: "hs_em"
      - dataset: name in ctx.datasets_cfg OR provide dataset_cls + params
      - iterations: [1,2,5,10,20,"final"]
      - bins: int
      - output: filename
      - order (optional): default 2
    Strategy: refit HS–EM with max_iter set to each 'k' and record beta.
    """

    def prepare(self) -> Dict[str, Any]:
        datasets_yaml = "../configs/datasets.yaml"
        loader = StandardLoader(
            registry_yaml=datasets_yaml,
            transforms=[ScaleX01(), StandardizeTarget()]
        )

        p = self.spec.params
        use_csv = bool(p.get("use_csv", True))
        output = p.get("output", "scaling.png")

        method_key = p.get("method", "hs_em")
        ds_key = p.get("dataset", "synth_spikes_s005")
        sigma = float(p.get("sigma", 0.05))

        show_fit = p.get("show_fit", False)
        show_beta = p.get("show_beta", False)
        num_col = p.get("num_col", 4)

        em_iterations_list = [s for s in p["em_iterations_list"]]
        num_spikes = int(p.get("num_spikes", 3))
        bins = int(p.get("bins", 60))
        n = int(p.get("n", 1024))
        seed = int(p.get("seed", 42))

        runners = get_method_runner()
        run = runners[method_key]

        # run
        extra_params = {'n': n, 'num_spikes': num_spikes, 'seed': seed, 'sigma':sigma}
        print(f'Currently working on: {extra_params}')
        data = loader.load_with_params(ds_key, **extra_params)
        x = data.x
        y = data.y
        y_true = data.y_true
        # inverse_target_fn = data.inverse_target_fn
        order = 2

        lambdas: List[np.ndarray] = []
        betas: List[np.ndarray] = []
        y_hats: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        labels: List[str] = []
        for k in em_iterations_list:
            params_base = dict(self.ctx.methods_cfg[method_key].get("params", {}))

            if k == "final":
                y_hat, meta = run(x, y, order, params_base, y_true)
                labels.append(f"end({str(meta['n_iter'])})")
            else:
                params_base["max_iter"] = int(k)
                y_hat, meta = run(x, y, order, params_base, y_true)
                labels.append(str(k))
            lambdas.append(meta.get('lambda2'))
            betas.append(meta.get('beta'))
            y_hats.append(y_hat)
            ys.append(y)
            # print(y_hats)

        return {"title": self.spec.title,
                "lambdas": lambdas,
                "betas": betas,
                "labels": labels,
                "bins": bins,
                "output": output,
                "y_hats": y_hats,
                "y": y,
                "y_true": y_true,
                "method_title": self.spec.method_title,
                "show_fit": show_fit,
                "show_beta": show_beta,
                "num_col": num_col,
                }

    def render(self, prepared: Dict[str, Any]) -> Path:
        betas = prepared["betas"]
        lambdas = prepared["lambdas"]
        show_fit = prepared["show_fit"]
        show_beta = prepared["show_beta"]
        num_col = prepared["num_col"]

        y_hats = prepared["y_hats"]
        y_hat = y_hats[len(y_hats) - 1]
        y = prepared["y"]
        y_true = prepared["y_true"]
        labels = prepared["labels"]
        print(labels)
        cols = self.ctx.colours

        nodes = [
            0,
            0.15,
            0.25,
            0.6,
            0.85,
            1.0,
        ]
        colors = [
            cols[2],
            cols[1],
            cols[0],
            cols[5],
            cols[4],
            cols[3],
        ]
        custom_cmap = LinearSegmentedColormap.from_list("beta_values", list(zip(nodes, colors)))

        ncols = min(num_col, len(labels))
        nrows = int(np.ceil(len(labels) / ncols))

        if show_fit and show_beta:
            fig, axes = plt.subplots(nrows*3, ncols, figsize=(ncols * 3, nrows * 6), squeeze=True)
        else:
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2), squeeze=True)

        plt.subplots_adjust(
            left=0.05,  # the left side of the subplots of the figure
            right=0.9,  # the right side of the subplots of the figure
            bottom=0.05,  # the bottom of the subplots of the figure
            top=0.95,  # the top of the subplots of the figure
            wspace=0.01,  # the amount of width reserved for blank space between subplots
            hspace=0.01  # the amount of height reserved for white space between subplots
        )

        norm = SymLogNorm(linthresh=1e-10, linscale=1, vmin=-1e1, vmax=1e1, base=10)

        for i, (beta, lab) in enumerate(zip(betas, labels)):
            if show_fit and show_beta:
                ax_beta = axes[0][i]
                ax_fit_beta = axes[1][i]
                ax_fit = axes[2][i]
            elif show_beta:
                ax_beta = axes[i // ncols][i % ncols]
            elif show_fit:
                ax_fit = axes[i // ncols][i % ncols]
            if show_beta:
                c = custom_cmap(norm(beta))
                im = ax_beta.scatter(np.arange(len(beta)), beta, lw=np.minimum(np.absolute(beta) * 200, 1), edgecolors=c, facecolors='none',
                                s=np.minimum(np.absolute(beta) * 5000, 100.0), alpha=0.8, norm=norm, label='$\\beta_j$',
                                zorder=1)
                ax_beta.set_ylim([-1e1, 1e2])
                if i % ncols == 0:  # and i//nrows==(nrows-2):
                    ax_beta.set_ylabel(r"$\hat{\beta}$")
                else:
                    ax_beta.yaxis.set_visible(False)
                ax_beta.set_yscale('symlog', linthresh=1e-5, linscale=1, base=10)

                ax_beta.set_title(f"i={lab}", y=0.85)
                ax_beta.tick_params(labelsize=8)

                if i // ncols == (nrows - 1):  # and i%ncols==(ncols-2):
                    ax_beta.set_xlabel(r"index ($_j$)")
                else:
                    ax_beta.xaxis.set_visible(False)
            if show_fit:
                ax_fit.set_ylim([-5, 4])
                ax_fit.plot(np.arange(len(y_true)), y_true, lw=1.5, alpha=0.8, color=cols[2], label='True signal', zorder=3)
                ax_fit.plot(np.arange(len(y_hats[i])), y_hats[i], lw=1.5, alpha=0.8, color=cols[1], label=r'B$\ell_1$-TF', zorder=2)

                if i==0:
                    ax_fit.legend(loc="lower left", frameon=False, fontsize=8)

                if i // ncols == (nrows - 1):  # and i%ncols==(ncols-2):
                    ax_fit.set_xlabel(r"index ($_j$)")
                else:
                    ax_fit.xaxis.set_visible(False)

                ax_fit.yaxis.set_visible(False)

            if show_fit and not show_beta:
                ax_fit.set_title(f"i={lab}", y=0.85)

            if show_beta and show_fit:
                ax_fit_beta.set_ylim([-5, 4])
                ax_fit_beta.plot(np.arange(len(y_hats[i])), y_hats[i], lw=2, alpha=0.8, color=cols[1],
                            label=r'B$\ell_1$-TF', zorder=2)
                c = custom_cmap(norm(beta))
                im = ax_fit_beta.scatter(np.arange(len(beta)), y_hats[i][2:], lw=np.minimum(np.absolute(beta) * 200, 1),
                                    edgecolors=c,
                                    s=np.minimum(np.absolute(beta) * 5000, 100.0), alpha=0.8,
                                    label='Knot points',
                                    zorder=1)
                im.set_facecolors('none')
                # ax_fit_beta.tick_params(labelsize=8)
                # ax_fit_beta.yaxis.tick_right()
                #
                ax_fit_beta.yaxis.set_visible(False)
                ax_fit.yaxis.set_visible(False)
                if i == 0:
                    ax_fit_beta.legend(loc="lower left", frameon=False, fontsize=8)

                if i // ncols == (nrows - 1):  # and i%ncols==(ncols-2):
                    ax_fit_beta.set_xlabel(r"index ($_j$)")
                else:
                    ax_fit_beta.xaxis.set_visible(False)
            # else:
            #     raise NotImplementedError('We done seem to be showing anything')




            # ax.yaxis.set_visible(False)
        fig.suptitle(rf"{prepared["title"]}")

        # fig_solo = plt.figure(figsize=(6, 6))
        # ax_solo = fig_solo.add_subplot(111)
        #
        # final_beta = betas[len(betas) - 1]
        #
        # ax_solo.plot(np.arange(len(y_hat)), y_hat, lw=0.5, color='black', zorder=-999)
        # # ax_solo.plot(np.arange(len(y_hat)), y_hat, lw=1.5, color=self.ctx.colours[4], label=prepared["method_title"])
        # ax_solo.scatter(np.arange(len(y)), y, s=3, alpha=0.2, lw=0.1, facecolor='none', edgecolors='black', zorder=-999)
        # # ax_solo.scatter(np.arange(len(y)), y, s=3, alpha=1, lw=0.1, facecolor='none', edgecolors=self.ctx.colours[4])
        # ax_solo_b = ax_solo.twinx()
        # ax_solo_b.scatter(np.arange(len(final_beta)), final_beta, c=final_beta, cmap=custom_cmap,
        #                   s=np.minimum(np.maximum(np.absolute(final_beta) * 10, 0.001) * 1000, 150.0), alpha=0.2,
        #                   norm=norm, zorder=1)
        # ax_solo_b.set_yscale('symlog', linthresh=1e-10, linscale=1, base=10)
        # # ax_solo.legend(loc="best", frameon=False, fontsize=8)
        # fig_solo.suptitle(f'{prepared["method_title"]} - ' + r'Final $\beta_j$ values and final fit.')
        # plt.tight_layout()
        # print(betas[0])

        # tidy unused axes
        for j in range(len(labels), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        # save_png(fig_solo, self.ctx.output_dir / ('hat_vs_' + prepared["output"]), dpi=self.ctx.dpi,
        #          overwrite=self.ctx.overwrite)
        # fig.show()
        return save_png(fig, self.ctx.output_dir / prepared["output"], dpi=self.ctx.dpi, overwrite=self.ctx.overwrite)
