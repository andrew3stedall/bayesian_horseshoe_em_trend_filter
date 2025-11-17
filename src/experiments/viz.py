from __future__ import annotations
from pathlib import Path
from typing import Optional, Mapping, Sequence, Tuple, Union
import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]

def save_fit_plot_multi(
    x: ArrayLike,
    y: ArrayLike,
    y_hats: Union[Mapping[str, ArrayLike], Sequence[Tuple[str, ArrayLike]]],
    path_png: str | Path,
    y_true: Optional[ArrayLike] = None,
    title: Optional[str] = None,
    dpi: int = 150,
    sort_x: bool = True,
    scatter_size: int = 12,
    scatter_alpha: float = 0.30,
    line_width: float = 1.4,
) -> None:
    """
    Save a PNG that contrasts multiple model fits on the same chart.

    Parameters
    ----------
    x, y : array-like
        Coordinates and observations. 1-D and same length.
    y_hats : mapping or sequence
        Either a dict {label: y_hat} or a list of (label, y_hat) pairs.
        Each y_hat must be 1-D and same length as x.
    path_png : str | Path
        Output file path (.png). Parent dirs are created if needed.
    y_true : array-like, optional
        Ground-truth curve to overlay (e.g., synthetic signal f(x)).
    title : str, optional
        Figure title.
    dpi : int
        Output resolution.
    sort_x : bool
        If True, sort by x before plotting lines (recommended).
    scatter_size, scatter_alpha : controls for the y scatter.
    line_width : float
        Line width for model curves and y_true.
    """
    # Headless-safe import (matches your original function)
    import matplotlib
    try:
        if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
            matplotlib.use("Agg", force=True)
    except Exception:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # Normalise inputs
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # Convert y_hats to an ordered list of (label, yhat)
    if isinstance(y_hats, Mapping):
        items = list(y_hats.items())  # preserves insertion order (Py3.7+)
    else:
        items = list(y_hats)

    # Validate and ravel
    yh_list: list[Tuple[str, np.ndarray]] = []
    n = x.size
    for label, arr in items:
        arr = np.asarray(arr).ravel()
        if arr.size != n:
            raise ValueError(f"y_hat for '{label}' has length {arr.size}, expected {n}")
        yh_list.append((label, arr))

    y_true_arr = None
    if y_true is not None:
        y_true_arr = np.asarray(y_true).ravel()
        if y_true_arr.size != n:
            raise ValueError(f"y_true has length {y_true_arr.size}, expected {n}")

    # Optional sorting by x for clean line plots
    if sort_x:
        idx = np.argsort(x)
        x_plot = x[idx]
        y_scatter = y[idx]  # scatter looks cleaner ordered too
        yh_list = [(lbl, arr[idx]) for lbl, arr in yh_list]
        if y_true_arr is not None:
            y_true_arr = y_true_arr[idx]
    else:
        x_plot = x
        y_scatter = y

    # Plot
    fig = plt.figure(figsize=(8.5, 3.8))
    ax = fig.add_subplot(111)

    # Raw data scatter
    ax.scatter(x_plot, y_scatter, s=scatter_size, alpha=scatter_alpha, label="observations")

    # Optional ground-truth curve (e.g., synthetic f(x))
    if y_true_arr is not None:
        ax.plot(x_plot, y_true_arr, linewidth=line_width, linestyle="--", label="ground truth")

    # Model fits
    # (Let matplotlib handle the color cycle so multiple curves are distinguishable.)
    for label, yhat in yh_list:
        ax.plot(x_plot, yhat, linewidth=line_width, label=label)

    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", frameon=False, ncol=1)
    ax.grid(False)

    path_png = Path(path_png)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def save_fit_plot(x: np.ndarray,
                  y: np.ndarray,
                  y_hat: np.ndarray,
                  path_png: str | Path,
                  y_true: Optional[np.ndarray] = None,
                  title: Optional[str] = None,
                  dpi: int = 150) -> None:
    """
    Save a PNG: scatter of y vs x and red line for y_hat vs x.
    """
    # Headless-safe import
    import matplotlib
    try:
        # If a non-interactive backend is already set, leave it
        if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
            matplotlib.use("Agg", force=True)
    except Exception:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    y_hat = np.asarray(y_hat).ravel()

    fig = plt.figure(figsize=(8, 3.5))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=12, alpha=0.3, label="y", color='grey')

    ax.plot(x, y_hat, linewidth=1.0, color="red", label="fit")
    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", frameon=False)
    ax.grid(False)

    path_png = Path(path_png)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

