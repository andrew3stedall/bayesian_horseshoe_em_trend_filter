from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from .base import DataSource, DataBundle
from .utils import make_x_grid, center_and_validate

@dataclass
class SyntheticInhomogeneous(DataSource):
    n: int
    sigma: float
    seed: Optional[int] = None
    name: str = "synthetic_inhomogeneous"

    def _f_inhomogeneous(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        return np.exp(-7.5 * x) * np.cos(10.0 * np.pi * x)

    def load(self) -> DataBundle:
        rng = np.random.default_rng(self.seed)
        x = make_x_grid(self.n, 0.0, 1.0)
        y_true = self._f_inhomogeneous(x)
        y = y_true + self.sigma * rng.standard_normal(self.n)
        x, y, meta_c = center_and_validate(x, y)
        y_true -= meta_c['removed_mean']
        meta: Dict[str, Any] = {"sigma": self.sigma, "seed": self.seed, "center_meta": meta_c}
        return DataBundle(name=self.name, x=x, y=y, y_true=y_true, meta=meta)


@dataclass
class SyntheticCustom(DataSource):
    n: int
    sigma: float
    seed: Optional[int] = None
    name: str = "synthetic_custom"

    def _true_signal(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        ramp = np.random.uniform(0, 1) * x
        burst_center = np.random.uniform(0, 1)
        burst = np.random.uniform(0, 1) * np.exp(-((x - burst_center) ** 2) / (2 * np.random.uniform(0.01, 0.1)))
        drift = np.random.uniform(-0.25, 0.25) * np.sin(2 * np.pi * 1.3 * x)
        kinks = np.zeros_like(x)
        for c in [np.random.uniform(0.1, 0.3), np.random.uniform(0.3, 0.5), np.random.uniform(0.5, 0.7),
                  np.random.uniform(0.7, 0.9)]:
            kinks += np.random.uniform(0.01, 0.2) * np.maximum(0.0, x - c)
        return ramp + burst + drift + kinks

    def load(self) -> DataBundle:
        rng = np.random.default_rng(self.seed)
        x = make_x_grid(self.n, 0.0, 1.0)
        y_true = self._true_signal(x, rng)
        y = y_true + self.sigma * rng.standard_normal(self.n)
        x, y, meta_c = center_and_validate(x, y)
        y_true -= meta_c['removed_mean']
        meta: Dict[str, Any] = {"sigma": self.sigma, "seed": self.seed, "center_meta": meta_c}
        return DataBundle(name=self.name, x=x, y=y, y_true=y_true, meta=meta)

@dataclass
class SyntheticSpikes(DataSource):
    """
    Synthetic time series where the signal is mostly smooth with two genuine spikes
    that are part of the trend (not outliers).

    Primary output is a DataFrame with columns:
    - t:   normalized time in [0, 1]
    - y:   baseline + spikes + noise
    - baseline: smoothed low-frequency baseline
    - spike_component: sum of Gaussian bumps
    - noise: additive Gaussian noise

    Spike metadata is provided as a separate DataFrame with:
    - center_index, width_points, amplitude
    """

    # Core controls
    n: int = 1000
    seed: Optional[int] = None
    sigma: float = 0.2
    smooth_scale: int = 80

    # Spike controls (exactly two spikes by default)
    num_spikes: int = 2
    spike_amp_range: tuple[float, float] = (2.0, 5.0)   # amplitude multipliers of baseline std
    spike_width_range: tuple[int, int] = (8, 25)        # std dev in points (Gaussian)
    center_bounds: tuple[float, float] = (0.15, 0.85)   # avoid edges
    name: str = "synthetic_spikes"

    # ------------- Public API expected from a DataSource -------------

    def load(self) -> DataBundle:
        # print(self.seed)
        rng = np.random.default_rng(self.seed)
        x = make_x_grid(self.n, 0.0, 1.0)
        n = int(self.n)

        # Low-frequency baseline
        baseline = (
            0.6 * np.sin(2 * np.pi * 1.2 * x) +
            0.4 * np.sin(2 * np.pi * 2.3 * x + 0.7) +
            0.3 * np.cos(2 * np.pi * 0.6 * x + 1.3)
        )

        # Smooth baseline with moving average
        k = max(int(self.smooth_scale), 1)
        kernel = np.ones(k) / k
        baseline_sm = np.convolve(baseline, kernel, mode="same")

        # Spikes: Gaussian bumps added to the trend
        low_idx = int(self.center_bounds[0] * n)
        high_idx = int(self.center_bounds[1] * n)
        num_spikes = int(self.num_spikes)
        if num_spikes < 1:
            num_spikes = 1
        centers = rng.integers(low=low_idx, high=high_idx, size=num_spikes)

        baseline_std = float(np.std(baseline_sm))
        idx = np.arange(n)
        spike_component = np.zeros(n, dtype=float)
        meta_rows = []

        for c in centers:
            width = int(rng.integers(self.spike_width_range[0], self.spike_width_range[1]))
            amp_mult = float(rng.uniform(self.spike_amp_range[0], self.spike_amp_range[1]))
            amplitude = amp_mult * baseline_std * int(rng.choice([-1, 1]))
            spike = amplitude * np.exp(-0.5 * ((idx - c) / max(width, 1)) ** 2)
            spike_component += spike
            meta_rows.append(
                {"center_index": int(c), "width_points": int(width), "amplitude": float(amplitude)}
            )

        y_true = baseline_sm + spike_component

        y = y_true + self.sigma * rng.standard_normal(self.n)
        x, y, meta_c = center_and_validate(x, y)
        y_true -= meta_c['removed_mean']
        meta: Dict[str, Any] = {"sigma": self.sigma, "seed": self.seed, "center_meta": meta_c}
        return DataBundle(name=self.name, x=x, y=y, y_true=y_true, meta=meta)