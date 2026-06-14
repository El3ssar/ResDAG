"""Benchmark configuration: shared hyper-parameters, the per-run config point,
and the context sweeps that make up the comparison matrix."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class HParams:
    """ESN hyper-parameters held identical across every library.

    These map onto each library's own knobs in the adapters; the *nominal*
    architecture is the same everywhere even though each implementation realizes
    it differently (dense vs sparse reservoir, CG vs direct ridge solve, ...).
    """

    dim: int = 3  # Lorenz-63 is 3-D (feedback == output dimension)
    spectral_radius: float = 0.9
    leak_rate: float = 0.3
    connectivity: float = 0.1  # reservoir sparsity / edge probability
    input_scaling: float = 1.0
    ridge: float = 1e-6  # L2 readout regularization
    activation: str = "tanh"
    seed: int = 1234
    vpt_threshold: float = 0.5  # normalized error threshold for valid prediction time


@dataclass(frozen=True)
class Point:
    """One concrete configuration in a sweep."""

    reservoir_size: int = 1000
    train_len: int = 5000  # timesteps used to fit the readout
    warmup: int = 200  # transient discarded before fitting / before generating
    horizon: int = 1000  # autoregressive forecast length

    @property
    def label(self) -> str:
        return f"N{self.reservoir_size}_T{self.train_len}_H{self.horizon}"


# --- The comparison matrix -------------------------------------------------
# Each context fixes all but one axis and reports one metric ("train" timing or
# "forecast" timing). Reservoir sizes are kept modest enough to fit a 6 GB GPU.

_BASE = Point(reservoir_size=1000, train_len=5000, warmup=200, horizon=1000)


@dataclass(frozen=True)
class Context:
    key: str
    title: str
    description: str
    metric: str  # "train" or "forecast"
    points: tuple[Point, ...]
    axis: str  # which field varies (for the table's first column)


CONTEXTS: tuple[Context, ...] = (
    Context(
        key="train_size",
        title="Training vs reservoir size",
        description=(
            "Fit the ridge readout on a fixed-length sequence while growing the "
            "reservoir. Stresses the reservoir run + the readout solve."
        ),
        metric="train",
        axis="reservoir_size",
        points=tuple(replace(_BASE, reservoir_size=n) for n in (250, 500, 1000, 2000)),
    ),
    Context(
        key="train_data",
        title="Training on lots of data",
        description=(
            "Fix the reservoir at 1000 units and grow the training sequence. "
            "Stresses the per-timestep reservoir loop at scale."
        ),
        metric="train",
        axis="train_len",
        points=tuple(replace(_BASE, train_len=t) for t in (1000, 5000, 20000)),
    ),
    Context(
        key="forecast",
        title="Long-horizon autoregressive forecasting",
        description=(
            "After an identical warmup, generate ever-longer closed-loop "
            "forecasts. Stresses the single-step reservoir update in a tight loop."
        ),
        metric="forecast",
        axis="horizon",
        points=tuple(replace(_BASE, horizon=h) for h in (1000, 5000, 10000)),
    ),
    Context(
        key="precision",
        title="Forecast precision (valid prediction time)",
        description=(
            "Predictive *skill*, not speed: after an identical warmup, how many "
            "steps the closed-loop forecast tracks the true Lorenz trajectory "
            "before the normalized error crosses the threshold. Higher is better."
        ),
        metric="precision",
        axis="reservoir_size",
        points=tuple(replace(_BASE, reservoir_size=n, horizon=3000) for n in (500, 1000, 2000)),
    ),
)


# A trimmed matrix for smoke tests / CI.
QUICK_CONTEXTS: tuple[Context, ...] = (
    Context(
        key="train_size",
        title="Training vs reservoir size",
        description="quick",
        metric="train",
        axis="reservoir_size",
        points=(
            replace(_BASE, reservoir_size=200, train_len=1000),
            replace(_BASE, reservoir_size=500, train_len=1000),
        ),
    ),
    Context(
        key="forecast",
        title="Long-horizon autoregressive forecasting",
        description="quick",
        metric="forecast",
        axis="horizon",
        points=(replace(_BASE, reservoir_size=300, train_len=1000, horizon=500),),
    ),
    Context(
        key="precision",
        title="Forecast precision (valid prediction time)",
        description="quick",
        metric="precision",
        axis="reservoir_size",
        points=(replace(_BASE, reservoir_size=300, train_len=1000, horizon=1000),),
    ),
)
