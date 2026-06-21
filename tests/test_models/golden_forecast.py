"""Shared specs + helpers behind the golden-forecast regression fixtures.

This module is the single source of truth for issue #401's behavioural
contract.  It is imported by **both**

- the regression test (``tests/test_models/test_golden_forecast.py``), and
- the regen tool (``tools/regen_golden_forecasts.py``),

so the model specs, the canonical Lorenz data splits, the metric, and every
tolerance live in exactly one place — a fixture and the test it guards can
never drift on their definitions.

Contract
--------
For each premade model we commit a *golden* warmup → forecast trajectory,
produced by a fully-seeded build (``seed=`` deterministically fixes every
reservoir parameter — topology, feedback/input weights, bias) trained on a
canonical Lorenz-63 series, all in ``float64`` on CPU.  The test rebuilds each
model under the same seed and asserts two things about the live forecast:

1. **Short-horizon prefix match** — the first :data:`PREFIX_STEPS` steps equal
   the committed golden within :data:`PREFIX_ATOL` / :data:`PREFIX_RTOL`.  This
   is the precise structural guard: every historical forecast regression
   (slot-0, the warmup/forecast seam, the flat single-step engine) shifts these
   steps by order 0.1–1, while cross-platform BLAS / CG noise stays well below
   the tolerance.
2. **VPT floor** — the Valid Prediction Time (contiguous steps whose NRMSE
   against the true continuation stays below :data:`VPT_THRESHOLD`) must clear a
   committed, deliberately *conservative* floor.  A chaotic forecast's valid
   horizon swings ~40% across BLAS implementations, so this integer-valued tier
   is only a coarse "quality did not collapse" guard — not a tight bound (see
   :func:`vpt_floor`).

The data is regenerated deterministically from :func:`resdag.lorenz` +
:func:`resdag.utils.prepare_esn_data` (both separately tested) rather than dumped
into every fixture, so a fixture stays tiny and captures only the *expected
trajectory*.  A behavioural change to those canonical generators therefore also
trips this test until the goldens are regenerated — an intentional tripwire.

See Also
--------
resdag.hpo.losses.forecast_horizon : The HPO loss whose NRMSE / contiguous
    horizon semantics :func:`valid_prediction_time` mirrors (re-implemented here
    so the normal test suite never imports the optional HPO stack).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

import resdag as rd
from resdag.core import ESNModel, Input
from resdag.layers import CGReadoutLayer, NGReservoir
from resdag.models import classic_esn, ott_esn, power_augmented
from resdag.training import ESNTrainer

# --------------------------------------------------------------------------- #
# Canonical constants — the committed contract.  Changing any of these is a
# behavioural change that requires regenerating the fixtures.
# --------------------------------------------------------------------------- #

#: Directory holding the committed ``<model>.npz`` golden fixtures.
FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "golden_forecast"

DTYPE: torch.dtype = torch.float64  # CPU float64 — the deterministic reference.

DATA_SEED = 0  # seed for the Lorenz initial-condition perturbation.
MODEL_SEED = 12345  # seed fixing every reservoir parameter.

WARMUP_STEPS = 200  # reservoir warmup / synchronisation steps.
TRAIN_STEPS = 2500  # readout-fit steps.
HORIZON = 600  # autonomous forecast horizon (== validation length).

# The committed golden is compared over a short prefix.  ``PREFIX_ATOL`` is
# calibrated to sit between two empirically separated scales: cross-platform
# BLAS / CG drift in the trained-ESN forecast is ~3e-3 (measured on CI runners
# with a different BLAS than the generating machine), while every forecast-path
# regression this test guards against (slot-0, the warmup/forecast seam, the
# flat single-step engine) shifts the prefix by order 0.1–1.  ``atol = 3e-2``
# clears the noise by ~10x and trips a real regression by ~10x.  ``rtol`` is
# kept at 0 because the standardised Lorenz series crosses zero, where a
# relative tolerance is meaningless; ``atol`` on a unit-std signal is the
# principled knob.  The well-conditioned ridge (see ``READOUT_ALPHA``) keeps the
# actual cross-platform drift far below ``atol``.
PREFIX_STEPS = 20  # deterministic prefix compared to the golden.
PREFIX_ATOL = 3e-2  # absolute tolerance for the prefix match (units: std of the signal).
PREFIX_RTOL = 0.0  # relative tolerance — disabled (zero-crossing signal).

# Readout ridge regularisation for the ESN models.  The factory default 1e-6 is
# nearly unregularised, so the conjugate-gradient ridge solve is ill-conditioned
# and converges to a *machine-dependent* readout — the forecast then drifts by
# percent-level across BLAS implementations.  A well-conditioned 1e-2 makes the
# solve reproduce across machines (and, on Lorenz, forecasts at least as well).
READOUT_ALPHA = 1e-2

VPT_THRESHOLD = 0.3  # NRMSE threshold defining a "valid" forecast step.

# The VPT floor is a deliberately *conservative, cross-platform-robust collapse
# detector*, not a tight per-machine bound.  A chaotic forecast's valid horizon
# swings widely across BLAS implementations — e.g. power_augmented measures
# VPT 263 on the generating machine but 144 on the CI runners — because a tiny
# weight/readout difference amplifies over hundreds of autoregressive steps.
# So the floor is pinned at a fraction of the measured VPT (never below a hard
# minimum), low enough to survive that swing yet high enough to catch a forecast
# that has collapsed to garbage.  The precise structural guard is the prefix
# match (:data:`PREFIX_ATOL`), which *is* reproducible across machines.
VPT_FLOOR_FRAC = 0.3  # committed floor = this fraction of the measured VPT...
VPT_FLOOR_MIN = 20  # ...but at least this many steps (a clear "not collapsed" bar).

# Schema version stamped into every fixture; bump on incompatible format changes.
FIXTURE_VERSION = 1


# --------------------------------------------------------------------------- #
# Model specs
# --------------------------------------------------------------------------- #


def _build_classic() -> ESNModel:
    """Classic ESN (input-concat), fully seeded, float64."""
    model = classic_esn(
        reservoir_size=300,
        feedback_size=3,
        output_size=3,
        spectral_radius=0.9,
        topology="erdos_renyi",
        readout_alpha=READOUT_ALPHA,
        seed=MODEL_SEED,
    )
    model.double()  # in-place; cast to float64 reference precision
    return model


def _build_ott() -> ESNModel:
    """Ott state-augmented ESN, fully seeded, float64."""
    model = ott_esn(
        reservoir_size=300,
        feedback_size=3,
        output_size=3,
        spectral_radius=0.9,
        topology="erdos_renyi",
        readout_alpha=READOUT_ALPHA,
        seed=MODEL_SEED,
    )
    model.double()
    return model


def _build_power() -> ESNModel:
    """Power-augmented ESN, fully seeded, float64."""
    model = power_augmented(
        reservoir_size=300,
        feedback_size=3,
        output_size=3,
        spectral_radius=0.9,
        topology="erdos_renyi",
        readout_alpha=READOUT_ALPHA,
        seed=MODEL_SEED,
    )
    model.double()
    return model


def _build_ngrc() -> ESNModel:
    """NG-RC forecaster: ``Input -> NGReservoir -> CGReadoutLayer``, float64.

    NG-RC has no random weights, so the build is deterministic regardless of the
    seed; ``torch.manual_seed`` is set anyway for symmetry with the ESN builders.
    The readout emits ``input_dim`` features so the single output matches the
    feedback dimension — the precondition for autoregressive forecasting.
    """
    torch.manual_seed(MODEL_SEED)
    inp = Input(shape=(100, 3))
    reservoir = NGReservoir(input_dim=3, k=2, s=1, p=2)
    features = reservoir(inp)
    readout = CGReadoutLayer(reservoir.feature_dim, 3, alpha=1e-3, name="output")(features)
    model = ESNModel(inp, readout)
    model.double()
    return model


@dataclass(frozen=True)
class GoldenSpec:
    """One premade model under the golden-forecast contract.

    Parameters
    ----------
    name : str
        Fixture stem (``tests/fixtures/golden_forecast/<name>.npz``) and pytest id.
    builder : Callable[[], ESNModel]
        Returns a fresh, untrained, ``float64`` model seeded with
        :data:`MODEL_SEED`.
    description : str
        One-line architecture summary, recorded in the fixture metadata.
    """

    name: str
    builder: Callable[[], ESNModel]
    description: str


SPECS: list[GoldenSpec] = [
    GoldenSpec("classic_esn", _build_classic, "Input -> Reservoir -> Concat(Input) -> Readout"),
    GoldenSpec("ott_esn", _build_ott, "Ott square-even state augmentation"),
    GoldenSpec("power_augmented", _build_power, "Power (exponent=2.0) state augmentation"),
    GoldenSpec("ngrc", _build_ngrc, "NG-RC k=2,s=1,p=2 delay-embedding forecaster"),
]


# --------------------------------------------------------------------------- #
# Data / forecast / metric helpers
# --------------------------------------------------------------------------- #

Splits = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def make_lorenz_splits() -> Splits:
    """Return the canonical, deterministic Lorenz-63 splits used by every model.

    Built from :func:`resdag.lorenz` (NumPy float64 RK4) and split with
    :func:`resdag.utils.prepare_esn_data`, whose autoregressive-seam alignment
    makes ``forecast(forecast_warmup, horizon=HORIZON)`` line up element-for-element
    with the returned ``val`` (the ground-truth continuation used for VPT).

    Returns
    -------
    tuple of torch.Tensor
        ``(warmup, train, target, forecast_warmup, val)`` of dtype
        :data:`DTYPE`, each shaped ``(1, *, 3)``.
    """
    n = WARMUP_STEPS + TRAIN_STEPS + HORIZON + 1
    data = rd.lorenz(n, dt=0.02, normalize="standard", seed=DATA_SEED, dtype=DTYPE)
    splits = rd.utils.prepare_esn_data(
        data,
        warmup_steps=WARMUP_STEPS,
        train_steps=TRAIN_STEPS,
        val_steps=HORIZON,
    )
    # ``return_stats`` defaults to False, so this is always the 5-tuple form.
    return cast(Splits, splits)


def train_and_forecast(model: ESNModel, splits: Splits) -> torch.Tensor:
    """Fit the readout and return the autonomous forecast over :data:`HORIZON`.

    Parameters
    ----------
    model : ESNModel
        A fresh model from a :class:`GoldenSpec` builder.
    splits : tuple of torch.Tensor
        The output of :func:`make_lorenz_splits`.

    Returns
    -------
    torch.Tensor
        Forecast of shape ``(1, HORIZON, 3)`` in :data:`DTYPE`.
    """
    warmup, train, target, forecast_warmup, val = splits
    ESNTrainer(model).fit(
        warmup_inputs=(warmup,),
        train_inputs=(train,),
        targets={"output": target},
    )
    forecast = model.forecast(forecast_warmup, horizon=val.shape[1])
    assert isinstance(forecast, torch.Tensor)  # single-output models here
    return forecast


def valid_prediction_time(
    pred: torch.Tensor,
    truth: torch.Tensor,
    threshold: float = VPT_THRESHOLD,
) -> int:
    """Contiguous valid forecast horizon (Valid Prediction Time).

    Per-step error is NRMSE — RMSE normalised by the per-dimension standard
    deviation of ``truth`` over ``(batch, time)`` — collapsed across the batch
    with the median, mirroring :func:`resdag.hpo.losses.forecast_horizon`
    (re-implemented here so the normal suite never imports the optional HPO
    stack).  The VPT is the number of steps, counting from the start, whose
    error stays below ``threshold``.

    Parameters
    ----------
    pred, truth : torch.Tensor
        Forecast and ground truth of shape ``(batch, time, dim)``.
    threshold : float, default=:data:`VPT_THRESHOLD`
        NRMSE threshold below which a step is "valid".

    Returns
    -------
    int
        Valid Prediction Time in steps (``0`` if the very first step already
        exceeds the threshold).
    """
    p = pred.detach().cpu().double().numpy()
    t = truth.detach().cpu().double().numpy()
    scale = t.std(axis=(0, 1), keepdims=True)
    scale = np.where(scale == 0.0, 1.0, scale)
    err = np.sqrt((((p - t) / scale) ** 2).mean(axis=2))  # (batch, time)
    e_t = np.median(err, axis=0)  # (time,)
    below = e_t < threshold
    if not below[0]:
        return 0
    return int(np.argmax(~below)) if (~below).any() else int(below.size)


def vpt_floor(measured_vpt: int) -> int:
    """Conservative, cross-platform-robust VPT floor (see the module constants).

    Returns a fraction (:data:`VPT_FLOOR_FRAC`) of the measured VPT, never below
    :data:`VPT_FLOOR_MIN`.  Pinned well below the measured value on purpose: the
    valid horizon of a chaotic forecast varies ~40% across BLAS implementations,
    so a tight floor would be machine-specific.  This tier only catches a
    collapse; the prefix match is the precise guard.
    """
    return max(VPT_FLOOR_MIN, round(VPT_FLOOR_FRAC * measured_vpt))


# --------------------------------------------------------------------------- #
# Fixture I/O
# --------------------------------------------------------------------------- #


def fixture_path(name: str) -> Path:
    """Path to the committed fixture for model ``name``."""
    return FIXTURES_DIR / f"{name}.npz"


def build_payload(spec: GoldenSpec, splits: Splits) -> dict[str, Any]:
    """Build, train, and forecast ``spec`` into a fixture payload dict.

    Returns
    -------
    dict
        ``{"golden_forecast": np.ndarray, "meta": dict}`` ready for
        :func:`save_fixture`.  Used by the regen tool and the in-test
        self-consistency check.
    """
    *_, val = splits
    model = spec.builder()
    forecast = train_and_forecast(model, splits)
    measured = valid_prediction_time(forecast, val)
    meta = {
        "fixture_version": FIXTURE_VERSION,
        "name": spec.name,
        "description": spec.description,
        "dtype": "float64",
        "device": "cpu",
        "data_seed": DATA_SEED,
        "model_seed": MODEL_SEED,
        "warmup_steps": WARMUP_STEPS,
        "train_steps": TRAIN_STEPS,
        "horizon": HORIZON,
        "prefix_steps": PREFIX_STEPS,
        "prefix_atol": PREFIX_ATOL,
        "prefix_rtol": PREFIX_RTOL,
        "vpt_threshold": VPT_THRESHOLD,
        "vpt": int(measured),
        "vpt_floor": int(vpt_floor(measured)),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    return {"golden_forecast": forecast.detach().cpu().double().numpy(), "meta": meta}


def save_fixture(path: Path, payload: dict[str, Any]) -> None:
    """Write a payload to ``path`` as a compressed ``.npz``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        golden_forecast=payload["golden_forecast"],
        meta=np.array(json.dumps(payload["meta"], sort_keys=True)),
    )


def load_fixture(name: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a committed fixture: ``(golden_forecast, meta)``."""
    with np.load(fixture_path(name)) as data:
        golden = data["golden_forecast"]
        meta = json.loads(str(data["meta"]))
    return golden, meta
