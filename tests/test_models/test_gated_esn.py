"""Contract of the gated ("reservoir GRU") ESN factory (resdag.models.gated_esn).

Pins down :func:`resdag.models.gated_esn.gated_esn`:

- architecture (a GatedReservoirLayer + input concatenation + ridge readout),
- forward shapes (feedback-only and driven),
- frozen vs trainable plumbing,
- an end-to-end train (ESNTrainer.fit) + finite Lorenz forecast with a
  non-trivial valid prediction time,
- save_full / load_full round-trip.

Imported directly from the module file (public ``__init__`` wiring is owned
elsewhere).
"""

import tempfile
from pathlib import Path

import pytest
import torch

import resdag as rd
from resdag import ESNModel
from resdag.layers.readouts import RidgeReadoutLayer
from resdag.layers.reservoirs.gated import GatedReservoirLayer
from resdag.metrics import valid_prediction_time
from resdag.models.gated_esn import gated_esn
from resdag.training import ESNTrainer


class TestGatedESNArchitecture:
    """Structure and forward shapes of the gated_esn factory."""

    def test_basic_instantiation(self) -> None:
        """The model wires a GatedReservoirLayer and a ridge readout."""
        model = gated_esn(reservoir_size=50, feedback_size=2, output_size=3)
        assert model is not None
        submodules = [m for _, m in model.named_modules()]
        assert any(isinstance(m, GatedReservoirLayer) for m in submodules)
        assert any(isinstance(m, RidgeReadoutLayer) for m in submodules)

    def test_forward_pass_shape(self) -> None:
        """Feedback-only forward maps ``(B, T, F)`` -> ``(B, T, output)``."""
        model = gated_esn(reservoir_size=50, feedback_size=2, output_size=3)
        out = model(torch.randn(4, 20, 2))
        assert out.shape == (4, 20, 3)

    def test_driven_forward_shape(self) -> None:
        """An ``input_size`` model takes ``(feedback, driver)``."""
        model = gated_esn(reservoir_size=40, feedback_size=3, output_size=3, input_size=2)
        out = model(torch.randn(2, 15, 3), torch.randn(2, 15, 2))
        assert out.shape == (2, 15, 3)

    def test_concat_input_readout_features(self) -> None:
        """With concat_input the readout sees feedback + reservoir features."""
        model = gated_esn(reservoir_size=50, feedback_size=2, output_size=3)
        readout = next(m for _, m in model.named_modules() if isinstance(m, RidgeReadoutLayer))
        assert readout.in_features == 2 + 50

    def test_no_concat_input_readout_features(self) -> None:
        """Without concat_input the readout sees only reservoir features."""
        model = gated_esn(reservoir_size=50, feedback_size=2, output_size=3, concat_input=False)
        readout = next(m for _, m in model.named_modules() if isinstance(m, RidgeReadoutLayer))
        assert readout.in_features == 50

    def test_frozen_by_default(self) -> None:
        """The reservoir is frozen by default."""
        model = gated_esn(reservoir_size=32, feedback_size=3, output_size=3)
        reservoir = next(m for _, m in model.named_modules() if isinstance(m, GatedReservoirLayer))
        assert reservoir.trainable is False
        assert all(not p.requires_grad for p in reservoir.cell.parameters())

    def test_trainable_reservoir_plumbed(self) -> None:
        """``trainable=True`` reaches the underlying cell."""
        model = gated_esn(reservoir_size=32, feedback_size=3, output_size=3, trainable=True)
        reservoir = next(m for _, m in model.named_modules() if isinstance(m, GatedReservoirLayer))
        assert reservoir.trainable is True
        assert all(p.requires_grad for p in reservoir.cell.parameters())

    def test_invalid_output_size_raises(self) -> None:
        """A non-positive output_size is rejected."""
        with pytest.raises(ValueError, match="output_size must be a positive integer"):
            gated_esn(reservoir_size=32, feedback_size=3, output_size=0)

    def test_unknown_readout_raises(self) -> None:
        """An unknown readout string is rejected before construction."""
        with pytest.raises(ValueError, match="Unknown readout"):
            gated_esn(reservoir_size=32, feedback_size=3, output_size=3, readout="bogus")


class TestGatedESNForecast:
    """End-to-end train + finite forecast on Lorenz-63."""

    @staticmethod
    def _lorenz_splits() -> tuple[torch.Tensor, ...]:
        dt = 0.02
        warmup_steps, train_steps, horizon = 500, 3000, 400
        n = warmup_steps + train_steps + horizon + 1
        data = rd.lorenz(n, dt=dt, normalize="standard", seed=42, dtype=torch.float64)
        return rd.utils.prepare_esn_data(
            data,
            warmup_steps=warmup_steps,
            train_steps=train_steps,
            val_steps=horizon,
        )

    def test_frozen_trains_and_forecasts_finitely(self) -> None:
        """A frozen gated_esn fits its readout and forecasts finite values.

        VPT is reported via ``print`` and only asserted to be non-trivial: a
        frozen gated reservoir legitimately underperforms an ESN, so the bar is
        a low honest floor (the trajectory tracks for a handful of steps), not a
        competitive horizon.
        """
        torch.manual_seed(0)
        warmup, train, target, fwarmup, val = self._lorenz_splits()

        model = gated_esn(
            reservoir_size=400,
            feedback_size=3,
            output_size=3,
            spectral_radius=0.9,
            seed=13,
        ).double()
        ESNTrainer(model).fit(
            warmup_inputs=(warmup,),
            train_inputs=(train,),
            targets={"output": target},
        )
        forecast = model.forecast(fwarmup, horizon=val.shape[1])
        assert isinstance(forecast, torch.Tensor)
        assert forecast.shape == (1, val.shape[1], 3)
        assert torch.isfinite(forecast).all()

        vpt = int(valid_prediction_time(forecast, val, threshold=0.3))
        print(f"[gated_esn] frozen Lorenz VPT = {vpt} steps")
        # Honest, conservative floor: the frozen gated reservoir at least tracks
        # the attractor for several steps before diverging.
        assert vpt >= 5

    def test_save_full_load_full_roundtrip(self) -> None:
        """save_full / load_full reproduce the forecast bit-exactly."""
        torch.manual_seed(0)
        warmup, train, target, fwarmup, _ = self._lorenz_splits()
        model = gated_esn(reservoir_size=200, feedback_size=3, output_size=3, seed=13).double()
        ESNTrainer(model).fit(
            warmup_inputs=(warmup,),
            train_inputs=(train,),
            targets={"output": target},
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gated.pt"
            model.save_full(path)
            restored = ESNModel.load_full(path, map_location="cpu", trusted=True)

        f_orig = model.forecast(fwarmup, horizon=50)
        f_rest = restored.forecast(fwarmup, horizon=50)
        assert torch.equal(f_orig, f_rest)
