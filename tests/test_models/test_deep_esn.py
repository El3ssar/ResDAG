"""Deep (stacked-reservoir) ESN factory contracts (resdag.models.deep_esn).

Pins down the DeepESN wiring: an N-layer reservoir stack where each layer's state
drives the next, a readout over the concatenation of all layers' states (or the
last layer only), a per-layer leak-rate schedule, and the sizing knobs
(reservoir_sizes list vs scalar + n_layers). Also checks it trains and produces a
finite autoregressive forecast.
"""

import pytest
import torch

from resdag.layers import Concatenate
from resdag.layers.cells import ESNCell
from resdag.layers.readouts import ReadoutLayer, RidgeReadoutLayer
from resdag.models import deep_esn
from resdag.training import ESNTrainer


class TestDeepESNConstruction:
    """Stack sizing, wiring, and readout feature dimensions."""

    def test_scalar_size_with_n_layers(self) -> None:
        """A scalar reservoir_sizes broadcasts to n_layers layers."""
        model = deep_esn(reservoir_sizes=40, n_layers=3, feedback_size=2, output_size=2)

        cells = [m for m in model.modules() if isinstance(m, ESNCell)]
        assert len(cells) == 3
        assert all(c.reservoir_size == 40 for c in cells)

    def test_explicit_size_list(self) -> None:
        """A per-layer list sizes each layer independently."""
        model = deep_esn(reservoir_sizes=[50, 30, 20], feedback_size=2, output_size=2)

        cells = [m for m in model.modules() if isinstance(m, ESNCell)]
        assert [c.reservoir_size for c in cells] == [50, 30, 20]

    def test_interlayer_feedback_dims(self) -> None:
        """Layer i's feedback dim is the previous reservoir size (feedback_size first)."""
        model = deep_esn(reservoir_sizes=[50, 30, 20], feedback_size=3, output_size=3)

        cells = [m for m in model.modules() if isinstance(m, ESNCell)]
        # First layer sees the raw 3-dim feedback; each next sees the prior state.
        assert [c.feedback_size for c in cells] == [3, 50, 30]

    def test_concat_all_states_readout_in_features(self) -> None:
        """The default readout sees the concatenation of every layer's state."""
        model = deep_esn(reservoir_sizes=[50, 30, 20], feedback_size=2, output_size=2)

        assert any(isinstance(m, Concatenate) for m in model.modules())
        readout = next(m for m in model.modules() if isinstance(m, ReadoutLayer))
        assert readout.in_features == 100  # 50 + 30 + 20

    def test_last_state_only_readout_in_features(self) -> None:
        """concat_all_states=False reads out from the last layer alone."""
        model = deep_esn(
            reservoir_sizes=[50, 30, 20],
            feedback_size=2,
            output_size=2,
            concat_all_states=False,
        )

        readout = next(m for m in model.modules() if isinstance(m, ReadoutLayer))
        assert readout.in_features == 20  # last layer only

    def test_default_readout_is_ridge(self) -> None:
        """The default readout is the exact Cholesky ridge solver."""
        model = deep_esn(reservoir_sizes=30, n_layers=2, feedback_size=2, output_size=2)
        readouts = [m for m in model.modules() if isinstance(m, ReadoutLayer)]
        assert readouts and all(isinstance(r, RidgeReadoutLayer) for r in readouts)

    def test_single_layer_no_concat(self) -> None:
        """A one-layer stack has no Concatenate and reads out from that layer."""
        model = deep_esn(reservoir_sizes=[40], feedback_size=2, output_size=2)

        assert not any(isinstance(m, Concatenate) for m in model.modules())
        readout = next(m for m in model.modules() if isinstance(m, ReadoutLayer))
        assert readout.in_features == 40


class TestDeepESNLeakSchedule:
    """Per-layer leak-rate schedule."""

    def test_scalar_leak_shared(self) -> None:
        """A scalar leak_rate is shared by every layer."""
        model = deep_esn(
            reservoir_sizes=30, n_layers=3, feedback_size=2, output_size=2, leak_rate=0.4
        )

        cells = [m for m in model.modules() if isinstance(m, ESNCell)]
        assert all(c.leak_rate == pytest.approx(0.4) for c in cells)

    def test_per_layer_leak_schedule(self) -> None:
        """A list leak_rate sets a decreasing-with-depth schedule per layer."""
        model = deep_esn(
            reservoir_sizes=30,
            n_layers=3,
            feedback_size=2,
            output_size=2,
            leak_rate=[1.0, 0.6, 0.3],
        )

        cells = [m for m in model.modules() if isinstance(m, ESNCell)]
        assert [pytest.approx(c.leak_rate) for c in cells] == [1.0, 0.6, 0.3]

    def test_leak_schedule_length_mismatch_raises(self) -> None:
        """A per-layer leak schedule must have one entry per layer."""
        with pytest.raises(ValueError, match="leak_rate schedule length"):
            deep_esn(
                reservoir_sizes=30,
                n_layers=3,
                feedback_size=2,
                output_size=2,
                leak_rate=[1.0, 0.5],  # only 2 for 3 layers
            )


class TestDeepESNValidation:
    """Error paths for the sizing/leak arguments."""

    def test_scalar_size_without_n_layers_raises(self) -> None:
        with pytest.raises(ValueError, match="n_layers is required"):
            deep_esn(reservoir_sizes=40, feedback_size=2, output_size=2)

    def test_n_layers_contradicts_list_raises(self) -> None:
        with pytest.raises(ValueError, match="contradicts"):
            deep_esn(reservoir_sizes=[40, 40], n_layers=3, feedback_size=2, output_size=2)

    def test_empty_sizes_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one layer"):
            deep_esn(reservoir_sizes=[], feedback_size=2, output_size=2)

    def test_non_positive_size_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            deep_esn(reservoir_sizes=[40, 0], feedback_size=2, output_size=2)

    def test_non_positive_output_size_raises(self) -> None:
        with pytest.raises(ValueError, match="output_size"):
            deep_esn(reservoir_sizes=30, n_layers=2, feedback_size=2, output_size=0)


class TestDeepESNForward:
    """Forward, training, and autoregressive forecast."""

    def test_forward_shape(self) -> None:
        model = deep_esn(reservoir_sizes=[40, 40], feedback_size=3, output_size=3)
        x = torch.randn(2, 20, 3)
        assert model(x).shape == (2, 20, 3)

    def test_fit_and_forecast_finite(self) -> None:
        """A deep ESN fits its readout and forecasts finitely (feedback dim == output dim)."""
        torch.manual_seed(0)
        model = deep_esn(
            reservoir_sizes=[60, 60],
            feedback_size=3,
            output_size=3,
            leak_rate=[1.0, 0.6],
            spectral_radius=0.9,
            readout_alpha=1e-4,
        )
        warmup = torch.randn(1, 60, 3)
        train_in = torch.randn(1, 200, 3)
        target = torch.randn(1, 200, 3)

        ESNTrainer(model).fit(
            warmup_inputs=(warmup,),
            train_inputs=(train_in,),
            targets={"output": target},
        )
        preds = model.forecast(warmup, horizon=50)

        assert preds.shape == (1, 50, 3)
        assert torch.isfinite(preds).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDeepESNGPU:
    """Deep ESN runs on GPU."""

    def test_forward_on_gpu(self) -> None:
        model = deep_esn(reservoir_sizes=[40, 40], feedback_size=2, output_size=2).cuda()
        x = torch.randn(2, 20, 2).cuda()
        out = model(x)
        assert out.is_cuda
        assert out.shape == (2, 20, 2)
