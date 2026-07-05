"""Regression tests: write-through delegation of live hyperparameters.

Historically the reservoir layers delegated attribute *reads* to the inner cell
via ``__getattr__`` but sent *writes* to the layer's own ``__dict__`` (plain
``nn.Module.__setattr__``), creating a shadow attribute.  So

    layer.leak_rate = 0.5        # read back 0.5, but the cell keeps using 1.0
    reservoir.trainable = True   # never flipped requires_grad on any parameter

silently diverged from the cell's live computation.  ``reservoir.trainable =
True`` in particular is the verbatim BPTT-enable recipe in the ``forecast``
docstring, and it was a silent no-op — the optimizer never touched the reservoir
weights and no error was raised.

These tests pin the fix:

- ``trainable`` is a live property on ``ESNCell`` / ``ReadoutLayer`` whose setter
  flips ``requires_grad_`` on every parameter;
- ``BaseReservoirLayer.__setattr__`` forwards writes for the live-config
  allow-list (``trainable``, ``leak_rate``, ``noise``) to the cell, while
  Parameters / Modules / Tensors / ``_``-prefixed internals keep registering
  through ``nn.Module.__setattr__``;
- structurally-baked config (``spectral_radius``, ``reservoir_size``,
  ``feedback_size``, ``input_size``) raises rather than silently shadowing;
- none of this breaks nn.Module registration, ``.to()`` / ``.double()``,
  ``state_dict``, pickling, or the ``pytorch_symbolic`` build path.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from resdag.layers import ESNLayer, NGReservoirLayer
from resdag.layers.readouts import CGReadoutLayer
from resdag.layers.readouts.base import ReadoutLayer


class TestTrainableLiveSwitch:
    """``layer.trainable = ...`` flips requires_grad on every parameter."""

    def test_starts_frozen(self) -> None:
        """A default (non-trainable) layer has all parameters frozen."""
        layer = ESNLayer(reservoir_size=40, feedback_size=3, seed=0)
        assert layer.trainable is False
        assert not any(p.requires_grad for p in layer.parameters())

    def test_layer_write_flips_requires_grad(self) -> None:
        """Setting trainable on the LAYER unfreezes all cell parameters."""
        layer = ESNLayer(reservoir_size=40, feedback_size=3, seed=0)

        layer.trainable = True

        assert layer.trainable is True  # read-back on the layer
        assert layer.cell.trainable is True  # forwarded to the cell
        params = list(layer.parameters())
        assert params  # sanity: there ARE parameters
        assert all(p.requires_grad for p in params)

    def test_cell_write_flips_requires_grad(self) -> None:
        """Setting trainable directly on the CELL also flips requires_grad."""
        layer = ESNLayer(reservoir_size=40, feedback_size=3, seed=0)

        layer.cell.trainable = True

        assert all(p.requires_grad for p in layer.parameters())

    def test_false_refreezes(self) -> None:
        """trainable = False re-freezes every parameter."""
        layer = ESNLayer(reservoir_size=40, feedback_size=3, trainable=True, seed=0)
        assert all(p.requires_grad for p in layer.parameters())

        layer.trainable = False

        assert layer.trainable is False
        assert not any(p.requires_grad for p in layer.parameters())

    def test_constructor_trainable_true_is_applied(self) -> None:
        """trainable=True at construction unfreezes (property setter runs post-init)."""
        layer = ESNLayer(reservoir_size=40, feedback_size=3, trainable=True, seed=0)
        assert all(p.requires_grad for p in layer.parameters())

    def test_bptt_step_updates_reservoir_weights(self) -> None:
        """After trainable=True a real optimizer step moves the recurrent weights.

        This is the documented BPTT recipe (``reservoir.trainable = True``): it
        must genuinely train the reservoir, not silently no-op.
        """
        layer = ESNLayer(reservoir_size=40, feedback_size=3, seed=0)
        layer.trainable = True

        before = layer.weight_hh.detach().clone()
        opt = torch.optim.SGD(layer.parameters(), lr=0.1)
        out = layer(torch.randn(2, 8, 3))
        out.pow(2).mean().backward()

        assert layer.weight_hh.grad is not None
        assert layer.weight_hh.grad.abs().sum() > 0
        opt.step()
        assert not torch.allclose(before, layer.weight_hh.detach())


class TestReadoutTrainableLiveSwitch:
    """ReadoutLayer.trainable has the same live semantics (readouts/base.py)."""

    def test_starts_frozen(self) -> None:
        readout = CGReadoutLayer(20, 3, name="output")
        assert readout.trainable is False
        assert not any(p.requires_grad for p in readout.parameters())

    def test_write_flips_requires_grad(self) -> None:
        readout = CGReadoutLayer(20, 3, name="output")
        readout.trainable = True
        assert readout.trainable is True
        assert all(p.requires_grad for p in readout.parameters())

    def test_false_refreezes(self) -> None:
        readout = CGReadoutLayer(20, 3, trainable=True)
        assert all(p.requires_grad for p in readout.parameters())
        readout.trainable = False
        assert not any(p.requires_grad for p in readout.parameters())

    def test_bias_false_only_weight(self) -> None:
        """A no-bias readout has only ``weight``; the switch still flips it."""
        readout = ReadoutLayer(10, 2, bias=False)
        assert readout.bias is None
        readout.trainable = True
        assert readout.weight.requires_grad is True

    def test_is_fitted_buffer_survives(self) -> None:
        """The is_fitted buffer is unaffected by the trainable refactor."""
        readout = CGReadoutLayer(20, 3, name="output")
        assert readout.is_fitted is False
        assert "_is_fitted" in readout.state_dict()


class TestLeakRateWriteThrough:
    """``layer.leak_rate = ...`` changes the cell's live computation."""

    def test_write_reaches_cell(self) -> None:
        layer = ESNLayer(reservoir_size=40, feedback_size=3, leak_rate=1.0, seed=7)
        layer.leak_rate = 0.5
        assert layer.leak_rate == 0.5
        assert layer.cell.leak_rate == 0.5

    def test_write_changes_output(self) -> None:
        """A post-init leak_rate write actually alters the produced sequence."""
        feedback = torch.randn(2, 12, 3)

        baseline = ESNLayer(reservoir_size=40, feedback_size=3, leak_rate=1.0, seed=7)
        out_default = baseline(feedback)

        changed = ESNLayer(reservoir_size=40, feedback_size=3, leak_rate=1.0, seed=7)
        changed.leak_rate = 0.5
        changed.reset_state()
        out_leaked = changed(feedback)

        assert not torch.allclose(out_default, out_leaked)

    def test_write_matches_constructor_value(self) -> None:
        """Writing leak_rate post-init == building with that leak_rate."""
        feedback = torch.randn(2, 12, 3)

        constructed = ESNLayer(reservoir_size=40, feedback_size=3, leak_rate=0.3, seed=11)
        out_constructed = constructed(feedback)

        mutated = ESNLayer(reservoir_size=40, feedback_size=3, leak_rate=1.0, seed=11)
        mutated.leak_rate = 0.3
        mutated.reset_state()
        out_mutated = mutated(feedback)

        assert torch.allclose(out_constructed, out_mutated)

    def test_vector_write_forwards_not_registered_as_buffer(self) -> None:
        """A Tensor-valued leak_rate write forwards to the cell, not a layer buffer.

        The ``__setattr__`` allow-list must route a per-neuron leak vector to the
        cell (where it is validated + buffered) rather than letting the generic
        ``nn.Module`` path register a stray ``leak_rate`` buffer on the layer.
        """
        n = 40
        layer = ESNLayer(reservoir_size=n, feedback_size=3, leak_rate=1.0, seed=7)
        leak = torch.rand(n) * 0.5 + 0.4

        layer.leak_rate = leak
        # Reached the cell as a validated vector...
        assert layer.cell._is_vector_leak is True
        assert torch.allclose(layer.cell.leak_rate, leak)
        # ...and did NOT shadow as a plain buffer on the layer itself.
        assert "leak_rate" not in layer._buffers

    def test_vector_write_matches_constructor_value(self) -> None:
        """Writing a vector leak post-init == building with that vector leak."""
        n = 40
        feedback = torch.randn(2, 12, 3)
        leak = torch.rand(n) * 0.5 + 0.3

        constructed = ESNLayer(reservoir_size=n, feedback_size=3, leak_rate=leak, seed=11)
        out_constructed = constructed(feedback)

        mutated = ESNLayer(reservoir_size=n, feedback_size=3, leak_rate=1.0, seed=11)
        mutated.leak_rate = leak
        mutated.reset_state()
        out_mutated = mutated(feedback)

        assert torch.allclose(out_constructed, out_mutated)


class TestNoiseWriteThrough:
    """``layer.noise = ...`` reaches the cell's train-mode noise injector."""

    def test_write_reaches_cell(self) -> None:
        layer = ESNLayer(reservoir_size=40, feedback_size=3, noise=0.0, seed=3)
        layer.noise = 0.3
        assert layer.noise == 0.3
        assert layer.cell.noise == 0.3

    def test_write_perturbs_train_output(self) -> None:
        feedback = torch.randn(2, 12, 3)
        layer = ESNLayer(reservoir_size=40, feedback_size=3, noise=0.0, seed=3)
        layer.train()
        out_clean = layer(feedback)

        layer.noise = 0.3
        layer.reset_state()
        out_noisy = layer(feedback)

        assert not torch.allclose(out_clean, out_noisy)


class TestBakedAttrsRaise:
    """Structurally-baked config cannot be mutated live — it raises."""

    @pytest.mark.parametrize(
        "attr, value",
        [
            ("spectral_radius", 0.5),
            ("reservoir_size", 99),
            ("feedback_size", 9),
            ("input_size", 4),
        ],
    )
    def test_baked_attr_write_raises(self, attr: str, value: object) -> None:
        """Assigning a baked attr raises AttributeError, never silently shadows."""
        layer = ESNLayer(
            reservoir_size=40, feedback_size=3, input_size=2, spectral_radius=0.9, seed=1
        )
        with pytest.raises(AttributeError, match="baked|cannot be changed"):
            setattr(layer, attr, value)

    def test_baked_attr_not_shadowed(self) -> None:
        """A rejected write leaves the original (cell) value intact — no shadow."""
        layer = ESNLayer(reservoir_size=40, feedback_size=3, spectral_radius=0.9, seed=1)
        with pytest.raises(AttributeError):
            layer.spectral_radius = 0.5
        # Still reads the construction value via delegation, not a shadow.
        assert layer.spectral_radius == 0.9


class TestSetattrDoesNotBreakModule:
    """The __setattr__ override must not disturb nn.Module registration."""

    def test_parameters_register(self) -> None:
        layer = ESNLayer(reservoir_size=40, feedback_size=3, seed=0)
        names = {n for n, _ in layer.named_parameters()}
        assert "cell.weight_feedback" in names
        assert "cell.weight_hh" in names
        assert "cell.bias_h" in names

    def test_state_dict_roundtrip(self) -> None:
        a = ESNLayer(reservoir_size=40, feedback_size=3, seed=0)
        b = ESNLayer(reservoir_size=40, feedback_size=3, seed=1)
        assert not torch.allclose(a.weight_hh, b.weight_hh)
        b.load_state_dict(a.state_dict())
        assert torch.allclose(a.weight_hh, b.weight_hh)

    def test_double_preserves_and_runs(self) -> None:
        layer = ESNLayer(reservoir_size=40, feedback_size=3, seed=0).double()
        out = layer(torch.randn(2, 6, 3, dtype=torch.float64))
        assert out.dtype == torch.float64

    def test_state_buffer_still_registers(self) -> None:
        """The non-persistent ``state`` buffer still registers (not shadowed)."""
        layer = ESNLayer(reservoir_size=40, feedback_size=3, seed=0)
        layer(torch.randn(2, 5, 3))
        assert "state" in dict(layer.named_buffers())
        assert "state" not in layer.state_dict()

    def test_detach_flag_settable(self) -> None:
        """A plain non-cell layer attribute (detach_state_between_calls) still sets."""
        layer = ESNLayer(reservoir_size=40, feedback_size=3, seed=0)
        layer.detach_state_between_calls = False
        assert layer.detach_state_between_calls is False

    def test_pickle_roundtrip_preserves_trainable(self) -> None:
        """Whole-cell pickling round-trips the trainable state (backing field)."""
        import pickle

        layer = ESNLayer(reservoir_size=30, feedback_size=3, trainable=True, seed=0)
        restored = pickle.loads(pickle.dumps(layer.cell))
        assert restored.trainable is True
        assert all(p.requires_grad for p in restored.parameters())

    def test_pytorch_symbolic_build_and_bptt_recipe(self) -> None:
        """The ps build path works and the BPTT recipe unfreezes an embedded layer."""
        import pytorch_symbolic as ps

        from resdag import ESNModel

        inp = ps.Input((10, 3))
        res = ESNLayer(50, feedback_size=3, seed=0)(inp)
        readout = CGReadoutLayer(50, 3, name="output")(res)
        model = ESNModel(inp, readout)

        out = model(torch.randn(2, 10, 3))
        assert out.shape == (2, 10, 3)

        reservoir = next(m for m in model.modules() if isinstance(m, ESNLayer))
        reservoir.trainable = True
        assert all(p.requires_grad for p in reservoir.parameters())

    def test_save_full_load_full_roundtrip(self) -> None:
        """save_full / load_full survive the property + __setattr__ changes."""
        import pytorch_symbolic as ps

        from resdag import ESNModel

        inp = ps.Input((10, 3))
        res = ESNLayer(50, feedback_size=3, seed=0)(inp)
        readout = CGReadoutLayer(50, 3, name="output")(res)
        model = ESNModel(inp, readout)
        x = torch.randn(2, 10, 3)
        expected = model(x)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model_full.pt"
            model.save_full(path)
            restored = ESNModel.load_full(path)

        restored.reset_reservoirs()
        model.reset_reservoirs()
        assert torch.allclose(restored(x), expected, atol=1e-6)


class TestNGReservoirSetattr:
    """NGReservoirLayer (weightless cell) is unaffected by the write-through logic."""

    def test_arbitrary_attr_falls_through(self) -> None:
        """A name the NGCell does not own registers on the layer as usual."""
        ng = NGReservoirLayer(input_dim=3, k=2, s=1, p=2)
        ng.some_flag = True
        assert ng.some_flag is True

    def test_readonly_property_still_readonly(self) -> None:
        """The layer's own read-only ``feature_dim`` property stays read-only."""
        ng = NGReservoirLayer(input_dim=3, k=2, s=1, p=2)
        with pytest.raises(AttributeError):
            ng.feature_dim = 99

    def test_detach_flag_settable(self) -> None:
        ng = NGReservoirLayer(input_dim=3, k=2, s=1, p=2)
        ng.detach_state_between_calls = False
        assert ng.detach_state_between_calls is False

    def test_forward_still_works(self) -> None:
        ng = NGReservoirLayer(input_dim=3, k=2, s=1, p=2)
        out = ng(torch.randn(2, 20, 3))
        assert out.shape[0] == 2 and out.shape[1] == 20


class TestLiveConfigForwardingHardening:
    """Consolidation-pass regressions: tensor-valued forwarding + live validation."""

    def test_tensor_valued_leak_rate_reaches_cell_not_shadow(self) -> None:
        """A Tensor-valued config write forwards to the cell instead of shadowing.

        The ``__setattr__`` guard used to route *any* Tensor straight to
        ``nn.Module.__setattr__``, so ``layer.leak_rate = torch.tensor(0.4)``
        silently shadowed on the layer while the cell kept its old value — the
        exact shadow bug the write-through fix set out to kill, resurfacing for
        tensor values.
        """
        layer = ESNLayer(reservoir_size=30, feedback_size=2, leak_rate=1.0, seed=1)
        layer.leak_rate = torch.tensor(0.4)
        assert layer.cell.leak_rate == pytest.approx(0.4)
        assert "leak_rate" not in layer.__dict__  # no shadow on the layer

    def test_baked_attr_raises_even_for_tensor_value(self) -> None:
        layer = ESNLayer(reservoir_size=30, feedback_size=2, seed=1)
        with pytest.raises(AttributeError, match="baked into the reservoir weights"):
            layer.spectral_radius = torch.tensor(0.5)

    def test_parameter_and_module_writes_still_register(self) -> None:
        layer = ESNLayer(reservoir_size=30, feedback_size=2, seed=1)
        layer.extra = torch.nn.Linear(2, 2)
        assert "extra" in dict(layer.named_modules())
        layer.new_param = torch.nn.Parameter(torch.zeros(3))
        assert "new_param" in dict(layer.named_parameters())

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 2.0])
    def test_live_leak_rate_out_of_range_raises(self, bad: float) -> None:
        layer = ESNLayer(reservoir_size=30, feedback_size=2, leak_rate=1.0, seed=1)
        with pytest.raises(ValueError, match="leak_rate must be in"):
            layer.leak_rate = bad
        assert layer.cell.leak_rate == 1.0  # unchanged after the rejected write

    def test_live_negative_noise_raises(self) -> None:
        layer = ESNLayer(reservoir_size=30, feedback_size=2, noise=0.0, seed=1)
        with pytest.raises(ValueError, match="noise must be non-negative"):
            layer.noise = -1.0
        assert layer.cell.noise == 0.0

    def test_construction_still_validates_leak_and_noise(self) -> None:
        with pytest.raises(ValueError, match="leak_rate must be in"):
            ESNLayer(reservoir_size=20, feedback_size=2, leak_rate=1.5, seed=1)
        with pytest.raises(ValueError, match="noise must be non-negative"):
            ESNLayer(reservoir_size=20, feedback_size=2, noise=-0.5, seed=1)
