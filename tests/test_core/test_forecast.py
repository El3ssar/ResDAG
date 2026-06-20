"""Forecast pipeline tests — autoregressive semantics, validation, shapes.

The contract under test (see :meth:`resdag.core.ESNModel.forecast`):

- every returned slot is a *genuine* autoregressive step — the loop feeds the
  model its own previous output, seeded by ``initial_feedback`` or the last
  warmup output.  No teacher-forced frame is emitted, so slot 0 is a real
  forecast (not a copy of the warmup output) and ``return_warmup=True`` does not
  duplicate the warmup/forecast seam;
- driver alignment matches training (``target = feedback shifted by 1``): step
  ``t`` consumes ``forecast_inputs[:, t]``, so the driver series starts right
  after the warmup window and must supply at least ``horizon`` timesteps.
"""

import sys

import pytest
import torch

from resdag.core import ESNModel, Input
from resdag.layers import CGReadoutLayer, Concatenate, ESNLayer
from resdag.layers.transforms import FeaturePartitioner

# torch.compile is unsupported on Python 3.15+ (mirrors tests/test_core/test_compile_and_precision.py).
COMPILE_SUPPORTED = torch.__version__ >= "2.0.0" and sys.version_info < (3, 15)
requires_compile = pytest.mark.skipif(not COMPILE_SUPPORTED, reason="torch.compile not supported")


def _build_driven_model(feedback_size: int = 1, driver_size: int = 2) -> tuple[ESNModel, ESNLayer]:
    torch.manual_seed(42)
    feedback = Input(shape=(20, feedback_size))
    driver = Input(shape=(20, driver_size))
    reservoir_layer = ESNLayer(
        reservoir_size=32,
        feedback_size=feedback_size,
        input_size=driver_size,
        spectral_radius=0.9,
    )
    states = reservoir_layer(feedback, driver)
    readout = CGReadoutLayer(32, feedback_size, name="output")(states)
    model = ESNModel([feedback, driver], readout)
    return model, reservoir_layer


def _build_multi_output_model(feedback_size: int = 2, aux_size: int = 3) -> ESNModel:
    """Two-output model; the first output (``main``) is used as feedback."""
    torch.manual_seed(42)
    feedback = Input(shape=(20, feedback_size))
    states = ESNLayer(reservoir_size=32, feedback_size=feedback_size, spectral_radius=0.9)(feedback)
    main = CGReadoutLayer(32, feedback_size, name="main")(states)
    aux = CGReadoutLayer(32, aux_size, name="aux")(states)
    return ESNModel(feedback, [main, aux])


def _manual_autoregressive(model: ESNModel, warmup: torch.Tensor, horizon: int) -> torch.Tensor:
    """Recompute a single-output forecast by hand from the same warmed state."""
    model.reset_reservoirs()
    warm = model.warmup(warmup, return_outputs=True)
    assert isinstance(warm, torch.Tensor)
    current = warm[:, -1:, :]
    steps = []
    with torch.no_grad():
        for _ in range(horizon):
            out = model(current)
            steps.append(out)
            current = out
    return torch.cat(steps, dim=1)


class TestAutoregressiveSlotZero:
    """Slot 0 must be a genuine autoregressive step, not the warmup output."""

    def test_horizon_one_differs_from_warmup_last_output(self, make_tiny_model) -> None:
        """``forecast(wu, horizon=1)`` is a real step, not an echo of warmup."""
        model = make_tiny_model(feedback_size=3)
        warmup = torch.randn(1, 20, 3)

        model.reset_reservoirs()
        warm = model.warmup(warmup, return_outputs=True)
        assert isinstance(warm, torch.Tensor)
        last_warmup_output = warm[:, -1:, :].clone()

        model.reset_reservoirs()
        preds = model.forecast(warmup, horizon=1)

        assert preds.shape == (1, 1, 3)
        assert not torch.allclose(preds[:, 0, :], last_warmup_output[:, 0, :], atol=1e-6)

    def test_matches_manual_autoregressive_recompute(self, make_tiny_model) -> None:
        """For ``horizon >= 2``, every slot equals a hand-rolled AR loop."""
        model = make_tiny_model(feedback_size=3)
        warmup = torch.randn(1, 20, 3)
        horizon = 8

        model.reset_reservoirs()
        preds = model.forecast(warmup, horizon=horizon)
        manual = _manual_autoregressive(model, warmup, horizon)

        assert preds.shape == (1, horizon, 3)
        assert torch.allclose(preds, manual, atol=1e-6)


class TestInitialFeedback:
    """``initial_feedback`` seeds the first step and is validated."""

    def test_initial_feedback_changes_slot_zero(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3)
        warmup = torch.randn(1, 20, 3)

        model.reset_reservoirs()
        default = model.forecast(warmup, horizon=4)

        model.reset_reservoirs()
        custom = model.forecast(warmup, horizon=4, initial_feedback=torch.full((1, 1, 3), 5.0))

        assert not torch.allclose(default[:, 0, :], custom[:, 0, :], atol=1e-6)

    def test_initial_feedback_is_used_as_seed(self, make_tiny_model) -> None:
        """The first step is ``model(initial_feedback)`` from the warmed state."""
        model = make_tiny_model(feedback_size=3)
        warmup = torch.randn(1, 20, 3)
        seed = torch.randn(1, 1, 3)

        model.reset_reservoirs()
        preds = model.forecast(warmup, horizon=3, initial_feedback=seed)

        # Recompute by hand: warm the state, then drive with the custom seed.
        model.reset_reservoirs()
        model.warmup(warmup)
        with torch.no_grad():
            expected_first = model(seed)
        assert torch.allclose(preds[:, 0:1, :], expected_first, atol=1e-6)

    @pytest.mark.parametrize(
        "bad",
        [
            torch.randn(1, 3),  # missing the time axis (rank 2)
            torch.randn(2, 1, 3),  # wrong batch
            torch.randn(1, 1, 5),  # wrong feature dim
            torch.randn(1, 2, 3),  # more than one timestep
        ],
    )
    def test_invalid_initial_feedback_raises(self, make_tiny_model, bad) -> None:
        model = make_tiny_model(feedback_size=3)
        with pytest.raises(ValueError, match="initial_feedback"):
            model.forecast(torch.randn(1, 20, 3), horizon=3, initial_feedback=bad)


class TestHorizonValidation:
    """``horizon`` must be a positive integer."""

    @pytest.mark.parametrize("horizon", [0, -1, -7])
    def test_non_positive_horizon_raises(self, make_tiny_model, horizon) -> None:
        model = make_tiny_model(feedback_size=3)
        with pytest.raises(ValueError, match="horizon"):
            model.forecast(torch.randn(1, 20, 3), horizon=horizon)


class TestReturnWarmupSeam:
    """``return_warmup`` must not duplicate the warmup/forecast seam frame."""

    def test_no_duplicate_seam_and_length(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3)
        warmup_steps, horizon = 20, 15
        warmup = torch.randn(1, warmup_steps, 3)

        model.reset_reservoirs()
        full = model.forecast(warmup, horizon=horizon, return_warmup=True)
        assert full.shape == (1, warmup_steps + horizon, 3)

        # The last warmup frame and the first forecast frame must differ.
        seam_last_warmup = full[:, warmup_steps - 1, :]
        seam_first_forecast = full[:, warmup_steps, :]
        assert not torch.allclose(seam_last_warmup, seam_first_forecast, atol=1e-6)

        # The forecast tail equals a plain (no-warmup) forecast.
        model.reset_reservoirs()
        preds = model.forecast(warmup, horizon=horizon)
        assert torch.allclose(full[:, warmup_steps:, :], preds, atol=1e-6)


class TestDriverAlignment:
    """The autoregressive loop consumes drivers in training-consistent order."""

    def test_drivers_consumed_in_order_starting_after_warmup(self, monkeypatch) -> None:
        """Step ``t`` must see ``forecast_inputs[:, t]`` — the driver series
        continuing exactly where the warmup drivers ended.

        The flat forecast engine drives the reservoir through ``step_stateless``
        (which calls ``cell.project_inputs``) rather than the layer's
        ``__call__``, so the driver alignment is probed on ``project_inputs``:
        one 3-D call for the whole teacher-forced warmup, then one 2-D call per
        autoregressive step.
        """
        model, reservoir_layer = _build_driven_model()
        horizon = 6

        warmup_feedback = torch.randn(1, 20, 1)
        # Distinguishable driver values: warmup drivers 0..19, forecast drivers
        # continue at 100, 101, ...
        warmup_driver = torch.arange(20, dtype=torch.float32).view(1, 20, 1).expand(1, 20, 2)
        forecast_driver = (
            (100 + torch.arange(horizon, dtype=torch.float32))
            .view(1, horizon, 1)
            .expand(1, horizon, 2)
        )

        seen_drivers: list[torch.Tensor] = []
        original = reservoir_layer.cell.project_inputs

        def probe(inputs):
            # inputs = [feedback_slice, driver_slice]
            seen_drivers.append(inputs[1].detach().clone())
            return original(inputs)

        monkeypatch.setattr(reservoir_layer.cell, "project_inputs", probe)
        model.forecast(
            (warmup_feedback, warmup_driver),
            forecast_inputs=(forecast_driver.contiguous(),),
            horizon=horizon,
        )

        # First captured call is the (3-D, whole-sequence) teacher-forced warmup.
        assert seen_drivers[0].shape[1] == 20
        # The horizon autoregressive steps consume the forecast drivers in
        # order, with no gap and no skipped first value (2-D per-step slices).
        autoregressive = seen_drivers[1:]
        assert len(autoregressive) == horizon
        for t, captured in enumerate(autoregressive):
            expected = forecast_driver[:, t, :]
            assert torch.equal(captured, expected), (
                f"autoregressive step {t} consumed driver {captured.flatten()[0].item()}, "
                f"expected {expected.flatten()[0].item()}"
            )

    def test_accepts_extra_driver_steps_and_consumes_first_horizon(self, monkeypatch) -> None:
        """A longer driver tensor is accepted; only the first ``horizon`` are used."""
        model, reservoir_layer = _build_driven_model()
        horizon = 5

        warmup_feedback = torch.randn(1, 20, 1)
        warmup_driver = torch.randn(1, 20, 2)
        forecast_driver = torch.randn(1, horizon + 4, 2)

        seen_drivers: list[torch.Tensor] = []
        original = reservoir_layer.cell.project_inputs

        def probe(inputs):
            seen_drivers.append(inputs[1].detach().clone())
            return original(inputs)

        monkeypatch.setattr(reservoir_layer.cell, "project_inputs", probe)
        model.forecast(
            (warmup_feedback, warmup_driver),
            forecast_inputs=(forecast_driver,),
            horizon=horizon,
        )

        autoregressive = seen_drivers[1:]
        assert len(autoregressive) == horizon
        for t, captured in enumerate(autoregressive):
            assert torch.equal(captured, forecast_driver[:, t, :])

    def test_rejects_too_few_driver_steps(self) -> None:
        model, _ = _build_driven_model()

        warmup_feedback = torch.randn(1, 20, 1)
        warmup_driver = torch.randn(1, 20, 2)

        with pytest.raises(ValueError, match="forecast_inputs\\[0\\]"):
            model.forecast(
                (warmup_feedback, warmup_driver),
                forecast_inputs=(torch.randn(1, 3, 2),),
                horizon=10,
            )

    def test_missing_drivers_raises(self) -> None:
        model, _ = _build_driven_model()

        with pytest.raises(ValueError, match="forecast_inputs must be provided"):
            model.forecast(
                (torch.randn(1, 20, 1), torch.randn(1, 20, 2)),
                horizon=10,
            )


class TestFeedbackOnlyForecast:
    """Shape contracts for the no-driver path (regression guard)."""

    def test_forecast_shapes(self) -> None:
        torch.manual_seed(42)
        feedback = Input(shape=(20, 3))
        states = ESNLayer(reservoir_size=32, feedback_size=3)(feedback)
        readout = CGReadoutLayer(32, 3, name="output")(states)
        model = ESNModel(feedback, readout)

        predictions = model.forecast(torch.randn(1, 20, 3), horizon=15)
        assert predictions.shape == (1, 15, 3)

        full = model.forecast(torch.randn(1, 20, 3), horizon=15, return_warmup=True)
        assert full.shape == (1, 35, 3)


class TestMultiOutputForecast:
    """The multi-output path mirrors the single-output autoregressive contract."""

    def test_multi_output_shapes(self) -> None:
        model = _build_multi_output_model(feedback_size=2, aux_size=3)
        warmup = torch.randn(1, 20, 2)
        horizon = 7

        model.reset_reservoirs()
        preds = model.forecast(warmup, horizon=horizon)

        assert isinstance(preds, tuple) and len(preds) == 2
        assert preds[0].shape == (1, horizon, 2)
        assert preds[1].shape == (1, horizon, 3)

    def test_multi_output_matches_manual_recompute(self) -> None:
        model = _build_multi_output_model(feedback_size=2, aux_size=3)
        warmup = torch.randn(1, 20, 2)
        horizon = 5

        model.reset_reservoirs()
        preds = model.forecast(warmup, horizon=horizon)

        model.reset_reservoirs()
        warm = model.warmup(warmup, return_outputs=True)
        assert isinstance(warm, tuple)
        current = warm[0][:, -1:, :]
        main_steps, aux_steps = [], []
        with torch.no_grad():
            for _ in range(horizon):
                out = model(current)
                main_steps.append(out[0])
                aux_steps.append(out[1])
                current = out[0]

        assert torch.allclose(preds[0], torch.cat(main_steps, dim=1), atol=1e-6)
        assert torch.allclose(preds[1], torch.cat(aux_steps, dim=1), atol=1e-6)

    def test_multi_output_return_warmup_no_duplicate_seam(self) -> None:
        model = _build_multi_output_model(feedback_size=2, aux_size=3)
        warmup_steps, horizon = 20, 6
        warmup = torch.randn(1, warmup_steps, 2)

        model.reset_reservoirs()
        full = model.forecast(warmup, horizon=horizon, return_warmup=True)

        assert isinstance(full, tuple) and len(full) == 2
        assert full[0].shape == (1, warmup_steps + horizon, 2)
        assert full[1].shape == (1, warmup_steps + horizon, 3)
        # Feedback channel seam must not be duplicated.
        assert not torch.allclose(
            full[0][:, warmup_steps - 1, :], full[0][:, warmup_steps, :], atol=1e-6
        )


# ======================================================================
# Flattened single-step inference engine (issue #254)
# ======================================================================
#
# ``forecast`` is driven by a flat, graph-free per-step engine
# (:mod:`resdag.core._flat_inference`) instead of re-walking the symbolic graph
# each step. The contract is: the flat path is *numerically identical* to the
# old per-step ``self(*step_inputs)`` graph re-execution, the opt-in
# ``compile=True`` path matches the eager path and falls back cleanly, and the
# reset / detach / multi-reservoir state semantics are preserved.


def _build_multi_reservoir_model(feedback_size: int = 2) -> ESNModel:
    """Two parallel reservoirs feeding a shared readout (multi-reservoir DAG)."""
    torch.manual_seed(42)
    inp = Input(shape=(20, feedback_size))
    r1 = ESNLayer(reservoir_size=24, feedback_size=feedback_size, spectral_radius=0.9)(inp)
    r2 = ESNLayer(reservoir_size=16, feedback_size=feedback_size, spectral_radius=0.8)(inp)
    merged = Concatenate()(r1, r2)
    out = CGReadoutLayer(24 + 16, feedback_size, name="output")(merged)
    return ESNModel(inp, out)


def _legacy_graph_forecast(
    model: ESNModel,
    warmup_inputs: tuple[torch.Tensor, ...],
    horizon: int,
    forecast_inputs: tuple[torch.Tensor, ...] | None = None,
    *,
    reset: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Reference forecast via the *old* per-step graph re-execution path.

    Mirrors the pre-#254 loop exactly: reset, teacher-forced warmup, then one
    full ``self(*step_inputs)`` graph walk per autoregressive step on 3-D
    ``(batch, 1, features)`` tensors. Returns a tuple of per-output tensors so a
    single comparison helper covers single- and multi-output models.
    """
    if reset:
        model.reset_reservoirs()
    drivers = forecast_inputs if forecast_inputs is not None else ()
    has_drivers = len(warmup_inputs) > 1
    with torch.no_grad():
        warm = model.warmup(*warmup_inputs, return_outputs=True, reset=reset)
        warm_tuple = warm if isinstance(warm, tuple) else (warm,)
        current = warm_tuple[0][:, -1:, :]
        acc: list[list[torch.Tensor]] = [[] for _ in warm_tuple]
        for t in range(horizon):
            if has_drivers:
                step_inputs = (current, *(d[:, t : t + 1, :] for d in drivers))
            else:
                step_inputs = (current,)
            out = model(*step_inputs)
            outs = out if isinstance(out, tuple) else (out,)
            for i, o in enumerate(outs):
                acc[i].append(o)
            current = outs[0]
    return tuple(torch.cat(steps, dim=1) for steps in acc)


def _as_tuple(result: object) -> tuple[torch.Tensor, ...]:
    return result if isinstance(result, tuple) else (result,)


class TestFlatEngineParity:
    """Flat forecast must equal the legacy graph-re-execution path to 1e-10."""

    def test_feedback_only_bit_parity(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3)
        warmup = torch.randn(2, 25, 3, dtype=torch.float64)
        model.double()
        warmup = warmup.double()

        model.reset_reservoirs()
        flat = model.forecast(warmup, horizon=60)
        ref = _legacy_graph_forecast(model, (warmup,), 60)
        assert (flat - ref[0]).abs().max().item() < 1e-10

    def test_driven_bit_parity(self) -> None:
        model, _ = _build_driven_model(feedback_size=2, driver_size=3)
        model.double()
        warm_fb = torch.randn(2, 20, 2, dtype=torch.float64)
        warm_drv = torch.randn(2, 20, 3, dtype=torch.float64)
        fut_drv = torch.randn(2, 80, 3, dtype=torch.float64)

        model.reset_reservoirs()
        flat = model.forecast((warm_fb, warm_drv), forecast_inputs=(fut_drv,), horizon=70)
        ref = _legacy_graph_forecast(model, (warm_fb, warm_drv), 70, (fut_drv,))
        assert (flat - ref[0]).abs().max().item() < 1e-10

    def test_multi_output_bit_parity(self) -> None:
        model = _build_multi_output_model(feedback_size=2, aux_size=3)
        model.double()
        warmup = torch.randn(2, 20, 2, dtype=torch.float64)

        model.reset_reservoirs()
        flat = _as_tuple(model.forecast(warmup, horizon=55))
        ref = _legacy_graph_forecast(model, (warmup,), 55)
        for f, r in zip(flat, ref):
            assert (f - r).abs().max().item() < 1e-10

    def test_multi_reservoir_dag_bit_parity(self) -> None:
        model = _build_multi_reservoir_model(feedback_size=2).double()
        warmup = torch.randn(2, 20, 2, dtype=torch.float64)

        model.reset_reservoirs()
        flat = model.forecast(warmup, horizon=50)
        ref = _legacy_graph_forecast(model, (warmup,), 50)
        assert (flat - ref[0]).abs().max().item() < 1e-10
        # The DAG really does carry two reservoirs through the engine.
        assert len(model._get_flat_step().reservoir_layers) == 2

    def test_multi_output_unpack_layer_bit_parity(self) -> None:
        """A graph with an unpacked (sibling) node + a shape-sensitive transform.

        ``FeaturePartitioner`` returns a list and indexes ``input.shape`` as 3-D,
        and the ``a, b = parts`` unpack creates ``UnpackLayer`` siblings that the
        engine must splat. This is the case the per-step engine must feed full
        3-D slices, not squeezed 2-D ones.
        """
        torch.manual_seed(0)
        inp = Input(shape=(20, 4))
        states = ESNLayer(reservoir_size=32, feedback_size=4, spectral_radius=0.9)(inp)
        first, second = FeaturePartitioner(partitions=2, overlap=0)(states)
        merged = Concatenate()(first, second)
        out = CGReadoutLayer(32, 4, name="output")(merged)
        model = ESNModel(inp, out).double()
        warmup = torch.randn(2, 25, 4, dtype=torch.float64)

        model.reset_reservoirs()
        flat = model.forecast(warmup, horizon=40)
        ref = _legacy_graph_forecast(model, (warmup,), 40)
        assert (flat - ref[0]).abs().max().item() < 1e-10

    def test_initial_feedback_bit_parity(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3).double()
        warmup = torch.randn(1, 20, 3, dtype=torch.float64)
        seed = torch.randn(1, 1, 3, dtype=torch.float64)

        model.reset_reservoirs()
        warm = model.warmup(warmup, return_outputs=True)
        assert isinstance(warm, torch.Tensor)
        # Reference loop seeded with the same custom feedback.
        model.reset_reservoirs()
        model.warmup(warmup)
        ref_steps = []
        cur = seed
        with torch.no_grad():
            for _ in range(30):
                cur = model(cur)
                ref_steps.append(cur)
        ref = torch.cat(ref_steps, dim=1)

        model.reset_reservoirs()
        flat = model.forecast(warmup, horizon=30, initial_feedback=seed)
        assert (flat - ref).abs().max().item() < 1e-10


class TestFlatEngineStateSemantics:
    """reset / detach / state-writeback semantics survive the flat engine."""

    def test_reset_true_makes_calls_independent(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3)
        warmup = torch.randn(1, 20, 3)
        first = model.forecast(warmup, horizon=20)
        second = model.forecast(warmup, horizon=20)  # reset=True default
        assert torch.allclose(first, second)

    def test_reset_false_continues_from_state(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3)
        warmup = torch.randn(1, 20, 3)
        with_reset = model.forecast(warmup, horizon=20, reset=True)
        # A second reset=False forecast continues from the mutated state, so it
        # must differ from the fresh-reset result.
        no_reset = model.forecast(warmup, horizon=20, reset=False)
        assert not torch.allclose(with_reset, no_reset)

    def test_state_written_back_matches_legacy(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3).double()
        warmup = torch.randn(1, 20, 3, dtype=torch.float64)

        model.reset_reservoirs()
        model.forecast(warmup, horizon=40)
        flat_states = model.get_reservoir_states()

        _legacy_graph_forecast(model, (warmup,), 40)
        ref_states = model.get_reservoir_states()

        assert flat_states.keys() == ref_states.keys()
        for name in flat_states:
            assert (flat_states[name] - ref_states[name]).abs().max().item() < 1e-10

    def test_detach_flag_does_not_break_forecast(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3).double()
        for module in model.modules():
            if hasattr(module, "detach_state_between_calls"):
                module.detach_state_between_calls = False
        warmup = torch.randn(1, 20, 3, dtype=torch.float64)
        model.reset_reservoirs()
        flat = model.forecast(warmup, horizon=30)
        ref = _legacy_graph_forecast(model, (warmup,), 30)
        assert (flat - ref[0]).abs().max().item() < 1e-10


class TestFlatStepCache:
    """The compiled step is cached and rebuilt only when the graph changes."""

    def test_step_cached_across_forecasts(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3)
        assert model._get_flat_step() is model._get_flat_step()

    def test_cache_rebuilt_on_add_output(self) -> None:
        torch.manual_seed(0)
        inp = Input(shape=(20, 2))
        states = ESNLayer(reservoir_size=16, feedback_size=2, spectral_radius=0.9)(inp)
        main = CGReadoutLayer(16, 2, name="main")(states)
        model = ESNModel(inp, main)
        first = model._get_flat_step()
        assert first.n_outputs == 1

        # Promote an existing in-graph node (the reservoir states) to an extra
        # output; the cache, keyed on ``self.outputs`` identity, must rebuild.
        model.add_output(states)
        rebuilt = model._get_flat_step()
        assert rebuilt is not first
        assert rebuilt.n_outputs == 2

    def test_deepcopy_after_forecast_is_independent(self, make_tiny_model) -> None:
        """A copy taken after the cache is populated rebuilds on its own layers.

        The cache holds an ``exec``-built closure that ``deepcopy`` would treat
        as atomic (keeping the original layers); it must be excluded so the copy
        forecasts against its own weights.
        """
        import copy

        model = make_tiny_model(feedback_size=3).double()
        warmup = torch.randn(1, 20, 3, dtype=torch.float64)
        model.forecast(warmup, horizon=10)  # populate the cache

        clone = copy.deepcopy(model)
        assert torch.equal(model.forecast(warmup, horizon=15), clone.forecast(warmup, horizon=15))

        # Perturbing the original must not affect the independent clone.
        before = clone.forecast(warmup, horizon=15)
        for param in model.parameters():
            param.data.add_(1.0)
        assert torch.equal(clone.forecast(warmup, horizon=15), before)
        assert not torch.allclose(model.forecast(warmup, horizon=15), before)

    def test_save_full_round_trip_after_forecast(self, make_tiny_model, tmp_path) -> None:
        """save_full / load_full survive a forecast having populated the cache."""
        model = make_tiny_model(feedback_size=3).double()
        warmup = torch.randn(1, 20, 3, dtype=torch.float64)
        expected = model.forecast(warmup, horizon=15)

        path = tmp_path / "model_full.pt"
        model.save_full(path)  # would raise PicklingError if the cache leaked
        restored = ESNModel.load_full(path)
        assert (restored.forecast(warmup, horizon=15) - expected).abs().max().item() < 1e-10


class TestStepStateless:
    """The per-step reservoir primitive matches a length-1 forward_stateless."""

    def test_matches_forward_stateless_single_step(self) -> None:
        torch.manual_seed(0)
        layer = ESNLayer(reservoir_size=32, feedback_size=3, input_size=2, spectral_radius=0.9)
        layer.double()
        fb = torch.randn(4, 3, dtype=torch.float64)
        drv = torch.randn(4, 2, dtype=torch.float64)
        state = layer.cell.init_state(4, fb.device, torch.float64)

        out_step, state_step = layer.step_stateless([fb, drv], state)
        out_seq, state_seq = layer.forward_stateless([fb.unsqueeze(1), drv.unsqueeze(1)], state)
        assert (out_step - out_seq[:, 0, :]).abs().max().item() < 1e-12
        assert (state_step - state_seq).abs().max().item() < 1e-12


@requires_compile
class TestCompiledForecast:
    """``compile=True`` matches eager, spans chunks, and falls back cleanly."""

    def test_compile_matches_eager(self, device: torch.device) -> None:
        torch.manual_seed(1)
        inp = Input(shape=(20, 3))
        states = ESNLayer(reservoir_size=64, feedback_size=3, spectral_radius=0.9)(inp)
        out = CGReadoutLayer(64, 3, name="output")(states)
        model = ESNModel(inp, out).to(device)
        warmup = torch.randn(1, 30, 3, device=device)

        model.reset_reservoirs()
        eager = model.forecast(warmup, horizon=40)
        model.reset_reservoirs()
        compiled = model.forecast(warmup, horizon=40, compile=True)
        assert (eager - compiled).abs().max().item() < 1e-4

    def test_compile_spans_chunks_and_remainder(self, device: torch.device) -> None:
        # horizon deliberately not a multiple of the chunk size, to exercise
        # both the compiled chunk loop and the eager remainder.
        from resdag.core._flat_inference import DEFAULT_COMPILE_CHUNK

        torch.manual_seed(2)
        inp = Input(shape=(20, 2))
        states = ESNLayer(reservoir_size=48, feedback_size=2, spectral_radius=0.9)(inp)
        out = CGReadoutLayer(48, 2, name="output")(states)
        model = ESNModel(inp, out).to(device)
        warmup = torch.randn(1, 25, 2, device=device)
        horizon = DEFAULT_COMPILE_CHUNK + 7

        model.reset_reservoirs()
        eager = model.forecast(warmup, horizon=horizon)
        model.reset_reservoirs()
        compiled = model.forecast(warmup, horizon=horizon, compile=True)
        assert compiled.shape == (1, horizon, 2)
        assert (eager - compiled).abs().max().item() < 1e-4

    def test_compile_driven_model(self, device: torch.device) -> None:
        model, _ = _build_driven_model(feedback_size=2, driver_size=3)
        model = model.to(device)
        warm_fb = torch.randn(1, 20, 2, device=device)
        warm_drv = torch.randn(1, 20, 3, device=device)
        fut_drv = torch.randn(1, 90, 3, device=device)

        model.reset_reservoirs()
        eager = model.forecast((warm_fb, warm_drv), forecast_inputs=(fut_drv,), horizon=80)
        model.reset_reservoirs()
        compiled = model.forecast(
            (warm_fb, warm_drv), forecast_inputs=(fut_drv,), horizon=80, compile=True
        )
        assert (eager - compiled).abs().max().item() < 1e-4

    def test_falls_back_to_eager_when_compile_raises(self, make_tiny_model, monkeypatch) -> None:
        import resdag.core._flat_inference as fi

        def _boom(*args, **kwargs):
            raise RuntimeError("inductor said no")

        monkeypatch.setattr(fi.torch, "compile", _boom)

        model = make_tiny_model(feedback_size=3).double()
        warmup = torch.randn(1, 20, 3, dtype=torch.float64)
        model.reset_reservoirs()
        eager = model.forecast(warmup, horizon=30)
        model.reset_reservoirs()
        with pytest.warns(RuntimeWarning, match="falling back to the eager flat step"):
            fellback = model.forecast(warmup, horizon=30, compile=True)
        # Fallback path is numerically identical to the eager flat path.
        assert (eager - fellback).abs().max().item() < 1e-10

    def test_falls_back_when_torch_too_old(self, make_tiny_model, monkeypatch) -> None:
        from torch.torch_version import TorchVersion

        import resdag.core._flat_inference as fi

        monkeypatch.setattr(fi.torch, "__version__", TorchVersion("2.9.0"))

        model = make_tiny_model(feedback_size=3).double()
        warmup = torch.randn(1, 20, 3, dtype=torch.float64)
        model.reset_reservoirs()
        eager = model.forecast(warmup, horizon=20)
        model.reset_reservoirs()
        with pytest.warns(RuntimeWarning, match="needs torch>="):
            result = model.forecast(warmup, horizon=20, compile=True)
        assert (eager - result).abs().max().item() < 1e-10
