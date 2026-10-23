"""Parallel CPU forecast specialization agrees with the original sequence loop."""

from __future__ import annotations

import copy
import types

import pytest
import torch
from torch.nn.modules import module as module_runtime

import resdag as rd
from resdag.layers.cells import ESNCell
from resdag.layers.readouts import ReadoutLayer
from resdag.models.parallel_esn import ParallelESNModel


def _model(batch=1, *, trainable=False):
    graph = rd.parallel_esn(
        input_dim=16,
        core_width=4,
        halo=2,
        reservoir_size=9,
        spectral_radius=0.3,
        leak_rate=0.7,
        seed=7,
        trainable=trainable,
        dtype=torch.float64,
        readout_alpha=0.01,
    ).eval()
    generator = torch.Generator().manual_seed(19)
    observations = torch.randn(batch, 24, 32, generator=generator).double()[..., ::2]
    graph.fit(observations[:, :3], observations[:, 3:-1], observations[:, 4:])
    return graph, observations[:, -5:]


def _original_loop(
    model,
    history,
    *,
    horizon=5,
    initial_feedback=None,
    reset=True,
    return_warmup=False,
    alignment="next",
):
    """Independent original full-sequence calls, including boundary ownership."""
    with torch.no_grad():
        warmup = model(history, reset=reset)
        current = initial_feedback if initial_feedback is not None else warmup[:, -1:]
        boundary = alignment == "next" and initial_feedback is None
        values = [current] if boundary else []
        for _ in range(int(boundary), horizon):
            current = model(current)
            values.append(current)
        output = torch.cat(values, dim=1)
        if return_warmup:
            prefix = warmup[:, :-1] if boundary else warmup
            output = torch.cat((prefix, output), dim=1)
        return output


def _assert_matches(model, history, **kwargs):
    kwargs.setdefault("horizon", 5)
    reference = copy.deepcopy(model)
    expected = _original_loop(reference, history, **kwargs)
    actual = model.forecast(history, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(model.reservoir.state, reference.reservoir.state, rtol=0, atol=0)
    assert model.reservoir._state_user_set == reference.reservoir._state_user_set
    return actual


@pytest.mark.parametrize("batch,trainable", [(1, False), (3, True)])
@pytest.mark.parametrize(
    "scenario", ["next", "prefix", "legacy", "seed", "empty", "continue", "one"]
)
def test_forecast_matches_original_sequence_loop(batch, trainable, scenario):
    model, history = _model(batch, trainable=trainable)
    assert not history.is_contiguous()
    kwargs = {}
    if scenario == "prefix":
        kwargs["return_warmup"] = True
    elif scenario == "legacy":
        kwargs.update(alignment="legacy", return_warmup=True)
    elif scenario in ("seed", "empty"):
        kwargs["initial_feedback"] = history[:, -1:]
        if scenario == "empty":
            history = history[:, :0]
    elif scenario == "continue":
        model(history, reset=True)
        kwargs["reset"] = False
    elif scenario == "one":
        kwargs["horizon"] = 1
    _assert_matches(model, history, **kwargs)


def test_builtin_forecast_selects_single_frame_path_once(monkeypatch):
    model, history = _model()
    calls = []
    step = model._forecast_single_frame

    def counted(series):
        calls.append(series.shape)
        return step(series)

    monkeypatch.setattr(model, "_forecast_single_frame", counted)
    _assert_matches(model, history, horizon=5)
    assert calls == [(1, 1, 16)] * 4


def _forbid_fast_path(model):
    def forbidden(*args):
        pytest.fail("Custom behavior must use the original sequence path")

    model._forecast_single_frame = forbidden


@pytest.mark.parametrize(
    "location",
    [
        "model_instance",
        "model_subclass",
        "window",
        "reservoir_instance",
        "reservoir_subclass",
        "cell_instance",
        "cell_subclass",
        "readout_instance",
        "readout_subclass",
        "readout_mro",
        "readout_dispatch",
        "activation",
    ],
)
def test_custom_methods_retain_native_sequence_dispatch(location):
    model, history = _model()
    _forbid_fast_path(model)
    if location == "model_instance":
        model.forward = types.MethodType(
            lambda self, *a, **k: ParallelESNModel.forward(self, *a, **k) + 0.01, model
        )
    elif location == "model_subclass":

        class CustomParallel(ParallelESNModel):
            def forward(self, *args, **kwargs):
                return super().forward(*args, **kwargs) + 0.01

        model.__class__ = CustomParallel
    elif location == "window":
        model.window = types.MethodType(
            lambda self, series: ParallelESNModel.window(self, series).flip(0), model
        )
    elif location == "reservoir_instance":
        model.reservoir.forward = types.MethodType(
            lambda self, *a, **k: rd.ESNLayer.forward(self, *a, **k) + 0.01, model.reservoir
        )
    elif location == "reservoir_subclass":

        class CustomReservoir(rd.ESNLayer):
            def forward(self, *args, **kwargs):
                return super().forward(*args, **kwargs) + 0.01

        model.reservoir.__class__ = CustomReservoir
    elif location == "cell_instance":
        model.reservoir.cell.project_inputs = types.MethodType(
            lambda self, inputs: ESNCell.project_inputs(self, inputs) + 0.01, model.reservoir.cell
        )
    elif location == "cell_subclass":

        class CustomCell(ESNCell):
            def step(self, projected, state):
                return super().step(projected + 0.01, state)

        model.reservoir.cell.__class__ = CustomCell
    elif location in ("readout_instance", "readout_subclass"):

        def mutating(self, values):
            # The full sequence path isolates this mutation from final state.
            values.add_(0.01)
            return ReadoutLayer.forward(self, values)

        if location == "readout_instance":
            model.readout.forward = types.MethodType(mutating, model.readout)
        else:

            class CustomReadout(type(model.readout)):
                forward = mutating

            model.readout.__class__ = CustomReadout
    elif location == "readout_mro":

        class MutatingLinear(torch.nn.Linear):
            def forward(self, values):
                values.add_(0.01)
                return super().forward(values)

        class CustomReadout(type(model.readout), MutatingLinear):
            pass

        model.readout.__class__ = CustomReadout
    elif location == "readout_dispatch":

        def mutating_call(self, values, *args, **kwargs):
            values.add_(0.01)
            return torch.nn.Module._call_impl(self, values, *args, **kwargs)

        model.readout._call_impl = types.MethodType(mutating_call, model.readout)
    else:
        model.reservoir.cell.activation_fn = lambda values: torch.tanh(values) * 0.9
    _assert_matches(model, history)


@pytest.mark.parametrize("location", ["model", "reservoir", "readout", "global_pre", "global_post"])
def test_local_and_global_hooks_retain_native_sequence_dispatch(location):
    model, history = _model()
    _forbid_fast_path(model)
    handle = None
    if location == "global_pre":
        handle = module_runtime.register_module_forward_pre_hook(
            lambda module, args: (args[0] + 0.01,) if isinstance(module, ReadoutLayer) else None
        )
    elif location == "global_post":
        handle = module_runtime.register_module_forward_hook(
            lambda module, args, output: output + 0.01 if isinstance(module, ReadoutLayer) else None
        )
    else:
        target = model if location == "model" else getattr(model, location)
        handle = target.register_forward_hook(lambda module, args, output: output + 0.01)
    try:
        _assert_matches(model, history)
    finally:
        handle.remove()


def test_protected_state_error_and_ownership_match_original():
    model, history = _model()
    state = torch.ones(8, 9, dtype=torch.float64)
    model.reservoir.set_state(state)
    stored = model.reservoir.state
    with pytest.raises(RuntimeError, match="batch size"):
        model.forecast(history, horizon=3, reset=False)
    assert model.reservoir.state is stored
    assert model.reservoir._state_user_set


@pytest.mark.parametrize("context", ["inference", "autocast"])
def test_inference_and_autocast_contexts_match_original(context):
    model, history = _model()
    if context == "autocast":
        _forbid_fast_path(model)
    manager = (
        torch.inference_mode()
        if context == "inference"
        else torch.autocast("cpu", dtype=torch.bfloat16)
    )
    with manager:
        _assert_matches(model, history)


def test_forward_ad_primal_and_tangent_match_original_sequence_loop():
    model, history = _model()
    reference = copy.deepcopy(model)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(history, torch.ones_like(history) * 0.2)
        expected = _original_loop(reference, dual, horizon=4)
        actual = model.forecast(dual, horizon=4)
        expected = torch.autograd.forward_ad.unpack_dual(expected)
        actual = torch.autograd.forward_ad.unpack_dual(actual)
        torch.testing.assert_close(actual.primal, expected.primal, rtol=0, atol=0)
        torch.testing.assert_close(actual.tangent, expected.tangent, rtol=0, atol=0)
