"""Independent patch-layout and admission oracles; no fitting or HPO studies."""

import copy
import dataclasses
import pickle

import pytest
import torch

import resdag as rd
from resdag.hpo._tuning_data import prepare_tuning_data
from resdag.hpo.resources import MemoryBudget, _memory_context, estimate_memory


def _parallel(batch=1, patches=8, *, graph_class=None):
    graph = rd.parallel_esn(
        input_dim=2 * patches,
        core_width=2,
        halo=1,
        reservoir_size=3,
        spectral_radius=0.2,
        seed=17,
        dtype=torch.float64,
    )
    if graph_class is not None:
        graph.__class__ = graph_class
    model = rd.Model(graph=graph, washout=2)
    series = torch.arange(batch * 20 * patches * 2, dtype=torch.float64).reshape(
        batch, 20, patches * 2
    )
    data = prepare_tuning_data(model, series, horizon=3)
    return model, data


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("patches", [8, 64])
def test_parallel_layout_counts_pooled_fit_rows_and_real_window_shapes(batch, patches):
    model, data = _parallel(batch, patches)
    context = _memory_context(model, data)
    assert context.train_samples == batch * next(iter(data.train.inputs.values())).shape[-2]
    assert context.refit_samples == batch * 20
    for phase in ("train", "refit"):
        prepared = getattr(data, phase)
        source = next(iter(prepared.inputs.values()))
        # Exercise the public indexing layout independently of the planner;
        # the planner itself must never call this method.
        start = max(model.washout, prepared.supervised_start)
        stop = prepared.supervised_stop
        actual_window = model.graph.window(source[:, start:stop])
        steps = stop - start
        rows = actual_window.shape[0] * steps
        assert getattr(context, f"readout_{phase}_samples") == (rows,)
        layout = context.sequence_layout[phase]
        assert layout["window"] == tuple(actual_window.shape) == (batch * patches, steps, 4)
        assert layout["projection"] == layout["features"] == (batch * patches, steps, 3)
        assert layout["state"] == (batch * patches, 3)
        assert layout["warmup_window"] == (batch * patches, start, 4)
        assert layout["warmup_projection"] == (batch * patches, start, 3)
    estimate = estimate_memory({}, context)
    pooled = batch * patches * (data.refit.supervised_stop - model.washout)
    assert estimate.details["readout_samples"] == (pooled,)
    components = estimate.details["parallel_trace_components"]
    # Every independently known patch-batched buffer gets the established
    # four-copy trace allowance, with state counted once per batch/patch.
    trace_rows = batch * patches * 20
    assert components == {
        "input_windows": 4 * trace_rows * 4 * 8,
        "target_windows": 4 * trace_rows * 4 * 8,
        "projected_inputs": 4 * trace_rows * 3 * 8,
        "features": 4 * trace_rows * 3 * 8,
        "core_targets": 4 * trace_rows * 2 * 8,
        "recurrent_state": 4 * batch * patches * 3 * 8,
    }
    assert estimate.details["feature_traces"] >= sum(components.values())
    legacy = estimate_memory(
        {},
        dataclasses.replace(
            context, readout_train_samples=(), readout_refit_samples=(), sequence_layout={}
        ),
    )
    assert estimate.details["solve_workspace"] > legacy.details["solve_workspace"]
    assert not MemoryBudget(legacy.total_bytes, worker_overhead=0).fits(estimate)
    # Lower sampled washout admits more fit rows without changing the public
    # observation-count fields or constructing a trial candidate.
    less_washout = estimate_memory({"washout": 0}, context)
    assert less_washout.details["readout_samples"] == (
        batch * patches * data.refit.supervised_stop,
    )
    assert less_washout.details["solve_workspace"] > estimate.details["solve_workspace"]


def test_layout_planning_executes_no_graph_window_projection_or_candidate_constructor(monkeypatch):
    model, data = _parallel()

    before = {
        name: (value.dtype, value.device, value.clone())
        for name, value in model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }

    def forbidden(*args, **kwargs):
        pytest.fail("resource planning executed or constructed a model")

    # Do not replace instance execution methods: those deliberately select
    # the generic fallback. Block all module calls and projection arithmetic.
    monkeypatch.setattr(torch.nn.Module, "__call__", forbidden)
    monkeypatch.setattr(torch.nn.functional, "linear", forbidden)
    monkeypatch.setattr(model, "to", forbidden)
    monkeypatch.setattr(rd.ESN, "__init__", forbidden)
    context = _memory_context(model, data)
    assert context.sequence_layout["kind"] == "parallel"
    estimate_memory({}, context)
    huge = dataclasses.replace(
        context,
        readout_train_samples=(),
        readout_refit_samples=(),
        sequence_layout={},
        preset_options={"reservoir_size": 10**9, "input_to_readout": False},
    )
    assert estimate_memory({}, huge).total_bytes >= 3 * 8 * 10**18
    for name, (dtype, device, value) in before.items():
        actual = model.state_dict()[name]
        assert actual.dtype == dtype and actual.device == device
        torch.testing.assert_close(actual, value, rtol=0, atol=0)


def test_pooled_row_metadata_agrees_with_rows_received_by_real_parallel_fit(monkeypatch):
    model, data = _parallel(batch=4, patches=8)
    context = _memory_context(model, data)
    inputs = next(iter(data.refit.inputs.values()))
    targets = next(iter(data.refit.targets.values()))
    received = []

    def capture(states, truth):
        received.append((tuple(states.shape), tuple(truth.shape)))

    # Run the real window/reservoir/target path, observing its solver boundary
    # without performing a readout solve or an HPO study.
    monkeypatch.setattr(model.graph.readout, "fit", capture)
    split = max(model.washout, data.refit.supervised_start)
    model.graph.fit(
        inputs[:, :split],
        inputs[:, split : data.refit.supervised_stop],
        targets[:, split - data.refit.supervised_start :],
    )
    assert received == [((32, 17, 3), (32, 17, 2))]
    assert context.readout_refit_samples == (received[0][0][0] * received[0][0][1],)


def test_supervision_support_limits_rows_even_when_sampled_washout_is_lower():
    model, data = _parallel()
    supported = dataclasses.replace(
        data.refit,
        targets={name: value[:, 5:17] for name, value in data.refit.targets.items()},
        supervised_start=5,
        supervised_stop=17,
    )
    data = dataclasses.replace(data, train=supported, refit=supported)
    context = _memory_context(model, data)
    assert context.train_samples == context.refit_samples == 20
    assert context.readout_train_samples == context.readout_refit_samples == (8 * 12,)
    assert context.sequence_layout["refit"]["window"] == (8, 12, 4)
    assert context.sequence_layout["refit"]["warmup_window"] == (8, 5, 4)
    assert estimate_memory({"washout": 0}, context).details["readout_samples"] == (8 * 12,)


@pytest.mark.parametrize("custom", ["subclass", "method", "hook"])
def test_custom_parallel_execution_retains_generic_fallback(custom):
    from resdag.models.parallel_esn import ParallelESNModel

    class CustomParallel(ParallelESNModel):
        pass

    model, data = _parallel(graph_class=CustomParallel if custom == "subclass" else None)
    handle = None
    if custom == "method":
        model.graph.window = lambda value: value
    elif custom == "hook":
        handle = model.graph.register_forward_hook(lambda *args: None)
    try:
        context = _memory_context(model, data)
    finally:
        if handle is not None:
            handle.remove()
    assert context.sequence_layout == {}
    assert context.readout_train_samples == context.readout_refit_samples == ()
    estimate = estimate_memory({}, context)
    assert estimate.details["readout_samples"] == (context.refit_samples,)
    assert estimate.details["parallel_trace_components"] == {}


def test_optional_layout_facts_are_immutable_serializable_and_visible_to_custom_estimators():
    model, data = _parallel()
    context = _memory_context(model, data)
    assert pickle.loads(pickle.dumps(context)) == context
    assert copy.deepcopy(context) == context
    with pytest.raises(TypeError, match="immutable"):
        context.sequence_layout["refit"]["window"] = (1, 1, 1)
    observed = []

    def estimator(params, facts):
        observed.append((facts.refit_samples, facts.readout_refit_samples, facts.sequence_layout))
        return 0

    result = estimate_memory({}, context, estimator=estimator)
    assert observed[0][:2] == (20, (8 * 17,))
    assert observed[0][2] is context.sequence_layout
    assert result.details["floor_applied"]
    old_context = dataclasses.replace(
        context, readout_train_samples=(), readout_refit_samples=(), sequence_layout={}
    )
    assert old_context.refit_samples == context.refit_samples
    assert estimate_memory({}, old_context).details["parallel_trace_components"] == {}


@pytest.mark.parametrize("skip", [None, True, False])
def test_preset_readout_width_excludes_driver_and_respects_classic_skip(skip, monkeypatch):
    model, data = _parallel()
    context = _memory_context(model, data)
    options = {"reservoir_size": 10, "feedback_size": 3}
    if skip is not None:
        options["input_to_readout"] = skip
    context = dataclasses.replace(
        context,
        input_shapes={"feedback": (1, 20, 3), "driver": (1, 20, 5)},
        target_shapes={"prediction": (1, 20, 3)},
        preset_options=options,
        readout_train_samples=(),
        readout_refit_samples=(),
        sequence_layout={},
    )
    monkeypatch.setattr(rd.ESN, "__init__", lambda *a, **k: pytest.fail("constructed preset"))
    result = estimate_memory({}, context)
    assert result.details["readout_features"] == (10 if skip is False else 13,)
    assert context.input_features == 8
    overridden = estimate_memory({"input_to_readout": False}, context)
    assert overridden.details["readout_features"] == (10,)
    assert overridden.details["solve_workspace"] <= result.details["solve_workspace"]
    expected_skip_trace = 0 if skip is False else 4 * result.details["trace_samples"] * 3 * 8
    assert (
        result.details["feature_traces"] - overridden.details["feature_traces"]
        == expected_skip_trace
    )


def test_lazy_preset_without_feedback_metadata_keeps_legacy_width_default():
    model, data = _parallel()
    context = dataclasses.replace(
        _memory_context(model, data),
        preset_options={"reservoir_size": 10},
        readout_train_samples=(),
        readout_refit_samples=(),
        sequence_layout={},
    )
    assert estimate_memory({}, context).details["readout_features"] == (10 + 16,)


def test_malformed_optional_readout_counts_fail_without_changing_observation_semantics():
    model, data = _parallel()
    context = _memory_context(model, data)
    for counts in ((-1,), (True,), (1, 2)):
        with pytest.raises(ValueError):
            dataclasses.replace(context, readout_refit_samples=counts)
    with pytest.raises(TypeError):
        dataclasses.replace(context, sequence_layout=[])
