"""Independent arithmetic/storage oracles for optional tuning resource planning."""

import copy
import dataclasses
import json
import pickle

import numpy as np
import pytest
import torch

import resdag as rd
from resdag.hpo._tuning_data import prepare_tuning_data
from resdag.hpo.resources import (
    MemoryBudget,
    MemoryContext,
    MemoryEstimate,
    Screening,
    estimate_memory,
    memory_context,
    raw_memory_context,
)


def _model(width=2, *, dtype=torch.float64):
    x = rd.reservoir_input(width, dtype=dtype)
    y = rd.AutoReadoutLayer(width, width).to(dtype=dtype)(x)
    return rd.Model(inputs={"state": x}, outputs={"prediction": y}, washout=1)


def _context():
    model = _model()
    series = torch.arange(80, dtype=torch.float64).reshape(40, 2)
    data = prepare_tuning_data(model, series, horizon=3)
    return memory_context(model, data)


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, 1),
        (np.int64(32), 32),
        ("8GB", 8_000_000_000),
        ("8GiB", 8 * 2**30),
        ("1.5 KiB", 1536),
        (" 2MB ", 2_000_000),
        ("0.5MiB", 524288),
        ("1B", 1),
        ("2TB", 2_000_000_000_000),
        ("1TiB", 2**40),
    ],
)
def test_budget_parses_decimal_and_binary_units_exactly(value, expected):
    budget = MemoryBudget(value, worker_overhead=0)
    assert budget.limit == budget.limit_bytes == expected
    assert budget.worker_overhead_bytes == 0


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(True),
        0,
        -1,
        1.0,
        None,
        "8",
        "8gb",
        "8GIB",
        "8 Gi",
        "1e3B",
        "NaN GB",
        "infGiB",
        "-1GiB",
        "0GiB",
        "0.1B",
        "1GB junk",
        "8GB\nignored",
    ],
)
def test_invalid_budget_is_rejected_before_scheduling(value):
    with pytest.raises((ValueError, TypeError)):
        MemoryBudget(value)


def test_budget_combines_host_device_and_per_worker_overhead_without_mutating_jobs():
    budget = MemoryBudget(100, device_limit=25, worker_overhead=10)
    first, second = MemoryEstimate(30, 10), MemoryEstimate(20, 20)
    assert budget.fits(first)
    assert not budget.fits(second, current=[first])  # device 30 > 25, total otherwise 100
    assert MemoryBudget(100, worker_overhead=10).fits(second, current=[first])
    assert not MemoryBudget(99, worker_overhead=10).fits(second, current=[first])
    assert not MemoryBudget(39, worker_overhead=0).fits(first)
    assert MemoryBudget(40, worker_overhead=0).fits(first)
    assert first.total_bytes == 40
    assert budget.device_limit_bytes == 25
    with pytest.raises(TypeError, match="MemoryEstimate"):
        budget.fits(40)


@pytest.mark.parametrize(
    "kw", [{"device_limit": 0}, {"worker_overhead": -1}, {"worker_overhead": "0.1B"}]
)
def test_invalid_secondary_limits_are_rejected(kw):
    with pytest.raises(ValueError):
        MemoryBudget("1GiB", **kw)


def test_large_byte_strings_are_exact_independent_of_decimal_context():
    import decimal

    with decimal.localcontext() as context:
        context.prec = 3
        value = "123456789012345678901234567890GiB"
        assert MemoryBudget(value).limit == 123456789012345678901234567890 * 2**30


def test_default_screen_promotes_eight_of_twenty_four_and_preserves_full_evaluation():
    screen = Screening()
    screen.validate(3, 3)
    assert screen.promoted_count(24) == 8
    assert [screen.promoted_count(n) for n in (0, 1, 2, 3, 4)] == [0, 1, 1, 1, 2]
    assert Screening(keep_fraction=0.1).promoted_count(10) == 1
    assert Screening(keep_fraction=1).promoted_count(24) == 24
    assert Screening(keep_fraction=0.5).promoted_count(10**400) == 10**400 // 2
    assert screen.cohort_size is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        screen.seed_count = 5


@pytest.mark.parametrize(
    "kw",
    [
        {"seed_count": 0},
        {"origin_count": -1},
        {"seed_count": True},
        {"origin_count": 1.5},
        {"cohort_size": 0},
        {"cohort_size": False},
        {"keep_fraction": 0},
        {"keep_fraction": 1.01},
        {"keep_fraction": True},
        {"keep_fraction": float("nan")},
        {"keep_fraction": float("inf")},
    ],
)
def test_invalid_screen_settings_fail_at_construction(kw):
    with pytest.raises(ValueError):
        Screening(**kw)


def test_screen_checks_full_repetition_counts_and_integer_cohort_size():
    with pytest.raises(ValueError, match="exceed"):
        Screening(seed_count=2).validate(1, 3)
    with pytest.raises(ValueError, match="exceed"):
        Screening(origin_count=4).validate(3, 3)
    with pytest.raises(ValueError):
        Screening().validate(True, 3)
    with pytest.raises(ValueError):
        Screening().promoted_count(2.0)


@pytest.mark.parametrize("shape", [(40, 2), (2, 40, 2)])
def test_context_records_full_refit_rows_and_actual_shared_storage(shape, monkeypatch):
    model = _model()
    series = torch.arange(np.prod(shape), dtype=torch.float64).reshape(shape)
    data = prepare_tuning_data(model, series, horizon=3)
    # A strided view keeps all of its backing allocation alive. Shared model
    # buffers likewise count once, including a nonpersistent buffer.
    backing = torch.zeros(37)
    model.register_buffer("resource_a", backing[2:5], persistent=False)
    model.register_buffer("resource_b", backing[20:23])
    before = {
        id(value.untyped_storage()): value.untyped_storage().nbytes()
        for value in (*model.parameters(), *model.buffers())
    }
    # Use stable storage handles rather than Python wrapper identity as oracle.
    storages = {}
    for value in (*model.parameters(), *model.buffers()):
        storages[value.untyped_storage()._cdata] = value.untyped_storage().nbytes()
    assert before
    monkeypatch.setattr(model.graph, "forward", lambda *a: pytest.fail("planning executed graph"))
    context = memory_context(model, data)
    assert context.model_host_bytes == sum(storages.values())
    assert context.model_device_bytes == 0
    assert context.refit_samples == (1 if len(shape) == 2 else shape[0]) * 40
    assert context.train_samples < context.refit_samples
    assert context.input_shapes == {"state": shape}
    assert context.batch_size == (1 if len(shape) == 2 else shape[0])
    assert context.dataset_host_bytes >= series.untyped_storage().nbytes()
    # The exact storage oracle covers aliases between training, targets,
    # forecast histories and validation truth, plus independent scales/times.
    tensors = [
        *data.train.inputs.values(),
        *data.train.targets.values(),
        *data.refit.inputs.values(),
        *data.refit.targets.values(),
        *data.scales.values(),
    ]
    for origin in data.origins:
        tensors += [*origin.history.values(), *origin.truth.values(), origin.times]
    unique = {value.untyped_storage()._cdata: value.untyped_storage().nbytes() for value in tensors}
    assert context.dataset_host_bytes == sum(unique.values())


def test_context_metadata_has_no_mutable_tensor_or_model_references_and_serializes():
    model = _model()
    data = prepare_tuning_data(model, torch.randn(40, 2).double(), horizon=3)
    options = {
        "reservoir_size": 10,
        "custom": {"items": [1, 2]},
        "tensor": torch.ones(5),
        "module": model,
    }
    context = memory_context(model, data, preset_options=options)
    assert isinstance(context, MemoryContext)
    options["custom"]["items"].append(3)
    assert context.preset_options["custom"]["items"] == (1, 2)
    assert context.preset_options["tensor"] == ("tensor", (5,), "torch.float32", "cpu")
    assert isinstance(context.preset_options["module"], tuple)
    with pytest.raises(TypeError, match="immutable"):
        context.input_shapes["state"] = (1, 1)
    with pytest.raises(TypeError, match="immutable"):
        context.preset_options["custom"]["items"] = (3,)
    restored = pickle.loads(pickle.dumps(context))
    assert restored == context
    assert copy.deepcopy(context) == context
    result = MemoryEstimate(1, details={"nested": {"list": [1, 2]}})
    assert json.loads(json.dumps(dataclasses.asdict(result))) == {
        "host_bytes": 1,
        "device_bytes": 0,
        "details": {"nested": {"list": [1, 2]}},
    }
    assert pickle.loads(pickle.dumps(result)) == result


def test_automatic_estimate_uses_full_refit_and_eight_byte_workspace_even_for_half_data():
    context = dataclasses.replace(_context(), dtype=torch.float16)
    estimate = estimate_memory({}, context)
    n, f, out = context.refit_samples, context.readout_features[0], context.readout_outputs[0]
    expected_workspace = 8 * (6 * n * (f + out) + 12 * f * f + 6 * min(n, f) ** 2 + 4 * f * out)
    assert estimate.details["samples"] == context.refit_samples
    assert estimate.details["solve_workspace"] == expected_workspace
    assert estimate.details["element_bytes"] == 2
    assert estimate.host_bytes >= expected_workspace + context.dataset_host_bytes
    assert estimate.device_bytes == 0
    assert (
        estimate_memory({}, dataclasses.replace(context, refit_samples=80)).host_bytes
        > estimate.host_bytes
    )


def test_accelerator_planning_includes_cpu_bridge_without_using_an_accelerator():
    context = dataclasses.replace(_context(), device=torch.device("cuda:17"))
    estimate = estimate_memory({}, context)
    assert estimate.device_bytes > 0
    assert estimate.details["cpu_bridge"] == estimate.details["solve_workspace"]
    assert estimate.host_bytes >= context.dataset_host_bytes + estimate.details["cpu_bridge"]


def test_preset_width_estimate_is_quadratic_and_never_constructs_a_reservoir(monkeypatch):
    context = dataclasses.replace(_context(), preset_options={"reservoir_size": 20})
    monkeypatch.setattr(rd.ESN, "__init__", lambda *a, **k: pytest.fail("constructed model"))
    small = estimate_memory({"reservoir_size": 20}, context)
    large = estimate_memory({"reservoir_size": 10**9}, context)
    assert large.host_bytes >= 3 * 8 * 10**18
    assert large.host_bytes > small.host_bytes
    assert not MemoryBudget("8GiB").fits(large)
    assert MemoryBudget("8GiB").fits(small)


def test_custom_factory_requires_explicit_estimator_and_does_not_mutate_params():
    context = _context()
    with pytest.raises(ValueError, match="custom factory.*memory_estimator"):
        estimate_memory({}, context, custom_factory=True)
    params = {"nested": {"width": 3}}

    def custom(values, facts):
        values["nested"]["width"] = 9
        assert facts is context
        return MemoryEstimate(100, 40, {"custom": True})

    estimated = estimate_memory(params, context, custom_factory=True, estimator=custom)
    assert estimated.host_bytes == max(100, context.dataset_host_bytes + context.model_host_bytes)
    assert estimated.device_bytes == 40
    assert estimated.details["requested_host_bytes"] == 100
    assert estimated.details["floor_applied"] is True
    assert params == {"nested": {"width": 3}}
    assert (
        estimate_memory({}, context, estimator=lambda p, c: np.int64(100)).details[
            "requested_host_bytes"
        ]
        == 100
    )


@pytest.mark.parametrize("result", [True, -1, 1.5, "1GiB", None])
def test_custom_estimator_result_must_be_estimate_or_nonnegative_integer(result):
    with pytest.raises(ValueError, match="custom memory estimate"):
        estimate_memory({}, _context(), estimator=lambda p, c: result)


def test_planning_does_not_change_global_torch_settings_or_model_values(monkeypatch):
    model = _model()
    data = prepare_tuning_data(model, torch.randn(40, 2).double(), horizon=3)
    state = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }
    default_dtype, threads = torch.get_default_dtype(), torch.get_num_threads()

    def forbidden(*args, **kwargs):
        pytest.fail("resource planning changed global settings or model precision")

    for name in ("set_default_dtype", "set_num_threads", "set_num_interop_threads"):
        monkeypatch.setattr(torch, name, forbidden)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", forbidden)
    monkeypatch.setattr(model, "to", forbidden)
    context = memory_context(model, data)
    estimate_memory({}, context)
    assert torch.get_default_dtype() == default_dtype and torch.get_num_threads() == threads
    for name, expected in state.items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kw", [{"host_bytes": -1}, {"host_bytes": True}, {"host_bytes": 1, "device_bytes": 1.0}]
)
def test_invalid_estimates_are_rejected(kw):
    with pytest.raises(ValueError):
        MemoryEstimate(**kw)


def test_context_rejects_invalid_dtype_layout_counts_and_head_dimensions():
    context = _context()
    for kw in (
        {"dtype": torch.int64},
        {"input_shapes": {"state": (40,)}},
        {"input_shapes": {"state": (0, 2)}},
        {"refit_samples": 0},
        {"readout_outputs": ()},
        {"dataset_host_bytes": -1},
    ):
        with pytest.raises(ValueError):
            dataclasses.replace(context, **kw)


def _raw_data():
    backing = torch.arange(24, dtype=torch.float64).reshape(1, 12, 2)
    driver = torch.ones(1, 12, 3, dtype=torch.float64)
    return {
        "warmup": backing[:, :2],
        "train": backing[:, 2:7],
        "target": backing[:, 3:8],
        "f_warmup": backing[:, :8],
        "val": backing[:, 8:],
        "warmup_forcing": driver[:, :2],
        "train_forcing": driver[:, 2:7],
        "f_warmup_forcing": driver[:, :8],
        "forecast_forcing": driver[:, 8:],
        "steps": 4,
    }


def test_raw_context_deduplicates_views_reserves_full_prefix_and_needs_estimator():
    data = _raw_data()
    context = raw_memory_context(data)
    assert context.input_shapes == {"train": (1, 7, 2), "train_forcing": (1, 7, 3)}
    assert context.target_shapes == {"target": (1, 5, 2)}
    assert context.train_samples == context.refit_samples == 7
    assert context.dataset_host_bytes == (12 * 2 + 12 * 3) * 8
    assert context.retained_forecast_bytes == 1 * 4 * 2 * 8
    assert context.max_horizon == 4
    assert context.readout_features == context.readout_outputs == ()
    with pytest.raises(ValueError, match="custom factory"):
        estimate_memory({}, context, custom_factory=True)
    with pytest.raises(ValueError, match="Unknown readout"):
        estimate_memory({}, context)
    estimate = estimate_memory({}, context, custom_factory=True, estimator=lambda p, c: 0)
    assert estimate.host_bytes == context.dataset_host_bytes + 64
    assert estimate.details["minimum_host_bytes"] == estimate.host_bytes
    assert estimate.details["requested_host_bytes"] == 0
    assert estimate.details["floor_applied"] is True


def test_raw_context_includes_each_alias_copy_when_crossing_devices_without_allocation():
    data = _raw_data()
    context = raw_memory_context(data, device="cuda:3", horizon=3)
    transferred = sum(
        value.numel() * value.element_size()
        for value in data.values()
        if isinstance(value, torch.Tensor)
    )
    assert context.dataset_host_bytes == (12 * 2 + 12 * 3) * 8
    assert context.dataset_device_bytes == transferred
    assert context.retained_forecast_bytes == 3 * 2 * 8
    estimate = estimate_memory(
        {}, context, custom_factory=True, estimator=lambda p, c: MemoryEstimate(0, 0)
    )
    assert estimate.host_bytes == context.dataset_host_bytes
    assert estimate.device_bytes == transferred + context.retained_forecast_bytes
    assert estimate.details["minimum_device_bytes"] == estimate.device_bytes


def test_custom_estimates_remain_complete_totals_above_known_floor():
    context = raw_memory_context(_raw_data())
    estimate = estimate_memory(
        {}, context, estimator=lambda p, c: MemoryEstimate(10**8, 10**7, {"recipe": "custom"})
    )
    assert (estimate.host_bytes, estimate.device_bytes) == (10**8, 10**7)
    assert estimate.details["floor_applied"] is False
    assert estimate.details["recipe"] == "custom"


@pytest.mark.parametrize("name", ["train", "target", "val"])
def test_raw_context_rejects_missing_or_wrong_tensor_layout(name):
    data = _raw_data()
    data[name] = data[name].squeeze(0)
    with pytest.raises(ValueError, match="Raw memory context requires"):
        raw_memory_context(data)


def test_raw_zero_warmup_and_explicit_working_precision():
    data = _raw_data()
    data["warmup"] = data["warmup"][:, :0]
    context = raw_memory_context(data, dtype=torch.float32, horizon=2)
    assert context.retained_forecast_bytes == 1 * 2 * 2 * 4
    with pytest.raises(ValueError, match="dtype"):
        raw_memory_context(data, dtype=torch.int64)


def test_dense_and_sparse_model_buffers_count_actual_storage_including_sparse_indices():
    model = _model()
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0, 2], [1, 0]]), torch.tensor([1.0, 2.0]), (3, 3)
    )
    model.register_buffer("sparse_resource", sparse)
    data = prepare_tuning_data(model, torch.ones(40, 2).double(), horizon=3)
    expected = sum(
        value.untyped_storage().nbytes()
        for value in (
            *model.parameters(),
            *[v for v in model.buffers() if v.layout == torch.strided],
        )
    )
    expected += (
        sparse._indices().untyped_storage().nbytes() + sparse._values().untyped_storage().nbytes()
    )
    assert memory_context(model, data).model_host_bytes == expected


def test_retained_full_objective_forecasts_are_explicit_and_scale_with_horizon():
    context = _context()
    base = estimate_memory({}, context)
    retained = context.batch_size * 100 * context.output_features * 8
    full = estimate_memory({}, dataclasses.replace(context, retained_forecast_bytes=retained))
    assert full.details["forecast_outputs"] == 3 * retained
    assert full.host_bytes > base.host_bytes


def test_history_replay_reserve_is_additive_to_output_retention_and_custom_floor():
    context = dataclasses.replace(_context(), retained_forecast_bytes=512)
    before = estimate_memory({}, context)
    replay = dataclasses.replace(context, replay_history_bytes=4096)
    after = estimate_memory({}, replay)
    assert after.host_bytes - before.host_bytes == 4096
    assert after.details["replay_history"] == 4096
    assert after.details["forecast_outputs"] == before.details["forecast_outputs"]
    custom = estimate_memory({}, replay, estimator=lambda p, c: 0)
    assert custom.host_bytes == context.dataset_host_bytes + context.model_host_bytes + 512 + 4096
    assert custom.details["replay_history_bytes"] == 4096
    accelerator = dataclasses.replace(replay, device=torch.device("cuda:9"))
    custom_device = estimate_memory({}, accelerator, estimator=lambda p, c: 0)
    assert custom_device.device_bytes == 512 + 4096
    with pytest.raises(ValueError):
        dataclasses.replace(context, replay_history_bytes=-1)


def test_late_fast_forecast_history_reserves_warmup_traces_without_changing_solver_rows():
    model = _model()
    training = torch.arange(24, dtype=torch.float64).reshape(12, 2)
    validation = torch.arange(24, 2024, dtype=torch.float64).reshape(1000, 2)
    data = prepare_tuning_data(
        model, training, validation=(validation, None), horizon=50, n_origins=3
    )
    context = memory_context(model, data)
    assert context.refit_samples == context.train_samples == 12
    expected = max(value.shape[-2] for origin in data.origins for value in origin.history.values())
    assert expected > 900
    assert context.forecast_samples == expected
    estimate = estimate_memory({}, context)
    assert estimate.details["samples"] == 12  # solve uses fit rows, not validation history
    assert estimate.details["trace_samples"] == expected
    assert estimate.details["feature_traces"] == 4 * expected * context.activation_width * 8
    shorter = estimate_memory({}, dataclasses.replace(context, forecast_samples=0))
    assert estimate.details["solve_workspace"] == shorter.details["solve_workspace"]
    assert estimate.host_bytes > shorter.host_bytes
