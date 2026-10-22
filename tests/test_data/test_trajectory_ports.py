"""Independent physical-time oracles for named optional trajectory ports."""

import builtins
import copy
import io

import numpy as np
import pytest
import torch

from resdag.data import _trajectory_ports as adapter


@pytest.fixture
def trajectory():
    return pytest.importorskip("tsdynamics").Trajectory


def _series(trajectory, times, *, name="x", function=lambda t: 2 * t + 3):
    times = np.asarray(times, dtype=np.float64)
    return trajectory(times, function(times)[:, None], variables=(name,))


def _prepare(training, targets=None, **options):
    return adapter.prepare_trajectory_ports(
        training,
        input_names=("state",),
        input_widths=(1,),
        output_names=("next",),
        output_widths=(1,),
        prediction_leads=(1,),
        targets=targets,
        **options,
    )


def _validate(context, *, names=("state",), outputs=("next",), leads=(1,)):
    return adapter.validate_trajectory_ports_context(
        context, names, (1,) * len(names), outputs, (1,) * len(outputs), leads
    )


def test_tensor_port_recognition_never_imports_optional_package(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("tsdynamics"):
            raise AssertionError("tensor-only paths must remain optional-import free")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert not adapter.has_trajectory_port({"state": torch.ones(3, 1)})
    assert not adapter.has_trajectory_port(torch.ones(3, 1))


@pytest.mark.parametrize("method", ["linear", "pchip", "cubic"])
def test_bare_irregular_default_keeps_sample_count_and_endpoints(trajectory, method):
    source = _series(trajectory, [2, 2.2, 2.9, 4])
    prepared = _prepare(source, method=method)
    expected = np.linspace(2, 4, 4)
    np.testing.assert_allclose(prepared.times, expected, atol=1e-15, rtol=0)
    np.testing.assert_allclose(prepared.inputs[0][:, 0], 2 * expected + 3, atol=1e-14)
    np.testing.assert_allclose(prepared.targets[0][:, 0], 2 * expected[1:] + 3, atol=1e-14)
    assert (prepared.supervised_start, prepared.supervised_stop) == (0, 3)
    assert prepared.context["version"] == 2
    assert prepared.context["clock"]["reference_input"] == "state"
    assert prepared.context["clock"]["dt"] == pytest.approx(2 / 3)


def test_declared_input_order_controls_interval_and_joint_target_support(trajectory):
    state = _series(trajectory, np.arange(11), name="position")
    driver = _series(trajectory, np.arange(1.25, 10.26, 0.5), name="forcing")
    one = _series(trajectory, np.arange(8, 13), name="one", function=lambda t: t**2)
    two = _series(trajectory, np.arange(7, 13), name="two", function=lambda t: -t)
    prepared = adapter.prepare_trajectory_ports(
        {"driver": driver, "state": state},
        input_names=("state", "driver"),
        input_widths=(1, 1),
        output_names=("near", "far"),
        output_widths=(1, 1),
        prediction_leads=(1, 2),
        targets={"far": two, "near": one},
    )
    np.testing.assert_array_equal(prepared.times, np.arange(2, 11))
    assert (prepared.supervised_start, prepared.supervised_stop) == (5, 7)
    np.testing.assert_array_equal(prepared.targets[0][:, 0], [64, 81])
    np.testing.assert_array_equal(prepared.targets[1][:, 0], [-9, -10])
    assert all(len(values) == 9 for values in prepared.inputs)
    assert prepared.context["outputs"]["near"]["variables"] == ["one"]
    assert prepared.context["outputs"]["far"]["held_out_samples"] == 2
    assert prepared.context["clock"]["dt"] == 1
    _validate(prepared.context, names=("state", "driver"), outputs=("near", "far"), leads=(1, 2))


@pytest.mark.parametrize("method", ["linear", "pchip", "cubic"])
def test_target_future_suffix_never_changes_interpolation_coefficients(trajectory, method):
    observed = _series(trajectory, np.arange(11))
    target_times = np.array([0, 1.5, 4, 6.25, 9.5, 10.5, 12])
    target_values = np.sin(target_times)
    baseline = trajectory(target_times, target_values[:, None], variables=("target",))
    changed = target_values.copy()
    changed[target_times > 10] = [1e30, np.nan]
    perturbed = trajectory(target_times, changed[:, None], variables=("target",))
    first = _prepare(observed, baseline, method=method)
    second = _prepare(observed, perturbed, method=method)
    np.testing.assert_array_equal(first.targets[0], second.targets[0])
    assert first.supervised_stop == 9  # last usable target source ends at 9.5
    assert first.context == second.context
    assert first.context["outputs"]["next"]["held_out_samples"] == 2


def test_explicit_step_right_anchors_common_domain_without_extrapolation(trajectory):
    prepared = _prepare(_series(trajectory, [2, 2.2, 2.9, 4]), dt=0.75)
    np.testing.assert_array_equal(prepared.times, [2.5, 3.25, 4])
    np.testing.assert_allclose(prepared.inputs[0][:, 0], [8, 9.5, 11])
    assert prepared.context["clock"]["requested_dt"] == 0.75


@pytest.mark.parametrize("which", ["training", "targets"])
def test_mixed_clocked_and_tensor_ports_reject(trajectory, which):
    source = _series(trajectory, np.arange(6))
    options = dict(
        input_names=("a", "b"),
        input_widths=(1, 1),
        output_names=("one", "two"),
        output_widths=(1, 1),
        prediction_leads=(1, 1),
    )
    training = {"a": source, "b": source}
    targets = {"one": source, "two": source}
    (training if which == "training" else targets)["b" if which == "training" else "two"] = (
        torch.ones(6, 1)
    )
    with pytest.raises(TypeError, match="mixed Tensor/Trajectory"):
        adapter.prepare_trajectory_ports(training, targets=targets, **options)


def test_history_uses_fitted_interval_and_checks_component_order(trajectory):
    fitted = _prepare(_series(trajectory, np.arange(6), name="position"))
    history = _series(trajectory, [20, 20.25, 21.5, 23], name="position")
    prepared = adapter.prepare_trajectory_port_history(
        history,
        input_names=("state",),
        input_widths=(1,),
        output_names=("next",),
        output_widths=(1,),
        context=fitted.context,
        dt=0.1,
    )
    np.testing.assert_array_equal(prepared.times, [20, 21, 22, 23])
    assert prepared.context["supervision"] is None
    assert prepared.context["outputs"] == fitted.context["outputs"]
    bad = _series(trajectory, [20, 21, 22], name="velocity")
    with pytest.raises(ValueError, match="components/order"):
        adapter.prepare_trajectory_port_history(
            bad,
            input_names=("state",),
            input_widths=(1,),
            output_names=("next",),
            output_widths=(1,),
            context=fitted.context,
        )


def test_drivers_query_required_ticks_instead_of_reanchoring(trajectory):
    driver = _series(trajectory, [10, 10.4, 11.1, 12.2, 13])
    values = adapter.prepare_trajectory_port_drivers(
        {"driver": driver},
        input_names=("driver",),
        input_widths=(1,),
        last_time=10,
        dt=1,
        steps=3,
    )
    np.testing.assert_allclose(values[0][:, 0], [25, 27, 29])


@pytest.mark.parametrize("origin", [0, 1e9])
def test_singleton_driver_uses_step_relative_clock_tolerance(trajectory, origin):
    options = dict(input_names=("driver",), input_widths=(1,), last_time=origin, dt=1, steps=1)
    values = adapter.prepare_trajectory_port_drivers(
        {"driver": _series(trajectory, [origin + 1])}, **options
    )
    np.testing.assert_array_equal(values[0], [[2 * (origin + 1) + 3]])
    with pytest.raises(ValueError, match="extrapolation"):
        adapter.prepare_trajectory_port_drivers(
            {"driver": _series(trajectory, [origin + 2])}, **options
        )


def test_empty_driver_horizon_one_and_unknown_clock_preflight(trajectory):
    empty = _series(trajectory, [])
    options = dict(input_names=("driver",), input_widths=(1,), dt=1, steps=0)
    values = adapter.prepare_trajectory_port_drivers({"driver": empty}, last_time=10, **options)
    assert values[0].shape == (0, 1)
    with pytest.raises(ValueError, match="timestamped or cached"):
        adapter.prepare_trajectory_port_drivers({"driver": empty}, last_time=None, **options)


def test_forecast_shared_timeline_and_per_head_target_metadata(trajectory):
    observed = _series(trajectory, np.arange(6), name="position")
    fitted = _prepare(observed)
    context = fitted.context
    context["outputs"]["far"] = copy.deepcopy(context["outputs"]["next"])
    context["outputs"]["far"]["variables"] = ["velocity"]
    times = adapter.trajectory_port_forecast_times(context, 2, history_length=6, return_warmup=True)
    np.testing.assert_array_equal(times, np.arange(1, 8))
    results = adapter.forecast_trajectory_ports(
        {"far": -times[:, None], "next": times[:, None]},
        context,
        output_names=("next", "far"),
        times=times,
    )
    assert list(results) == ["next", "far"]
    assert results["next"].variables == ("position",)
    assert results["far"].variables == ("velocity",)
    np.testing.assert_array_equal(results["next"].t, results["far"].t)
    np.testing.assert_array_equal(results["far"].y[:, 0], -times)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update(version=1),
        lambda c: c["clock"].update(reference_input="other"),
        lambda c: c["clock"].update(n_samples=99),
        lambda c: c["inputs"]["state"].update(variables=["x", "y"]),
        lambda c: c["inputs"]["state"]["source"].update(unsafe=object()),
        lambda c: c["outputs"]["next"]["observation"].update(last_time=99),
        lambda c: c["supervision"].update(start=4),
        lambda c: c["supervision"]["prediction_leads"].update(next=2),
    ],
)
def test_context_rejects_structural_and_physical_corruption(trajectory, mutate):
    context = _prepare(_series(trajectory, np.arange(5))).context
    mutate(context)
    with pytest.raises(ValueError):
        _validate(context)


def test_current_context_is_weights_only_portable_and_not_aliased(trajectory):
    context = _prepare(_series(trajectory, np.arange(5))).context
    stream = io.BytesIO()
    torch.save(context, stream)
    stream.seek(0)
    restored = _validate(torch.load(stream, weights_only=True))
    assert restored == context
    restored["inputs"]["state"]["variables"][0] = "changed"
    assert context["inputs"]["state"]["variables"] == ["x"]


@pytest.mark.parametrize("condition", ["no_overlap", "no_labels", "multi_without_targets"])
def test_empty_shared_support_has_actionable_error(trajectory, condition):
    first = _series(trajectory, np.arange(5))
    second = _series(trajectory, np.arange(10, 15))
    if condition == "no_labels":
        with pytest.raises(ValueError, match="no jointly supported"):
            _prepare(first, second)
        return
    with pytest.raises(ValueError, match="shared observed|explicit per-output"):
        adapter.prepare_trajectory_ports(
            {"a": first, "b": second if condition == "no_overlap" else first},
            input_names=("a", "b"),
            input_widths=(1, 1),
            output_names=("next",),
            output_widths=(1,),
            prediction_leads=(1,),
        )
