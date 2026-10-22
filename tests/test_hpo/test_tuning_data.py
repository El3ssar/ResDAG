"""Independent ordinal/physical-time and held-out-data tuning oracles."""

import copy

import numpy as np
import pytest
import torch
from torch import nn

import resdag as rd
from resdag.hpo._tuning_data import PreparedFit, model_signature, prepare_tuning_data


def _model(*, washout=2, leads=(1, 3), drivers=False, **options):
    state = rd.reservoir_input(1, dtype=torch.float64)
    inputs = {"state": state}
    features, width = state, 1
    if drivers:
        forcing = rd.reservoir_input(1, dtype=torch.float64)
        inputs["forcing"] = forcing
        features, width = rd.Concatenate()(state, forcing), 2
    heads = {
        name: rd.SVDReadoutLayer(width, 1, alpha=0, name=f"head_{name}").double()(features)
        for name in ("near", "far")
    }
    return rd.Model(
        inputs=inputs,
        outputs=heads,
        washout=washout,
        prediction_lead=dict(zip(heads, leads)),
        feedback=[rd.Feedback("far", "state", delay=leads[1])],
        **options,
    )


def _single(**options):
    x = rd.reservoir_input(1, dtype=torch.float64)
    y = rd.RidgeReadoutLayer(1, 1, alpha=0).double()(x)
    return rd.Model(inputs={"state": x}, outputs={"prediction": y}, washout=2, **options)


def _same_state(left, right):
    assert left.keys() == right.keys()
    for name, value in left.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, right[name], rtol=0, atol=0)
        else:
            assert value == right[name]


@pytest.mark.parametrize("batch", [None, 1, 2])
def test_tensor_two_leads_have_independent_origin_alignment_and_complete_refit_tail(batch):
    data = torch.arange(12, dtype=torch.float64)[:, None]
    if batch is not None:
        data = data.expand(batch, -1, -1).clone()
    labels = {"far": data + 3, "near": data + 1}
    model = _model()
    plan = prepare_tuning_data(model, data, labels, horizon=2)
    assert plan.train.supervised_stop == 6  # C=9, maxlead=3
    assert plan.refit.supervised_stop == 9
    assert len(plan.origins) == 2  # only complete windows, no duplicate origins
    torch.testing.assert_close(plan.train.inputs["state"], data[..., :9, :])
    for name in model.output_names:
        torch.testing.assert_close(plan.train.targets[name], labels[name][..., :6, :])
        torch.testing.assert_close(plan.scales[name], torch.tensor([np.sqrt(1.25)]).double())
    plan.fit(model)
    torch.testing.assert_close(model.input_scalers[0]._stat0, torch.tensor([[[3.5]]]).double())
    assert model._history_0[..., -1, :].eq(8).all()
    for origin in plan.origins:
        predictions = origin.forecast(model)
        for name in model.output_names:
            expected = origin.times.double()[..., None]
            if batch is not None:
                expected = expected.expand(batch, -1, -1)
            torch.testing.assert_close(origin.truth[name], expected)
            torch.testing.assert_close(predictions[name], expected, atol=1e-10, rtol=0)
    plan.fit(model, refit=True)
    assert model._history_0[..., -1, :].eq(11).all()
    result = model.forecast(horizon=2)
    expected = torch.tensor([[12.0], [13.0]]).double()
    if batch is not None:
        expected = expected.expand(batch, -1, -1)
    for value in result.values():
        torch.testing.assert_close(value, expected, atol=1e-10, rtol=0)
    assert plan.report["refit_discarded_target_rows"] == {"near": 3, "far": 3}


def test_validation_labels_cannot_change_fitted_weights_transforms_or_score_scales():
    x = torch.arange(40, dtype=torch.float64)[:, None]
    labels = {"near": x + 1, "far": x + 3}
    changed = {name: values.clone() for name, values in labels.items()}
    # Physical validation begins at 32. Each head has its own origin index.
    changed["near"][31:] += 1e9
    changed["far"][29:] -= 1e9
    left, right = _model(), _model()
    original = prepare_tuning_data(left, x, labels, horizon=3)
    perturbed = prepare_tuning_data(right, x, changed, horizon=3)
    original.fit(left)
    perturbed.fit(right)
    _same_state(left.state_dict(), right.state_dict())
    for name in left.output_names:
        torch.testing.assert_close(original.scales[name], perturbed.scales[name], rtol=0, atol=0)
    # Later trial washout cannot alter the frozen scoring scale.
    right.washout = 4
    perturbed.fit(right)
    for name in left.output_names:
        torch.testing.assert_close(original.scales[name], perturbed.scales[name], rtol=0, atol=0)


def test_unsupported_explicit_future_target_suffix_is_never_read_or_refitted():
    x = torch.arange(40, dtype=torch.float64)[:, None]
    labels = {"near": x + 1, "far": x + 3}
    labels["near"][-1:] = torch.nan
    labels["far"][-3:] = torch.inf
    model = _model()
    plan = prepare_tuning_data(model, x, labels, horizon=4)
    plan.fit(model, refit=True)
    for value in model.forecast(horizon=3).values():
        torch.testing.assert_close(
            value, torch.arange(40, 43).double()[:, None], atol=1e-10, rtol=0
        )


@pytest.mark.parametrize("explicit", [False, True])
def test_explicit_validation_is_held_out_from_final_refit(explicit):
    train = torch.arange(12, dtype=torch.float64)[:, None]
    validation = torch.arange(12, 20, dtype=torch.float64)[:, None]
    if explicit:
        model = _model()
        targets = {"near": train + 1, "far": train + 3}
        val_targets = {"near": validation + 1, "far": validation + 3}
    else:
        model, targets, val_targets = _single(), None, None
    plan = prepare_tuning_data(
        model, train, targets, validation=(validation, val_targets), horizon=2
    )
    assert plan.report["first_validation_index"] == (15 if explicit else 12)
    assert plan.refit is plan.train
    assert not plan.report["refit_includes_validation"]
    plan.fit(model, refit=True)
    assert model._history_0[-1, 0].item() == 11
    first = plan.origins[0]
    for name, prediction in first.forecast(model).items():
        torch.testing.assert_close(prediction, first.truth[name], atol=1e-10, rtol=0)
    for value in plan.scales.values():
        assert value.item() < 10  # no validation statistics


@pytest.mark.parametrize("horizon", [1, 2, 5])
def test_driven_polynomial_recurrence_checks_driver_count_and_future_truth(horizon):
    time = torch.arange(50, dtype=torch.float64)[:, None]

    def position(t):
        return t * (t + 1) / 2

    values = {"forcing": time + 1, "state": position(time)}
    targets = {"far": position(time + 3), "near": position(time + 1)}
    model = _model(drivers=True)
    plan = prepare_tuning_data(model, values, targets, horizon=horizon)
    plan.fit(model)
    for origin in plan.origins:
        if horizon == 1:
            assert origin.future_inputs is None
        else:
            assert list(origin.future_inputs) == ["forcing"]
            assert origin.future_inputs["forcing"].shape == (horizon - 1, 1)
            torch.testing.assert_close(
                origin.future_inputs["forcing"][:, 0], origin.times[:-1].double() + 1
            )
        actual = origin.forecast(model)
        for name in model.output_names:
            expected = position(origin.times.double()[:, None])
            torch.testing.assert_close(origin.truth[name], expected)
            torch.testing.assert_close(actual[name], expected, atol=1e-9, rtol=0)


class _Cumulative(nn.Module):
    def forward(self, x):
        return x.cumsum(dim=1)


class _RecordRows(nn.Module):
    supports_framewise = True

    def __init__(self):
        super().__init__()
        self.register_buffer("summary", torch.zeros(3, dtype=torch.float64))

    def fit(self, data):
        self.summary.copy_(
            torch.stack((data.new_tensor(data.shape[1]), data[:, 0].mean(), data[:, -1].mean()))
        )
        return self

    def forward(self, data):
        return data

    def inverse(self, data):
        return data


def test_private_prepared_fit_keeps_temporal_prefix_and_absolute_washout_once():
    x = rd.reservoir_input(1, dtype=torch.float64)
    y = rd.RidgeReadoutLayer(1, 1, alpha=0).double()(_Cumulative()(x))
    incoming, outgoing = _RecordRows(), _RecordRows()
    model = rd.Model(
        inputs={"state": x},
        outputs={"prediction": y},
        washout=2,
        input_transforms={"state": incoming},
        output_transforms={"prediction": outgoing},
    )
    observations = torch.arange(1, 13, dtype=torch.float64)[:, None]
    all_labels = observations.cumsum(0) * 2 + 7
    prepared = PreparedFit({"state": observations}, {"prediction": all_labels[4:9]}, 4, 9, False)
    prepared.fit(model)
    torch.testing.assert_close(incoming.summary, torch.tensor([5.0, 5.0, 9.0]).double())
    torch.testing.assert_close(outgoing.summary, torch.tensor([5.0, 37.0, 97.0]).double())
    # A sliced warmup would lose the first four cumulative contributions.
    torch.testing.assert_close(model(observations), all_labels, atol=1e-10, rtol=0)
    torch.testing.assert_close(model._history_0, observations)


def test_preparation_is_read_only_and_failed_private_fit_preserves_existing_model():
    model = _single()
    x = torch.arange(30, dtype=torch.float64)[:, None]
    model.fit(x)
    before = copy.deepcopy(model.state_dict())
    snapshot = x.clone()
    plan = prepare_tuning_data(model, x, horizon=3)
    _same_state(model.state_dict(), before)
    bad = PreparedFit(
        plan.train.inputs, {"prediction": torch.full((2, 1), torch.nan).double()}, 0, 2, True
    )
    with pytest.raises(ValueError):
        bad.fit(model)
    _same_state(model.state_dict(), before)
    torch.testing.assert_close(x, snapshot, rtol=0, atol=0)


def test_tensor_preparation_never_converts_to_numpy_or_cpu(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("tensor data left its device")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    x = torch.arange(40, dtype=torch.float64)[:, None]
    model = _single()
    plan = prepare_tuning_data(model, x, horizon=3)
    plan.fit(model)
    for origin in plan.origins:
        prediction = origin.forecast(model)["prediction"]
        torch.testing.assert_close(prediction, origin.truth["prediction"], atol=1e-10, rtol=0)


@pytest.mark.parametrize(
    "options",
    [
        {"horizon": 0},
        {"horizon": True},
        {"horizon": 99},
        {"n_origins": 0},
        {"validation_fraction": float("nan")},
        {"validation_fraction": True},
        {"validation": []},
    ],
)
def test_invalid_protocol_is_rejected_before_any_model_fit(options, monkeypatch):
    model = _single()
    monkeypatch.setattr(
        model, "_fit_prepared", lambda *a, **kw: pytest.fail("preflight fitted a model")
    )
    with pytest.raises((ValueError, TypeError)):
        prepare_tuning_data(model, torch.arange(30).double()[:, None], **options)


def test_candidate_signature_keeps_task_fixed_but_allows_washout_and_dtype():
    model = _model()
    plan = prepare_tuning_data(
        model,
        torch.arange(30).double()[:, None],
        {
            "near": torch.arange(30).double()[:, None] + 1,
            "far": torch.arange(30).double()[:, None] + 3,
        },
        horizon=2,
    )
    changed_washout = _model(washout=4).float()
    assert model_signature(model) == model_signature(changed_washout)
    plan.fit(changed_washout)
    assert changed_washout._history_0.dtype == torch.float32
    assert plan.scales_for(changed_washout)["near"].dtype == torch.float32
    with pytest.raises(ValueError, match="prediction leads"):
        plan.validate_model(_model(leads=(1, 2)))


def test_constant_channel_scoring_scale_is_one_in_original_units():
    plan = prepare_tuning_data(_single(), torch.ones(30, 1).double(), horizon=2)
    torch.testing.assert_close(plan.scales["prediction"], torch.ones(1).double())


@pytest.fixture
def ts():
    return pytest.importorskip("tsdynamics")


def _trajectory(ts, times, values=None, name="position"):
    times = np.asarray(times, dtype=np.float64)
    values = times if values is None else values
    return ts.Trajectory(times, np.asarray(values).reshape(-1, 1), variables=(name,))


@pytest.mark.parametrize("method", ["linear", "pchip", "cubic"])
def test_bare_trajectory_matches_tensor_oracle_and_refit_future_clock(ts, method):
    source = _trajectory(ts, np.arange(40))
    model = _single(resample_method=method)
    plan = prepare_tuning_data(model, source, horizon=3)
    tensor = prepare_tuning_data(
        _single(resample_method=method), torch.arange(40).double()[:, None], horizon=3
    )
    torch.testing.assert_close(plan.train.inputs["state"], tensor.train.inputs["state"])
    torch.testing.assert_close(plan.scales["prediction"], tensor.scales["prediction"])
    plan.fit(model)
    for clocked, ordinal in zip(plan.origins, tensor.origins):
        torch.testing.assert_close(clocked.times, ordinal.times.double())
        torch.testing.assert_close(
            clocked.forecast(model)["prediction"], ordinal.truth["prediction"], atol=1e-10, rtol=0
        )
    plan.fit(model, refit=True)
    result = model.forecast(horizon=3)
    assert result.variables == ("position",)
    np.testing.assert_array_equal(result.t, [40, 41, 42])
    np.testing.assert_allclose(result.y[:, 0], result.t, atol=1e-10, rtol=0)
    # Ordinary Tensor refitting must still remove trajectory output defaults.
    tensor.fit(model, refit=True)
    assert isinstance(model.forecast(horizon=1), torch.Tensor)


@pytest.mark.parametrize("method", ["linear", "pchip", "cubic"])
def test_irregular_raw_sources_are_sliced_before_interpolation(ts, method):
    times = np.arange(0, 40, 0.6)
    original = _trajectory(ts, times, np.sin(times / 4))
    model = _single(resample_dt=1, resample_method=method)
    plan = prepare_tuning_data(model, original, horizon=3)
    changed = original.y.copy()
    changed[original.t >= plan.report["first_validation_time"]] += 1e7
    perturbed = _trajectory(ts, times, changed)
    other = _single(resample_dt=1, resample_method=method)
    second = prepare_tuning_data(other, perturbed, horizon=3)
    torch.testing.assert_close(
        plan.train.inputs["state"], second.train.inputs["state"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        plan.train.targets["prediction"], second.train.targets["prediction"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        plan.scales["prediction"], second.scales["prediction"], rtol=0, atol=0
    )
    plan.fit(model)
    second.fit(other)
    _same_state(model.state_dict(), other.state_dict())


@pytest.mark.parametrize("method", ["linear", "pchip", "cubic"])
def test_named_staggered_clocks_preroll_and_target_suffix_are_safe(ts, method):
    inputs = {
        "forcing": _trajectory(ts, np.arange(-1.75, 41, 2), np.arange(-1.75, 41, 2) + 1, "force"),
        "state": _trajectory(ts, np.arange(41)),
    }
    target_times = np.arange(46)
    targets = {
        "far": _trajectory(ts, target_times, name="far_position"),
        "near": _trajectory(ts, target_times, name="near_position"),
    }
    model = _model(drivers=True, resample_method=method)
    plan = prepare_tuning_data(model, inputs, targets, horizon=3)
    assert plan.origins[0].skip == 1
    assert plan.origins[0].times[0].item() == 32.25
    assert plan.origins[0].horizon == 4
    changed_targets = {
        name: _trajectory(
            ts,
            target_times,
            np.where(target_times > 40, np.nan, target_times),
            name=f"{name}_position",
        )
        for name in targets
    }
    other = _model(drivers=True, resample_method=method)
    changed = prepare_tuning_data(other, inputs, changed_targets, horizon=3)
    for name in model.output_names:
        torch.testing.assert_close(
            plan.train.targets[name], changed.train.targets[name], rtol=0, atol=0
        )
        torch.testing.assert_close(
            plan.origins[0].truth[name], changed.origins[0].truth[name], rtol=0, atol=0
        )
    plan.fit(model)
    for origin in plan.origins:
        actual = origin.forecast(model)
        assert origin.future_inputs["forcing"].shape[0] == origin.horizon - 1
        for name in model.output_names:
            torch.testing.assert_close(origin.truth[name][:, 0], origin.times)
            torch.testing.assert_close(actual[name], origin.truth[name], atol=1e-9, rtol=0)
    plan.fit(model, refit=True)
    result = model.forecast(horizon=1)
    for name in model.output_names:
        np.testing.assert_array_equal(result[name].t, [41])
        assert result[name].variables == (f"{name}_position",)


def test_explicit_trajectory_validation_never_enters_refit_and_must_be_later(ts):
    training = _trajectory(ts, np.arange(20))
    validation = _trajectory(ts, np.arange(20, 30))
    model = _single()
    plan = prepare_tuning_data(model, training, validation=(validation, None), horizon=3)
    assert plan.refit is plan.train
    plan.fit(model, refit=True)
    np.testing.assert_array_equal(model.forecast(horizon=1).t, [20])
    for origin in plan.origins:
        torch.testing.assert_close(
            origin.forecast(model)["prediction"], origin.truth["prediction"], atol=1e-10, rtol=0
        )
    with pytest.raises(ValueError, match="strictly after"):
        prepare_tuning_data(_single(), training, validation=(training, None), horizon=2)


def test_batch_feature_scales_use_training_labels_and_keep_cases_separate():
    x = rd.reservoir_input(2, dtype=torch.float64)
    y = rd.RidgeReadoutLayer(2, 2, alpha=0).double()(x)
    model = rd.Model(inputs={"state": x}, outputs={"prediction": y}, washout=2)
    time = torch.arange(40).double()
    values = torch.stack((time, time * 10), -1).expand(2, -1, -1).clone()
    values[1] += torch.tensor([100.0, -50.0])
    plan = prepare_tuning_data(model, values, horizon=3)
    expected = values[:, 3:32].std((0, 1), correction=0)
    torch.testing.assert_close(plan.scales["prediction"], expected)
    assert plan.origins[0].truth["prediction"].shape == (2, 3, 2)
    assert plan.origins[0].history["state"].shape == (2, 32, 2)


@pytest.mark.parametrize("horizon", [2, 6])
def test_horizon_shorter_than_all_prediction_leads_needs_no_future_drivers(horizon):
    def position(t):
        return t * (t + 1) / 2

    t = torch.arange(60).double()[:, None]
    model = _model(drivers=True, leads=(3, 5))
    plan = prepare_tuning_data(
        model,
        {"state": position(t), "forcing": t + 1},
        {"near": position(t + 3), "far": position(t + 5)},
        horizon=horizon,
    )
    plan.fit(model)
    for origin in plan.origins:
        if horizon == 2:
            assert origin.future_inputs is None
        else:
            assert origin.future_inputs["forcing"].shape == (3, 1)
        for name, prediction in origin.forecast(model).items():
            torch.testing.assert_close(prediction, origin.truth[name], atol=1e-8, rtol=0)


def test_late_trajectory_target_support_retains_absolute_supervision_start(ts):
    inputs = _trajectory(ts, np.arange(40))
    targets = {
        "near": _trajectory(ts, np.arange(8, 46), name="near_position"),
        "far": _trajectory(ts, np.arange(5, 46), name="far_position"),
    }
    model = _model()
    plan = prepare_tuning_data(model, inputs, targets, horizon=3)
    assert plan.train.supervised_start == 7
    assert plan.train.supervised_stop == 29
    plan.fit(model)
    torch.testing.assert_close(model.input_scalers[0]._stat0, torch.tensor([[[17.5]]]).double())
    torch.testing.assert_close(model.output_scalers[0]._stat0, torch.tensor([[[18.5]]]).double())
    torch.testing.assert_close(model.output_scalers[1]._stat0, torch.tensor([[[20.5]]]).double())
    for origin in plan.origins:
        for name, prediction in origin.forecast(model).items():
            torch.testing.assert_close(prediction, origin.truth[name], atol=1e-10, rtol=0)
    plan.fit(model, refit=True)
    assert model._history_0[-1, 0].item() == 39


def test_explicit_trajectory_validation_target_component_order_must_match_training(ts):
    x = rd.reservoir_input(2, dtype=torch.float64)
    y = rd.SVDReadoutLayer(2, 2, alpha=0).double()(x)
    model = rd.Model(inputs={"state": x}, outputs={"prediction": y}, washout=2)
    train_time, val_time = np.arange(20.0), np.arange(20.0, 30.0)
    names = ("position", "velocity")
    train = ts.Trajectory(
        train_time, np.stack((train_time, 2 * train_time), axis=1), variables=names
    )
    validation = ts.Trajectory(
        val_time, np.stack((val_time, 2 * val_time), axis=1), variables=names
    )
    reversed_target = ts.Trajectory(
        val_time, np.stack((2 * val_time, val_time), axis=1), variables=names[::-1]
    )
    with pytest.raises(ValueError, match="target components/order.*prediction"):
        prepare_tuning_data(
            model, train, train, validation=(validation, reversed_target), horizon=2
        )
    plan = prepare_tuning_data(model, train, train, validation=(validation, validation), horizon=2)
    assert plan.refit is plan.train
    first = plan.origins[0]
    torch.testing.assert_close(
        first.truth["prediction"], torch.stack((first.times, 2 * first.times), dim=1)
    )
