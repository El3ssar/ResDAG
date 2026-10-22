"""Public tuning workflows and independent transaction/reproducibility oracles."""

import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

import resdag as rd

optuna = pytest.importorskip("optuna")


def _affine(*, cls=rd.Model, washout=2):
    x = rd.reservoir_input(1, dtype=torch.float64)
    y = rd.SVDReadoutLayer(1, 1, alpha=0).double()(x)
    return cls(inputs={"state": x}, outputs={"prediction": y}, washout=washout)


def _reservoir(width=4, seed=0):
    x = rd.reservoir_input(1, dtype=torch.float64)
    hidden = rd.ESNLayer(width, feedback_size=1, dtype=torch.float64, seed=seed)(x)
    y = rd.SVDReadoutLayer(width, 1, alpha=1e-5).double()(hidden)
    return rd.Model(inputs={"state": x}, outputs={"prediction": y}, washout=2)


def _no_search(trial):
    return {}


def _alpha_search(trial):
    return {"alpha": trial.suggest_float("alpha", 1e-8, 1e-3, log=True)}


def _rng_state():
    python = random.getstate()
    numpy = np.random.get_state()
    return python, (numpy[0], numpy[1].copy(), *numpy[2:]), torch.random.get_rng_state().clone()


def _assert_rng_equal(left, right):
    assert left[0] == right[0]
    assert left[1][0] == right[1][0]
    np.testing.assert_array_equal(left[1][1], right[1][1])
    assert left[1][2:] == right[1][2:]
    torch.testing.assert_close(left[2], right[2], rtol=0, atol=0)


def _same_state(left, right):
    assert left.keys() == right.keys()
    for name, value in left.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, right[name], rtol=0, atol=0)
        else:
            assert value == right[name]


def test_public_tune_refits_all_development_data_and_starts_after_complete_tail():
    data = torch.arange(40).double()[:, None]
    model = _affine()
    identity = id(model)
    returned = model.tune(data, preset="quick", n_trials=1, search_space=_no_search, horizon=3)
    assert returned is model and id(returned) == identity
    assert model.is_fitted
    assert model.tuning_report.attempts == 1
    assert model.tuning_report.best_params == {}
    assert model.tuning_report.score == 3
    assert model.tuning_report.validation["refit_includes_validation"]
    assert model._history_0[-1, 0].item() == 39
    # Final fit origins are 2..38, after validation selected the configuration.
    torch.testing.assert_close(model.input_scalers[0]._stat0, torch.tensor([[[20.0]]]).double())
    torch.testing.assert_close(
        model.forecast(horizon=3), torch.arange(40, 43).double()[:, None], atol=1e-10, rtol=0
    )


def test_explicit_validation_is_never_refitted_and_reported_as_held_out():
    data = torch.arange(30).double()[:, None]
    validation = torch.arange(30, 40).double()[:, None]
    model = _affine()
    model.tune(
        data,
        validation=(validation, None),
        preset="quick",
        n_trials=1,
        search_space=_no_search,
        horizon=3,
    )
    assert model._history_0[-1, 0].item() == 29
    assert not model.tuning_report.validation["refit_includes_validation"]
    assert model.tuning_report.validation["first_validation_index"] == 30
    torch.testing.assert_close(
        model.forecast(horizon=2), torch.tensor([[30.0], [31.0]]).double(), atol=1e-10, rtol=0
    )


def test_structural_recipe_reuses_seed_bank_origins_and_scales_for_every_configuration():
    source = torch.sin(torch.arange(50).double() / 4)[:, None]
    candidates, evaluations = [], []

    def factory(params, seed):
        draw = torch.rand(3)
        candidates.append((params["width"], seed, draw.clone()))
        return _reservoir(params["width"], seed)

    def space(trial):
        return {"width": trial.suggest_int("width", 3, 5), "fixed_setting": 17}

    def objective(context):
        evaluations.append(
            (
                context.seed,
                context.origin,
                context.truth["prediction"].clone(),
                context.scales["prediction"].clone(),
            )
        )
        return -float(
            (context.predictions["prediction"] - context.truth["prediction"]).square().mean()
        )

    study = optuna.create_study(direction="maximize")
    study.enqueue_trial({"width": 3})
    study.enqueue_trial({"width": 5})
    model = _reservoir()
    model.tune(
        source,
        factory=factory,
        search_space=space,
        objective=objective,
        seeds=(7, 11),
        n_origins=2,
        n_trials=2,
        horizon=3,
        study=study,
    )
    assert [(width, seed) for width, seed, _ in candidates[:4]] == [
        (3, 7),
        (3, 11),
        (5, 7),
        (5, 11),
    ]
    assert candidates[-1][:2] == (model.tuning_report.best_params["width"], 7)
    assert model.tuning_report.best_params["fixed_setting"] == 17
    assert model.tuning_report.seeds == (7, 11)
    assert len(evaluations) == 8
    torch.testing.assert_close(candidates[0][2], candidates[2][2], rtol=0, atol=0)
    torch.testing.assert_close(candidates[1][2], candidates[3][2], rtol=0, atol=0)
    assert not torch.equal(candidates[0][2], candidates[1][2])
    for i, (seed, origin, truth, scale) in enumerate(evaluations):
        assert (seed, origin) == ((7, 0), (7, 1), (11, 0), (11, 1))[i % 4]
        torch.testing.assert_close(truth, evaluations[i % 2][2], rtol=0, atol=0)
        torch.testing.assert_close(scale, evaluations[0][3], rtol=0, atol=0)
    width = next(
        layer for layer in model.graph.modules() if isinstance(layer, rd.ESNLayer)
    ).cell.reservoir_size
    assert width == model.tuning_report.best_params["width"]
    assert model._history_0[-1, 0].item() == source[-1, 0].item()
    assert len(model.tuning_report.runs) == 4
    assert study.best_trial.value == model.tuning_report.score
    study.set_user_attr("caller_still_owns_study", True)
    assert study.user_attrs["caller_still_owns_study"]


def test_original_outer_hooks_identity_and_training_mode_survive_tuning():
    model = _affine().eval()
    calls = []
    hook = model.register_forward_hook(lambda module, args, result: calls.append(module))
    original_hooks = tuple(model._forward_hooks.values())
    try:
        assert (
            model.tune(
                torch.arange(30).double()[:, None],
                preset="quick",
                n_trials=1,
                search_space=_no_search,
                horizon=2,
            )
            is model
        )
        assert not model.training
        assert tuple(model._forward_hooks.values()) == original_hooks
        prediction = model(torch.tensor([[30.0]]).double())
        assert calls[-1] is model
        torch.testing.assert_close(prediction, torch.tensor([[31.0]]).double(), atol=1e-10, rtol=0)
    finally:
        hook.remove()
    assert not model._forward_hooks


class _RejectFinalFit(rd.Model):
    def _fit_prepared(self, training, targets, **kwargs):
        value = next(iter(training.values()))
        if value.shape[-2] == 40:
            raise RuntimeError("independent final-refit failure")
        return super()._fit_prepared(training, targets, **kwargs)


def test_final_refit_failure_preserves_existing_model_parameters_hooks_cache_and_rng():
    data = torch.arange(40).double()[:, None]
    model = _affine().fit(data[:20]).eval()
    before = copy.deepcopy(model.state_dict())
    parameter_ids = [id(value) for value in model.parameters()]
    hook = model.register_forward_hook(lambda *args: None)
    original_hooks = tuple(model._forward_hooks.values())
    study = optuna.create_study(direction="maximize")

    def factory(params, seed):
        random.random()
        np.random.random(2)
        torch.rand(2)
        return _affine(cls=_RejectFinalFit)

    rng = _rng_state()
    try:
        with pytest.raises(RuntimeError, match="final-refit failure"):
            model.tune(
                data,
                factory=factory,
                search_space=_no_search,
                seeds=(3,),
                n_origins=1,
                n_trials=1,
                horizon=2,
                study=study,
            )
        _same_state(model.state_dict(), before)
        assert parameter_ids == [id(value) for value in model.parameters()]
        assert tuple(model._forward_hooks.values()) == original_hooks
        assert not model.training and not hasattr(model, "tuning_report")
        assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
        _assert_rng_equal(_rng_state(), rng)
    finally:
        hook.remove()


def test_success_restores_all_caller_rng_streams_even_when_recipe_and_objective_draw():
    def factory(params, seed):
        random.random()
        np.random.random(2)
        torch.rand(2)
        return _affine()

    def objective(context):
        random.random()
        np.random.random(2)
        torch.rand(2)
        return 1.0

    model = _affine()
    rng = _rng_state()
    model.tune(
        torch.arange(30).double()[:, None],
        factory=factory,
        objective=objective,
        search_space=_no_search,
        seeds=(2, 9),
        n_origins=2,
        n_trials=2,
        horizon=2,
    )
    _assert_rng_equal(_rng_state(), rng)


def test_failed_configurations_consume_attempt_budget_without_creating_extra_trials():
    def factory(params, seed):
        if params["fail"]:
            raise ValueError("deliberately invalid architecture")
        return _affine()

    def space(trial):
        return {"fail": trial.suggest_categorical("fail", [True, False])}

    study = optuna.create_study(direction="maximize")
    study.enqueue_trial({"fail": True})
    study.enqueue_trial({"fail": False})
    model = _affine()
    model.tune(
        torch.arange(30).double()[:, None],
        factory=factory,
        search_space=space,
        study=study,
        seeds=(0,),
        n_origins=1,
        n_trials=2,
        horizon=2,
    )
    assert len(study.trials) == model.tuning_report.attempts == 2
    assert [trial.state for trial in study.trials] == [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.COMPLETE,
    ]
    assert model.tuning_report.best_params == {"fail": False}
    assert "deliberately invalid architecture" in study.trials[0].user_attrs["error"]


def test_existing_study_resumes_on_same_now_fitted_public_model():
    model = _affine()
    data = torch.arange(30).double()[:, None]
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=7))
    options = dict(search_space=_alpha_search, study=study, preset="quick", n_trials=1, horizon=2)
    model.tune(data, **options)
    first_hash = model.tuning_report.protocol_hash
    model.tune(data, **options)
    assert len(study.trials) == 2
    assert model.tuning_report.protocol_hash == first_hash
    assert model.tuning_report.attempts == 1
    before = len(study.trials)
    with pytest.raises(ValueError, match="protocol differs"):
        model.tune(data + 1, **options)
    assert len(study.trials) == before


def test_differentiable_carried_states_do_not_prevent_tuning():
    model = _reservoir()
    data = torch.sin(torch.arange(40).double() / 4)[:, None]
    model.fit(data)
    reservoirs = [layer for layer in model.graph.modules() if isinstance(layer, rd.ESNLayer)]
    for reservoir in reservoirs:
        reservoir.detach_state_between_calls = False
    model(data[:4].clone().requires_grad_(True))
    assert any(layer.state is not None and layer.state.grad_fn is not None for layer in reservoirs)
    model.tune(data, preset="quick", n_trials=1, search_space=_no_search, horizon=2)
    assert model.is_fitted
    assert torch.isfinite(model.forecast(horizon=2)).all()


def test_lazy_esn_tune_builds_a_real_pytorch_module_ready_for_a_larger_pipeline():
    model = rd.ESN(reservoir_size=8, washout=2, readout="svd", dtype=torch.float64)
    assert not model.is_built
    data = torch.sin(torch.arange(40).double() / 4)[:, None]
    assert model.tune(data, preset="quick", n_trials=1, search_space=_no_search, horizon=2) is model
    assert model.is_built and model.is_fitted
    assert model.workflow._history_0[-1, 0].item() == data[-1, 0].item()
    pipeline = nn.Sequential(model, nn.Linear(1, 2).double())
    sample = data[:4].clone().requires_grad_(True)
    prediction = pipeline(sample)
    assert prediction.shape == (4, 2)
    prediction.square().sum().backward()
    assert sample.grad is not None and torch.isfinite(sample.grad).all()


def test_public_named_trajectory_tuning_keeps_units_component_names_and_future_dates():
    ts = pytest.importorskip("tsdynamics")
    t = np.arange(40, dtype=np.float64)
    source = ts.Trajectory(t, t[:, None], variables=("position",))
    x = rd.reservoir_input(1, dtype=torch.float64)
    near = rd.SVDReadoutLayer(1, 1, alpha=0).double()(x)
    far = rd.SVDReadoutLayer(1, 1, alpha=0).double()(x)
    model = rd.Model(
        inputs={"state": x},
        outputs={"near": near, "far": far},
        washout=2,
        prediction_lead={"near": 1, "far": 3},
        feedback=[rd.Feedback("far", "state", delay=3)],
    )
    target_times = np.arange(45, dtype=np.float64)
    targets = {
        name: ts.Trajectory(
            target_times,
            np.where(target_times <= 39, target_times, np.nan)[:, None],
            variables=(f"{name}_position",),
        )
        for name in ("far", "near")
    }
    model.tune(
        {"state": source}, targets, preset="quick", n_trials=1, search_space=_no_search, horizon=3
    )
    assert model.tuning_report.validation["refit_discarded_target_rows"] == {"near": 3, "far": 3}
    for name, forecast in model.forecast(horizon=3).items():
        assert forecast.variables == (f"{name}_position",)
        np.testing.assert_array_equal(forecast.t, [40, 41, 42])
        np.testing.assert_allclose(forecast.y[:, 0], forecast.t, atol=1e-10, rtol=0)


def _gain_objective(context, *, gain):
    return gain


class _ConfiguredObjective:
    def __init__(self, gain):
        self.gain = gain

    def __call__(self, context):
        return self.gain


@pytest.mark.parametrize("kind", ["partial", "instance"])
def test_changed_callable_configuration_cannot_reuse_old_objective_scores(kind):
    from functools import partial

    objective = (
        partial(_gain_objective, gain=1.0) if kind == "partial" else _ConfiguredObjective(1.0)
    )
    changed = partial(_gain_objective, gain=7.0) if kind == "partial" else _ConfiguredObjective(7.0)
    model = _affine()
    data = torch.arange(30).double()[:, None]
    study = optuna.create_study(direction="maximize")
    options = dict(preset="quick", n_trials=1, search_space=_no_search, horizon=2, study=study)
    model.tune(data, objective=objective, **options)
    with pytest.raises(ValueError, match="protocol differs"):
        model.tune(data, objective=changed, **options)
    assert len(study.trials) == 1


def test_search_callback_rng_draws_are_isolated_from_callers():
    def space(trial):
        random.random()
        np.random.random(2)
        torch.rand(2)
        return {}

    model = _affine()
    before = _rng_state()
    model.tune(
        torch.arange(30).double()[:, None],
        preset="quick",
        n_trials=1,
        search_space=space,
        horizon=2,
    )
    _assert_rng_equal(_rng_state(), before)


def test_presets_change_repetition_count_without_changing_the_evaluation_horizon():
    data = torch.arange(50).double()[:, None]
    summaries = []
    for preset in ("quick", "balanced", "thorough"):
        model = _affine()
        model.tune(data, preset=preset, n_trials=1, search_space=_no_search)
        report = model.tuning_report
        summaries.append((report.validation["horizon"], len(report.validation["origins"])))
        assert report.seeds == (0,)  # fixed DAG weights are not fake seed replicates
    assert summaries == [(5, 1), (5, 3), (5, 5)]


class _ConfiguredPower(nn.Module):
    supports_framewise = True

    def __init__(self, power):
        super().__init__()
        self.power = power

    def forward(self, inputs):
        return inputs.pow(self.power)


def test_plain_custom_layer_configuration_changes_cannot_silently_resume_old_model():
    x = rd.reservoir_input(1, dtype=torch.float64)
    y = rd.SVDReadoutLayer(1, 1, alpha=0).double()(_ConfiguredPower(1)(x))
    model = rd.Model(inputs={"state": x}, outputs={"prediction": y}, washout=2)
    data = torch.arange(1, 31).double()[:, None] / 30
    study = optuna.create_study(direction="maximize")
    options = dict(preset="quick", n_trials=1, search_space=_no_search, horizon=2, study=study)
    model.tune(data, **options)
    layer = next(layer for layer in model.graph.modules() if isinstance(layer, _ConfiguredPower))
    layer.power = 2
    with pytest.raises(ValueError, match="model changed|protocol differs"):
        model.tune(data, **options)
    assert len(study.trials) == 1
    assert layer.power == 2


@pytest.mark.parametrize("custom_recipe", [False, True])
def test_tuning_inside_inference_mode_produces_an_ordinary_differentiable_model(custom_recipe):
    model = _affine()
    data = torch.arange(30).double()[:, None]

    def factory(params, seed):
        return _affine()

    options = {"factory": factory} if custom_recipe else {}
    with torch.inference_mode():
        assert (
            model.tune(
                data, preset="quick", n_trials=1, search_space=_no_search, horizon=2, **options
            )
            is model
        )
    assert all(not value.is_inference() for value in model.parameters())
    assert all(not value.is_inference() for value in model.buffers())
    sample = torch.tensor([[30.0]]).double().requires_grad_(True)
    model(sample).sum().backward()
    torch.testing.assert_close(sample.grad, torch.ones_like(sample), atol=1e-10, rtol=0)
