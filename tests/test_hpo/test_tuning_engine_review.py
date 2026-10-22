"""Independent real-process and persistent-study tuning acceptance checks."""

import pytest
import torch
from torch import nn

optuna = pytest.importorskip("optuna")
pytest.importorskip("joblib")

import resdag as rd  # noqa: E402
from resdag.hpo.storage import _close_storage  # noqa: E402

_GLOBAL_SCORE_CONFIGURATION = {"value": 7.0}


def _global_score(context):
    return _GLOBAL_SCORE_CONFIGURATION["value"]


def _no_search(trial):
    return {}


@pytest.fixture
def owned_rdb(monkeypatch):
    import resdag.hpo.tuning as tuning

    original = tuning.resolve_storage
    backends = []

    def capture(*args, **kwargs):
        backend = original(*args, **kwargs)
        backends.append(backend)
        return backend

    monkeypatch.setattr(tuning, "resolve_storage", capture)
    try:
        yield backends
    finally:
        for backend in backends:
            _close_storage(backend)


def _affine(width=2, alpha=1e-8):
    x = rd.reservoir_input(1, dtype=torch.float64)
    projection = nn.Linear(1, width, bias=False).double()
    with torch.no_grad():
        projection.weight.copy_(torch.arange(1, width + 1).double()[:, None] / width)
    hidden = projection(x)
    prediction = rd.SVDReadoutLayer(width, 1, alpha=alpha).double()(hidden)
    return rd.Model(inputs={"state": x}, outputs={"prediction": prediction}, washout=2)


def _alpha_space(trial):
    return {"alpha": trial.suggest_float("alpha", 1e-9, 1e-6, log=True)}


def test_real_parallel_notebook_factory_respects_attempt_windows_and_installs_winner():
    data = torch.arange(50).double()[:, None]
    model = _affine()
    captured_scale = 1.0

    def notebook_factory(params, seed):
        width = params["width"]
        x = rd.reservoir_input(1, dtype=torch.float64)
        projection = nn.Linear(1, width, bias=False).double()
        with torch.no_grad():
            projection.weight.copy_(
                torch.arange(1, width + 1).double()[:, None] * captured_scale / width
            )
        y = rd.SVDReadoutLayer(width, 1, alpha=1e-8).double()(projection(x))
        return rd.Model(inputs={"state": x}, outputs={"prediction": y}, washout=2)

    def notebook_space(trial):
        return {"width": trial.suggest_int("width", 2, 4), "fixed_setting": 23}

    returned = model.tune(
        data,
        factory=notebook_factory,
        search_space=notebook_space,
        n_jobs=2,
        n_trials=3,
        seeds=(4, 9),
        n_origins=2,
        horizon=3,
    )
    assert returned is model
    report = model.tuning_report
    assert report.attempts == len(report.study.trials) == 3
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in report.study.trials)
    assert report.score == 3
    assert report.best_params["fixed_setting"] == 23
    assert len(report.runs) == 4
    assert {(run["seed"], run["origin"]) for run in report.runs} == {(4, 0), (4, 1), (9, 0), (9, 1)}
    assert (
        next(
            module for module in model.graph.modules() if isinstance(module, nn.Linear)
        ).out_features
        == report.best_params["width"]
    )
    torch.testing.assert_close(
        model.forecast(horizon=3), torch.arange(50, 53).double()[:, None], rtol=0, atol=1e-6
    )


def test_real_rdb_resume_from_fresh_equivalent_model_and_returned_study_is_usable(
    tmp_path, owned_rdb
):
    data = torch.arange(40).double()[:, None]
    path = str(tmp_path / "resume.db")
    options = dict(
        storage=path,
        study_name="independent-resume",
        search_space=_alpha_space,
        preset="quick",
        n_trials=1,
        horizon=2,
    )
    first = _affine()
    first.tune(data, **options)
    fingerprint = first.tuning_report.protocol_hash
    first.tuning_report.study.set_user_attr("returned_study_usable", True)
    assert first.tuning_report.study.user_attrs["returned_study_usable"]

    fresh = _affine()
    fresh.tune(data, **options)
    assert fresh.tuning_report.protocol_hash == fingerprint
    assert len(fresh.tuning_report.study.trials) == 2
    fresh.tune(data, **options)
    assert fresh.tuning_report.protocol_hash == fingerprint
    assert len(fresh.tuning_report.study.trials) == 3
    assert fresh.tuning_report.attempts == 1
    torch.testing.assert_close(
        fresh.forecast(horizon=2), torch.tensor([[40.0], [41.0]]).double(), rtol=0, atol=1e-5
    )


def test_changed_global_objective_configuration_cannot_reuse_prior_scores(
    tmp_path, monkeypatch, owned_rdb
):
    model = _affine()
    data = torch.arange(40).double()[:, None]
    options = dict(
        storage=str(tmp_path / "global-objective.db"),
        study_name="global-objective",
        objective=_global_score,
        search_space=_no_search,
        preset="quick",
        n_trials=1,
        horizon=2,
    )
    model.tune(data, **options)
    assert model.tuning_report.score == 7
    monkeypatch.setitem(_GLOBAL_SCORE_CONFIGURATION, "value", 1.0)
    with pytest.raises(ValueError, match="protocol differs"):
        model.tune(data, **options)
    assert len(model.tuning_report.study.trials) == 1


def test_inference_mode_tuning_produces_parameters_usable_by_autograd():
    model = _affine()
    data = torch.arange(40).double()[:, None]
    with torch.inference_mode():
        model.tune(data, search_space=_no_search, preset="quick", n_trials=1, horizon=2)
    assert not any(torch.is_inference(parameter) for parameter in model.parameters())
    sample = data[:4].clone().requires_grad_(True)
    model(sample).sum().backward()
    assert sample.grad is not None and torch.isfinite(sample.grad).all()


def test_custom_objective_inplace_work_cannot_mutate_caller_data_or_later_trial_scales():
    model = _affine()
    data = torch.arange(40).double()[:, None]
    original = data.clone()
    observed_scales = []

    def objective(context):
        observed_scales.append(context.scales["prediction"].clone())
        context.truth["prediction"].zero_()
        context.scales["prediction"].fill_(100)
        return 1.0

    model.tune(
        data,
        objective=objective,
        search_space=_no_search,
        preset="quick",
        n_trials=2,
        horizon=2,
    )
    torch.testing.assert_close(data, original, rtol=0, atol=0)
    assert len(observed_scales) == 2
    torch.testing.assert_close(observed_scales[0], observed_scales[1], rtol=0, atol=0)


@pytest.mark.parametrize("catch_exceptions", [True, False])
def test_real_parallel_candidate_failure_leaves_no_running_trials_and_caller_study_usable(
    catch_exceptions,
):
    model = _affine().fit(torch.arange(20).double()[:, None])
    before = model.forecast(horizon=2).clone()
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial({"invalid": True})
    study.enqueue_trial({"invalid": False})

    def factory(params, seed):
        if params["invalid"]:
            raise ValueError("independent parallel invalid candidate")
        return _affine()

    def space(trial):
        return {"invalid": trial.suggest_categorical("invalid", [True, False])}

    options = dict(
        factory=factory,
        search_space=space,
        study=study,
        seeds=(0,),
        n_origins=1,
        horizon=2,
        n_trials=2,
        n_jobs=2,
        catch_exceptions=catch_exceptions,
    )
    data = torch.arange(40).double()[:, None]
    if catch_exceptions:
        model.tune(data, **options)
        assert [trial.state for trial in study.trials] == [
            optuna.trial.TrialState.FAIL,
            optuna.trial.TrialState.COMPLETE,
        ]
        assert model.tuning_report.best_params == {"invalid": False}
    else:
        with pytest.raises(ValueError, match="independent parallel invalid candidate"):
            model.tune(data, **options)
        assert all(trial.state == optuna.trial.TrialState.FAIL for trial in study.trials)
        torch.testing.assert_close(model.forecast(horizon=2), before, rtol=0, atol=0)
        assert not hasattr(model, "tuning_report")
    assert len(study.trials) == 2
    assert all(trial.state != optuna.trial.TrialState.RUNNING for trial in study.trials)
    study.set_user_attr("caller_usable", True)
    assert study.user_attrs["caller_usable"]


def test_failed_install_disposes_retained_sqlite_memory_pool(monkeypatch):
    import sqlalchemy

    import resdag.hpo.tuning as tuning

    storage = optuna.storages.RDBStorage("sqlite:///:memory:")
    closed = []
    sqlalchemy.event.listen(
        storage.engine, "close", lambda connection, record: closed.append(connection)
    )
    monkeypatch.setattr(tuning, "resolve_storage", lambda *args, **kwargs: storage)
    primary = RuntimeError("independent winner installation failure")

    def fail_install(*args):
        raise primary

    monkeypatch.setattr(tuning, "_install", fail_install)
    model = _affine()
    try:
        with pytest.raises(RuntimeError, match="winner installation failure") as caught:
            model.tune(
                torch.arange(40).double()[:, None],
                storage="sqlite:///:memory:",
                search_space=_no_search,
                preset="quick",
                n_trials=1,
                horizon=2,
            )
        assert caught.value is primary
        assert closed
        assert not model.is_fitted
        assert not hasattr(model, "tuning_report")
    finally:
        _close_storage(storage)


def test_inplace_torch_layer_cannot_modify_protocol_inputs_or_training_labels():
    x = rd.reservoir_input(1, dtype=torch.float64)
    y = rd.SVDReadoutLayer(1, 1, alpha=0.1).double()(nn.ReLU(inplace=True)(x))
    model = rd.Model(inputs={"state": x}, outputs={"prediction": y}, preprocessing=None, washout=2)
    data = torch.linspace(-2, 2, 40).double()[:, None]
    original = data.clone()
    model.tune(data, search_space=_no_search, preset="quick", n_trials=2, horizon=2)
    torch.testing.assert_close(data, original, rtol=0, atol=0)
    assert len(model.tuning_report.study.trials) == 2
    assert all(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in model.tuning_report.study.trials
    )

    # Independent augmented least squares checks that in-place feature
    # computation also left the implicit next-sample labels untouched.
    features = original[:-1][2:].clamp_min(0)
    design = torch.cat((features, torch.ones_like(features)), dim=1)
    regularizer = torch.diag(torch.tensor([0.1**0.5, 0.0], dtype=torch.float64))
    augmented = torch.cat((design, regularizer), dim=0)
    labels = torch.cat((original[1:][2:], torch.zeros(2, 1, dtype=torch.float64)), dim=0)
    expected = torch.linalg.lstsq(augmented, labels, driver="gelsd").solution
    head = next(layer for layer in model.graph.modules() if isinstance(layer, rd.SVDReadoutLayer))
    torch.testing.assert_close(head.weight, expected[:1].T, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(head.bias, expected[-1], atol=1e-10, rtol=1e-10)
