"""Resume binds task code and the actual materialized historical observations."""

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from resdag.hpo import run_hpo
from resdag.hpo._provenance import callable_identity, fingerprint, trial_settings


def _factory(size=8):
    raise AssertionError("No numerical model is needed for task identity tests")


def _space(trial):
    return {"size": trial.suggest_int("size", 5, 10)}


def _loss(predicted, truth):
    return float(np.mean((predicted - truth) ** 2))


def _prepared_study(monkeypatch):
    run = importlib.import_module("resdag.hpo.run")
    calls = []

    def execute(study, runner, **kwargs):
        for _ in range(kwargs["n_trials"]):
            trial = study.ask()
            prepared = runner.prepare(trial)
            calls.append(prepared.number)
            study.tell(trial, 1.0)

    monkeypatch.setattr(run, "run_trials", execute)
    return calls


def _loader(path):
    def loader(trial):
        value = float(Path(path).read_text())
        values = torch.arange(30, dtype=torch.float32).reshape(1, -1, 1) * value
        return dict(
            warmup=values[:, :3],
            train=values[:, 3:16],
            target=values[:, 4:17],
            f_warmup=values[:, 16:20],
            val=values[:, 20:25],
        )

    return loader


def test_resume_rejects_changed_file_contents_even_when_callback_path_unchanged(
    tmp_path, monkeypatch
):
    calls = _prepared_study(monkeypatch)
    observations = tmp_path / "observations.txt"
    observations.write_text("1")
    loader = _loader(str(observations))
    storage = str(tmp_path / "study.db")
    options = dict(
        model_creator=_factory,
        search_space=_space,
        data_loader=loader,
        n_trials=1,
        storage=storage,
        study_name="task",
        loss=_loss,
        verbosity=0,
        seed=3,
    )
    first = run_hpo(**options)
    identity = fingerprint(callable_identity(loader))
    second = run_hpo(**options)
    assert len(second.trials) == 2
    assert trial_settings(first.trials[0]).keys() == {"size"}
    observations.write_text("2")
    assert fingerprint(callable_identity(loader)) == identity
    with pytest.raises(ValueError, match="Study data changed"):
        run_hpo(**options)
    assert calls == [0, 1]


def test_resume_changed_objective_rejected_before_preparing_attempt(tmp_path, monkeypatch):
    calls = _prepared_study(monkeypatch)
    path = tmp_path / "data.txt"
    path.write_text("1")
    options = dict(
        model_creator=_factory,
        search_space=_space,
        data_loader=_loader(str(path)),
        n_trials=1,
        storage=str(tmp_path / "study.db"),
        study_name="task",
        verbosity=0,
    )
    run_hpo(**options, loss=_loss)

    def changed(predicted, truth):
        return _loss(predicted, truth) + 1

    with pytest.raises(ValueError, match="protocol differs"):
        run_hpo(**options, loss=changed)
    assert calls == [0]


def test_unreplayable_live_study_callback_requires_explicit_trusted_context(tmp_path, monkeypatch):
    _prepared_study(monkeypatch)
    path = tmp_path / "data.txt"
    path.write_text("1")
    plain = _loader(str(path))

    def loader(trial):
        assert trial.study.study_name == "task"
        return plain(trial)

    options = dict(
        model_creator=_factory,
        search_space=_space,
        data_loader=loader,
        n_trials=1,
        storage=str(tmp_path / "study.db"),
        study_name="task",
        loss=_loss,
        verbosity=0,
    )
    run_hpo(**options)
    with pytest.raises(ValueError, match="trusted task_context"):
        run_hpo(**options)
    # A distinct study explicitly binds its external resources to a trusted version.
    options["study_name"] = "trusted"

    def trusted_loader(trial):
        _ = trial.study.study_name
        return plain(trial)

    options["data_loader"] = trusted_loader
    run_hpo(**options, task_context="dataset-v1")
    with pytest.warns(RuntimeWarning, match="trusting the explicit"):
        result = run_hpo(**options, task_context="dataset-v1")
    assert len(result.trials) == 2


def test_canonical_content_identity_ignores_tensor_storage_and_array_strides():
    values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    assert fingerprint(values) == fingerprint(values.clone())
    assert fingerprint(values) != fingerprint(values.double())
    assert fingerprint(values) != fingerprint(values + 1)
    array = np.arange(12).reshape(3, 4)
    assert fingerprint(array) == fingerprint(array.T.copy().T)
    assert fingerprint(array) != fingerprint(array.astype(np.float64))


class _CustomLayer(torch.nn.Module):
    def forward(self, inputs):
        return inputs + 1


def _uses_custom_layer():
    return _CustomLayer()


def test_global_custom_layer_active_method_change_alters_factory_identity(monkeypatch):
    before = fingerprint(callable_identity(_uses_custom_layer))

    def replacement(self, inputs):
        return inputs + 2

    # Retain the same class/module/qualified name and replace only its active
    # method: class definition text alone would silently reuse the old score.
    monkeypatch.setattr(_CustomLayer, "forward", replacement)
    assert fingerprint(callable_identity(_uses_custom_layer)) != before


def test_code_fallback_identity_does_not_depend_on_machine_path():
    namespace1, namespace2 = {}, {}
    source = "def callback(x):\n    return x * 2\n"
    exec(compile(source, "/machine-one/notebook.py", "exec"), namespace1)
    exec(compile(source, "/other-machine/same-notebook.py", "exec"), namespace2)
    assert callable_identity(namespace1["callback"]) == callable_identity(namespace2["callback"])


def test_callable_factory_instance_active_method_change_alters_identity(monkeypatch):
    class Factory:
        def __call__(self, size=8):
            return size + 1

    factory = Factory()
    before = fingerprint(callable_identity(factory))

    def replacement(self, size=8):
        return size + 2

    monkeypatch.setattr(Factory, "__call__", replacement)
    assert fingerprint(callable_identity(factory)) != before


def test_repeated_score_transfer_replays_original_evaluation_number(tmp_path, monkeypatch):
    _prepared_study(monkeypatch)
    path = tmp_path / "data.txt"
    path.write_text("1")
    plain = _loader(str(path))

    def loader(trial):
        return {key: value * (trial.number + 1) for key, value in plain(trial).items()}

    options = dict(
        model_creator=_factory,
        search_space=_space,
        data_loader=loader,
        n_trials=1,
        storage=str(tmp_path / "study.db"),
        loss=_loss,
        seed=3,
        verbosity=0,
    )
    source = run_hpo(**options, study_name="source")
    run_hpo(**options, study_name="intermediate")
    intermediate = run_hpo(
        **options, study_name="intermediate", transfer_from=source, transfer_mode="scores"
    )
    assert [trial.user_attrs["resdag_raw_evaluation_number"] for trial in intermediate.trials] == [
        0,
        0,
        2,
    ]
    final = run_hpo(
        **options, study_name="final", transfer_from=intermediate, transfer_mode="scores"
    )
    assert len(final.trials) == 4
    assert [trial.user_attrs["resdag_raw_evaluation_number"] for trial in final.trials] == [
        0,
        0,
        2,
        3,
    ]
