"""Study aliases must not turn transfer into self-import."""

import optuna
import pytest
from optuna.distributions import IntDistribution

from resdag.hpo._provenance import PROTOCOL_KEY
from resdag.hpo.storage import _close_storage
from resdag.hpo.transfer import transfer_trials


@pytest.fixture(params=["sqlite", "journal", "memory"])
def aliases(request, tmp_path):
    storages = []
    if request.param == "sqlite":
        url = f"sqlite:///{tmp_path / 'study.db'}"
        storage = optuna.storages.RDBStorage(url)
        alias_storage = optuna.storages.RDBStorage(url)
    elif request.param == "journal":
        from resdag.hpo.storage import resolve_storage

        storage = resolve_storage(str(tmp_path / "study.log"), n_workers=1)
        alias_storage = resolve_storage(str(tmp_path / "study.log"), n_workers=1)
    else:
        storage = optuna.storages.InMemoryStorage()
        alias_storage = storage
    storages.extend([storage, alias_storage])
    original = optuna.create_study(storage=storage, study_name="same-name")
    original.set_user_attr(PROTOCOL_KEY, "same-task")
    original.add_trial(
        optuna.trial.create_trial(
            value=1.0, params={"size": 2}, distributions={"size": IntDistribution(1, 3)}
        )
    )
    alias = optuna.load_study(storage=alias_storage, study_name=original.study_name)
    try:
        yield original, alias
    finally:
        for backend in storages:
            _close_storage(backend)


@pytest.mark.parametrize("mode", ["reevaluate", "scores"])
def test_separate_handles_to_same_study_cannot_import_their_own_trials(aliases, mode):
    destination, source = aliases
    before = destination.trials
    assert destination is not source
    with pytest.raises(ValueError, match="same study"):
        transfer_trials(destination, source, mode=mode, protocol_hash="same-task")
    assert destination.trials == source.trials == before


def test_distinct_studies_with_the_same_name_can_reuse_settings():
    source = optuna.create_study(study_name="same-name")
    target = optuna.create_study(study_name="same-name")
    source.add_trial(
        optuna.trial.create_trial(
            value=1.0, params={"size": 2}, distributions={"size": IntDistribution(1, 3)}
        )
    )
    assert transfer_trials(target, source) == 1
    assert len(target.trials) == 1
    assert target.trials[0].state == optuna.trial.TrialState.WAITING


@pytest.mark.parametrize("mode", ["reevaluate", "scores"])
def test_direct_same_object_is_rejected_even_without_completed_trials(mode):
    study = optuna.create_study()
    study.set_user_attr(PROTOCOL_KEY, "same-task")
    with pytest.raises(ValueError, match="same study"):
        transfer_trials(study, study, mode=mode, protocol_hash="same-task")
    assert study.trials == []


@pytest.mark.parametrize("owned_destination", [False, True])
def test_common_tuning_rejects_loaded_or_implicitly_reopened_destination_alias(
    tmp_path, owned_destination
):
    import torch

    import resdag as rd

    def build():
        x = rd.reservoir_input(1, dtype=torch.float64)
        y = rd.SVDReadoutLayer(1, 1, alpha=0).double()(x)
        return rd.Model(inputs={"state": x}, outputs={"next": y}, washout=0)

    def space(trial):
        return {}

    database = str(tmp_path / "shared.db")
    data = torch.arange(30, dtype=torch.float64)[:, None]
    options = dict(preset="quick", horizon=2, n_trials=1, search_space=space)
    first = build().tune(data, storage=database, study_name="same", **options)
    source = first.tuning_report.study
    alias_storage = optuna.storages.RDBStorage(f"sqlite:///{database}")
    alias = optuna.load_study(study_name="same", storage=alias_storage)
    destination = (
        {"storage": database, "study_name": "same"} if owned_destination else {"study": alias}
    )
    try:
        with pytest.raises(ValueError, match="same study"):
            build().tune(data, warm_start=source, **destination, **options)
        assert len(source.trials) == len(alias.trials) == 1
    finally:
        _close_storage(alias_storage)


@pytest.mark.parametrize("mode", ["reevaluate", "scores"])
def test_standard_optuna_copy_study_remains_a_distinct_transfer_target(aliases, mode):
    source, _ = aliases
    # Seed a previous transfer marker, which Optuna then copies verbatim.
    assert transfer_trials(source, optuna.create_study()) == 0
    storage = optuna.storages.InMemoryStorage()
    optuna.copy_study(
        from_study_name=source.study_name,
        from_storage=source._storage,
        to_storage=storage,
        to_study_name="copied",
    )
    copied = optuna.load_study(study_name="copied", storage=storage)
    assert copied.user_attrs == source.user_attrs
    assert transfer_trials(copied, source, mode=mode, protocol_hash="same-task") == 1
    assert len(copied.trials) == 2


def test_interfering_identity_check_fails_without_copying_trials(monkeypatch):
    destination, source = optuna.create_study(), optuna.create_study()
    reads = 0
    original = optuna.Study.user_attrs.fget

    def attrs(study):
        nonlocal reads
        if study is destination:
            reads += 1
            return {"resdag_transfer_identity_check_v1": "interfering-check"}
        return original(study)

    monkeypatch.setattr(optuna.Study, "user_attrs", property(attrs))
    with pytest.raises(RuntimeError, match="Concurrent study identity check"):
        transfer_trials(destination, source)
    assert reads == 1
    assert destination.trials == source.trials == []
