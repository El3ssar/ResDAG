"""Classic feature policies survive automatic rebuilding and winner installation."""

from pathlib import Path

import pytest
import torch

import resdag as rd
from resdag.models._recipes import builtin_recipe

optuna = pytest.importorskip("optuna")


def _series():
    time = torch.linspace(0, 6, 54, dtype=torch.float64)
    return torch.stack((1 + time.sin(), -2 + 0.4 * time.cos()), -1)


def _model(kind, input_to_readout):
    options = dict(input_to_readout=input_to_readout, readout="svd", seed=17, dtype=torch.float64)
    if kind == "facade":
        return rd.ESN(8, input_size=2, washout=2, **options)
    return rd.Model(graph=rd.classic_esn(8, 2, 2, **options), washout=2)


def _head(model):
    return next(module for module in model.graph.modules() if isinstance(module, rd.ReadoutLayer))


def _memory(model):
    return next(module for module in model.graph.modules() if isinstance(module, rd.ESNLayer))


@pytest.mark.parametrize("kind", ["classic", "facade"])
@pytest.mark.parametrize("input_to_readout", [False, True])
def test_rebuilding_switches_features_without_changing_seeded_reservoir(kind, input_to_readout):
    from resdag.hpo.tuning import _preset_recipe

    template = _model(kind, not input_to_readout)
    rebuilt = _preset_recipe(template)({"input_to_readout": input_to_readout}, 17)
    assert _head(rebuilt).in_features == 8 + (2 if input_to_readout else 0)
    assert builtin_recipe(rebuilt.graph)["options"]["input_to_readout"] is input_to_readout
    for name, value in _memory(template).state_dict().items():
        assert torch.equal(value, _memory(rebuilt).state_dict()[name])
    if kind == "facade":
        assert rebuilt._factory_options["input_to_readout"] is input_to_readout


@pytest.mark.parametrize("kind", ["classic", "facade"])
@pytest.mark.parametrize("input_to_readout", [False, True])
def test_default_search_winner_artifact_and_retune_keep_selected_feature_policy(
    kind, input_to_readout, tmp_path
):
    # A queued unit trial exercises the real default space without selecting on
    # forecasting accuracy. Both feature choices use the same fixed controls.
    params = dict(
        reservoir_size=8,
        spectral_radius=0.4,
        leak_rate=0.7,
        alpha=1e-4,
        input_scaling=0.2,
        bias_scaling=0.1,
        mean_degree=3,
        input_to_readout=input_to_readout,
    )
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())
    study.enqueue_trial(params)
    model = _model(kind, not input_to_readout).eval()
    identity = id(model)
    model.tune(_series(), n_trials=1, seeds=(17,), n_origins=1, horizon=3, study=study)
    assert id(model) == identity and model.is_fitted and not model.training
    assert model.tuning_report.best_params["input_to_readout"] is input_to_readout
    assert study.best_trial.distributions["input_to_readout"].choices == (True, False)
    assert _head(model).in_features == 8 + (2 if input_to_readout else 0)
    assert builtin_recipe(model.graph)["options"]["input_to_readout"] is input_to_readout
    if kind == "facade":
        assert model._factory_options["input_to_readout"] is input_to_readout
        selected = dict(model._factory_options)
        selected["alpha"] = selected.pop("readout_alpha")
        checkpoint_clone = rd.ESN(washout=2, dtype=torch.float64, **selected)
        checkpoint_clone.load_state_dict(model.state_dict())
        assert _head(checkpoint_clone).in_features == _head(model).in_features
    expected = model.forecast(history=_series()[-9:], horizon=3)
    path = tmp_path / "winner.pt"
    model.save(path)
    restored = rd.load(path)
    assert builtin_recipe(restored.graph)["options"]["input_to_readout"] is input_to_readout
    assert torch.equal(restored.forecast(history=_series()[-9:], horizon=3), expected)
    # A caller-provided space replaces the default and does not resample the
    # feature policy. The installed winner and saved recipe remain coherent.
    restored.tune(
        _series(),
        n_trials=1,
        seeds=(19,),
        n_origins=1,
        horizon=3,
        search_space=lambda trial: {"alpha": trial.suggest_float("alpha", 1e-4, 1e-4)},
    )
    assert restored.tuning_report.best_params == {"alpha": 1e-4}
    assert _head(restored).in_features == 8 + (2 if input_to_readout else 0)
    assert builtin_recipe(restored.graph)["options"]["input_to_readout"] is input_to_readout


def test_custom_factory_owns_feature_policy_and_default_search_stays_replaceable():
    calls = []

    def factory(params, seed):
        calls.append((dict(params), seed))
        return rd.Model(
            graph=rd.classic_esn(
                8,
                2,
                2,
                input_to_readout=False,
                readout="svd",
                readout_alpha=params["alpha"],
                seed=seed,
                dtype=torch.float64,
            ),
            washout=2,
        )

    model = _model("classic", True)
    model.tune(
        _series(),
        factory=factory,
        search_space=lambda trial: {"alpha": trial.suggest_float("alpha", 1e-4, 1e-4)},
        n_trials=1,
        seeds=(17,),
        n_origins=1,
        horizon=3,
    )
    assert calls and all(params == {"alpha": 1e-4} for params, _ in calls)
    assert _head(model).in_features == 8


@pytest.mark.parametrize("kind", ["classic", "facade"])
def test_edited_graph_cannot_be_replaced_by_automatic_feature_switch(kind):
    from resdag.hpo.tuning import _automatic_options, _default_space, _fixed_recipe

    model = _model(kind, True)
    with torch.no_grad():
        _memory(model).cell.weight_feedback.add_(0.01)
    assert builtin_recipe(model.graph) is None and _automatic_options(model) is None
    sampled = _default_space(model)(optuna.trial.FixedTrial({"alpha": 1e-4}))
    assert sampled == {"alpha": 1e-4}
    with pytest.raises(ValueError, match="Supply factory"):
        _fixed_recipe(model)({"input_to_readout": False}, 17)


@pytest.mark.parametrize("kind", ["classic", "facade"])
def test_pre_option_recipe_remains_eligible_for_both_feature_choices(kind):
    from resdag.hpo.tuning import _default_space, _preset_recipe

    directory = Path(__file__).parents[1] / "fixtures" / "input_to_readout_ca40fe4b"
    restored = rd.load(directory / f"{kind}.pt")
    trial = optuna.trial.FixedTrial(
        dict(
            reservoir_size=8,
            spectral_radius=0.4,
            leak_rate=0.7,
            alpha=1e-4,
            input_scaling=0.2,
            bias_scaling=0.1,
            mean_degree=3,
            input_to_readout=False,
        )
    )
    assert _default_space(restored)(trial)["input_to_readout"] is False
    assert trial.distributions["input_to_readout"].choices == (True, False)
    rebuilt = _preset_recipe(restored)({"input_to_readout": False}, 17)
    assert _head(rebuilt).in_features == 5
    assert builtin_recipe(rebuilt.graph)["options"]["input_to_readout"] is False
