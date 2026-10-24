"""Independent public-lifecycle checks for automatic classical-model tuning.

These verify reconstruction/customization contracts, not forecasting skill.
The recorded phase-five accuracy studies used explicit user factories.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from torch import nn

import resdag as rd

optuna = pytest.importorskip("optuna")


def _series(*, dtype=torch.float64, device="cpu"):
    time = torch.linspace(0.0, 8.0, 80, dtype=dtype, device=device)
    return torch.stack((1.0 + time.sin(), -2.0 + 0.4 * time.cos()), -1)


def _built(kind="classic", *, width=12, dtype=torch.float64, **kwargs):
    options = dict(seed=17, readout="svd", dtype=dtype, **kwargs)
    if kind == "facade":
        return rd.ESN(width, input_size=2, washout=3, **options)
    factory = rd.classic_esn if kind == "classic" else rd.ott_esn
    return rd.Model(graph=factory(width, 2, 2, **options), washout=3)


def _memory(model):
    return next(module for module in model.graph.modules() if isinstance(module, rd.ESNLayer))


def _head(model):
    return next(module for module in model.graph.modules() if isinstance(module, rd.ReadoutLayer))


def _small_space(trial):
    return {"reservoir_size": 16, "spectral_radius": 0.4, "leak_rate": 0.7, "alpha": 1e-4}


def _alpha_space(trial):
    return {"alpha": 1e-4}


@pytest.mark.parametrize("kind", ["classic", "ott", "facade"])
def test_small_real_automatic_tune_refit_forecast_roundtrip_and_retune(kind, tmp_path):
    data = _series()
    model = _built(kind, dtype=torch.float32).double().eval()
    outer = id(model)
    model.tune(
        data,
        preset="quick",
        n_trials=1,
        seeds=(7, 19),
        n_origins=2,
        search_space=_small_space,
        horizon=4,
    )
    assert id(model) == outer and model.is_fitted and not model.training
    assert _memory(model).reservoir_size == 16
    assert _head(model).in_features == 18
    assert _memory(model).seed == 7
    assert len(model.tuning_report.runs) == 4
    assert {(run["seed"], run["origin"]) for run in model.tuning_report.runs} == {
        (7, 0),
        (7, 1),
        (19, 0),
        (19, 1),
    }
    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())
    assert model.tuning_report.best_params == _small_space(None)
    expected = model.forecast(history=data[-20:], horizon=4)
    assert expected.shape == (4, 2) and torch.isfinite(expected).all()
    checkpoint = tmp_path / f"{kind}.pt"
    model.save(checkpoint)
    restored = rd.load(checkpoint, map_location="cpu")
    torch.testing.assert_close(
        restored.forecast(history=data[-20:], horizon=4), expected, rtol=0, atol=0
    )
    # Eligibility must survive fitting, dtype conversion and portable reconstruction.
    restored.tune(
        data,
        preset="quick",
        n_trials=1,
        seeds=(11, 23),
        n_origins=1,
        search_space=lambda trial: {"reservoir_size": 18, "alpha": 1e-3},
        horizon=3,
    )
    assert _memory(restored).reservoir_size == 18
    assert len(restored.tuning_report.runs) == 2
    assert torch.isfinite(restored.forecast(horizon=3)).all()


@pytest.mark.parametrize("kind", ["classic", "ott", "facade"])
def test_default_search_distributions_match_approved_controls_and_uniform_width(kind):
    from resdag.hpo.tuning import _default_space

    model = _built(kind, width=16)
    trial = optuna.trial.FixedTrial(
        {
            "reservoir_size": 8,
            "spectral_radius": 0.05,
            "leak_rate": 0.05,
            "alpha": 1e-12,
            "input_scaling": 1e-3,
            "bias_scaling": 1e-3,
            "mean_degree": 3,
            **({"input_to_readout": False} if kind != "ott" else {}),
        }
    )
    values = _default_space(model)(trial)
    expected = {
        "reservoir_size",
        "spectral_radius",
        "leak_rate",
        "alpha",
        "input_scaling",
        "bias_scaling",
        "mean_degree",
    }
    if kind != "ott":
        expected.add("input_to_readout")
        assert values["input_to_readout"] is False
        assert trial.distributions["input_to_readout"].choices == (True, False)
    assert set(values) == expected
    distributions = trial.distributions
    assert isinstance(distributions["reservoir_size"], optuna.distributions.IntDistribution)
    assert (distributions["reservoir_size"].low, distributions["reservoir_size"].high) == (8, 32)
    for name, low, high, logarithmic in (
        ("spectral_radius", 0.05, 1.5, False),
        ("leak_rate", 0.05, 1.0, False),
        ("alpha", 1e-12, 1e-2, True),
        ("input_scaling", 1e-3, 2.0, True),
        ("bias_scaling", 1e-3, 1.0, True),
    ):
        distribution = distributions[name]
        assert (distribution.low, distribution.high, distribution.log) == (low, high, logarithmic)
    # Mean degree is expected row entries, including the possible self-loop.
    assert distributions["mean_degree"].low == 3
    assert distributions["mean_degree"].high == min(12, values["reservoir_size"])


@pytest.mark.parametrize("kind", ["classic", "ott"])
@pytest.mark.parametrize("width", [8, 32, 64])
def test_lowest_default_density_is_viable_for_small_reservoirs_and_multiple_seeds(kind, width):
    model = _built(kind, width=width)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())
    study.enqueue_trial(
        {
            "reservoir_size": max(8, width // 2),
            "spectral_radius": 0.05,
            "leak_rate": 0.05,
            "alpha": 1e-12,
            "input_scaling": 1e-3,
            "bias_scaling": 1e-3,
            "mean_degree": 3,
        }
    )
    model.tune(
        _series(),
        preset="quick",
        n_trials=1,
        seeds=tuple(range(16)),
        n_origins=1,
        horizon=2,
        study=study,
        catch_exceptions=False,
    )
    assert study.best_trial.state == optuna.trial.TrialState.COMPLETE
    assert len(model.tuning_report.runs) == 16
    assert all(run["complete"] for run in model.tuning_report.runs)
    assert _memory(model).weight_hh.count_nonzero() > 0
    assert torch.isfinite(model.forecast(horizon=3)).all()


class _Units(nn.Module):
    supports_framewise = True

    def __init__(self, gain):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(float(gain), dtype=torch.float64))

    def fit(self, training):
        return self

    def forward(self, values):
        return values * self.gain

    def inverse(self, values):
        return values / self.gain


@pytest.mark.parametrize("family", [rd.classic_esn, rd.ott_esn])
def test_rebuild_preserves_current_units_named_ports_feedback_clock_and_workflow_settings(family):
    incoming, outgoing = _Units(2), _Units(5)
    model = rd.Model(
        graph=family(12, 2, 2, seed=17, readout="svd", dtype=torch.float64, readout_name="head"),
        input_names=("state",),
        output_names=("next",),
        washout=3,
        preprocessing=None,
        input_transforms={"state": incoming},
        output_transforms={"next": outgoing},
        prediction_lead={"next": 2},
        feedback=[rd.Feedback("next", "state", delay=2)],
        resample_dt=0.5,
    ).eval()
    # These are live changes after construction, not the original constructor recipe.
    model.washout = 7
    model.resample_dt, model.resample_method = 0.25, "pchip"
    with torch.no_grad():
        incoming.gain.fill_(4)
        outgoing.gain.fill_(7)
    model.tune(
        _series(), preset="quick", n_trials=1, search_space=_small_space, seeds=(7, 19), horizon=3
    )
    assert model.washout == 7 and model.preprocessing is None
    assert model.resample_dt == 0.25 and model.resample_method == "pchip"
    assert model.input_names == ("state",) and model.output_names == ("next",)
    assert model.prediction_leads == {"next": 2}
    assert model.feedback == (rd.Feedback("next", "state", delay=2),)
    assert model.input_scalers[0].gain.item() == 4
    assert model.output_scalers[0].gain.item() == 7
    assert model.input_scalers[0].gain.requires_grad
    assert not model.training and _head(model).name == "head"
    assert torch.isfinite(model.forecast(history=_series()[-20:], horizon=3)).all()


def test_facade_rebuild_uses_live_workflow_values_instead_of_stale_constructor_options():
    model = _built("facade").eval()
    model.workflow.washout = 7
    model.workflow.resample_dt, model.workflow.resample_method = 0.125, "pchip"
    model.tune(
        _series(), preset="quick", n_trials=1, search_space=_small_space, seeds=(7,), horizon=3
    )
    assert model.washout == 7
    assert model.workflow.resample_dt == 0.125 and model.workflow.resample_method == "pchip"
    assert not model.training


def _custom_cycle(rows, cols=None, **kwargs):
    cols = rows if cols is None else cols
    assert rows == cols
    return torch.roll(torch.eye(rows, dtype=torch.float64), shifts=1, dims=1)


def _constant_weights(rows, cols, value=0.25, **kwargs):
    return torch.full((rows, cols), value, dtype=torch.float64)


@pytest.mark.parametrize("family", [rd.classic_esn, rd.ott_esn])
def test_custom_initializers_activation_and_readout_options_survive_builtin_rebuilding(family):
    from resdag.hpo.tuning import _default_space

    model = rd.Model(
        graph=family(
            12,
            2,
            2,
            topology=_custom_cycle,
            feedback_initializer=(_constant_weights, {"value": 0.2}),
            bias_initializer=(_constant_weights, {"value": 0.125}),
            activation="sigmoid",
            spectral_radius=0.4,
            readout="svd",
            readout_bias=False,
            readout_kwargs={"rcond": 1e-9},
            seed=17,
            dtype=torch.float64,
        ),
        washout=3,
    )
    trial = optuna.trial.FixedTrial(
        {
            "reservoir_size": 16,
            "spectral_radius": 0.4,
            "leak_rate": 0.7,
            "alpha": 1e-4,
            **({"input_to_readout": False} if family is rd.classic_esn else {}),
        }
    )
    sampled = _default_space(model)(trial)
    assert "mean_degree" not in sampled and "bias_scaling" not in sampled
    assert "input_scaling" not in sampled  # arbitrary callback has no scaling contract
    model.tune(
        _series(), preset="quick", n_trials=1, search_space=_small_space, seeds=(7, 19), horizon=3
    )
    memory = _memory(model)
    assert memory.activation == "sigmoid" and memory.reservoir_size == 16
    torch.testing.assert_close(
        memory.weight_hh, _custom_cycle(16, 16) * 0.4, atol=1e-14, rtol=1e-14
    )
    torch.testing.assert_close(
        memory.weight_feedback, torch.full((16, 2), 0.2, dtype=torch.float64), atol=0, rtol=0
    )
    torch.testing.assert_close(
        memory.bias_h, torch.full((16,), 0.125, dtype=torch.float64), atol=0, rtol=0
    )
    assert _head(model).rcond == 1e-9 and _head(model).bias is None


def _scaled_hidden(module, inputs, output):
    return output * 1.25


def _bounded_activation(values):
    return values.tanh() * 0.5


@pytest.mark.parametrize("edit", ["weight", "hook", "activation", "readout", "module"])
def test_user_graph_edits_take_fixed_graph_path_and_are_never_discarded(edit):
    model = _built()
    memory, head = _memory(model), _head(model)
    if edit == "weight":
        with torch.no_grad():
            memory.weight_hh[0, 0].add_(0.125)
    elif edit == "hook":
        memory.register_forward_hook(_scaled_hidden)
    elif edit == "activation":
        memory.cell.activation_fn = _bounded_activation
    elif edit == "readout":
        head.rcond = 0.25
    else:
        model.graph.add_module("user_annotation", nn.Linear(2, 2, dtype=torch.float64))
    weights = memory.weight_hh.detach().clone()
    model.tune(_series(), preset="balanced", n_trials=1, search_space=_alpha_space, horizon=3)
    assert len(model.tuning_report.seeds) == 1
    assert model.tuning_report.best_params == {"alpha": 1e-4}
    torch.testing.assert_close(_memory(model).weight_hh, weights, atol=0, rtol=0)
    if edit == "hook":
        assert _scaled_hidden in _memory(model)._forward_hooks.values()
    elif edit == "activation":
        assert _memory(model).cell.activation_fn is _bounded_activation
    elif edit == "readout":
        assert _head(model).rcond == 0.25
    elif edit == "module":
        assert isinstance(model.graph.user_annotation, nn.Linear)


def test_genuinely_custom_named_multioutput_dag_remains_fixed_without_an_explicit_factory():
    data = _series()
    source = rd.reservoir_input(2, dtype=torch.float64)
    hidden = rd.ESNLayer(10, feedback_size=2, seed=17, dtype=torch.float64)(source)
    position = rd.SVDReadoutLayer(10, 2, alpha=1e-4).double()(hidden)
    energy = rd.SVDReadoutLayer(10, 1, alpha=1e-4).double()(hidden)
    model = rd.Model(
        inputs={"state": source},
        outputs={"position": position, "energy": energy},
        washout=3,
        feedback=[rd.Feedback("position", "state")],
    )
    old_weights = _memory(model).weight_hh.detach().clone()
    model.tune(
        data[:-1],
        targets={"position": data[1:], "energy": data[1:].square().sum(-1, keepdim=True)},
        preset="balanced",
        n_trials=1,
        search_space=_alpha_space,
        horizon=3,
    )
    assert len(model.tuning_report.seeds) == 1
    torch.testing.assert_close(_memory(model).weight_hh, old_weights, atol=0, rtol=0)
    predicted = model.forecast(history=data[-15:], horizon=3)
    assert set(predicted) == {"position", "energy"}
    assert predicted["position"].shape == (3, 2) and predicted["energy"].shape == (3, 1)
    with pytest.raises(ValueError, match="factory"):
        model.tune(
            data[:-1],
            targets={"position": data[1:], "energy": data[1:].square().sum(-1, keepdim=True)},
            preset="quick",
            seeds=(7, 19),
            n_trials=1,
            search_space=_alpha_space,
            horizon=3,
        )


@pytest.mark.parametrize("family", [rd.classic_esn, rd.ott_esn])
def test_recipe_registration_matches_manual_dag_weights_and_caller_rng(family):
    torch.manual_seed(8128)
    automatic = family(
        12,
        2,
        2,
        seed=17,
        spectral_radius=0.4,
        readout="svd",
        readout_alpha=1e-4,
        dtype=torch.float64,
    )
    after_automatic = torch.random.get_rng_state().clone()
    torch.manual_seed(8128)
    source = rd.reservoir_input(2, dtype=torch.float64)
    hidden = rd.ESNLayer(12, feedback_size=2, seed=17, spectral_radius=0.4, dtype=torch.float64)(
        source
    )
    if family is rd.ott_esn:
        hidden = rd.SelectiveExponentiation(index=0, exponent=2.0)(hidden)
    output = rd.SVDReadoutLayer(14, 2, alpha=1e-4, name="output").double()(
        rd.Concatenate()(source, hidden)
    )
    manual = rd.ESNModel(source, output)
    assert torch.equal(after_automatic, torch.random.get_rng_state())
    for left, right in zip(automatic.parameters(), manual.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_ordinary_classical_model_construction_never_imports_optional_optuna(tmp_path):
    code = """
import importlib.abc
import sys
import torch
class RejectOptuna(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "optuna" or fullname.startswith("optuna."):
            raise AssertionError("ordinary model construction imported Optuna")
sys.meta_path.insert(0, RejectOptuna())
import resdag as rd
for factory in (rd.classic_esn, rd.ott_esn):
    factory(12, 2, 2, seed=17, readout="svd", dtype=torch.float64)
assert "optuna" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_automatic_recipe_survives_cpu_to_mps_and_retains_float32_placement():
    graph = rd.classic_esn(
        12,
        2,
        2,
        seed=17,
        dtype=torch.float32,
        readout="cg",
        readout_kwargs={"gram_dtype": torch.float32, "use_float64": False},
    )
    model = rd.Model(graph=graph, washout=3).to("mps")
    data = _series(dtype=torch.float32, device="mps")
    model.tune(
        data,
        preset="quick",
        n_trials=1,
        seeds=(7, 19),
        search_space=lambda trial: {"reservoir_size": 16, "alpha": 0.1},
        horizon=2,
    )
    assert _memory(model).reservoir_size == 16
    prediction = model.forecast(history=data[-15:], horizon=2)
    assert prediction.device.type == "mps" and prediction.dtype == torch.float32
    assert torch.isfinite(prediction).all()
    assert all(parameter.device.type == "mps" for parameter in model.parameters())


@pytest.mark.parametrize("which", ["recurrent", "readout"])
def test_direct_parameter_gradient_edits_are_not_lost_when_weights_are_unchanged(which):
    model = _built()
    parameter = _memory(model).weight_hh if which == "recurrent" else _head(model).weight
    parameter.requires_grad_(True)
    initial_bytes = parameter.detach().clone()
    model.tune(_series(), preset="balanced", n_trials=1, search_space=_alpha_space, horizon=3)
    restored = _memory(model).weight_hh if which == "recurrent" else _head(model).weight
    assert restored.requires_grad
    assert len(model.tuning_report.seeds) == 1
    if which == "recurrent":
        torch.testing.assert_close(restored, initial_bytes, atol=0, rtol=0)
    model.reset_state()
    sample = _series()[-5:].clone().requires_grad_()
    model(sample).square().mean().backward()
    assert restored.grad is not None and torch.isfinite(restored.grad).all()


@pytest.mark.parametrize("kind", ["classic", "ott", "facade"])
def test_dtype_move_cannot_revalidate_previously_edited_reservoir_weights(kind):
    model = _built(kind, dtype=torch.float32)
    with torch.no_grad():
        _memory(model).weight_hh[0, 0].add_(0.125)
    expected = _memory(model).weight_hh.detach().double().clone()
    model.double()
    model.tune(_series(), preset="balanced", n_trials=1, search_space=_alpha_space, horizon=3)
    assert len(model.tuning_report.seeds) == 1
    torch.testing.assert_close(_memory(model).weight_hh, expected, rtol=0, atol=0)


class _UserPipeline(rd.Model):
    def __init__(self, graph):
        super().__init__(graph=graph, washout=3)
        self.post = nn.Linear(2, 2, dtype=torch.float64, bias=False)
        with torch.no_grad():
            self.post.weight.copy_(torch.eye(2, dtype=torch.float64) * 0.75)

    def forward(self, values, **kwargs):
        return self.post(super().forward(values, **kwargs))


@pytest.mark.parametrize("customization", ["subclass", "outer_parameter"])
def test_custom_workflow_components_are_not_replaced_by_a_plain_model_recipe(customization):
    graph = rd.classic_esn(12, 2, 2, seed=17, readout="svd", dtype=torch.float64)
    if customization == "subclass":
        model = _UserPipeline(graph)
        expected = model.post.weight.detach().clone()
    else:
        model = rd.Model(graph=graph, washout=3)
        model.register_parameter("user_gain", nn.Parameter(torch.tensor(0.75, dtype=torch.float64)))
    model.tune(_series(), preset="balanced", n_trials=1, search_space=_alpha_space, horizon=3)
    assert len(model.tuning_report.seeds) == 1
    if customization == "subclass":
        assert type(model) is _UserPipeline
        torch.testing.assert_close(model.post.weight, expected, rtol=0, atol=0)
    else:
        assert model.user_gain.item() == 0.75 and model.user_gain.requires_grad
    model.reset_state()
    assert torch.isfinite(model(_series()[-5:])).all()


def test_outer_facade_hook_is_preserved_by_single_seed_fixed_graph_tuning():
    model = _built("facade")
    model.register_forward_hook(_scaled_hidden)
    weights = _memory(model).weight_hh.detach().clone()
    model.tune(_series(), preset="balanced", n_trials=1, search_space=_alpha_space, horizon=3)
    assert len(model.tuning_report.seeds) == 1
    assert _scaled_hidden in model._forward_hooks.values()
    torch.testing.assert_close(_memory(model).weight_hh, weights, atol=0, rtol=0)
    model.reset_state()
    sample = _series()[-5:]
    expected = model.workflow(sample)
    model.reset_state()
    torch.testing.assert_close(model(sample), expected * 1.25, atol=0, rtol=0)


@pytest.mark.parametrize("customization", ["module", "parameter", "buffer"])
def test_lazy_facade_custom_components_select_fixed_graph_before_initial_build(customization):
    model = rd.ESN(12, seed=17, readout="svd", washout=3, dtype=torch.float64)
    if customization == "module":
        model.add_module("user_projection", nn.Linear(2, 2, dtype=torch.float64))
        expected = model.user_projection.weight.detach().clone()
    elif customization == "parameter":
        model.register_parameter("user_gain", nn.Parameter(torch.tensor(0.75)))
    else:
        model.register_buffer("user_offset", torch.tensor(0.25))
    assert not model.is_built
    with pytest.raises(ValueError, match="factory"):
        model.tune(
            _series(),
            preset="quick",
            seeds=(7, 19),
            n_trials=1,
            search_space=_alpha_space,
            horizon=3,
        )
    assert not model.is_built  # validation must not partially initialize the caller
    model.tune(_series(), preset="balanced", n_trials=1, search_space=_alpha_space, horizon=3)
    assert len(model.tuning_report.seeds) == 1 and model.is_fitted
    if customization == "module":
        torch.testing.assert_close(model.user_projection.weight, expected, atol=0, rtol=0)
    elif customization == "parameter":
        assert model.user_gain.item() == 0.75 and model.user_gain.requires_grad
    else:
        assert model.user_offset.item() == 0.25
    assert torch.isfinite(model.forecast(horizon=3)).all()


@pytest.mark.parametrize(
    "preset,seeds,origins", [("quick", 1, 1), ("balanced", 3, 3), ("thorough", 5, 5)]
)
def test_automatic_presets_keep_the_existing_seed_and_origin_budgets(preset, seeds, origins):
    model = _built()
    model.tune(_series(), preset=preset, n_trials=1, search_space=_alpha_space, horizon=2)
    assert len(model.tuning_report.seeds) == seeds
    assert len(model.tuning_report.runs) == seeds * origins
    assert {run["origin"] for run in model.tuning_report.runs} == set(range(origins))
    assert all(run["complete"] for run in model.tuning_report.runs)


@pytest.mark.parametrize("family", [rd.classic_esn, rd.ott_esn])
def test_conditional_search_controls_apply_to_the_new_initializer_topology_and_readout(family):
    # Simultaneously replace configurable components and apply shared controls.
    # All 16 possible row entries requested => an exactly dense new looped graph.
    model = rd.Model(graph=family(12, 2, 2, seed=17, readout="svd", dtype=torch.float64), washout=3)
    params = {
        "reservoir_size": 16,
        "feedback_initializer": ("uniform", {"input_scaling": 100.0}),
        "input_scaling": 0.2,
        "topology": ("erdos_renyi", {"p": 0.01, "self_loops": True}),
        "mean_degree": 16,
        "readout_kwargs": {"alpha": 1e5, "rcond": 1e-8},
        "alpha": 1e-4,
    }
    model.tune(
        _series(),
        preset="quick",
        n_trials=1,
        search_space=lambda trial: params,
        horizon=3,
        catch_exceptions=False,
    )
    memory, head = _memory(model), _head(model)
    assert memory.weight_hh.count_nonzero() == 16 * 16
    assert 0.1 < memory.weight_feedback.abs().max().item() <= 0.2
    assert head.alpha == 1e-4 and head.rcond == 1e-8
