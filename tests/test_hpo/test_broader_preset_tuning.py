"""Focused construction oracles for built-in tuning recipes."""

from __future__ import annotations

import copy

import pytest
import torch

import resdag as rd
from resdag.init.input_feedback import get_input_feedback
from resdag.init.topology import get_topology
from resdag.models._recipes import (
    builtin_recipe,
    erdos_renyi_options,
    input_scaling_supported,
    sampled_options,
)

optuna = pytest.importorskip("optuna")


@pytest.mark.parametrize("object_spec", [False, True])
def test_density_mapping_preserves_er_flags_seed_and_original_spec(object_spec):
    spec = ("erdos_renyi", {"p": 0.1, "directed": False, "self_loops": False, "seed": 91})
    if object_spec:
        spec = get_topology(spec[0], **spec[1])
    options = {"reservoir_size": 20, "recurrent_initializer": spec}
    supplied = sampled_options(options, {"reservoir_size": 8, "mean_degree": 3})
    _, updated, loops = erdos_renyi_options(supplied)
    settings = updated.graph_kwargs if object_spec else updated[1]
    assert not loops and settings == {
        "p": 3 / 7,
        "directed": False,
        "self_loops": False,
        "seed": 91,
    }
    original = spec.graph_kwargs if object_spec else spec[1]
    assert original["p"] == 0.1
    assert "topology" not in supplied


@pytest.mark.parametrize("object_spec", [False, True])
def test_input_strength_preserves_initializer_family_bounds_connectivity_and_seed(object_spec):
    settings = {"low": -0.2, "high": 0.7, "input_scaling": 0.9, "connectivity": 0.4, "seed": 19}
    spec = get_input_feedback("uniform", **settings) if object_spec else ("uniform", settings)
    supplied = sampled_options({"feedback_initializer": spec}, {"input_scaling": 0.25})
    updated = supplied["feedback_initializer"]
    values = vars(updated) if object_spec else updated[1]
    for key in ("low", "high", "connectivity", "seed"):
        assert values[key] == settings[key]
    assert values["input_scaling"] == 0.25
    assert (spec.input_scaling if object_spec else spec[1]["input_scaling"]) == 0.9


def test_explicit_initializer_readout_and_width_overrides_precede_shared_controls():
    supplied = sampled_options(
        {"reservoir_size": 20, "readout_kwargs": {"alpha": 0.7}},
        {
            "reservoir_size": 8,
            "topology": ("erdos_renyi", {"self_loops": False}),
            "mean_degree": 3,
            "feedback_initializer": ("bernoulli", {"input_scaling": 0.9, "p": 0.7}),
            "input_scaling": 0.2,
            "readout_kwargs": {"alpha": 0.8, "rcond": 1e-9},
            "alpha": 1e-5,
        },
    )
    assert supplied["topology"][1]["p"] == 3 / 7
    assert supplied["feedback_initializer"] == ("bernoulli", {"input_scaling": 0.2, "p": 0.7})
    assert supplied["readout_kwargs"] == {"rcond": 1e-9}
    assert supplied["readout_alpha"] == 1e-5
    with pytest.raises(ValueError, match="mean_degree"):
        sampled_options({"reservoir_size": 8}, {"topology": "cycle", "mean_degree": 3})
    with pytest.raises(ValueError, match="input_scaling"):
        sampled_options(
            {}, {"feedback_initializer": lambda weight: weight.zero_(), "input_scaling": 0.2}
        )


def test_registered_initializer_without_scaling_contract_is_not_replaced():
    assert not input_scaling_supported({"feedback_initializer": "zero"})

    def callback(weight):
        return weight.zero_()

    assert not input_scaling_supported({"feedback_initializer": callback})
    assert (
        sampled_options({"feedback_initializer": callback}, {})["feedback_initializer"] is callback
    )


@pytest.mark.parametrize("operation", ["to_empty", "custom_apply", "edited_then_double"])
def test_value_changing_tensor_operations_cannot_launder_recipe_eligibility(operation):
    graph = rd.ott_esn(8, 1, 1, seed=5, dtype=torch.float32)
    assert builtin_recipe(graph) is not None
    if operation == "to_empty":
        graph.to_empty(device="cpu")
    elif operation == "custom_apply":
        graph._apply(lambda value: value * 2)
    else:
        with torch.no_grad():
            graph.ESNLayer_1.weight_hh.add_(0.25)
        graph.double()
    assert builtin_recipe(graph) is None
    assert builtin_recipe(copy.deepcopy(graph)) is None


def test_selective_module_modes_survive_rebuilding():
    from resdag.hpo.tuning import _preset_recipe

    model = rd.ESN(8, input_size=1, seed=5, washout=3)
    model.graph.ESNLayer_1.eval()
    candidate = _preset_recipe(model)({"reservoir_size": 12}, 7)
    assert candidate.training and candidate.workflow.training
    assert not candidate.graph.ESNLayer_1.training
    assert candidate.graph.AutoReadoutLayer_1.training


def test_lazy_outer_customization_uses_one_fixed_seed_and_retains_module():
    from resdag.hpo.tuning import _automatic_options

    model = rd.ESN(8, washout=3)
    model.add_module("postprocessor", torch.nn.Identity())
    assert _automatic_options(model) is None
    model.tune(
        torch.sin(torch.arange(60)[:, None] * 0.1),
        preset="balanced",
        n_trials=1,
        horizon=2,
        search_space=lambda trial: {"alpha": 1e-4},
    )
    assert model.postprocessor is not None
    assert model.tuning_report.seeds == (0,)


def test_shared_memory_preserves_values_and_automatic_recipe():
    graph = rd.classic_esn(8, 1, 1, seed=5, dtype=torch.float64)
    before = {name: value.clone() for name, value in graph.state_dict().items()}
    graph.share_memory()
    assert builtin_recipe(graph) is not None
    assert all(parameter.is_shared() for parameter in graph.parameters())
    for name, value in graph.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
