"""Feature-switch oracles for classic graphs and the public ESN workflow."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional

import resdag as rd
from resdag.artifacts import save_model
from resdag.models._recipes import builtin_recipe


def _series():
    time = torch.linspace(0, 6, 54, dtype=torch.float64)
    return torch.stack((1 + time.sin(), -2 + 0.4 * time.cos()), -1)


def _memory(graph):
    return next(module for module in graph.modules() if isinstance(module, rd.ESNLayer))


def _head(graph):
    return next(module for module in graph.modules() if isinstance(module, rd.ReadoutLayer))


def _classic(**options):
    return rd.classic_esn(5, 2, 2, seed=17, dtype=torch.float64, **options)


@pytest.mark.parametrize("kind", ["classic", "facade"])
@pytest.mark.parametrize("invalid", [0, 1, None, "False", torch.tensor(False)])
def test_input_to_readout_requires_an_actual_bool_before_construction(kind, invalid):
    before = torch.random.get_rng_state().clone()
    with pytest.raises(TypeError, match="input_to_readout must be a bool"):
        if kind == "classic":
            _classic(input_to_readout=invalid)
        else:
            rd.ESN(5, input_to_readout=invalid)
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("kind", ["classic", "facade"])
def test_omitted_switch_preserves_explicit_true_weights_rng_and_predictions(kind):
    def build(**options):
        if kind == "classic":
            return rd.Model(graph=_classic(**options), washout=2)
        return rd.ESN(5, input_size=2, washout=2, seed=17, dtype=torch.float64, **options)

    torch.manual_seed(129)
    implicit = build()
    consumed = torch.random.get_rng_state().clone()
    torch.manual_seed(129)
    explicit = build(input_to_readout=True)
    assert torch.equal(torch.random.get_rng_state(), consumed)
    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    for name, value in implicit.state_dict().items():
        other = explicit.state_dict()[name]
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, other)
        else:
            assert value == other
    implicit.fit(_series())
    explicit.fit(_series())
    assert torch.equal(implicit(_series()[:8]), explicit(_series()[:8]))
    assert torch.equal(
        implicit.forecast(history=_series()[-9:], horizon=4),
        explicit.forecast(history=_series()[-9:], horizon=4),
    )


@pytest.mark.parametrize("input_to_readout", [False, True])
@pytest.mark.parametrize("driver", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
def test_features_outputs_and_gradients_match_independent_recurrence(
    input_to_readout, driver, batch
):
    graph = rd.classic_esn(
        3,
        2,
        2,
        input_size=1 if driver else None,
        input_to_readout=input_to_readout,
        topology="simple_cycle",
        leak_rate=0.4,
        trainable=True,
        readout_kwargs={"trainable": True},
        seed=7,
        dtype=torch.float64,
    )
    memory, head = _memory(graph), _head(graph)
    assert head.in_features == 3 + (2 if input_to_readout else 0)
    with torch.no_grad():
        memory.cell.weight_hh.copy_(torch.arange(9, dtype=torch.float64).reshape(3, 3) / 40)
        memory.cell.weight_feedback.copy_(
            torch.tensor([[0.2, -0.3], [0.1, 0.4], [-0.2, 0.1]], dtype=torch.float64)
        )
        memory.cell.bias_h.copy_(torch.tensor([0.1, -0.1, 0.05], dtype=torch.float64))
        if driver:
            memory.cell.weight_input.copy_(torch.tensor([[0.3], [-0.2], [0.1]]))
        head.weight.copy_(
            torch.arange(head.weight.numel(), dtype=torch.float64).reshape_as(head.weight) / 9
        )
        head.bias.copy_(torch.tensor([0.15, -0.2], dtype=torch.float64))
    shape = (batch, 4, 2)
    feedback = torch.linspace(-0.5, 0.7, torch.tensor(shape).prod().item(), dtype=torch.float64)
    feedback = feedback.reshape(shape).requires_grad_()
    inputs = [feedback]
    if driver:
        driving = (feedback[..., :1].detach() * 0.3 + 0.4).requires_grad_()
        inputs.append(driving)
    actual = graph(*inputs)

    fb = feedback
    dr = driving if driver else None
    state = feedback.new_zeros((fb.shape[0], 3))
    expected_steps = []
    for step in range(fb.shape[1]):
        preactivation = functional.linear(
            fb[:, step], memory.cell.weight_feedback, memory.cell.bias_h
        ) + functional.linear(state, memory.cell.weight_hh)
        if dr is not None:
            preactivation = preactivation + functional.linear(dr[:, step], memory.cell.weight_input)
        state = 0.6 * state + 0.4 * torch.tanh(preactivation)
        features = torch.cat((fb[:, step], state), -1) if input_to_readout else state
        expected_steps.append(functional.linear(features, head.weight, head.bias))
    expected = torch.stack(expected_steps, 1)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    variables = (*inputs, *graph.parameters())
    actual_gradients = torch.autograd.grad(actual.square().sum(), variables)
    expected_gradients = torch.autograd.grad(expected.square().sum(), variables)
    for observed, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(observed, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", ["constructor", "build", "fit"])
def test_false_switch_survives_eager_and_lazy_construction(mode):
    model = rd.ESN(
        5,
        input_size=2 if mode == "constructor" else None,
        input_to_readout=False,
        washout=2,
        seed=17,
        dtype=torch.float64,
    )
    if mode == "build":
        model.build(2)
    model.fit(_series())
    assert _head(model.graph).in_features == 5
    assert builtin_recipe(model.graph)["options"]["input_to_readout"] is False
    assert model._factory_options["input_to_readout"] is False
    assert torch.isfinite(model.forecast(horizon=4)).all()


@pytest.mark.parametrize("kind", ["classic", "facade"])
@pytest.mark.parametrize("input_to_readout", [False, True])
def test_portable_artifact_preserves_feature_recipe_and_exact_forecast(
    kind, input_to_readout, tmp_path
):
    if kind == "classic":
        source = rd.Model(graph=_classic(input_to_readout=input_to_readout), washout=2)
    else:
        source = rd.ESN(
            5, input_to_readout=input_to_readout, washout=2, seed=17, dtype=torch.float64
        )
    source.fit(_series())
    expected = source.forecast(history=_series()[-9:], horizon=4)
    path = tmp_path / "features.pt"
    save_model(source, path)
    restored = rd.load(path)
    assert _head(restored.graph).in_features == 5 + (2 if input_to_readout else 0)
    assert builtin_recipe(restored.graph)["options"]["input_to_readout"] is input_to_readout
    assert torch.equal(restored.forecast(history=_series()[-9:], horizon=4), expected)


def test_false_checkpoint_loads_matching_lazy_preset_and_rejects_opposite_architecture():
    options = dict(washout=2, seed=17, dtype=torch.float64)
    source = rd.ESN(5, input_to_readout=False, **options).fit(_series())
    restored = rd.ESN(5, input_to_readout=False, **options)
    restored.load_state_dict(source.state_dict())
    assert _head(restored.graph).in_features == 5 and restored.is_fitted
    assert torch.equal(restored.forecast(horizon=4), source.forecast(horizon=4))
    wrong = rd.ESN(5, input_to_readout=True, **options)
    before = copy.deepcopy(wrong.state_dict())
    with pytest.raises(RuntimeError, match="architecture differs"):
        wrong.load_state_dict(source.state_dict())
    assert not wrong.is_built and wrong.state_dict() == before


def test_unbuilt_artifact_preserves_false_for_later_fit(tmp_path):
    source = rd.ESN(5, input_to_readout=False, washout=2, seed=17)
    path = tmp_path / "lazy.pt"
    save_model(source, path)
    restored = rd.load(path)
    assert not restored.is_built and restored._factory_options["input_to_readout"] is False
    restored.fit(_series())
    assert _head(restored.graph).in_features == 5


@pytest.mark.parametrize("kind", ["classic", "facade"])
def test_actual_pre_option_defaults_load_without_schema_migration(kind):
    directory = Path(__file__).parents[1] / "fixtures" / "input_to_readout_ca40fe4b"
    provenance = json.loads((directory / "provenance.json").read_text())
    assert provenance["source_ref"] == "ca40fe4b9806a97ec11813a791b6f1cb11c58d0f"
    assert (
        hashlib.sha256(provenance["writer_source"].encode()).hexdigest()
        == provenance["writer_sha256"]
    )
    for name, digest in provenance["files"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    options = dict(seed=17, dtype=torch.float64)
    if kind == "facade":
        current = rd.ESN(5, washout=2, **options)
    else:
        current = rd.Model(graph=rd.classic_esn(5, 2, 2, **options), washout=2)
    current.load_state_dict(torch.load(directory / f"{kind}-state.pt", weights_only=True))
    restored = rd.load(directory / f"{kind}.pt")
    reference = torch.load(directory / f"{kind}-forecast.pt", weights_only=True)
    actual = current.forecast(history=_series()[-9:], horizon=4)
    portable = restored.forecast(history=_series()[-9:], horizon=4)
    assert torch.equal(actual, portable)
    # The old artifact was produced on macOS; BLAS implementations may round
    # differently on other supported hosts while evaluating the same equation.
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    assert _head(restored.graph).in_features == 7
    recipe = builtin_recipe(restored.graph)
    assert recipe is not None and recipe["family"] == "classic"
    assert "input_to_readout" not in recipe["options"]
    if kind == "facade":
        assert current._factory_options["input_to_readout"] is True
        assert restored._factory_options["input_to_readout"] is True
