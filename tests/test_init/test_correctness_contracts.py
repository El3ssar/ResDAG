"""Independent regression checks for initializer values and calling contracts."""

from __future__ import annotations

import runpy
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from resdag.init.graphs.multi_cycle import multi_cycle_graph
from resdag.init.graphs.regular import regular_adjacency, regular_graph
from resdag.init.graphs.simple_cycle_jumps import simple_cycle_jumps_graph
from resdag.init.graphs.spectral_cascade import spectral_cascade_graph
from resdag.init.input_feedback import FunctionInitializer, get_input_feedback
from resdag.init.input_feedback.registry import (
    _INPUT_FEEDBACK_REGISTRY,
    register_input_feedback,
    show_input_initializers,
)
from resdag.init.topology import GraphTopology, MatrixTopology, get_topology
from resdag.init.topology import base as topology_base
from resdag.init.topology.registry import (
    _TOPOLOGY_REGISTRY,
    register_matrix_topology,
    show_topologies,
)
from resdag.init.utils.resolve import resolve_initializer, resolve_topology

# Snapshot built-ins before tests register any user callbacks.
_BUILTIN_TOPOLOGIES = dict(_TOPOLOGY_REGISTRY)
_BUILTIN_INITIALIZERS = dict(_INPUT_FEEDBACK_REGISTRY)
_CATALOG = runpy.run_path(str(Path(__file__).parents[2] / "hooks/docs_autogen.py"))


@pytest.mark.parametrize(
    "name,params",
    [
        ("dendrocycle_input", {"c": 0.5}),
        ("ring_window", {"c": 0.5, "window": 3}),
    ],
)
def test_required_named_options_match_configured_initializer(name, params):
    options = {**params, **({"seed": 12} if name == "dendrocycle_input" else {})}
    resolved = resolve_initializer((name, options), seed=99)
    configured = get_input_feedback(name, **options)
    a, b = torch.empty(24, 3), torch.empty(24, 3)
    resolved.initialize(a)
    configured.initialize(b)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.fixture
def registered_inplace():
    name = "contract_uniform_inplace"
    register_matrix_topology(name)(torch.nn.init.uniform_)
    register_input_feedback(name)(torch.nn.init.uniform_)
    yield name
    _TOPOLOGY_REGISTRY.pop(name, None)
    _INPUT_FEEDBACK_REGISTRY.pop(name, None)


@pytest.mark.parametrize("kind", ["topology", "feedback"])
@pytest.mark.parametrize(
    "form", ["direct", "tuple", "registered", "registered_tuple", "configured"]
)
def test_inplace_seed_matches_independent_torch_generator(registered_inplace, kind, form):
    name = registered_inplace
    fn = torch.nn.init.uniform_
    forms = {
        "direct": fn,
        "tuple": (fn, {"a": -0.4, "b": 0.7}),
        "registered": name,
        "registered_tuple": (name, {"a": -0.4, "b": 0.7}),
        "configured": (
            get_topology(name, seed=71) if kind == "topology" else get_input_feedback(name, seed=71)
        ),
    }
    seed = 71
    if kind == "topology":
        actual = torch.empty(9, 9, dtype=torch.float64)
        initializer = resolve_topology(forms[form], seed=seed)
    else:
        actual = torch.empty(9, 3, dtype=torch.float64)
        initializer = resolve_initializer(forms[form], seed=seed)
    global_state = torch.random.get_rng_state().clone()
    initializer.initialize(actual)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    expected = torch.empty_like(actual)
    kwargs = {"a": -0.4, "b": 0.7} if form in ("tuple", "registered_tuple") else {}
    fn(expected, generator=torch.Generator().manual_seed(seed), **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["topology", "feedback"])
@pytest.mark.parametrize("registered", [False, True])
def test_explicit_callback_seed_wins_over_layer_seed(registered_inplace, kind, registered):
    spec = (registered_inplace if registered else torch.nn.init.uniform_, {"seed": 41})
    weight = torch.empty(8, 8)
    resolver = resolve_topology if kind == "topology" else resolve_initializer
    resolver(spec, seed=902).initialize(weight)
    expected = torch.empty_like(weight).uniform_(generator=torch.Generator().manual_seed(41))
    torch.testing.assert_close(weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["topology", "feedback"])
def test_bound_generator_advances_and_wins_over_layer_seed(kind):
    generator = torch.Generator().manual_seed(19)
    resolver = resolve_topology if kind == "topology" else resolve_initializer
    initializer = resolver((torch.nn.init.uniform_, {"generator": generator}), seed=902)
    first, second = torch.empty(8, 8), torch.empty(8, 8)
    initializer.initialize(first)
    initializer.initialize(second)
    reference = torch.Generator().manual_seed(19)
    torch.testing.assert_close(first, torch.empty_like(first).uniform_(generator=reference))
    torch.testing.assert_close(second, torch.empty_like(second).uniform_(generator=reference))
    assert not torch.equal(first, second)


@pytest.mark.parametrize(
    "name",
    ["pseudo_diagonal", "minimal_init", "dendrocycle_input", "weighted_init", "weighted_minimal"],
)
def test_numpy_backed_initializer_respects_global_torch_seed(name):
    kwargs = {"c": 0.5} if name == "dendrocycle_input" else {}
    torch.manual_seed(317)
    first = get_input_feedback(name, **kwargs).initialize(torch.empty(24, 3))
    torch.manual_seed(317)
    second = get_input_feedback(name, **kwargs).initialize(torch.empty(24, 3))
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("wrapper", ["matrix", "feedback", "graph"])
def test_float64_callback_precision_is_preserved(wrapper):
    value = np.float64(1.000000000123)
    if wrapper == "graph":

        def builder(n):
            graph = nx.DiGraph()
            graph.add_nodes_from(range(n))
            graph.add_edge(0, 0, weight=value)
            return graph

        initializer = GraphTopology(builder)
        expected = torch.zeros(3, 3, dtype=torch.float64)
        expected[0, 0] = value
    elif wrapper == "matrix":
        initializer = MatrixTopology(lambda n: np.full((n, n), value, dtype=">f8"))
        expected = torch.full((3, 3), value, dtype=torch.float64)
    else:
        initializer = FunctionInitializer(lambda rows, cols: np.full((rows, cols), value))
        expected = torch.full((3, 3), value, dtype=torch.float64)
    actual = initializer.initialize(torch.empty(3, 3, dtype=torch.float64))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_postponed_annotations_and_registered_defaults_are_discoverable(capsys):
    name = "contract_annotated"

    def callback(n: int, gain: float = 2.0) -> torch.Tensor:
        return gain * torch.eye(n)

    try:
        register_matrix_topology(name, gain=0.4)(callback)
        register_input_feedback(name, gain=0.4)(callback)
        show_topologies(name)
        show_input_initializers(name)
        output = capsys.readouterr().out
        assert output.count("0.4") == 2
        assert "float" in output
    finally:
        _TOPOLOGY_REGISTRY.pop(name, None)
        _INPUT_FEEDBACK_REGISTRY.pop(name, None)


def test_dense_spectral_scale_does_not_trust_stationary_subdominant_probe(monkeypatch):
    monkeypatch.setattr(topology_base, "_DENSE_EIGVALS_MAX_N", 8)
    n = 12
    probe = torch.randn(n, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    dominant = torch.arange(1, n + 1, dtype=torch.float64)
    dominant -= (dominant @ probe) / (probe @ probe) * probe
    dominant /= dominant.norm()
    matrix = torch.eye(n, dtype=torch.float64) + 9 * torch.outer(dominant, dominant)
    assert topology_base.estimate_spectral_radius(matrix) == pytest.approx(10, abs=1e-10)
    scaled = topology_base.scale_to_spectral_radius(matrix, 0.9)
    assert torch.linalg.eigvals(scaled).abs().max().item() == pytest.approx(0.9, abs=1e-12)


@pytest.mark.parametrize("n", range(1, 11))
def test_regular_lattice_has_exact_degree_and_independent_adjacency(n):
    for k in range(n):
        if n * k % 2:
            with pytest.raises(ValueError, match="even"):
                regular_adjacency(n, k)
            continue
        expected = nx.Graph()
        expected.add_nodes_from(range(n))
        for offset in range(1, k // 2 + 1):
            expected.add_edges_from((i, (i + offset) % n) for i in range(n))
        if k % 2:
            expected.add_edges_from((i, i + n // 2) for i in range(n // 2))
        for directed in (False, True):
            actual = regular_adjacency(n, k, directed=directed, random_weights=False)
            np.testing.assert_array_equal(actual != 0, nx.to_numpy_array(expected) != 0)
            assert np.all(np.count_nonzero(actual, axis=1) == k)
            graph = regular_graph(n, k, directed=directed, seed=7)
            degree = graph.out_degree if directed else graph.degree
            assert all(count == k for _, count in degree)


def test_cycle_and_jump_weights_add_without_duplicate_jump_pairs():
    graph = simple_cycle_jumps_graph(4, jump_length=1, r_c=1.1, r_l=0.3)
    expected = np.zeros((4, 4))
    for i in range(4):
        expected[i, (i + 1) % 4] += 1.1
        expected[i, (i + 1) % 4] += 0.3
        expected[(i + 1) % 4, i] += 0.3
    np.testing.assert_allclose(nx.to_numpy_array(graph), expected)
    pair = nx.to_numpy_array(simple_cycle_jumps_graph(2, 1, r_c=1.1, r_l=0.3))
    np.testing.assert_allclose(pair, [[0, 1.4], [1.4, 0]])


def test_zero_spectral_cascade_is_actually_zero():
    graph = spectral_cascade_graph(10, spectral_radius=0)
    assert not np.any(nx.to_numpy_array(graph))


@pytest.mark.parametrize("kwargs", [{"n": 0, "k": 3}, {"n": 12, "k": 0}, {"n": 12, "k": -1}])
def test_multicycle_rejects_invalid_counts_before_division(kwargs):
    with pytest.raises(ValueError):
        multi_cycle_graph(**kwargs)


@pytest.mark.parametrize(
    "category,name,entry",
    [("topologies", name, entry) for name, entry in _BUILTIN_TOPOLOGIES.items()]
    + [("initializers", name, entry) for name, entry in _BUILTIN_INITIALIZERS.items()],
)
def test_every_generated_catalog_recipe_executes(category, name, entry):
    import re

    page_builder = _CATALOG["_topology_page" if category == "topologies" else "_initializer_page"]
    page = page_builder(name, entry)
    snippets = re.findall(r"```python\n(.*?)\n```", page, flags=re.DOTALL)
    assert snippets
    for snippet in snippets:
        namespace = {}
        exec(compile(snippet, f"generated:{category}/{name}.md", "exec"), namespace)
        layer = namespace["reservoir"]
        output = layer(torch.zeros(1, 2, layer.cell.feedback_size))
        assert torch.isfinite(output).all()


@pytest.mark.parametrize("kind", ["topology", "feedback"])
def test_build_style_generator_callback_receives_layer_seed(kind):
    def square(n, *, generator=None):
        return torch.rand(n, n, generator=generator)

    def rectangular(rows, cols, *, generator=None):
        return torch.rand(rows, cols, generator=generator)

    resolver = resolve_topology if kind == "topology" else resolve_initializer
    callback = square if kind == "topology" else rectangular
    actual = torch.empty(3, 3)
    resolver(callback, seed=18).initialize(actual)
    expected = torch.rand(3, 3, generator=torch.Generator().manual_seed(18))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_target_radius_is_rejected(value):
    with pytest.raises(ValueError, match="target_radius"):
        topology_base.scale_to_spectral_radius(torch.eye(3), value)


@pytest.mark.parametrize("magnitude", [1e-9, 1e-30])
def test_small_nonzero_radius_can_be_scaled_without_ratio_overflow(magnitude):
    matrix = magnitude * torch.eye(3, dtype=torch.float32)
    scaled = topology_base.scale_to_spectral_radius(matrix, 1e10)
    assert torch.isfinite(scaled).all()
    torch.testing.assert_close(scaled, 1e10 * torch.eye(3))
