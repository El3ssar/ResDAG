"""Topology initialization contracts (resdag.init.topology + resdag.init.graphs).

Pins down:

- ``GraphTopology``: weight initialization from graph functions, spectral
  radius scaling, validation, dtype/device handling, and the dendrocycle
  rounding edge cases,
- the topology registry (``get_topology`` / ``show_topologies``),
- ``MatrixTopology`` and ``register_matrix_topology``: build-style and
  in-place callables, NumPy/NetworkX coercion, error reporting,
- callable topology specs through ``resolve_topology`` and end-to-end use
  in ``ESNLayer``.
"""

import importlib

import networkx as nx
import numpy as np
import pytest
import torch

import resdag.init.graphs as graphs_pkg
from resdag.init.graphs import (
    barabasi_albert_graph,
    dendrocycle_graph,
    erdos_renyi_graph,
    ring_chord_graph,
)
from resdag.init.topology import (
    GraphTopology,
    MatrixTopology,
    get_topology,
    register_matrix_topology,
    show_topologies,
)
from resdag.init.utils import resolve_initializer, resolve_topology
from resdag.layers import ESNLayer


def block_diagonal(n: int, blocks: int = 4) -> torch.Tensor:
    """Build-style matrix callable used across the matrix-topology tests."""
    w = torch.zeros(n, n)
    size = n // blocks
    for b in range(blocks):
        s = b * size
        w[s : s + size, s : s + size] = torch.randn(size, size)
    return w


# ---------------------------------------------------------------------------
# Public-API export consistency (issue #137)
# ---------------------------------------------------------------------------


class TestGraphsPackageExports:
    """``resdag.init.graphs.__all__`` must agree with the bound module symbols.

    Regression guard for the drift class where ``__all__`` advertises a name
    that is never bound at module level (issue #137: ``__all__`` listed
    ``chord_dendrocycle_graph`` while the import bound
    ``dendrocycle_with_chords_graph``). Such drift silently breaks both
    targeted imports and ``from resdag.init.graphs import *``.
    """

    def test_every_all_name_is_bound(self) -> None:
        """Every name advertised in ``__all__`` resolves to a module attribute."""
        missing = [name for name in graphs_pkg.__all__ if not hasattr(graphs_pkg, name)]

        assert not missing, f"__all__ advertises names not bound on the module: {missing}"

    def test_every_all_name_is_importable(self) -> None:
        """Every ``__all__`` name imports via ``from ... import <name>``."""
        for name in graphs_pkg.__all__:
            module = importlib.import_module("resdag.init.graphs")
            assert hasattr(module, name), f"cannot import name {name!r} from resdag.init.graphs"

    def test_star_import_succeeds(self) -> None:
        """``from resdag.init.graphs import *`` executes without error.

        A phantom ``__all__`` entry raises ``AttributeError`` at star-import
        time, so successful execution proves every advertised name is bound.
        """
        namespace: dict[str, object] = {}

        exec("from resdag.init.graphs import *", namespace)

        for name in graphs_pkg.__all__:
            assert name in namespace, f"star import did not bind {name!r}"


# ---------------------------------------------------------------------------
# GraphTopology
# ---------------------------------------------------------------------------


class TestGraphTopology:
    """GraphTopology weight initialization."""

    def test_initialization_basic(self) -> None:
        """Basic graph topology initialization."""
        topology = GraphTopology(erdos_renyi_graph, {"p": 0.1, "directed": True, "seed": 42})
        weight = torch.empty(50, 50)

        result = topology.initialize(weight)

        assert result is weight  # Should return the same tensor
        assert weight.shape == (50, 50)
        assert not torch.all(weight == 0)  # Should have been initialized

    def test_initialization_with_spectral_radius(self) -> None:
        """Initialization with spectral radius scaling."""
        topology = GraphTopology(erdos_renyi_graph, {"p": 0.2, "directed": True, "seed": 42})
        weight = torch.empty(50, 50)
        target_radius = 0.9

        topology.initialize(weight, spectral_radius=target_radius)

        # Verify spectral radius is close to target
        eigenvalues = torch.linalg.eigvals(weight)
        actual_radius = torch.max(torch.abs(eigenvalues)).item()

        assert abs(actual_radius - target_radius) < 0.01

    def test_non_square_weight_raises_error(self) -> None:
        """Non-square weights raise ValueError."""
        topology = GraphTopology(erdos_renyi_graph, {"p": 0.1})
        weight = torch.empty(50, 100)

        with pytest.raises(ValueError, match="must be square"):
            topology.initialize(weight)

    def test_different_graph_functions(self) -> None:
        """Initialization with different graph functions."""
        topologies = [
            GraphTopology(erdos_renyi_graph, {"p": 0.1, "directed": True, "seed": 42}),
            GraphTopology(ring_chord_graph, {"L": 1, "w": 0.5, "alpha": 1.0}),
        ]

        for topology in topologies:
            weight = torch.empty(30, 30)
            result = topology.initialize(weight, spectral_radius=0.9)

            assert result.shape == (30, 30)
            assert not torch.all(result == 0)


class TestTopologyRegistry:
    """Topology registry lookups."""

    def test_show_topologies_list(self) -> None:
        """Listing available topologies."""
        topologies = show_topologies()

        assert isinstance(topologies, list)
        assert len(topologies) > 0
        assert "erdos_renyi" in topologies
        assert "watts_strogatz" in topologies

    def test_get_topology_by_name(self) -> None:
        """Getting topology by name."""
        topology = get_topology("erdos_renyi", p=0.15, seed=42)

        assert isinstance(topology, GraphTopology)
        assert topology.graph_kwargs["p"] == 0.15
        assert topology.graph_kwargs["seed"] == 42

    def test_get_topology_unknown_raises_error(self) -> None:
        """Unknown topology name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown topology"):
            get_topology("nonexistent_topology")

    def test_get_topology_with_defaults(self) -> None:
        """Getting topology with default parameters."""
        topology = get_topology("erdos_renyi")
        weight = torch.empty(40, 40)

        result = topology.initialize(weight, spectral_radius=0.95)

        assert result.shape == (40, 40)

    def test_get_topology_override_defaults(self) -> None:
        """Overriding default parameters."""
        topology = get_topology("erdos_renyi", p=0.5, directed=False)

        assert topology.graph_kwargs["p"] == 0.5
        assert topology.graph_kwargs["directed"] is False


class TestGraphTopologyEdgeCases:
    """Edge cases for graph topology initialization."""

    def test_very_small_graph(self) -> None:
        """Very small graphs initialize without error."""
        topology = GraphTopology(erdos_renyi_graph, {"p": 0.5, "directed": True, "seed": 42})
        weight = torch.empty(3, 3)

        topology.initialize(weight)

        assert weight.shape == (3, 3)

    def test_initialize_on_device(self, device: torch.device) -> None:
        """Initialization works on tensors living on any device."""
        topology = GraphTopology(erdos_renyi_graph, {"p": 0.1, "directed": True, "seed": 42})
        weight = torch.empty(50, 50, device=device)

        result = topology.initialize(weight, spectral_radius=0.9)

        assert result.device.type == device.type
        assert result.shape == (50, 50)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_different_dtypes(self, dtype: torch.dtype) -> None:
        """Initialization preserves the target tensor's dtype."""
        topology = GraphTopology(erdos_renyi_graph, {"p": 0.1, "directed": True, "seed": 42})

        weight = torch.empty(30, 30, dtype=dtype)
        result = topology.initialize(weight)

        assert result.dtype == dtype


class TestDendrocycleRounding:
    """Dendrocycle graph rounding edge cases."""

    def test_dendrocycle_rounding_edge_case(self) -> None:
        """Dendrocycle handles rounding correctly when c+d≈1."""
        # Cases where rounding could cause C + D to exceed n
        # e.g., c=0.503, d=0.497 → C=201, D=199 → C+D=400 ✓
        # but c=0.5025, d=0.4975 could round differently
        topology = GraphTopology(dendrocycle_graph, {"c": 0.5025, "d": 0.4975, "seed": 42})
        weight = torch.empty(400, 400)

        # Should not raise ValueError about graph size mismatch
        result = topology.initialize(weight, spectral_radius=0.9)

        assert result.shape == (400, 400)

    def test_dendrocycle_various_parameter_combinations(self) -> None:
        """Dendrocycle with c,d combinations that could cause rounding issues."""
        n = 400
        test_cases = [
            (0.499, 0.499),  # Close to equal split
            (0.503, 0.495),  # Slightly uneven
            (0.5025, 0.4975),  # Very close to 1.0
            (0.501, 0.498),  # Another close case
            (0.333, 0.332),  # Thirds (rounding issues)
            (0.666, 0.333),  # Two thirds
        ]

        for c, d in test_cases:
            if c + d > 1.0:
                continue  # Skip invalid combinations

            topology = GraphTopology(dendrocycle_graph, {"c": c, "d": d, "seed": 42})
            weight = torch.empty(n, n)

            # Should create exactly n nodes
            result = topology.initialize(weight, spectral_radius=0.9)
            assert result.shape == (n, n), f"Failed for c={c}, d={d}"

    def test_dendrocycle_systematic_scan(self) -> None:
        """Systematically test dendrocycle across parameter space."""
        n = 400
        c_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        d_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

        for c in c_values:
            for d in d_values:
                if c + d > 1.0 or c + d < 0.0:
                    continue  # Skip invalid combinations

                topology = GraphTopology(dendrocycle_graph, {"c": c, "d": d, "seed": 42})
                weight = torch.empty(n, n)

                # Should create exactly n nodes regardless of rounding
                result = topology.initialize(weight, spectral_radius=0.9)
                assert result.shape == (n, n), f"Failed for c={c}, d={d}"


# ---------------------------------------------------------------------------
# MatrixTopology
# ---------------------------------------------------------------------------


class TestMatrixTopology:
    """MatrixTopology: matrix builders and in-place initializer callables."""

    def test_build_style_tensor(self) -> None:
        """A build-style callable returning a tensor fills the weight."""
        topology = MatrixTopology(block_diagonal, {"blocks": 2})
        weight = torch.empty(20, 20)
        topology.initialize(weight)

        # Off-diagonal blocks must be exactly zero
        assert torch.all(weight[:10, 10:] == 0)
        assert torch.all(weight[10:, :10] == 0)
        assert not torch.all(weight[:10, :10] == 0)

    def test_build_style_numpy(self) -> None:
        """A callable returning a NumPy array is coerced to a tensor."""
        topology = MatrixTopology(lambda n: np.eye(n, dtype=np.float64))
        weight = torch.empty(8, 8)
        topology.initialize(weight)

        assert torch.allclose(weight, torch.eye(8))
        assert weight.dtype == torch.float32

    def test_build_style_graph_return(self) -> None:
        """A callable returning a NetworkX graph is coerced to its adjacency."""
        topology = MatrixTopology(lambda n: nx.cycle_graph(n))
        weight = torch.empty(6, 6)
        topology.initialize(weight)

        # Cycle graph adjacency: each node connects to two neighbours
        assert torch.allclose(weight.sum(dim=0), torch.full((6,), 2.0))

    def test_inplace_style_torch_nn_init(self) -> None:
        """An in-place torch.nn.init callable is applied to the weight."""
        topology = MatrixTopology(torch.nn.init.orthogonal_)
        weight = torch.empty(16, 16)
        topology.initialize(weight)

        assert torch.allclose(weight @ weight.T, torch.eye(16), atol=1e-5)

    def test_spectral_radius_scaling(self) -> None:
        """Spectral radius scaling applies to matrix-built weights."""
        topology = MatrixTopology(lambda n: torch.randn(n, n))
        weight = torch.empty(32, 32)
        topology.initialize(weight, spectral_radius=0.7)

        radius = torch.max(torch.abs(torch.linalg.eigvals(weight))).item()
        assert radius == pytest.approx(0.7, abs=1e-4)

    def test_wrong_shape_raises(self) -> None:
        """A builder returning the wrong shape raises ValueError."""
        topology = MatrixTopology(lambda n: torch.randn(n, n + 1))
        with pytest.raises(ValueError, match="shape"):
            topology.initialize(torch.empty(10, 10))

    def test_invalid_return_type_raises(self) -> None:
        """A builder returning a non-matrix raises ValueError."""
        topology = MatrixTopology(lambda n: "not a matrix")
        with pytest.raises(ValueError, match="cannot"):
            topology.initialize(torch.empty(4, 4))

    def test_unusable_callable_raises_informative(self) -> None:
        """A callable matching neither convention raises an informative error."""

        def needs_three_positional(a: object, b: object, c: object) -> object:
            return a + b + c

        topology = MatrixTopology(needs_three_positional)
        with pytest.raises(ValueError, match="neither initializer convention"):
            topology.initialize(torch.empty(4, 4))

    def test_non_square_weight_raises(self) -> None:
        """Non-square targets raise ValueError."""
        topology = MatrixTopology(block_diagonal)
        with pytest.raises(ValueError, match="square"):
            topology.initialize(torch.empty(4, 6))


class TestMatrixRegistry:
    """register_matrix_topology + built-in matrix topologies."""

    def test_register_and_get(self) -> None:
        """A registered matrix topology resolves through get_topology."""

        @register_matrix_topology("_test_blockdiag", blocks=2)
        def _blockdiag(n: int, blocks: int = 2) -> torch.Tensor:
            return block_diagonal(n, blocks)

        topology = get_topology("_test_blockdiag")
        assert isinstance(topology, MatrixTopology)

        weight = torch.empty(10, 10)
        topology.initialize(weight)
        assert torch.all(weight[:5, 5:] == 0)

    def test_builtin_orthogonal(self) -> None:
        """The built-in 'orthogonal' matrix topology builds orthogonal weights."""
        topology = get_topology("orthogonal", seed=42)
        assert isinstance(topology, MatrixTopology)

        weight = torch.empty(24, 24)
        topology.initialize(weight)
        assert torch.allclose(weight @ weight.T, torch.eye(24), atol=1e-5)

    def test_orthogonal_seed_reproducible(self) -> None:
        """Seeded orthogonal initialization is reproducible."""
        w1, w2 = torch.empty(12, 12), torch.empty(12, 12)
        get_topology("orthogonal", seed=7).initialize(w1)
        get_topology("orthogonal", seed=7).initialize(w2)
        assert torch.equal(w1, w2)

    def test_duplicate_name_raises(self) -> None:
        """Registering an existing name raises ValueError."""
        with pytest.raises(ValueError, match="already registered"):
            register_matrix_topology("orthogonal")(lambda n: torch.eye(n))

    def test_graph_topologies_unaffected(self) -> None:
        """Graph-topology names still resolve to GraphTopology."""
        topology = get_topology("erdos_renyi", p=0.2, seed=1)
        assert isinstance(topology, GraphTopology)


class TestTopologyResolverCallables:
    """resolve_topology accepts bare callables and (callable, params) tuples."""

    def test_bare_callable_topology(self) -> None:
        """A bare callable resolves to MatrixTopology."""
        resolved = resolve_topology(block_diagonal)
        assert isinstance(resolved, MatrixTopology)

    def test_callable_tuple_topology(self) -> None:
        """A (callable, params) tuple resolves with the given kwargs."""
        resolved = resolve_topology((block_diagonal, {"blocks": 5}))
        assert isinstance(resolved, MatrixTopology)
        assert resolved.matrix_kwargs == {"blocks": 5}


class TestESNLayerTopologyIntegration:
    """End-to-end use of matrix topologies through ESNLayer."""

    def test_layer_with_callable_topology(self) -> None:
        """A (callable, params) topology spec drives the recurrent weights."""
        layer = ESNLayer(
            reservoir_size=20,
            feedback_size=3,
            topology=(block_diagonal, {"blocks": 2}),
            spectral_radius=0.9,
        )

        assert torch.all(layer.weight_hh[:10, 10:] == 0)
        radius = torch.max(torch.abs(torch.linalg.eigvals(layer.weight_hh))).item()
        assert radius == pytest.approx(0.9, abs=1e-4)

    def test_layer_with_torch_init_everywhere(self) -> None:
        """torch.nn.init callables work as topology and initializer specs."""
        layer = ESNLayer(
            reservoir_size=16,
            feedback_size=3,
            topology=torch.nn.init.orthogonal_,
            feedback_initializer=torch.nn.init.xavier_uniform_,
        )

        out = layer(torch.randn(2, 10, 3))
        assert out.shape == (2, 10, 16)

    def test_layer_with_named_matrix_topology(self) -> None:
        """A named matrix topology with params works through the layer."""
        layer = ESNLayer(
            reservoir_size=16,
            feedback_size=3,
            topology=("orthogonal", {"seed": 3}),
        )
        w = layer.weight_hh.data
        assert torch.allclose(w @ w.T, torch.eye(16), atol=1e-5)


# ---------------------------------------------------------------------------
# Seed reproducibility (issue #134)
# ---------------------------------------------------------------------------


class TestSeedReproducibility:
    """Reproducibility of topology/initializer weights via torch RNG and seed."""

    def test_graph_topology_reproducible_under_torch_manual_seed(self) -> None:
        """Back-to-back graph-topology layers under torch.manual_seed match.

        Acceptance criterion: ``create_rng(None)`` derives its NumPy seed from
        torch's global RNG, so the string-form graph topology is reproducible
        without any explicit ``seed``.
        """
        torch.manual_seed(0)
        a = ESNLayer(50, feedback_size=3, topology="erdos_renyi", spectral_radius=0.9)
        torch.manual_seed(0)
        b = ESNLayer(50, feedback_size=3, topology="erdos_renyi", spectral_radius=0.9)

        assert torch.equal(a.weight_hh, b.weight_hh)

    def test_matrix_topology_reproducible_under_torch_manual_seed(self) -> None:
        """Back-to-back matrix-topology layers under torch.manual_seed match."""
        torch.manual_seed(0)
        a = ESNLayer(40, feedback_size=3, topology="orthogonal", spectral_radius=0.9)
        torch.manual_seed(0)
        b = ESNLayer(40, feedback_size=3, topology="orthogonal", spectral_radius=0.9)

        assert torch.equal(a.weight_hh, b.weight_hh)

    def test_different_torch_seeds_give_different_graph_weights(self) -> None:
        """Different torch global seeds yield different graph-topology weights."""
        torch.manual_seed(0)
        a = ESNLayer(50, feedback_size=3, topology="erdos_renyi", spectral_radius=0.9)
        torch.manual_seed(123)
        b = ESNLayer(50, feedback_size=3, topology="erdos_renyi", spectral_radius=0.9)

        assert not torch.equal(a.weight_hh, b.weight_hh)

    def test_seed_kwarg_reproducible_without_tuple_form(self) -> None:
        """``ESNLayer(..., topology='erdos_renyi', seed=42)`` is reproducible.

        Acceptance criterion: a top-level ``seed`` argument makes the
        string-form topology reproducible without the ``(name, {'seed': ...})``
        tuple form, independent of the torch global RNG state.
        """
        torch.manual_seed(1)
        a = ESNLayer(50, feedback_size=3, topology="erdos_renyi", seed=42, spectral_radius=0.9)
        torch.manual_seed(2)
        b = ESNLayer(50, feedback_size=3, topology="erdos_renyi", seed=42, spectral_radius=0.9)

        assert torch.equal(a.weight_hh, b.weight_hh)

    def test_seed_kwarg_different_seeds_differ(self) -> None:
        """Different ``seed`` values yield different recurrent weights."""
        a = ESNLayer(50, feedback_size=3, topology="erdos_renyi", seed=42)
        b = ESNLayer(50, feedback_size=3, topology="erdos_renyi", seed=43)

        assert not torch.equal(a.weight_hh, b.weight_hh)

    def test_seed_kwarg_round_trips_to_topology(self) -> None:
        """A ``seed`` kwarg round-trips into the resolved topology kwargs."""
        topology = resolve_topology("erdos_renyi", seed=123)

        assert isinstance(topology, GraphTopology)
        assert topology.graph_kwargs["seed"] == 123

    def test_explicit_tuple_seed_wins_over_seed_kwarg(self) -> None:
        """An explicit seed inside a tuple spec overrides the ``seed`` kwarg."""
        a = ESNLayer(50, feedback_size=3, topology=("erdos_renyi", {"seed": 7}), seed=42)
        b = ESNLayer(50, feedback_size=3, topology=("erdos_renyi", {"seed": 7}), seed=999)

        assert torch.equal(a.weight_hh, b.weight_hh)

    def test_seed_kwarg_matrix_topology_reproducible(self) -> None:
        """The ``seed`` kwarg drives the orthogonal matrix topology too."""
        a = ESNLayer(40, feedback_size=3, topology="orthogonal", seed=5)
        b = ESNLayer(40, feedback_size=3, topology="orthogonal", seed=5)

        assert torch.equal(a.weight_hh, b.weight_hh)

    def test_resolve_topology_seed_skipped_for_pre_resolved_object(self) -> None:
        """A pre-resolved TopologyInitializer is returned untouched by seed."""
        topology = get_topology("erdos_renyi")
        resolved = resolve_topology(topology, seed=42)

        # Identity preserved and the seed kwarg is not injected (its registered
        # default of None is left intact rather than overwritten with 42).
        assert resolved is topology
        assert topology.graph_kwargs["seed"] is None

    def test_resolve_initializer_seed_round_trips(self) -> None:
        """A ``seed`` kwarg round-trips into a seed-accepting initializer."""
        initializer = resolve_initializer("random", seed=99)

        assert initializer.seed == 99  # type: ignore[attr-defined]

    def test_resolve_initializer_seed_ignored_when_unsupported(self) -> None:
        """A ``seed`` kwarg is silently ignored by initializers without seed."""
        # xavier_uniform_ takes (tensor, gain) — no seed param — must not raise.
        initializer = resolve_initializer(torch.nn.init.xavier_uniform_, seed=42)

        weight = torch.empty(8, 4)
        initializer.initialize(weight)  # type: ignore[union-attr]
        assert weight.shape == (8, 4)

    def test_seed_kwarg_reproduces_feedback_initializer(self) -> None:
        """The ``seed`` kwarg also makes seed-accepting feedback init reproducible."""
        a = ESNLayer(40, feedback_size=3, topology="zeros", feedback_initializer="random", seed=8)
        b = ESNLayer(40, feedback_size=3, topology="zeros", feedback_initializer="random", seed=8)

        assert torch.equal(a.weight_feedback, b.weight_feedback)


class TestResolverGeneratorSeed:
    """The resolvers accept a ``torch.Generator`` seed (reduced to its int)."""

    def test_resolve_topology_accepts_generator(self) -> None:
        """A generator seed round-trips into a seed-accepting graph topology."""
        gen = torch.Generator().manual_seed(321)
        resolved = resolve_topology("erdos_renyi", seed=gen)

        assert resolved.graph_kwargs["seed"] == 321  # type: ignore[attr-defined]

    def test_resolve_topology_generator_matches_int(self) -> None:
        """A generator seeded with N resolves to the same matrix as ``seed=N``."""
        gen = torch.Generator().manual_seed(55)
        from_gen = resolve_topology("erdos_renyi", seed=gen)
        from_int = resolve_topology("erdos_renyi", seed=55)

        w_gen = torch.empty(30, 30)
        w_int = torch.empty(30, 30)
        from_gen.initialize(w_gen)  # type: ignore[union-attr]
        from_int.initialize(w_int)  # type: ignore[union-attr]

        assert torch.equal(w_gen, w_int)

    def test_resolve_initializer_accepts_generator(self) -> None:
        """A generator seed round-trips into a seed-accepting initializer."""
        gen = torch.Generator().manual_seed(99)
        initializer = resolve_initializer("random", seed=gen)

        assert initializer.seed == 99  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Barabási-Albert generator correctness (issue #138)
# ---------------------------------------------------------------------------


class TestBarabasiAlbertGraph:
    """The BA generator must produce valid scale-free graphs.

    Regression guard for the multiset bug where ``rng.choice(targets,
    replace=False)`` over a degree-frequency multiset only guaranteed distinct
    *positions* (not distinct node ids), so a new node could draw the same id
    twice and end up with fewer than ``m`` neighbours, leaving the graph
    edge-deficient and not scale-free (issue #138).
    """

    def test_undirected_edge_count_exact_over_100_seeds(self) -> None:
        """Every undirected BA graph at n=30, m=3 has exactly 3 + 3*(n-3) edges.

        The BA model adds an ``m``-clique (``m*(m-1)//2`` edges) and then ``m``
        edges per remaining node, so the undirected edge count is fixed at
        ``m*(m-1)//2 + m*(n-m)``. For n=30, m=3 that is ``3 + 3*27 = 84``.
        """
        n, m = 30, 3
        expected = m * (m - 1) // 2 + m * (n - m)  # 84 == 3 + 3*(n-3)
        assert expected == 3 + 3 * (n - 3)

        for seed in range(100):
            graph = barabasi_albert_graph(n, m=m, directed=False, seed=seed)
            assert graph.number_of_edges() == expected, f"seed {seed} edge-deficient"

    def test_each_new_node_has_m_distinct_neighbors(self) -> None:
        """Each node added after the seed clique has exactly ``m`` distinct neighbours."""
        n, m = 40, 4
        for seed in range(20):
            graph = barabasi_albert_graph(n, m=m, directed=False, seed=seed)
            # Nodes 0..m-1 form the initial clique; nodes m..n-1 are grown in.
            for node in range(m, n):
                # An undirected node's neighbour set already de-duplicates ids,
                # but a missing edge would show up as degree < m here.
                assert graph.degree(node) >= m, f"node {node} under-connected (seed {seed})"
                # The m edges this node *created* must point at m distinct ids: a
                # repeated draw would leave its created-edge count below m, which
                # the exact total edge count cross-checks.
            assert graph.number_of_edges() == m * (m - 1) // 2 + m * (n - m)

    def test_degree_distribution_approximately_power_law(self) -> None:
        """On a large graph the degree distribution is heavy-tailed (scale-free).

        A valid BA graph has a power-law tail ``P(k) ~ k^-3``: a small fraction of
        hubs accumulate very high degree while most nodes stay near ``m``. The
        broken multiset generator collapsed toward a far narrower distribution, so
        we assert a heavy tail via the max/mean degree ratio and a sane mean.
        """
        n, m = 2000, 3
        graph = barabasi_albert_graph(n, m=m, directed=False, seed=0)

        degrees = np.array([d for _, d in graph.degree()], dtype=float)

        # Mean undirected degree is 2*E/n -> 2*m as n grows.
        mean_degree = degrees.mean()
        assert mean_degree == pytest.approx(2 * m, abs=0.1)

        # Every grown node attaches m distinct neighbours, so min degree >= m.
        assert degrees.min() >= m

        # Heavy tail: at least one hub far above the mean (uniform attachment
        # would keep this ratio small).
        assert degrees.max() / mean_degree > 5.0

    def test_seed_is_reproducible(self) -> None:
        """The same seed yields identical edge sets; different seeds differ."""
        a = barabasi_albert_graph(50, m=3, directed=False, seed=7)
        b = barabasi_albert_graph(50, m=3, directed=False, seed=7)
        c = barabasi_albert_graph(50, m=3, directed=False, seed=8)

        assert set(a.edges()) == set(b.edges())
        assert set(a.edges()) != set(c.edges())

    def test_directed_doubles_edges(self) -> None:
        """``directed=True`` adds both i->t and t->i, doubling the edge count."""
        n, m = 30, 3
        undirected_edges = m * (m - 1) // 2 + m * (n - m)
        graph = barabasi_albert_graph(n, m=m, directed=True, seed=0)

        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_edges() == 2 * undirected_edges

    def test_m_equals_one_is_a_tree(self) -> None:
        """``m=1`` produces a connected tree (n-1 edges), the BA chain limit."""
        n = 25
        graph = barabasi_albert_graph(n, m=1, directed=False, seed=3)

        assert graph.number_of_edges() == n - 1
        assert nx.is_connected(graph)

    def test_invalid_m_raises(self) -> None:
        """``m < 1`` or ``m >= n`` raises ValueError."""
        with pytest.raises(ValueError, match="m must be"):
            barabasi_albert_graph(10, m=0)
        with pytest.raises(ValueError, match="m must be"):
            barabasi_albert_graph(10, m=10)
