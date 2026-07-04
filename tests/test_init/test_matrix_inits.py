"""Deterministic-minimal and structured matrix initializers (RC.jl parity).

Pins down the eight recurrent-matrix builders ported from
``ReservoirComputing.jl`` (``src/inits/inits_reservoir.jl`` and
``inits_components.jl``):

- deterministic-minimal (Rodan & Tiňo): ``delay_line``,
  ``delay_line_backward``, ``simple_cycle``, ``cycle_jumps``,
  ``self_loop_cycle``,
- structured: ``toeplitz``, ``band``, ``block_diagonal``.

For each: the exact nonzero placement at small ``n`` (matched to the RC.jl
component loops), registry resolution by name, and end-to-end use through
``ESNLayer`` with spectral-radius scaling.
"""

import warnings

import pytest
import torch

import resdag.init.matrices  # noqa: F401 - triggers topology registration
from resdag.init.matrices import (
    band,
    block_diagonal,
    cycle_jumps,
    delay_line,
    delay_line_backward,
    self_loop_cycle,
    simple_cycle,
    toeplitz,
)
from resdag.init.topology import MatrixTopology, get_topology
from resdag.init.utils import resolve_topology
from resdag.layers import ESNLayer

MINIMAL_NAMES = [
    "delay_line",
    "delay_line_backward",
    "simple_cycle",
    "cycle_jumps",
    "self_loop_cycle",
]
STRUCTURED_NAMES = ["toeplitz", "band", "block_diagonal"]
ALL_NAMES = MINIMAL_NAMES + STRUCTURED_NAMES

# Topologies whose raw matrix has a nonzero spectral radius, so the layer-level
# ``spectral_radius`` rescale hits its target. ``delay_line`` is nilpotent
# (strictly sub-diagonal -> all eigenvalues zero) and is excluded.
SCALABLE_NAMES = [n for n in ALL_NAMES if n != "delay_line"]


# ---------------------------------------------------------------------------
# Registry + public API
# ---------------------------------------------------------------------------


class TestRegistration:
    """Each new builder registers and resolves as a MatrixTopology."""

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_get_topology_resolves(self, name: str) -> None:
        """get_topology returns a MatrixTopology for every new name."""
        topology = get_topology(name)
        assert isinstance(topology, MatrixTopology)

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_resolve_topology_by_name(self, name: str) -> None:
        """resolve_topology finds each name (string spec)."""
        assert isinstance(resolve_topology(name), MatrixTopology)

    def test_public_api_exports(self) -> None:
        """Every builder is exported from resdag.init.matrices.__all__."""
        import resdag.init.matrices as matrices_pkg

        for symbol in [
            "delay_line",
            "delay_line_backward",
            "simple_cycle",
            "cycle_jumps",
            "self_loop_cycle",
            "toeplitz",
            "band",
            "block_diagonal",
        ]:
            assert symbol in matrices_pkg.__all__, symbol
            assert hasattr(matrices_pkg, symbol), symbol


# ---------------------------------------------------------------------------
# Exact nonzero patterns (matched to RC.jl component loops, n=6)
# ---------------------------------------------------------------------------


class TestDeterministicMinimalPatterns:
    """n=6 placements match the RC.jl inits_components.jl loops exactly."""

    def test_delay_line_subdiagonal(self) -> None:
        """delay_line! -> M[i+shift, i] = weight (single sub-diagonal)."""
        m = delay_line(6, weight=0.1, shift=1)
        expected = torch.zeros(6, 6)
        idx = torch.arange(5)
        expected[idx + 1, idx] = 0.1
        assert torch.equal(m, expected)

    def test_delay_line_shift(self) -> None:
        """shift=2 moves the delay line to the second sub-diagonal."""
        m = delay_line(6, weight=1.0, shift=2)
        nz = set(zip(*torch.nonzero(m, as_tuple=True)))  # type: ignore[arg-type]
        assert {(int(i), int(j)) for i, j in nz} == {(2, 0), (3, 1), (4, 2), (5, 3)}

    def test_delay_line_backward_sub_and_super(self) -> None:
        """delay_line! + backward_connection! -> sub- and super-diagonal."""
        m = delay_line_backward(6, weight=0.1, backward_weight=0.5, shift=1, backward_shift=1)
        expected = torch.zeros(6, 6)
        idx = torch.arange(5)
        expected[idx + 1, idx] = 0.1  # sub-diagonal (delay)
        expected[idx, idx + 1] = 0.5  # super-diagonal (backward)
        assert torch.equal(m, expected)

    def test_simple_cycle_subdiagonal_and_corner(self) -> None:
        """simple_cycle! -> sub-diagonal + wrap-around corner M[0, n-1]."""
        m = simple_cycle(6, weight=0.1)
        expected = torch.zeros(6, 6)
        idx = torch.arange(5)
        expected[idx + 1, idx] = 0.1
        expected[0, 5] = 0.1  # RC.jl M[1, end]
        assert torch.equal(m, expected)

    def test_cycle_jumps_pattern(self) -> None:
        """cycle_jumps -> simple_cycle + symmetric jumps at the RC.jl walk indices.

        For n=6, jump_size=3 (divisible) the add_jumps! walk connects nodes 0
        and 3 symmetrically, on top of the simple cycle.
        """
        m = cycle_jumps(6, cycle_weight=0.1, jump_weight=0.5, jump_size=3)
        # Simple cycle entries.
        expected = simple_cycle(6, weight=0.1)
        # Symmetric jump 0 <-> 3.
        expected[0, 3] = 0.5
        expected[3, 0] = 0.5
        assert torch.equal(m, expected)

    def test_cycle_jumps_jump_indices_nondivisible(self) -> None:
        """Non-divisible n drops the final partial jump (RC.jl walk semantics).

        n=7, jump_size=3: the walk is 0 -> 3 -> 6, then 6+3=9 > 6 stops without
        wrapping, so only the 0<->3 and 3<->6 symmetric jumps are added.
        """
        m = cycle_jumps(7, cycle_weight=0.1, jump_weight=0.5, jump_size=3)
        jumps = {(int(i), int(j)) for i, j in zip(*torch.where(m == 0.5))}  # type: ignore[arg-type]
        assert jumps == {(0, 3), (3, 0), (3, 6), (6, 3)}

    def test_self_loop_cycle_diagonal_plus_cycle(self) -> None:
        """selfloop_cycle -> self_loop! (diagonal) + simple_cycle!."""
        m = self_loop_cycle(6, cycle_weight=0.1, self_loop_weight=0.2)
        expected = simple_cycle(6, weight=0.1)
        diag = torch.arange(6)
        expected[diag, diag] = 0.2
        assert torch.equal(m, expected)


class TestStructuredPatterns:
    """n=6 placements for the structured builders."""

    def test_toeplitz_constant_per_diagonal(self) -> None:
        """Flat Toeplitz (decay=1) is constant on every diagonal."""
        m = toeplitz(6, weight=0.1, decay=1.0)
        assert torch.allclose(m, torch.full((6, 6), 0.1))

    def test_toeplitz_decay_and_symmetry(self) -> None:
        """decay applies geometrically to |offset|; result is symmetric."""
        m = toeplitz(6, weight=1.0, decay=0.5)
        assert m[0, 0] == pytest.approx(1.0)
        assert m[0, 1] == pytest.approx(0.5)
        assert m[0, 2] == pytest.approx(0.25)
        assert torch.equal(m, m.T)  # |offset| is symmetric

    def test_toeplitz_bandwidth_zeros_outside_band(self) -> None:
        """bandwidth restricts to |offset| <= bandwidth."""
        m = toeplitz(6, weight=0.1, decay=1.0, bandwidth=1)
        assert m[0, 2] == 0.0
        assert m[0, 1] == pytest.approx(0.1)
        assert m[2, 2] == pytest.approx(0.1)

    def test_band_tridiagonal(self) -> None:
        """bandwidth=1 is a tridiagonal constant band."""
        m = band(6, weight=0.1, bandwidth=1)
        for i in range(6):
            for j in range(6):
                if abs(i - j) <= 1:
                    assert m[i, j] == pytest.approx(0.1)
                else:
                    assert m[i, j] == 0.0
        assert torch.equal(m, m.T)

    def test_band_zero_bandwidth_is_diagonal(self) -> None:
        """bandwidth=0 degenerates to a scaled identity."""
        m = band(6, weight=0.3, bandwidth=0)
        assert torch.equal(m, 0.3 * torch.eye(6))

    def test_block_diagonal_constant_blocks(self) -> None:
        """RC.jl block_diagonal -> solid constant-weight diagonal blocks."""
        m = block_diagonal(6, weight=2.0, block_size=2)
        expected = torch.zeros(6, 6)
        for b in range(3):
            s = b * 2
            expected[s : s + 2, s : s + 2] = 2.0
        assert torch.equal(m, expected)

    def test_block_diagonal_leftover_rows_zero(self) -> None:
        """A block_size that does not divide n leaves trailing rows/cols zero."""
        m = block_diagonal(7, weight=1.0, block_size=3)  # 2 blocks, 1 leftover
        assert torch.all(m[6, :] == 0)
        assert torch.all(m[:, 6] == 0)
        assert torch.all(m[:3, :3] == 1.0)
        assert torch.all(m[3:6, 3:6] == 1.0)

    def test_block_diagonal_default_is_identity(self) -> None:
        """RC.jl defaults (weight=1, block_size=1) give the identity."""
        m = block_diagonal(6)
        assert torch.equal(m, torch.eye(6))


# ---------------------------------------------------------------------------
# Output contract (dtype, shape, determinism, validation)
# ---------------------------------------------------------------------------


class TestOutputContract:
    """dtype/shape match orthogonal.py/fsi.py; builders are deterministic."""

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_dtype_and_shape(self, name: str) -> None:
        """Every builder returns an (n, n) float32 tensor (like orthogonal/fsi)."""
        w = torch.zeros(12, 12)
        get_topology(name).initialize(w, spectral_radius=None)
        assert w.shape == (12, 12)

    @pytest.mark.parametrize(
        "builder", [delay_line, delay_line_backward, simple_cycle, cycle_jumps]
    )
    def test_builder_returns_float32(self, builder) -> None:  # type: ignore[no-untyped-def]
        """Direct builder calls return float32 tensors."""
        m = builder(8)
        assert isinstance(m, torch.Tensor)
        assert m.dtype == torch.float32
        assert m.shape == (8, 8)

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_deterministic_no_seed(self, name: str) -> None:
        """Structural builders are bit-identical across calls with no seed."""
        w1, w2 = torch.zeros(10, 10), torch.zeros(10, 10)
        get_topology(name).initialize(w1, spectral_radius=None)
        get_topology(name).initialize(w2, spectral_radius=None)
        assert torch.equal(w1, w2)

    def test_delay_line_shift_out_of_range(self) -> None:
        """shift must satisfy 1 <= shift < n."""
        with pytest.raises(ValueError, match="shift"):
            delay_line(5, shift=5)

    def test_cycle_jumps_jump_size_out_of_range(self) -> None:
        """jump_size must satisfy 1 <= jump_size < n."""
        with pytest.raises(ValueError, match="jump_size"):
            cycle_jumps(5, jump_size=5)

    def test_band_bandwidth_out_of_range(self) -> None:
        """bandwidth must satisfy 0 <= bandwidth < n."""
        with pytest.raises(ValueError, match="bandwidth"):
            band(5, bandwidth=5)

    def test_block_size_out_of_range(self) -> None:
        """block_size must satisfy 1 <= block_size <= n."""
        with pytest.raises(ValueError, match="block_size"):
            block_diagonal(5, block_size=6)


# ---------------------------------------------------------------------------
# End-to-end through ESNLayer (spectral-radius scaling)
# ---------------------------------------------------------------------------


class TestESNLayerIntegration:
    """ESNLayer builds each topology by name and scales the spectral radius."""

    @pytest.mark.parametrize("name", SCALABLE_NAMES)
    def test_layer_scales_spectral_radius(self, name: str) -> None:
        """A scalable topology rescales to the requested spectral radius."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no prescaled-override warning
            layer = ESNLayer(
                reservoir_size=200, feedback_size=3, topology=name, spectral_radius=0.9
            )
        radius = torch.max(torch.abs(torch.linalg.eigvals(layer.weight_hh))).item()
        assert radius == pytest.approx(0.9, abs=1e-3)

    def test_delay_line_is_nilpotent(self) -> None:
        """delay_line is nilpotent: radius 0, unaffected by spectral_radius.

        A strictly sub-diagonal matrix has all-zero eigenvalues, so the shared
        rescale is a no-op (current radius ~ 0). The layer still builds and runs.
        """
        layer = ESNLayer(
            reservoir_size=64, feedback_size=3, topology="delay_line", spectral_radius=0.9
        )
        radius = torch.max(torch.abs(torch.linalg.eigvals(layer.weight_hh))).item()
        assert radius == pytest.approx(0.0, abs=1e-6)
        # Non-zero sub-diagonal weights survive (structure preserved).
        assert torch.count_nonzero(layer.weight_hh) == 63

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_layer_forward_runs(self, name: str) -> None:
        """Every topology drives a working forward pass through the layer."""
        layer = ESNLayer(reservoir_size=32, feedback_size=3, topology=name)
        out = layer(torch.randn(2, 8, 3))
        assert out.shape == (2, 8, 32)

    def test_layer_with_param_tuple(self) -> None:
        """(name, params) spec threads kwargs through to the builder."""
        layer = ESNLayer(
            reservoir_size=20,
            feedback_size=3,
            topology=("block_diagonal", {"weight": 1.0, "block_size": 10}),
            spectral_radius=0.9,
        )
        # Two 10x10 blocks; off-block entries stay zero.
        assert torch.all(layer.weight_hh[:10, 10:] == 0)
        assert torch.all(layer.weight_hh[10:, :10] == 0)
