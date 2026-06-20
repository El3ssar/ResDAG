"""
Unit tests for NGCell and NGReservoir.

Tests cover:
- Construction and feature_dim / state_size computation
- Forward pass output shapes
- Correctness of delay-tap extraction (O_lin)
- Correctness of polynomial monomials (O_nonlin)
- include_constant / include_linear flags
- k=1 edge case (no delay buffer)
- s>1 edge case (non-unit tap spacing)
- State management API (reset, get, set, set_random)
- Gradient flow
- Warning for large feature_dim
"""

import itertools
import math
import warnings

import pytest
import torch

from resdag.layers import NGCell, NGReservoir

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def n_monomials(D: int, p: int) -> int:
    """C(D+p-1, p)."""
    return math.comb(D + p - 1, p)


def expected_feature_dim(
    input_dim: int,
    k: int,
    p: int,
    include_constant: bool = True,
    include_linear: bool = True,
) -> int:
    D = input_dim * k
    return int(include_constant) + int(include_linear) * D + n_monomials(D, p)


# ---------------------------------------------------------------------------
# NGCell — construction
# ---------------------------------------------------------------------------


class TestNGCellConstruction:
    """NGCell instantiation and dimension computation."""

    def test_default_params(self) -> None:
        cell = NGCell(input_dim=3)
        assert cell.input_dim == 3
        assert cell.k == 2
        assert cell.s == 1
        assert cell.p == 2
        assert cell.include_constant is True
        assert cell.include_linear is True

    def test_feature_dim_lorenz63(self) -> None:
        """Lorenz63: d=3, k=2, p=2, const+linear → feature_dim=28."""
        cell = NGCell(input_dim=3, k=2, s=1, p=2, include_constant=True, include_linear=True)
        # D=6, n_nonlin=C(7,2)=21, total=1+6+21=28
        assert cell.feature_dim == 28

    def test_feature_dim_double_scroll(self) -> None:
        """Double-scroll: d=2, k=2, p=3, no-const + linear → feature_dim=24."""
        cell = NGCell(input_dim=2, k=2, s=1, p=3, include_constant=False, include_linear=True)
        # D=4, n_nonlin=C(6,3)=20, total=0+4+20=24
        assert cell.feature_dim == 24

    @pytest.mark.parametrize(
        "input_dim, k, p, const, lin",
        [
            (3, 1, 2, True, True),
            (3, 2, 2, True, True),
            (3, 3, 2, False, True),
            (5, 2, 3, True, False),
            (2, 4, 2, False, False),
        ],
    )
    def test_feature_dim_formula(
        self, input_dim: int, k: int, p: int, const: bool, lin: bool
    ) -> None:
        cell = NGCell(input_dim=input_dim, k=k, p=p, include_constant=const, include_linear=lin)
        assert cell.feature_dim == expected_feature_dim(input_dim, k, p, const, lin)

    def test_state_size_k1(self) -> None:
        cell = NGCell(input_dim=3, k=1)
        assert cell.state_size == 0

    def test_state_size_k2_s1(self) -> None:
        cell = NGCell(input_dim=3, k=2, s=1)
        assert cell.state_size == 1

    def test_state_size_k3_s2(self) -> None:
        cell = NGCell(input_dim=3, k=3, s=2)
        assert cell.state_size == 4  # (3-1)*2 = 4

    def test_monomial_indices_shape(self) -> None:
        cell = NGCell(input_dim=3, k=2, p=2)
        D = 6
        expected_n = n_monomials(D, 2)
        assert cell.monomial_indices.shape == (expected_n, 2)

    def test_monomial_indices_content(self) -> None:
        """Verify monomial indices match itertools.combinations_with_replacement."""
        cell = NGCell(input_dim=2, k=1, p=2)
        D = 2
        expected = [list(combo) for combo in itertools.combinations_with_replacement(range(D), 2)]
        actual = cell.monomial_indices.tolist()
        assert actual == expected

    def test_delay_indices_k1(self) -> None:
        cell = NGCell(input_dim=3, k=1)
        assert cell.delay_indices.shape == (0,)

    def test_delay_indices_k2_s1(self) -> None:
        cell = NGCell(input_dim=3, k=2, s=1)
        # j=1: (k-1-j)*s = 0
        assert cell.delay_indices.tolist() == [0]

    def test_delay_indices_k3_s2(self) -> None:
        cell = NGCell(input_dim=3, k=3, s=2)
        # j=1: (3-1-1)*2=2; j=2: (3-1-2)*2=0
        assert cell.delay_indices.tolist() == [2, 0]

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            NGCell(input_dim=3, k=0)

    def test_invalid_s_raises(self) -> None:
        with pytest.raises(ValueError, match="s must be >= 1"):
            NGCell(input_dim=3, s=0)

    def test_invalid_p_raises(self) -> None:
        with pytest.raises(ValueError, match="p must be >= 1"):
            NGCell(input_dim=3, p=0)

    def test_large_feature_dim_warning(self) -> None:
        # d=10, k=5, p=4 → D=50, n_monomials=C(53,4)=292,825 → >10000
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            NGCell(input_dim=10, k=5, p=4)
        assert len(w) == 1
        assert "10,000" in str(w[0].message)


# ---------------------------------------------------------------------------
# NGCell — init_state
# ---------------------------------------------------------------------------


class TestNGCellInitState:
    def test_init_state_shape_k2(self) -> None:
        cell = NGCell(input_dim=3, k=2, s=1)
        state = cell.init_state(batch_size=4, device="cpu", dtype=torch.float32)
        assert state.shape == (4, 1, 3)
        assert torch.all(state == 0)

    def test_init_state_shape_k1(self) -> None:
        cell = NGCell(input_dim=3, k=1)
        state = cell.init_state(batch_size=2, device="cpu", dtype=torch.float32)
        assert state.shape == (2, 0, 3)

    def test_init_state_dtype(self) -> None:
        cell = NGCell(input_dim=3)
        state = cell.init_state(1, "cpu", torch.float64)
        assert state.dtype == torch.float64


# ---------------------------------------------------------------------------
# NGCell — forward output shapes
# ---------------------------------------------------------------------------


class TestNGCellForwardShape:
    def test_output_shape_k2_p2(self) -> None:
        cell = NGCell(input_dim=3, k=2, p=2)
        x = torch.randn(4, 3)
        state = cell.init_state(4, "cpu", torch.float32)
        features, new_state = cell([x], state)
        assert features.shape == (4, cell.feature_dim)
        assert new_state.shape == (4, cell.state_size, 3)

    def test_output_shape_k1(self) -> None:
        cell = NGCell(input_dim=5, k=1, p=2)
        x = torch.randn(2, 5)
        state = cell.init_state(2, "cpu", torch.float32)
        features, new_state = cell([x], state)
        assert features.shape == (2, cell.feature_dim)
        assert new_state.shape == (2, 0, 5)

    def test_output_shape_k3_s2(self) -> None:
        cell = NGCell(input_dim=3, k=3, s=2, p=2)
        x = torch.randn(8, 3)
        state = cell.init_state(8, "cpu", torch.float32)
        features, new_state = cell([x], state)
        assert features.shape == (8, cell.feature_dim)
        assert new_state.shape == (8, 4, 3)

    @pytest.mark.parametrize("batch", [1, 3, 16])
    def test_various_batch_sizes(self, batch: int) -> None:
        cell = NGCell(input_dim=4, k=2, p=2)
        x = torch.randn(batch, 4)
        state = cell.init_state(batch, "cpu", torch.float32)
        features, _ = cell([x], state)
        assert features.shape == (batch, cell.feature_dim)


# ---------------------------------------------------------------------------
# NGCell — delay tap correctness
# ---------------------------------------------------------------------------


class TestNGCellDelayTaps:
    """Verify O_lin contains correct time-delayed vectors."""

    def test_o_lin_k2_s1_after_one_step(self) -> None:
        """After 1 step from zeros: O_lin = [x0, zeros]."""
        cell = NGCell(input_dim=2, k=2, s=1, include_constant=False, include_linear=True, p=2)
        x0 = torch.tensor([[1.0, 2.0]])
        state = cell.init_state(1, "cpu", torch.float32)  # (1, 1, 2) all zeros

        features, new_state = cell([x0], state)

        # The linear part lives at features[:, :D] when include_constant=False
        D = 2 * 2
        o_lin = features[:, :D]
        # O_lin = [x0, X_{i-1}=zeros] = [1, 2, 0, 0]
        expected_o_lin = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
        assert torch.allclose(o_lin, expected_o_lin)

    def test_o_lin_k2_s1_after_two_steps(self) -> None:
        """After step 0 then step 1: step 1's O_lin = [x1, x0]."""
        cell = NGCell(input_dim=2, k=2, s=1, include_constant=False, include_linear=True, p=2)
        x0 = torch.tensor([[1.0, 2.0]])
        x1 = torch.tensor([[3.0, 4.0]])
        state = cell.init_state(1, "cpu", torch.float32)

        _, state = cell([x0], state)
        features, _ = cell([x1], state)

        D = 4
        o_lin = features[:, :D]
        # O_lin = [x1, x0] = [3, 4, 1, 2]
        expected = torch.tensor([[3.0, 4.0, 1.0, 2.0]])
        assert torch.allclose(o_lin, expected)

    def test_o_lin_k3_s2_sequential(self) -> None:
        """k=3, s=2: O_lin at step i = [x_i, x_{i-2}, x_{i-4}]."""
        d = 2
        cell = NGCell(input_dim=d, k=3, s=2, include_constant=False, include_linear=True, p=2)
        # Feed 5 steps: X0..X4, then check step 4's O_lin
        xs = [torch.tensor([[float(i), float(i + 10)]]) for i in range(5)]
        state = cell.init_state(1, "cpu", torch.float32)

        for i, x in enumerate(xs):
            features, state = cell([x], state)

        # At step 4 (i=4): O_lin = [x4, x2, x0]
        D = d * 3
        o_lin = features[:, :D]
        expected = torch.tensor([[4.0, 14.0, 2.0, 12.0, 0.0, 10.0]])
        assert torch.allclose(o_lin, expected)

    def test_buffer_update_preserves_history(self) -> None:
        """Buffer shifts correctly: new state = old[1:] + current x."""
        cell = NGCell(input_dim=2, k=2, s=1)
        x0 = torch.tensor([[1.0, 0.0]])
        x1 = torch.tensor([[2.0, 0.0]])
        state = cell.init_state(1, "cpu", torch.float32)

        _, s1 = cell([x0], state)
        assert torch.allclose(s1, torch.tensor([[[1.0, 0.0]]]))  # buffer now holds x0

        _, s2 = cell([x1], s1)
        assert torch.allclose(s2, torch.tensor([[[2.0, 0.0]]]))  # buffer now holds x1


# ---------------------------------------------------------------------------
# NGCell — monomial correctness
# ---------------------------------------------------------------------------


class TestNGCellMonomials:
    """Verify O_nonlin contains correct polynomial monomials."""

    def test_monomials_p2_d2(self) -> None:
        """d=2, k=1, p=2: monomials are x0^2, x0*x1, x1^2."""
        cell = NGCell(input_dim=2, k=1, p=2, include_constant=False, include_linear=False)
        x = torch.tensor([[3.0, 4.0]])
        state = cell.init_state(1, "cpu", torch.float32)

        features, _ = cell([x], state)
        # Expected: [9, 12, 16]
        expected = torch.tensor([[9.0, 12.0, 16.0]])
        assert torch.allclose(features, expected)

    def test_monomials_p2_match_combinations(self) -> None:
        """All monomials match itertools.combinations_with_replacement."""
        d = 3
        cell = NGCell(input_dim=d, k=1, p=2, include_constant=False, include_linear=False)
        x = torch.randn(1, d)
        state = cell.init_state(1, "cpu", torch.float32)

        features, _ = cell([x], state)
        x_vals = x[0]

        expected_vals = [
            x_vals[i] * x_vals[j] for i, j in itertools.combinations_with_replacement(range(d), 2)
        ]
        expected = torch.tensor([expected_vals])
        assert torch.allclose(features, expected, atol=1e-6)

    def test_monomials_p3(self) -> None:
        """Degree-3 monomials: x^3 when d=1."""
        cell = NGCell(input_dim=1, k=1, p=3, include_constant=False, include_linear=False)
        x = torch.tensor([[2.0]])
        state = cell.init_state(1, "cpu", torch.float32)

        features, _ = cell([x], state)
        # Only monomial: x^3 = 8
        assert torch.allclose(features, torch.tensor([[8.0]]))

    def test_const_and_linear_excluded(self) -> None:
        """With include_constant=False and include_linear=False, output = monomials only."""
        cell = NGCell(input_dim=2, k=1, p=2, include_constant=False, include_linear=False)
        assert cell.feature_dim == n_monomials(2, 2)

        x = torch.tensor([[2.0, 3.0]])
        state = cell.init_state(1, "cpu", torch.float32)
        features, _ = cell([x], state)
        assert features.shape == (1, cell.feature_dim)

    def test_constant_prepended(self) -> None:
        """First element of O_total is 1.0 when include_constant=True."""
        cell = NGCell(input_dim=3, k=1, p=2, include_constant=True, include_linear=True)
        x = torch.randn(4, 3)
        state = cell.init_state(4, "cpu", torch.float32)
        features, _ = cell([x], state)
        assert torch.all(features[:, 0] == 1.0)

    def test_linear_part_position(self) -> None:
        """With constant, linear part occupies columns [1, 1+D)."""
        d, k = 3, 2
        D = d * k
        cell = NGCell(input_dim=d, k=k, p=2, include_constant=True, include_linear=True)
        x0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        x1 = torch.randn(2, d)
        state = cell.init_state(2, "cpu", torch.float32)
        _, state = cell([x0], state)
        features, _ = cell([x1], state)
        # Column 0 is constant, columns [1, 1+D) are linear
        o_lin_from_features = features[:, 1 : 1 + D]
        # Manually build expected o_lin: [x1, x0]
        expected_o_lin = torch.cat([x1, x0], dim=-1)
        assert torch.allclose(o_lin_from_features, expected_o_lin)


# ---------------------------------------------------------------------------
# NGCell — p=1 must not duplicate the linear block (issue #151)
# ---------------------------------------------------------------------------


class TestNGCellP1NoDuplicateColumns:
    """Regression tests for issue #151.

    With ``p == 1`` the degree-1 monomials are exactly the columns of
    ``O_lin``.  Emitting both blocks duplicated every linear column and made
    the readout design matrix rank-deficient.  The nonlinear block must be
    dropped when ``p == 1`` and ``include_linear`` is ``True``.
    """

    def test_no_duplicate_columns_p1_with_linear(self) -> None:
        """``NGCell(p=1, include_linear=True)`` emits no duplicate columns."""
        cell = NGCell(input_dim=3, k=2, p=1, include_constant=True, include_linear=True)
        x = torch.randn(4, 3)
        # Use a couple of steps so the delay buffer is non-trivially filled.
        state = cell.init_state(4, "cpu", torch.float32)
        _, state = cell([torch.randn(4, 3)], state)
        features, _ = cell([x], state)

        # No two feature columns may be identical (constant col is unique).
        cols = features.t()  # (feature_dim, batch)
        for i in range(cols.shape[0]):
            for j in range(i + 1, cols.shape[0]):
                assert not torch.allclose(
                    cols[i], cols[j]
                ), f"columns {i} and {j} are identical for p=1"

    def test_feature_dim_p1_with_linear_no_nonlin_block(self) -> None:
        """``feature_dim`` drops the monomial block for p=1 + include_linear."""
        # D = input_dim * k = 6. With the duplicate fix: 1 (const) + 6 (lin).
        cell = NGCell(input_dim=3, k=2, p=1, include_constant=True, include_linear=True)
        D = 3 * 2
        assert cell.feature_dim == 1 + D  # no extra n_monomials term

    def test_o_lin_and_nonlin_blocks_not_identical_p1(self) -> None:
        """O_lin and any nonlinear block must not be identical for p=1.

        Before the fix the output was ``[const || O_lin || O_nonlin]`` with
        ``O_nonlin == O_lin`` (a verbatim duplicate of the linear block).
        After the fix there is no separate nonlinear block, so the output is
        exactly ``[const || O_lin]`` with nothing trailing the linear columns.
        """
        D = 3 * 2
        cell = NGCell(input_dim=3, k=2, p=1, include_constant=True, include_linear=True)
        x = torch.randn(2, 3)
        state = cell.init_state(2, "cpu", torch.float32)
        _, state = cell([torch.randn(2, 3)], state)
        features, _ = cell([x], state)

        # The only block of D columns is O_lin; nothing trails it (the bug
        # produced a trailing copy of O_lin here).
        assert features.shape[1] == 1 + D
        trailing = features[:, 1 + D :]
        assert trailing.shape[1] == 0

    def test_p1_without_linear_keeps_degree1_monomials(self) -> None:
        """``p=1, include_linear=False`` keeps the degree-1 monomials."""
        # With no linear block, the degree-1 monomials are the only delay
        # features and do not overlap anything.
        D = 3 * 2
        cell = NGCell(input_dim=3, k=2, p=1, include_constant=False, include_linear=False)
        assert cell.feature_dim == D  # exactly the degree-1 monomials

        x = torch.randn(2, 3)
        state = cell.init_state(2, "cpu", torch.float32)
        _, state = cell([torch.randn(2, 3)], state)
        features, _ = cell([x], state)
        assert features.shape == (2, D)

    def test_reservoir_full_rank_design_matrix_p1(self) -> None:
        """The NGReservoir feature matrix is full column rank for p=1."""
        layer = NGReservoir(input_dim=3, k=2, p=1, include_constant=True, include_linear=True)
        x = torch.randn(1, 64, 3)
        feats = layer(x)[0]  # (time, feature_dim)
        rank = torch.linalg.matrix_rank(feats.double())
        assert int(rank) == layer.feature_dim


# ---------------------------------------------------------------------------
# NGCell — no learnable parameters
# ---------------------------------------------------------------------------


class TestNGCellNoParams:
    def test_no_parameters(self) -> None:
        cell = NGCell(input_dim=3, k=2, p=2)
        assert list(cell.parameters()) == []

    def test_has_buffers(self) -> None:
        cell = NGCell(input_dim=3, k=2, p=2)
        buffers = dict(cell.named_buffers())
        assert "monomial_indices" in buffers
        assert "delay_indices" in buffers


# ---------------------------------------------------------------------------
# NGReservoir — construction and properties
# ---------------------------------------------------------------------------


class TestNGReservoirConstruction:
    def test_default_construction(self) -> None:
        layer = NGReservoir(input_dim=3)
        assert layer.input_dim == 3
        assert layer.feature_dim == 28
        assert layer.state_size == 1
        assert layer.warmup_length == 1
        assert layer.state is None

    def test_delegate_to_cell(self) -> None:
        layer = NGReservoir(input_dim=3, k=3, s=2, p=2)
        assert layer.state_size == layer.cell.state_size
        assert layer.feature_dim == layer.cell.feature_dim
        assert layer.warmup_length == layer.cell.state_size

    def test_getattr_delegation(self) -> None:
        """Unknown attributes delegate to cell."""
        layer = NGReservoir(input_dim=3)
        assert layer.k == 2
        assert layer.s == 1
        assert layer.p == 2


# ---------------------------------------------------------------------------
# NGReservoir — forward pass shapes
# ---------------------------------------------------------------------------


class TestNGReservoirForwardShape:
    def test_output_shape(self) -> None:
        layer = NGReservoir(input_dim=3, k=2, p=2)
        x = torch.randn(4, 50, 3)
        out = layer(x)
        assert out.shape == (4, 50, layer.feature_dim)

    def test_output_shape_k1(self) -> None:
        layer = NGReservoir(input_dim=5, k=1, p=2)
        x = torch.randn(2, 30, 5)
        out = layer(x)
        assert out.shape == (2, 30, layer.feature_dim)

    def test_output_shape_k3_s2(self) -> None:
        layer = NGReservoir(input_dim=3, k=3, s=2, p=2)
        x = torch.randn(1, 20, 3)
        out = layer(x)
        assert out.shape == (1, 20, layer.feature_dim)

    def test_non_3d_input_raises(self) -> None:
        layer = NGReservoir(input_dim=3)
        with pytest.raises(ValueError, match="3D"):
            layer(torch.randn(4, 3))

    def test_state_updated_after_forward(self) -> None:
        layer = NGReservoir(input_dim=3)
        x = torch.randn(2, 10, 3)
        layer(x)
        assert layer.state is not None
        assert layer.state.shape == (2, layer.state_size, 3)

    @pytest.mark.parametrize("batch", [1, 4, 16])
    def test_various_batch_sizes(self, batch: int) -> None:
        layer = NGReservoir(input_dim=3)
        x = torch.randn(batch, 10, 3)
        out = layer(x)
        assert out.shape == (batch, 10, layer.feature_dim)


# ---------------------------------------------------------------------------
# NGReservoir — state management
# ---------------------------------------------------------------------------


class TestNGReservoirStateManagement:
    def test_state_none_before_forward(self) -> None:
        layer = NGReservoir(input_dim=3)
        assert layer.get_state() is None

    def test_state_set_after_forward(self) -> None:
        layer = NGReservoir(input_dim=3)
        layer(torch.randn(2, 10, 3))
        state = layer.get_state()
        assert state is not None
        assert state.shape == (2, 1, 3)

    def test_get_state_returns_clone(self) -> None:
        layer = NGReservoir(input_dim=3)
        layer(torch.randn(2, 5, 3))
        s1 = layer.get_state()
        s2 = layer.get_state()
        assert s1 is not s2
        assert torch.allclose(s1, s2)

    def test_reset_state_no_args(self) -> None:
        layer = NGReservoir(input_dim=3)
        layer(torch.randn(2, 5, 3))
        layer.reset_state()
        assert layer.state is None

    def test_reset_state_with_batch_size(self) -> None:
        layer = NGReservoir(input_dim=3)
        layer(torch.randn(2, 5, 3))  # Initialize state (and device/dtype)
        layer.reset_state(batch_size=4)
        assert layer.state is not None
        assert layer.state.shape == (4, 1, 3)
        assert torch.all(layer.state == 0)

    def test_reset_state_before_forward(self) -> None:
        """reset_state(batch_size=N) before any forward → CPU float32."""
        layer = NGReservoir(input_dim=3, k=2)
        layer.reset_state(batch_size=3)
        assert layer.state is not None
        assert layer.state.shape == (3, 1, 3)

    def test_set_state_valid(self) -> None:
        layer = NGReservoir(input_dim=3)
        custom_state = torch.randn(2, 1, 3)
        layer.set_state(custom_state)
        assert layer.state is not None
        assert torch.allclose(layer.state, custom_state)

    def test_set_state_wrong_shape_raises(self) -> None:
        layer = NGReservoir(input_dim=3)
        bad_state = torch.randn(2, 2, 3)  # state_size=1, not 2
        with pytest.raises(ValueError, match="validate_state"):
            layer.set_state(bad_state)

    def test_set_random_state(self) -> None:
        layer = NGReservoir(input_dim=3)
        layer.reset_state(batch_size=2)
        initial_state = layer.get_state()
        layer.set_random_state()
        # Random state is very unlikely to equal zero init
        assert not torch.allclose(layer.state, initial_state)

    def test_set_random_state_before_init_raises(self) -> None:
        layer = NGReservoir(input_dim=3)
        with pytest.raises(RuntimeError, match="not initialised"):
            layer.set_random_state()

    def test_set_random_state_lazy_initialises(self) -> None:
        """``set_random_state(batch_size=...)`` lazily initialises before
        randomising (P2.3)."""
        layer = NGReservoir(input_dim=3, k=2, s=1)
        layer.set_random_state(batch_size=4)
        assert layer.state is not None
        assert layer.state.shape[0] == 4

    def test_state_persists_across_forward_passes(self) -> None:
        """State at end of forward is used as init for next forward."""
        layer = NGReservoir(input_dim=2, k=2, s=1)

        # Run two separate sequences and verify different from fresh-run
        torch.manual_seed(0)
        x1 = torch.randn(1, 3, 2)
        x2 = torch.randn(1, 3, 2)

        layer.reset_state()
        layer(x1)
        out2_continued = layer(x2)

        # Re-run with fresh state for x2 — should differ from continued
        layer.reset_state()
        layer(x1)
        layer.reset_state()
        out2_fresh = layer(x2)

        # Continuing from x1's state vs fresh state should give different results
        assert not torch.allclose(out2_continued, out2_fresh)

    def test_warmup_length_k1_is_zero(self) -> None:
        layer = NGReservoir(input_dim=3, k=1)
        assert layer.warmup_length == 0

    def test_warmup_length_k3_s2(self) -> None:
        layer = NGReservoir(input_dim=3, k=3, s=2)
        assert layer.warmup_length == 4


# ---------------------------------------------------------------------------
# NGReservoir — state is a non-persistent buffer (issue #132)
# ---------------------------------------------------------------------------


class TestNGReservoirStateBuffer:
    """The 3-D NG-RC delay buffer is a non-persistent buffer (issue #132).

    The delay buffer must move with the module under ``.to()`` / ``.double()``
    (so a warmed-up buffer survives a device/dtype change instead of being
    silently zero-reinitialised) while staying out of ``state_dict()``.
    """

    def test_state_is_registered_buffer(self) -> None:
        """``state`` lives in ``named_buffers()`` but not ``state_dict()``."""
        layer = NGReservoir(input_dim=3, k=3, s=2)
        layer(torch.randn(2, 20, 3))  # warm up so the buffer fills

        assert "state" in dict(layer.named_buffers())
        assert "state" not in layer.state_dict()

    def test_state_not_in_state_dict_before_warmup(self) -> None:
        """A ``None`` delay buffer never appears in ``state_dict()``."""
        layer = NGReservoir(input_dim=3, k=3, s=2)
        assert layer.state is None
        assert "state" not in layer.state_dict()

    def test_state_does_not_require_grad(self) -> None:
        """The delay buffer must not carry gradients by default."""
        layer = NGReservoir(input_dim=3, k=3, s=2)
        layer(torch.randn(2, 20, 3))
        assert layer.state.requires_grad is False

    def test_double_preserves_warmed_3d_buffer(self) -> None:
        """``.double()`` moves and preserves the 3-D buffer (no zero-reinit)."""
        layer = NGReservoir(input_dim=3, k=3, s=2)  # state_size = (k-1)*s = 4
        layer(torch.randn(2, 20, 3))  # warm up on CPU/float32
        warmed = layer.get_state().clone()

        assert layer.state.dtype == torch.float32
        assert layer.state.shape == (2, 4, 3)
        layer = layer.double()

        # 3-D shape and values preserved; dtype cast to float64.
        assert layer.state.dtype == torch.float64
        assert layer.state.shape == (2, 4, 3)
        assert (layer.state != 0).any()
        assert torch.allclose(layer.state, warmed.double())

    def test_double_then_forward_keeps_continuity(self) -> None:
        """Warm → ``.double()`` → forward continues from the moved buffer.

        The NG-RC delay buffer holds recent inputs, so continuity is observed in
        the *output features*: the first step of the post-move forward consumes
        the warmed delay tap, which must differ from a cold (zero-buffer) start.
        Comparing buffers directly would be misleading here — a buffer that only
        holds the last input is content-addressed by recent inputs.
        """
        torch.manual_seed(0)
        layer = NGReservoir(input_dim=2, k=2, s=1)  # state_size = 1
        warm = torch.randn(1, 6, 2)
        probe = torch.randn(1, 4, 2)

        layer(warm)  # warm the delay buffer on CPU/float32
        layer = layer.double()
        out_warm = layer(probe.double())  # consumes the moved warmed buffer

        # Cold reference: same probe from a zero buffer.
        layer.reset_state()
        out_cold = layer(probe.double())

        assert out_warm.dtype == torch.float64
        # The first feature step must reflect the warmed buffer (no zero-reinit);
        # a silently re-zeroed state would make the warm and cold runs identical.
        assert not torch.allclose(out_warm[:, 0], out_cold[:, 0])

    def test_maybe_init_state_reinits_on_batch_change(self) -> None:
        """A genuine batch-size change still triggers a fresh zero buffer."""
        layer = NGReservoir(input_dim=3, k=3, s=2)
        layer(torch.randn(2, 20, 3))
        assert layer.state.shape[0] == 2

        layer(torch.randn(5, 20, 3))  # different batch
        assert layer.state.shape[0] == 5

    @pytest.mark.gpu
    @cuda_required
    def test_to_cuda_preserves_warmed_buffer(self) -> None:
        """Warm on CPU, ``.to('cuda')``: the 3-D buffer moves with values intact."""
        layer = NGReservoir(input_dim=3, k=3, s=2)
        layer(torch.randn(2, 20, 3))
        warmed = layer.get_state().clone()

        layer = layer.to("cuda")
        assert layer.state.device.type == "cuda"
        assert layer.state.shape == (2, 4, 3)
        assert torch.allclose(layer.state.cpu(), warmed)


# ---------------------------------------------------------------------------
# NGReservoir — causality & correctness
# ---------------------------------------------------------------------------


class TestNGReservoirCausality:
    """Verify sequential processing matches step-by-step cell calls."""

    def test_matches_manual_cell_scan(self) -> None:
        """NGReservoir.forward must produce same output as manual cell scan."""
        d, k, s, p = 3, 2, 1, 2
        layer = NGReservoir(input_dim=d, k=k, s=s, p=p)
        x = torch.randn(2, 10, d)

        # Run via layer
        layer.reset_state()
        layer_out = layer(x)

        # Run manually via cell
        cell = layer.cell
        state = cell.init_state(2, "cpu", torch.float32)
        manual_out_list = []
        for t in range(10):
            feats, state = cell([x[:, t, :]], state)
            manual_out_list.append(feats)
        manual_out = torch.stack(manual_out_list, dim=1)

        assert torch.allclose(layer_out, manual_out, atol=1e-6)

    def test_first_steps_use_zero_buffer(self) -> None:
        """Before buffer is filled, delay taps are zero."""
        layer = NGReservoir(input_dim=2, k=2, s=1, include_constant=False, include_linear=True, p=2)
        # The buffer starts at zero. At step 0, X_{i-1} = 0.
        x0 = torch.tensor([[[5.0, 6.0]]])  # (1, 1, 2)
        layer.reset_state()
        out0 = layer(x0)  # shape (1, 1, feature_dim)

        D = 4
        o_lin_0 = out0[0, 0, :D]
        # O_lin at t=0 = [x0, zeros] = [5, 6, 0, 0]
        assert torch.allclose(o_lin_0, torch.tensor([5.0, 6.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# NGReservoir — vectorized whole-sequence forward (issue #255)
# ---------------------------------------------------------------------------


def _manual_cell_scan(layer: NGReservoir, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-step scan over the inner cell from a fresh zero buffer."""
    cell = layer.cell
    batch = x.shape[0]
    state = cell.init_state(batch, x.device, x.dtype)
    outs = []
    for t in range(x.shape[1]):
        feats, state = cell([x[:, t, :]], state)
        outs.append(feats)
    return torch.stack(outs, dim=1), state


class TestNGReservoirVectorizedForward:
    """The whole-sequence forward must match the per-step loop exactly."""

    @pytest.mark.parametrize("k", [1, 2, 3, 5])
    @pytest.mark.parametrize("s", [1, 2, 3])
    @pytest.mark.parametrize("p", [1, 2, 3])
    @pytest.mark.parametrize("const", [True, False])
    @pytest.mark.parametrize("lin", [True, False])
    def test_zero_max_diff_vs_per_step_loop(
        self, k: int, s: int, p: int, const: bool, lin: bool
    ) -> None:
        """``0.0`` max-abs-diff vs the per-step loop, including the warmup region.

        Covers ``k>=1, s>=1, p>=1`` and both ``include_constant`` /
        ``include_linear`` flag combinations.  float64 makes any non-bit-exact
        path visible.
        """
        d = 3
        layer = NGReservoir(input_dim=d, k=k, s=s, p=p, include_constant=const, include_linear=lin)
        torch.manual_seed(k * 100 + s * 10 + p)
        # seq_len exceeds the warmup length for every k here, so the comparison
        # spans both the unfilled-buffer (warmup) region and the filled region.
        x = torch.randn(2, 13, d, dtype=torch.float64)

        layer.reset_state()
        vectorized = layer(x)
        manual, _ = _manual_cell_scan(layer, x)

        max_diff = (vectorized - manual).abs().max().item()
        assert max_diff == 0.0, f"max-abs-diff {max_diff} != 0.0 for k={k} s={s} p={p}"

    def test_final_state_matches_per_step_loop(self) -> None:
        """Final delay buffer after a batch forward equals the per-step buffer."""
        d, k, s, p = 3, 3, 2, 2
        layer = NGReservoir(input_dim=d, k=k, s=s, p=p)
        torch.manual_seed(7)
        x = torch.randn(2, 15, d, dtype=torch.float64)

        layer.reset_state()
        layer(x)
        _, manual_state = _manual_cell_scan(layer, x)

        assert layer.state is not None
        assert torch.equal(layer.state, manual_state)

    def test_warmup_region_exact(self) -> None:
        """The first ``warmup_length`` steps (unfilled buffer) match exactly."""
        d, k, s, p = 2, 3, 2, 2
        layer = NGReservoir(input_dim=d, k=k, s=s, p=p, include_constant=False, include_linear=True)
        torch.manual_seed(3)
        x = torch.randn(1, 8, d, dtype=torch.float64)

        layer.reset_state()
        vectorized = layer(x)
        manual, _ = _manual_cell_scan(layer, x)

        warm = layer.warmup_length
        assert warm > 0
        assert torch.equal(vectorized[:, :warm, :], manual[:, :warm, :])

    def test_streaming_step_continues_after_batch_forward(self) -> None:
        """A per-step cell stream continues seamlessly after a batch forward.

        Run the batch fast path over a prefix, then drive the remaining steps
        through the per-step ``cell.forward`` starting from the layer's stored
        delay buffer.  The streamed tail must match the fully-vectorized run.
        """
        d, k, s, p = 3, 3, 2, 2
        layer = NGReservoir(input_dim=d, k=k, s=s, p=p)
        torch.manual_seed(11)
        full = torch.randn(2, 30, d, dtype=torch.float64)

        # Reference: vectorized run over the whole sequence.
        layer.reset_state()
        reference = layer(full)

        # Batch forward over the prefix, then stream the tail via the cell.
        layer.reset_state()
        layer(full[:, :20, :])
        cell = layer.cell
        state = layer.get_state()
        assert state is not None
        streamed = []
        for t in range(20, 30):
            feats, state = cell([full[:, t, :]], state)
            streamed.append(feats)
        streamed_tail = torch.stack(streamed, dim=1)

        assert torch.equal(streamed_tail, reference[:, 20:, :])

    def test_continued_batch_forward_matches_single_pass(self) -> None:
        """Two consecutive batch forwards equal one batch forward over the join."""
        d, k, s, p = 2, 2, 1, 3
        layer = NGReservoir(input_dim=d, k=k, s=s, p=p)
        torch.manual_seed(5)
        x = torch.randn(2, 24, d, dtype=torch.float64)

        layer.reset_state()
        single = layer(x)

        layer.reset_state()
        first = layer(x[:, :10, :])
        second = layer(x[:, 10:, :])
        joined = torch.cat([first, second], dim=1)

        assert torch.equal(joined, single)

    def test_no_python_loop_on_batch_forward_path(self) -> None:
        """No per-step Python loop runs on the batch-forward path.

        ``NGReservoir.forward`` is a single vectorized call, so a sequence of
        any length triggers exactly one ``NGCell.forward_sequence`` invocation
        and zero per-step ``NGCell.forward`` calls.
        """
        d = 3
        layer = NGReservoir(input_dim=d, k=2, s=1, p=2)
        x = torch.randn(1, 64, d)

        seq_calls = 0
        step_calls = 0
        orig_seq = NGCell.forward_sequence
        orig_step = NGCell.forward

        def counting_seq(self, *a, **kw):  # type: ignore[no-untyped-def]
            nonlocal seq_calls
            seq_calls += 1
            return orig_seq(self, *a, **kw)

        def counting_step(self, *a, **kw):  # type: ignore[no-untyped-def]
            nonlocal step_calls
            step_calls += 1
            return orig_step(self, *a, **kw)

        NGCell.forward_sequence = counting_seq  # type: ignore[method-assign]
        NGCell.forward = counting_step  # type: ignore[method-assign]
        try:
            layer.reset_state()
            layer(x)
        finally:
            NGCell.forward_sequence = orig_seq  # type: ignore[method-assign]
            NGCell.forward = orig_step  # type: ignore[method-assign]

        assert seq_calls == 1
        assert step_calls == 0

    def test_state_3d_validation_intact(self) -> None:
        """set_state / get_state 3-D validation survives the vectorized path."""
        d, k, s = 3, 3, 2
        layer = NGReservoir(input_dim=d, k=k, s=s)
        layer(torch.randn(2, 20, d))

        # get_state returns a 3-D clone.
        state = layer.get_state()
        assert state is not None
        assert state.dim() == 3
        assert state.shape == (2, layer.state_size, d)

        # set_state accepts a correctly-shaped 3-D buffer.
        good = torch.randn(2, layer.state_size, d)
        layer.set_state(good)
        assert torch.allclose(layer.state, good)

        # set_state rejects a wrong-shaped buffer.
        with pytest.raises(ValueError, match="validate_state"):
            layer.set_state(torch.randn(2, layer.state_size + 1, d))


# ---------------------------------------------------------------------------
# NGReservoir — gradients
# ---------------------------------------------------------------------------


class TestNGReservoirGradients:
    def test_gradients_flow_through(self) -> None:
        """Gradients flow from features back to the input."""
        layer = NGReservoir(input_dim=3, k=2, p=2)
        x = torch.randn(2, 5, 3, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_no_parameter_gradients(self) -> None:
        """No parameters → no parameter gradients to worry about."""
        layer = NGReservoir(input_dim=3, k=2, p=2)
        assert list(layer.parameters()) == []

    def test_gradients_k1(self) -> None:
        layer = NGReservoir(input_dim=4, k=1, p=2)
        x = torch.randn(1, 10, 4, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# NGReservoir — repr
# ---------------------------------------------------------------------------


class TestNGReservoirRepr:
    def test_repr_contains_cell_info(self) -> None:
        layer = NGReservoir(input_dim=3, k=2, s=1, p=2)
        r = repr(layer)
        assert "NGCell" in r
        assert "input_dim=3" in r
        assert "k=2" in r
        assert "feature_dim=28" in r


# ---------------------------------------------------------------------------
# Integration: NGReservoir + CGReadoutLayer composition
# ---------------------------------------------------------------------------


class TestNGRCWithReadout:
    def test_feature_dim_feeds_readout(self) -> None:
        """NGReservoir output can feed into CGReadoutLayer."""
        from resdag.layers import CGReadoutLayer

        d, out_dim = 3, 1
        layer = NGReservoir(input_dim=d, k=2, p=2)
        readout = CGReadoutLayer(in_features=layer.feature_dim, out_features=out_dim)

        x = torch.randn(2, 20, d)
        features = layer(x)
        pred = readout(features)
        assert pred.shape == (2, 20, out_dim)
