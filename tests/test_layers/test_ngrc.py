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

        expected_vals = [x_vals[i] * x_vals[j] for i, j in itertools.combinations_with_replacement(range(d), 2)]
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
        with pytest.raises(ValueError, match="State shape mismatch"):
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
        with pytest.raises(RuntimeError, match="not initialized"):
            layer.set_random_state()

    def test_state_persists_across_forward_passes(self) -> None:
        """State at end of forward is used as init for next forward."""
        layer = NGReservoir(input_dim=2, k=2, s=1)

        # Run two separate sequences and verify different from fresh-run
        torch.manual_seed(0)
        x1 = torch.randn(1, 3, 2)
        x2 = torch.randn(1, 3, 2)

        layer.reset_state()
        out1 = layer(x1)
        state_mid = layer.get_state()
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
