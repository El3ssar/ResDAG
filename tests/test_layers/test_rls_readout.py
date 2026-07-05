"""RLSReadout contracts: true-online recursive-least-squares readout.

Pins down :class:`RLSReadout`, the streaming readout that refines its weights one
sample at a time via the Recursive Least Squares recursion with an exponential
forgetting factor :math:`\\lambda`.

Coverage:

- Construction, inheritance, and constructor validation (``forgetting``,
  ``delta`` / ``alpha`` alias, positivity).
- Statistical acceptance: on a stationary linear problem ``y = W*·x + noise``,
  ``lambda=1`` RLS converges to the **ordinary least-squares** solution, and a
  fully-streamed RLS fit agrees with a fresh batch :class:`RidgeReadoutLayer`
  (small ``alpha``) fit to ~1e-3.
- Numerical stability: ``P`` stays symmetric and positive-definite.
- Streaming semantics: chunked ``partial_fit`` equals a whole-sequence fit;
  ``fit_online`` resets and re-fits.
- Forgetting: ``lambda < 1`` tracks a regime switch that ``lambda = 1`` cannot.
- Drop-in ``ESNTrainer`` / ``forecast`` compatibility on a small ESN.
- Base ``fit`` bookkeeping (2D/3D flattening, shape + finiteness guards).
- Device coverage.
"""

import pytest
import pytorch_symbolic as ps
import torch

from resdag import ESNLayer, ESNModel, RLSReadout
from resdag.layers.readouts import ReadoutLayer, RidgeReadoutLayer
from resdag.training import ESNTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_problem(
    n: int = 6000,
    f: int = 15,
    o: int = 3,
    noise: float = 0.01,
    bias: bool = False,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """A well-posed stationary problem ``Y = X W_true (+ b_true) + noise``.

    Returns ``(X, Y, W_true, b_true)`` in ``float64``; ``b_true`` is ``None``
    when ``bias`` is ``False``.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, f, dtype=torch.float64, generator=gen)
    w_true = torch.randn(f, o, dtype=torch.float64, generator=gen)
    y = x @ w_true
    b_true: torch.Tensor | None = None
    if bias:
        b_true = torch.randn(o, dtype=torch.float64, generator=gen)
        y = y + b_true
    y = y + noise * torch.randn(n, o, dtype=torch.float64, generator=gen)
    return x, y, w_true, b_true


# ---------------------------------------------------------------------------
# Instantiation / validation
# ---------------------------------------------------------------------------


class TestInstantiation:
    """Construction, inheritance, and constructor validation."""

    def test_is_readout_subclass(self) -> None:
        """RLSReadout is a ReadoutLayer (so ESNTrainer accepts it)."""
        readout = RLSReadout(in_features=20, out_features=3)
        assert isinstance(readout, ReadoutLayer)
        assert readout.in_features == 20
        assert readout.out_features == 3
        assert not readout.is_fitted
        assert readout.n_seen == 0

    def test_defaults(self) -> None:
        """Default forgetting is 1.0 and default delta is 1e-2."""
        readout = RLSReadout(10, 2)
        assert readout.forgetting == 1.0
        assert readout.delta == 1e-2

    def test_p_initialised_to_identity_over_delta(self) -> None:
        """P starts as I / delta (augmented by one row/col when bias is fitted)."""
        readout = RLSReadout(5, 2, bias=False, delta=0.25)
        expected = torch.eye(5, dtype=readout.P.dtype) / 0.25
        assert torch.allclose(readout.P, expected)
        # With a bias, the constant feature augments P to (F + 1).
        readout_b = RLSReadout(5, 2, bias=True, delta=0.25)
        assert readout_b.P.shape == (6, 6)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_forgetting_out_of_range_rejected(self, bad: float) -> None:
        """forgetting outside (0, 1] raises at construction."""
        with pytest.raises(ValueError, match=r"forgetting \(lambda\) must be in \(0, 1\]"):
            RLSReadout(10, 2, forgetting=bad)

    @pytest.mark.parametrize("bad", [0.0, -1e-3])
    def test_nonpositive_delta_rejected(self, bad: float) -> None:
        """delta <= 0 raises at construction."""
        with pytest.raises(ValueError, match="delta .* must be positive"):
            RLSReadout(10, 2, delta=bad)

    def test_alpha_aliases_delta(self) -> None:
        """alpha is accepted as an alias for delta (P_0 = I / alpha)."""
        readout = RLSReadout(10, 2, alpha=1e-3)
        assert readout.delta == 1e-3

    def test_alpha_and_delta_together_rejected(self) -> None:
        """Passing both alpha and delta raises (they name the same knob)."""
        with pytest.raises(ValueError, match="not both"):
            RLSReadout(10, 2, alpha=1e-3, delta=1e-2)

    def test_repr_contains_forgetting_and_delta(self) -> None:
        """repr surfaces the key hyperparameters."""
        r = repr(RLSReadout(10, 2, name="rls", forgetting=0.99, delta=1e-3))
        assert "forgetting=0.99" in r
        assert "delta=0.001" in r
        assert "name='rls'" in r


# ---------------------------------------------------------------------------
# Acceptance criterion 1: lambda=1 converges to ordinary least squares
# ---------------------------------------------------------------------------


class TestConvergesToOLS:
    """A growing-window (lambda=1) RLS fit recovers the OLS / ridge solution."""

    def test_lambda1_matches_ols_no_bias(self) -> None:
        """No-bias, lambda=1 RLS converges to the lstsq (OLS) solution tightly."""
        x, y, _, _ = _linear_problem(n=6000, f=15, o=3, bias=False, seed=1)
        w_ols = torch.linalg.lstsq(x, y).solution  # (F, O)

        readout = RLSReadout(15, 3, bias=False, forgetting=1.0, delta=1e-2, dtype=torch.float64)
        readout.fit_online(x, y)
        w_rls = readout.weight.detach().T  # (F, O)

        assert torch.allclose(w_rls, w_ols, atol=1e-4)

    def test_lambda1_matches_ridge_no_bias(self) -> None:
        """Streamed RLS agrees with a fresh batch ridge (small alpha) fit to ~1e-3."""
        x, y, _, _ = _linear_problem(n=6000, f=15, o=3, bias=False, seed=2)

        ridge = RidgeReadoutLayer(15, 3, bias=False, alpha=1e-8).double()
        ridge.fit(x, y)

        rls = RLSReadout(15, 3, bias=False, forgetting=1.0, delta=1e-2, dtype=torch.float64)
        rls.fit_online(x, y)

        assert torch.allclose(rls.weight.detach(), ridge.weight.detach(), atol=1e-3)

    def test_lambda1_matches_ridge_with_bias(self) -> None:
        """With a bias, streamed RLS agrees with batch ridge on weight and intercept."""
        x, y, _, b_true = _linear_problem(n=6000, f=15, o=3, bias=True, seed=3)
        assert b_true is not None

        ridge = RidgeReadoutLayer(15, 3, bias=True, alpha=1e-8).double()
        ridge.fit(x, y)

        rls = RLSReadout(15, 3, bias=True, forgetting=1.0, delta=1e-3, dtype=torch.float64)
        rls.fit_online(x, y)

        assert rls.bias is not None and ridge.bias is not None
        assert torch.allclose(rls.weight.detach(), ridge.weight.detach(), atol=1e-3)
        assert torch.allclose(rls.bias.detach(), ridge.bias.detach(), atol=1e-3)
        # The recovered intercept also tracks the ground-truth bias.
        assert torch.allclose(rls.bias.detach(), b_true, atol=1e-2)


# ---------------------------------------------------------------------------
# Acceptance criterion 2: P stays symmetric positive-definite
# ---------------------------------------------------------------------------


class TestInverseCorrelationStability:
    """The inverse-correlation matrix P remains symmetric and positive-definite."""

    def test_p_symmetric_after_fit(self) -> None:
        """P is exactly symmetric after a full streaming fit (re-symmetrised each step)."""
        x, y, _, _ = _linear_problem(n=2000, f=12, o=2, bias=True, seed=4)
        readout = RLSReadout(12, 2, bias=True, delta=1e-2)
        readout.fit_online(x, y)
        asym = (readout.P - readout.P.T).abs().max().item()
        assert asym == 0.0

    def test_p_positive_definite_after_fit(self) -> None:
        """All eigenvalues of P stay strictly positive (P is SPD)."""
        x, y, _, _ = _linear_problem(n=2000, f=12, o=2, bias=False, seed=5)
        readout = RLSReadout(12, 2, bias=False, delta=1e-2)
        readout.fit_online(x, y)
        evals = torch.linalg.eigvalsh(readout.P)
        assert float(evals.min().item()) > 0.0

    def test_p_positive_definite_with_forgetting(self) -> None:
        """Forgetting (lambda < 1) keeps P bounded and positive-definite."""
        x, y, _, _ = _linear_problem(n=3000, f=10, o=2, bias=False, seed=6)
        readout = RLSReadout(10, 2, bias=False, forgetting=0.99, delta=1e-2)
        readout.fit_online(x, y)
        evals = torch.linalg.eigvalsh(readout.P)
        assert float(evals.min().item()) > 0.0
        assert torch.isfinite(readout.P).all()

    def test_p_exactly_symmetric_for_non_multiple_length(self) -> None:
        """End-of-pass symmetrisation makes P exactly symmetric for any sample count.

        ``P`` is re-symmetrised only every ``_RESYM_EVERY`` steps (a perf fix, so
        the per-step cost stays O(F^2) instead of paying a transpose-copy every
        step). A sample count that is *not* a multiple of that interval would
        otherwise leave a tiny round-off asymmetry — the always-run end-of-pass
        symmetrisation guarantees exact symmetry regardless.
        """
        n = 3 * RLSReadout._RESYM_EVERY + 7  # deliberately not a multiple
        assert n % RLSReadout._RESYM_EVERY != 0
        x, y, _, _ = _linear_problem(n=n, f=9, o=2, bias=False, seed=14)
        readout = RLSReadout(9, 2, bias=False, delta=1e-2)
        readout.fit_online(x, y)
        assert (readout.P - readout.P.T).abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# Streaming semantics: chunked == whole; fit_online resets
# ---------------------------------------------------------------------------


class TestStreaming:
    """partial_fit carries state across calls; fit_online restarts cleanly."""

    def test_chunked_equals_whole(self) -> None:
        """Feeding the sequence in chunks equals feeding it whole (state carried)."""
        x, y, _, _ = _linear_problem(n=500, f=8, o=2, bias=True, seed=7)

        whole = RLSReadout(8, 2, bias=True, delta=1e-2)
        whole.fit_online(x, y)

        chunked = RLSReadout(8, 2, bias=True, delta=1e-2)
        chunked.reset_state()
        for i in range(0, x.shape[0], 50):
            chunked.partial_fit(x[i : i + 50], y[i : i + 50])

        assert whole.n_seen == chunked.n_seen == 500
        assert whole.bias is not None and chunked.bias is not None
        assert torch.allclose(whole.weight, chunked.weight, atol=1e-6)
        assert torch.allclose(whole.bias, chunked.bias, atol=1e-6)

    def test_partial_fit_returns_self_and_flips_fitted(self) -> None:
        """partial_fit returns self (chainable) and marks the layer fitted."""
        x, y, _, _ = _linear_problem(n=100, f=6, o=2, seed=8)
        readout = RLSReadout(6, 2, bias=False)
        out = readout.partial_fit(x, y)
        assert out is readout
        assert readout.is_fitted
        assert readout.n_seen == 100

    def test_fit_online_resets_before_fitting(self) -> None:
        """fit_online discards prior state so the fit depends only on its input."""
        x, y, _, _ = _linear_problem(n=800, f=8, o=2, bias=False, seed=9)
        readout = RLSReadout(8, 2, bias=False, delta=1e-2)
        # Pollute with an unrelated stream first.
        junk_x = torch.randn(800, 8, dtype=torch.float64)
        junk_y = torch.randn(800, 2, dtype=torch.float64)
        readout.partial_fit(junk_x, junk_y)
        readout.fit_online(x, y)
        assert readout.n_seen == 800

        fresh = RLSReadout(8, 2, bias=False, delta=1e-2)
        fresh.fit_online(x, y)
        assert torch.allclose(readout.weight, fresh.weight, atol=1e-10)

    def test_reset_state_semantics(self) -> None:
        """reset_state returns the layer to its freshly-constructed online state."""
        x, y, _, _ = _linear_problem(n=200, f=5, o=2, bias=True, seed=10)
        readout = RLSReadout(5, 2, bias=True, delta=0.1)
        readout.fit_online(x, y)
        assert readout.is_fitted and readout.n_seen == 200
        readout.reset_state()
        assert not readout.is_fitted
        assert readout.n_seen == 0
        assert torch.count_nonzero(readout.weight) == 0
        assert readout.bias is not None and torch.count_nonzero(readout.bias) == 0
        expected_P = torch.eye(6, dtype=readout.P.dtype) / 0.1
        assert torch.allclose(readout.P, expected_P)


# ---------------------------------------------------------------------------
# Forgetting: lambda < 1 tracks non-stationarity
# ---------------------------------------------------------------------------


class TestForgetting:
    """A forgetting factor lets RLS follow a regime switch batch ridge cannot."""

    def test_forgetting_tracks_regime_switch(self) -> None:
        """lambda<1 ends near the *recent* regime; lambda=1 averages both."""
        gen = torch.Generator().manual_seed(11)
        f, o, n = 10, 2, 4000
        x = torch.randn(n, f, dtype=torch.float64, generator=gen)
        w1 = torch.randn(f, o, dtype=torch.float64, generator=gen)
        w2 = torch.randn(f, o, dtype=torch.float64, generator=gen)
        y = torch.empty(n, o, dtype=torch.float64)
        y[: n // 2] = x[: n // 2] @ w1
        y[n // 2 :] = x[n // 2 :] @ w2  # regime switch halfway

        forgetful = RLSReadout(f, o, bias=False, forgetting=0.98, delta=1e-2)
        forgetful.fit_online(x, y)
        err_recent = (forgetful.weight.detach().T - w2).abs().max().item()

        growing = RLSReadout(f, o, bias=False, forgetting=1.0, delta=1e-2)
        growing.fit_online(x, y)
        err_growing_recent = (growing.weight.detach().T - w2).abs().max().item()

        # Forgetting locks onto the recent regime; growing-window averages both
        # and so is further from the recent regime.
        assert err_recent < 1e-3
        assert err_growing_recent > err_recent


# ---------------------------------------------------------------------------
# Base fit() bookkeeping (shared via ReadoutLayer)
# ---------------------------------------------------------------------------


class TestFitBookkeeping:
    """RLSReadout inherits the base fit() guards and 2D/3D handling."""

    def test_fit_3d_and_is_fitted(self) -> None:
        """3D inputs are flattened and is_fitted flips True."""
        torch.manual_seed(0)
        readout = RLSReadout(in_features=10, out_features=3)
        x = torch.randn(4, 25, 10)
        y = torch.randn(4, 25, 3)
        readout.fit(x, y)
        assert readout.is_fitted
        assert readout.weight.shape == (3, 10)

    def test_fit_wrong_in_features_raises(self) -> None:
        """A wrong in_features surfaces a ValueError."""
        readout = RLSReadout(in_features=20, out_features=5)
        with pytest.raises(ValueError, match="state feature dimension"):
            readout.fit(torch.randn(100, 16), torch.randn(100, 5))

    def test_partial_fit_wrong_out_features_raises(self) -> None:
        """A wrong out_features surfaces a ValueError from partial_fit."""
        readout = RLSReadout(in_features=10, out_features=3)
        with pytest.raises(ValueError, match="target feature dimension"):
            readout.partial_fit(torch.randn(50, 10), torch.randn(50, 2))

    def test_partial_fit_sample_mismatch_raises(self) -> None:
        """Mismatched sample counts raise from partial_fit."""
        readout = RLSReadout(in_features=10, out_features=3)
        with pytest.raises(ValueError, match="sample count mismatch"):
            readout.partial_fit(torch.randn(50, 10), torch.randn(40, 3))

    def test_non_finite_states_rejected(self) -> None:
        """NaN/Inf states are rejected before poisoning the recursion."""
        readout = RLSReadout(in_features=5, out_features=2)
        x = torch.randn(20, 5)
        x[0, 0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            readout.partial_fit(x, torch.randn(20, 2))

    def test_non_finite_targets_rejected(self) -> None:
        """NaN/Inf targets are rejected."""
        readout = RLSReadout(in_features=5, out_features=2)
        y = torch.randn(20, 2)
        y[3, 1] = float("inf")
        with pytest.raises(ValueError, match="non-finite"):
            readout.partial_fit(torch.randn(20, 5), y)

    def test_diverged_recursion_raises_finiteness_error(self) -> None:
        """A diverging recursion is caught: non-finite P/weights raise, not fitted silently.

        With a tiny forgetting factor, a near-singular ``delta`` prior, hugely
        unscaled inputs and float32 accumulation, the RLS recursion blows up to
        NaN/Inf. The finiteness guard turns that into a clear ``RuntimeError``
        (naming the likely causes) instead of returning ``is_fitted=True`` with a
        poisoned ``P`` / weights.
        """
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(3000, 40, dtype=torch.float32, generator=gen) * 1e4
        y = torch.randn(3000, 3, dtype=torch.float32, generator=gen) * 1e4
        readout = RLSReadout(40, 3, bias=False, forgetting=0.3, delta=1e-10, use_float64=False)
        with pytest.raises(RuntimeError, match="RLS recursion diverged"):
            readout.fit_online(x, y)

    def test_finiteness_guard_quiet_on_normal_fit(self) -> None:
        """The divergence guard does not fire on an ordinary, well-scaled fit."""
        x, y, _, _ = _linear_problem(n=1000, f=10, o=2, bias=True, seed=15)
        readout = RLSReadout(10, 2, bias=True, delta=1e-2)
        readout.fit_online(x, y)  # must not raise
        assert readout.is_fitted
        assert torch.isfinite(readout.P).all()
        assert torch.isfinite(readout.weight).all()

    def test_no_bias_leaves_intercept_none(self) -> None:
        """bias=False fits no intercept."""
        x, y, _, _ = _linear_problem(n=500, f=8, o=2, bias=False, seed=12)
        readout = RLSReadout(8, 2, bias=False)
        coefs, intercept = readout._fit_impl(x, y)
        assert intercept is None
        assert torch.isfinite(coefs).all()

    def test_fit_impl_resets_between_calls(self) -> None:
        """_fit_impl restarts the recursion, so batch fit is order-independent."""
        x, y, _, _ = _linear_problem(n=1000, f=8, o=2, bias=False, seed=13)
        readout = RLSReadout(8, 2, bias=False, delta=1e-2)
        readout.fit(x, y)
        w_first = readout.weight.detach().clone()
        # Fit a different problem, then the original again — no carryover.
        x2, y2, _, _ = _linear_problem(n=1000, f=8, o=2, bias=False, seed=99)
        readout.fit(x2, y2)
        readout.fit(x, y)
        assert torch.allclose(readout.weight.detach(), w_first, atol=1e-10)


# ---------------------------------------------------------------------------
# ESNTrainer / forecast compatibility (drop-in, no trainer changes)
# ---------------------------------------------------------------------------


class TestESNTrainerCompatibility:
    """RLSReadout trains end-to-end through ESNTrainer, keyed by name."""

    def test_trainer_fits_and_forecasts(self) -> None:
        """ESNTrainer fits the named readout via its pre-hook, forecast is finite."""
        torch.manual_seed(0)
        inp = ps.Input((30, 3))
        reservoir = ESNLayer(50, feedback_size=3, spectral_radius=0.9)(inp)
        readout = RLSReadout(50, 3, name="output", forgetting=1.0, delta=1e-2)(reservoir)
        model = ESNModel(inp, readout)

        warmup = torch.randn(1, 30, 3)
        train = torch.randn(1, 30, 3)
        targets = torch.randn(1, 30, 3)

        trainer = ESNTrainer(model)
        trainer.fit(
            warmup_inputs=(warmup,),
            train_inputs=(train,),
            targets={"output": targets},
        )

        readout_layer = next(m for m in model.modules() if isinstance(m, ReadoutLayer))
        assert readout_layer.is_fitted
        pred = model.forecast(warmup, horizon=20)
        assert pred.shape == (1, 20, 3)
        assert torch.isfinite(pred).all()

    def test_learns_a_solvable_esn_map(self) -> None:
        """RLS readout drives a low forecast error on a learnable sine map."""
        torch.manual_seed(0)
        t = torch.linspace(0, 40 * torch.pi, 800, dtype=torch.float32)
        series = torch.stack([torch.sin(t), torch.sin(0.5 * t), torch.cos(t)], dim=-1).unsqueeze(
            0
        )  # (1, 800, 3)

        inp = ps.Input((599, 3))
        reservoir = ESNLayer(200, feedback_size=3, spectral_radius=0.9, seed=0)(inp)
        readout = RLSReadout(200, 3, name="output", forgetting=1.0, delta=1e-4)(reservoir)
        model = ESNModel(inp, readout)

        warmup = series[:, :100]
        train_in = series[:, 100:699]
        train_tgt = series[:, 101:700]

        trainer = ESNTrainer(model)
        trainer.fit(
            warmup_inputs=(warmup,),
            train_inputs=(train_in,),
            targets={"output": train_tgt},
        )
        pred = model.forecast(series[:, :100], horizon=50)
        truth = series[:, 100:150]
        rel_err = (pred - truth).norm() / truth.norm()
        assert torch.isfinite(pred).all()
        assert float(rel_err.item()) < 0.5


# ---------------------------------------------------------------------------
# Device coverage
# ---------------------------------------------------------------------------


class TestDevice:
    """RLSReadout fits and predicts on every available device."""

    def test_fit_on_device(self, device: torch.device) -> None:
        """Fitting and predicting land on the target device."""
        torch.manual_seed(0)
        readout = RLSReadout(in_features=20, out_features=5).to(device)
        assert readout.P.device.type == device.type
        x = torch.randn(200, 20, device=device)
        y = torch.randn(200, 5, device=device)
        readout.fit(x, y)
        assert readout.is_fitted
        assert readout.weight.device.type == device.type
        pred = readout(torch.randn(10, 20, device=device))
        assert pred.device.type == device.type
        assert pred.shape == (10, 5)

    def test_construct_with_device_kwarg(self, device: torch.device) -> None:
        """The device constructor kwarg places parameters and P buffer."""
        readout = RLSReadout(8, 2, device=device)
        assert readout.weight.device.type == device.type
        assert readout.P.device.type == device.type
