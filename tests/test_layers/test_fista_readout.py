"""FISTAReadout contracts: L1 (lasso) / elastic-net readout via FISTA.

Pins down :class:`FISTAReadout`, the sparse / robust readout that solves an
L1 or elastic-net least-squares problem with the accelerated proximal-gradient
method FISTA (Beck & Teboulle 2009), batched over every output column at once.

Coverage:

- Construction, inheritance, and constructor validation.
- Objective parameterisation: the FISTA coefficients match
  ``sklearn.linear_model.Lasso`` / ``ElasticNet`` when
  ``alpha_resdag = 2 * N * alpha_sklearn`` (sklearn cross-check gated by
  ``importorskip``).
- Sparsity: a larger ``alpha`` zeros out strictly more coefficients.
- ``l1_ratio=0`` (pure ridge) converges to :class:`RidgeReadoutLayer`.
- Non-convergence warns; the diagnostics ``n_iter_`` / ``converged_`` are set.
- Drop-in ``ESNTrainer`` compatibility and finite forecasts.
- Device coverage and CPU/GPU-consistent (deterministic) results.
"""

import warnings

import pytest
import pytorch_symbolic as ps
import torch

from resdag import ESNLayer, ESNModel
from resdag.layers.readouts import FISTAReadout, ReadoutLayer, RidgeReadoutLayer
from resdag.training import ESNTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sparse_problem(
    n: int = 300,
    f: int = 25,
    t: int = 3,
    zero_frac: float = 0.6,
    noise: float = 0.01,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A well-posed linear problem ``Y = X W_true + noise`` with sparse ``W_true``."""
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(n, f, dtype=torch.float64, generator=gen)
    w_true = torch.randn(f, t, dtype=torch.float64, generator=gen)
    mask = torch.rand(f, t, generator=gen) < zero_frac
    w_true[mask] = 0.0
    y = X @ w_true + noise * torch.randn(n, t, dtype=torch.float64, generator=gen)
    return X, y


# ---------------------------------------------------------------------------
# Instantiation / validation
# ---------------------------------------------------------------------------


class TestInstantiation:
    """Construction, inheritance, and constructor validation."""

    def test_is_readout_subclass(self) -> None:
        """FISTAReadout is a ReadoutLayer (so ESNTrainer accepts it)."""
        readout = FISTAReadout(in_features=20, out_features=3)
        assert isinstance(readout, ReadoutLayer)
        assert readout.in_features == 20
        assert readout.out_features == 3
        assert not readout.is_fitted
        assert readout.n_iter_ == 0
        assert not readout.converged_

    def test_negative_alpha_rejected(self) -> None:
        """alpha < 0 raises at construction."""
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            FISTAReadout(10, 2, alpha=-1e-6)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_l1_ratio_out_of_range_rejected(self, bad: float) -> None:
        """l1_ratio outside [0, 1] raises at construction."""
        with pytest.raises(ValueError, match=r"l1_ratio must be in \[0, 1\]"):
            FISTAReadout(10, 2, l1_ratio=bad)

    def test_bad_max_iter_rejected(self) -> None:
        """max_iter < 1 raises at construction."""
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            FISTAReadout(10, 2, max_iter=0)

    def test_negative_tol_rejected(self) -> None:
        """tol < 0 raises at construction."""
        with pytest.raises(ValueError, match="tol must be non-negative"):
            FISTAReadout(10, 2, tol=-1e-6)

    def test_repr_contains_alpha_and_l1_ratio(self) -> None:
        """repr surfaces the key hyperparameters."""
        r = repr(FISTAReadout(10, 2, name="fr", alpha=1e-2, l1_ratio=0.7))
        assert "alpha=0.01" in r
        assert "l1_ratio=0.7" in r
        assert "name='fr'" in r


# ---------------------------------------------------------------------------
# Acceptance criterion 1: matches sklearn Lasso / ElasticNet
# ---------------------------------------------------------------------------


class TestMatchesSklearn:
    """FISTA coefficients match sklearn's coordinate descent to solver tolerance.

    Parameterisation: this layer minimises the *un-normalised* sum-of-squares
    objective, while sklearn normalises by ``1/(2N)``. The two coincide when
    ``alpha_resdag = 2 * N * alpha_sklearn`` (same ``l1_ratio``).
    """

    def test_lasso_matches_sklearn(self) -> None:
        """l1_ratio=1 matches sklearn.linear_model.Lasso coefficients + intercept."""
        linmod = pytest.importorskip("sklearn.linear_model")
        X, y = _sparse_problem()
        n = X.shape[0]
        sk_alpha = 1e-2

        lasso = linmod.Lasso(alpha=sk_alpha, fit_intercept=True, max_iter=200000, tol=1e-12)
        lasso.fit(X.numpy(), y.numpy())
        sk_coef = torch.from_numpy(lasso.coef_.T)  # (F, T)
        sk_int = torch.from_numpy(lasso.intercept_)  # (T,)

        readout = FISTAReadout(
            X.shape[1],
            y.shape[1],
            alpha=2 * n * sk_alpha,
            l1_ratio=1.0,
            max_iter=50000,
            tol=1e-11,
        )
        coefs, intercept = readout._fit_impl(X, y)

        assert intercept is not None
        assert torch.allclose(coefs, sk_coef, atol=1e-6)
        assert torch.allclose(intercept, sk_int, atol=1e-6)
        # The exact sparsity pattern (which coefficients are zero) agrees.
        assert torch.equal(coefs == 0, sk_coef == 0)

    @pytest.mark.parametrize("l1_ratio", [0.3, 0.5, 0.9])
    def test_elasticnet_matches_sklearn(self, l1_ratio: float) -> None:
        """Intermediate l1_ratio matches sklearn.linear_model.ElasticNet."""
        linmod = pytest.importorskip("sklearn.linear_model")
        X, y = _sparse_problem()
        n = X.shape[0]
        sk_alpha = 1e-2

        en = linmod.ElasticNet(
            alpha=sk_alpha,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=500000,
            tol=1e-12,
        )
        en.fit(X.numpy(), y.numpy())
        en_coef = torch.from_numpy(en.coef_.T)

        readout = FISTAReadout(
            X.shape[1],
            y.shape[1],
            alpha=2 * n * sk_alpha,
            l1_ratio=l1_ratio,
            max_iter=100000,
            tol=1e-11,
        )
        coefs, _ = readout._fit_impl(X, y)
        assert torch.allclose(coefs, en_coef, atol=1e-6)


# ---------------------------------------------------------------------------
# Acceptance criterion 2: sparsity increases with alpha
# ---------------------------------------------------------------------------


class TestSparsity:
    """The L1 penalty drives coefficients to *exactly* zero, more so with alpha."""

    def test_sparsity_monotone_in_alpha(self) -> None:
        """A larger alpha zeros out at least as many coefficients (lasso)."""
        X, y = _sparse_problem()
        n = X.shape[0]

        fracs = []
        for sk_alpha in [1e-3, 1e-2, 5e-2, 2e-1, 1.0]:
            readout = FISTAReadout(
                X.shape[1],
                y.shape[1],
                alpha=2 * n * sk_alpha,
                l1_ratio=1.0,
                max_iter=20000,
                tol=1e-9,
            )
            coefs, _ = readout._fit_impl(X, y)
            fracs.append(float((coefs == 0).float().mean()))

        # Monotone non-decreasing sparsity, and the strongest penalty is
        # strictly sparser than the weakest.
        assert all(b >= a - 1e-12 for a, b in zip(fracs, fracs[1:]))
        assert fracs[-1] > fracs[0]

    def test_large_alpha_zeros_irrelevant_coefficients(self) -> None:
        """With a genuinely sparse ground truth, lasso recovers exact zeros.

        Only a handful of columns of X actually drive the target; a moderate
        lasso penalty must zero the noise-only coefficients exactly.
        """
        gen = torch.Generator().manual_seed(3)
        n, f, t = 400, 30, 2
        X = torch.randn(n, f, dtype=torch.float64, generator=gen)
        w_true = torch.zeros(f, t, dtype=torch.float64)
        w_true[:4] = torch.randn(4, t, dtype=torch.float64, generator=gen)  # 4 active
        y = X @ w_true + 0.01 * torch.randn(n, t, dtype=torch.float64, generator=gen)

        readout = FISTAReadout(f, t, alpha=2 * n * 0.05, l1_ratio=1.0, max_iter=20000, tol=1e-10)
        coefs, _ = readout._fit_impl(X, y)
        # Substantial sparsity: most of the noise coefficients are zeroed.
        assert float((coefs == 0).float().mean()) > 0.5
        # The active columns keep a non-trivial coefficient.
        assert (coefs[:4].abs() > 1e-2).all()


# ---------------------------------------------------------------------------
# Acceptance criterion 3: l1_ratio=0 (pure ridge) matches RidgeReadoutLayer
# ---------------------------------------------------------------------------


class TestRidgeLimit:
    """l1_ratio=0 reduces to ridge and converges to RidgeReadoutLayer."""

    def test_l1_ratio_zero_matches_ridge(self) -> None:
        """FISTA(l1_ratio=0, alpha=2a) equals RidgeReadoutLayer(alpha=a).

        The FISTA L2 term carries sklearn's 0.5 factor
        (``0.5·alpha·(1-l1_ratio)·‖W‖²``) while ``RidgeReadoutLayer`` minimises
        ``alpha·‖W‖²`` with no 0.5, so ``alpha_fista = 2·alpha_ridge`` matches.
        """
        X, y = _sparse_problem(noise=0.05)
        alpha_ridge = 1.0

        ridge = RidgeReadoutLayer(X.shape[1], y.shape[1], alpha=alpha_ridge)
        rc, ri = ridge._fit_impl(X, y)

        fista = FISTAReadout(
            X.shape[1],
            y.shape[1],
            alpha=2 * alpha_ridge,
            l1_ratio=0.0,
            max_iter=50000,
            tol=1e-12,
        )
        fc, fi = fista._fit_impl(X, y)

        assert ri is not None and fi is not None
        assert torch.allclose(fc, rc, atol=1e-6)
        assert torch.allclose(fi, ri, atol=1e-6)
        assert fista.converged_


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------


class TestConvergence:
    """Convergence bookkeeping and the non-convergence warning."""

    def test_converges_sets_diagnostics(self) -> None:
        """A well-posed fit converges and records n_iter_ / converged_."""
        X, y = _sparse_problem()
        readout = FISTAReadout(X.shape[1], y.shape[1], alpha=1.0, max_iter=10000, tol=1e-8)
        readout._fit_impl(X, y)
        assert readout.converged_
        assert 0 < readout.n_iter_ < 10000

    def test_non_convergence_warns(self) -> None:
        """Hitting max_iter without meeting tol warns and marks not-converged."""
        X, y = _sparse_problem()
        readout = FISTAReadout(X.shape[1], y.shape[1], alpha=1.0, max_iter=2, tol=1e-15)
        with pytest.warns(RuntimeWarning, match="did not converge"):
            readout._fit_impl(X, y)
        assert not readout.converged_
        assert readout.n_iter_ == 2

    def test_power_iteration_step_is_stable(self) -> None:
        """A power-iteration Lipschitz estimate still yields a convergent fit.

        The estimate is inflated by ``power_iter_margin`` so the step stays
        below the true 1/L; the fit must remain finite and match the exact
        eigenvalue path closely.
        """
        X, y = _sparse_problem()
        n = X.shape[0]
        exact = FISTAReadout(
            X.shape[1], y.shape[1], alpha=2 * n * 1e-2, l1_ratio=1.0, max_iter=50000, tol=1e-11
        )
        ce, _ = exact._fit_impl(X, y)

        power = FISTAReadout(
            X.shape[1],
            y.shape[1],
            alpha=2 * n * 1e-2,
            l1_ratio=1.0,
            max_iter=50000,
            tol=1e-11,
            power_iters=30,
        )
        cp, _ = power._fit_impl(X, y)
        assert torch.isfinite(cp).all()
        assert torch.allclose(cp, ce, atol=1e-5)


# ---------------------------------------------------------------------------
# fit() bookkeeping shared via the base class
# ---------------------------------------------------------------------------


class TestFitBookkeeping:
    """FISTAReadout inherits the base fit() guards and 2D/3D handling."""

    def test_fit_3d_and_is_fitted(self) -> None:
        """3D inputs are flattened and is_fitted flips True."""
        torch.manual_seed(0)
        readout = FISTAReadout(in_features=10, out_features=3, alpha=1e-2)
        X = torch.randn(4, 25, 10)
        y = torch.randn(4, 25, 3)
        readout.fit(X, y)
        assert readout.is_fitted
        assert readout.weight.shape == (3, 10)

    def test_fit_wrong_in_features_raises(self) -> None:
        """A wrong in_features surfaces the base-class ValueError."""
        readout = FISTAReadout(in_features=20, out_features=5)
        with pytest.raises(ValueError, match="state feature dimension"):
            readout.fit(torch.randn(100, 16), torch.randn(100, 5))

    def test_no_bias_leaves_intercept_none(self) -> None:
        """bias=False fits no intercept and does not center."""
        X, y = _sparse_problem()
        readout = FISTAReadout(X.shape[1], y.shape[1], bias=False, alpha=1.0)
        coefs, intercept = readout._fit_impl(X, y)
        assert intercept is None
        assert torch.isfinite(coefs).all()


# ---------------------------------------------------------------------------
# ESNTrainer compatibility (drop-in, no trainer changes)
# ---------------------------------------------------------------------------


class TestESNTrainerCompatibility:
    """FISTAReadout trains end-to-end through ESNTrainer, keyed by name."""

    def test_trainer_fits_and_forecasts(self) -> None:
        """ESNTrainer fits the named readout via its pre-hook, forecast is finite."""
        torch.manual_seed(0)
        inp = ps.Input((30, 3))
        reservoir = ESNLayer(50, feedback_size=3, spectral_radius=0.9)(inp)
        readout = FISTAReadout(50, 3, name="output", alpha=1e-3)(reservoir)
        model = ESNModel(inp, readout)

        warmup = torch.randn(1, 30, 3)
        train = torch.randn(1, 30, 3)
        targets = torch.randn(1, 30, 3)

        trainer = ESNTrainer(model)
        # A short random train window need not fully converge; silence the
        # advisory warning so it does not fail the test under -W error.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
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


# ---------------------------------------------------------------------------
# Device coverage + determinism
# ---------------------------------------------------------------------------


class TestDevice:
    """FISTAReadout fits and predicts on every available device, deterministically."""

    def test_fit_on_device(self, device: torch.device) -> None:
        """Fitting and predicting land on the target device."""
        torch.manual_seed(0)
        readout = FISTAReadout(in_features=20, out_features=5, alpha=1e-2).to(device)
        X = torch.randn(100, 20, device=device)
        y = torch.randn(100, 5, device=device)
        readout.fit(X, y)
        assert readout.is_fitted
        assert readout.weight.device.type == device.type
        pred = readout(torch.randn(10, 20, device=device))
        assert pred.device.type == device.type
        assert pred.shape == (10, 5)

    def test_deterministic(self) -> None:
        """FISTA has no RNG: two fits on identical data give identical weights."""
        X, y = _sparse_problem()
        r1 = FISTAReadout(X.shape[1], y.shape[1], alpha=1.0, max_iter=5000, tol=1e-9)
        r2 = FISTAReadout(X.shape[1], y.shape[1], alpha=1.0, max_iter=5000, tol=1e-9)
        c1, _ = r1._fit_impl(X, y)
        c2, _ = r2._fit_impl(X, y)
        assert torch.equal(c1, c2)

    @pytest.mark.gpu
    def test_cpu_gpu_consistent(self) -> None:
        """A CPU and a CUDA fit of the same problem agree (deterministic solver)."""
        if not torch.cuda.is_available():  # pragma: no cover - guarded by marker
            pytest.skip("CUDA not available")
        X, y = _sparse_problem()
        cpu = FISTAReadout(
            X.shape[1],
            y.shape[1],
            alpha=2 * X.shape[0] * 1e-2,
            l1_ratio=1.0,
            max_iter=50000,
            tol=1e-11,
        )
        cc, _ = cpu._fit_impl(X, y)
        gpu = FISTAReadout(
            X.shape[1],
            y.shape[1],
            alpha=2 * X.shape[0] * 1e-2,
            l1_ratio=1.0,
            max_iter=50000,
            tol=1e-11,
        )
        gc, _ = gpu._fit_impl(X.cuda(), y.cuda())
        assert torch.allclose(cc, gc.cpu(), atol=1e-6)
