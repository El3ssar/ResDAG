"""Direct-solve readout contracts: Ridge / SVD / Pinv layers + _solve_utils.

Pins down the three direct-solve readouts introduced alongside the iterative
:class:`CGReadoutLayer`:

- ``RidgeReadoutLayer`` — Cholesky / LU solve of the regularised normal
  equations; matches CG to ``< 1e-8`` on well-conditioned fits.
- ``SVDReadoutLayer`` — SVD filter factors; handles ``alpha=0`` and
  rank-deficient Gram matrices where CG under-converges, with a lower train
  residual.
- ``PinvReadoutLayer`` — ``lstsq`` / ``pinv`` least squares with an ``rcond``
  cutoff.

Plus: the shared ``_solve_utils`` centering / precision helper that all three
reuse (no copy-paste of the CG centering block), and that all three are
exported and drop-in compatible with ``ESNTrainer``.
"""

import warnings

import pytest
import pytorch_symbolic as ps
import torch

from resdag import (
    CGReadoutLayer,
    ESNLayer,
    ESNModel,
    PinvReadoutLayer,
    ReadoutLayer,
    RidgeReadoutLayer,
    SVDReadoutLayer,
)
from resdag.layers.readouts import _solve_utils
from resdag.training import ESNTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def closed_form_ridge(
    X: torch.Tensor, y: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form ridge-with-intercept on centered data.

    Returns ``(coefs (F, T), intercept (T,))``.
    """
    x_mean = X.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    xc = X - x_mean
    yc = y - y_mean
    gram = xc.T @ xc + alpha * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
    coefs = torch.linalg.solve(gram, xc.T @ yc)
    intercept = (y_mean - x_mean @ coefs).squeeze(0)
    return coefs, intercept


def train_residual(coefs: torch.Tensor, intercept: torch.Tensor, X, y) -> torch.Tensor:
    """Sum-of-squares train residual ``||X W + b - y||^2``."""
    pred = X @ coefs + intercept
    return ((pred - y) ** 2).sum()


# ---------------------------------------------------------------------------
# Shared _solve_utils
# ---------------------------------------------------------------------------


class TestSolveUtils:
    """The shared centering / precision helper used by every direct solver."""

    def test_resolve_solve_dtype(self) -> None:
        """solve dtype is float64 when use_float64 else the input dtype."""
        assert _solve_utils.resolve_solve_dtype(torch.float32, True) == torch.float64
        assert _solve_utils.resolve_solve_dtype(torch.float32, False) == torch.float32

    def test_resolve_gram_dtype_auto_cpu(self) -> None:
        """Auto gram dtype is the solve dtype on CPU (float64 is cheap there)."""
        dt = _solve_utils.resolve_gram_dtype(
            torch.float32, torch.device("cpu"), torch.float64, None
        )
        assert dt == torch.float64

    def test_resolve_gram_dtype_explicit_override(self) -> None:
        """An explicit gram_dtype wins over the auto policy."""
        dt = _solve_utils.resolve_gram_dtype(
            torch.float32, torch.device("cpu"), torch.float64, torch.float32
        )
        assert dt == torch.float32

    def test_build_problem_centered_gram_matches_manual(self) -> None:
        """The analytic centered Gram equals the explicitly-centered Gram."""
        torch.manual_seed(0)
        X = torch.randn(120, 8, dtype=torch.float64) + 3.0
        y = torch.randn(120, 4, dtype=torch.float64)

        prob = _solve_utils.build_ridge_problem(
            X, y, fit_intercept=True, use_float64=True, gram_dtype=None
        )

        xc = X - X.mean(dim=0, keepdim=True)
        yc = y - y.mean(dim=0, keepdim=True)
        assert torch.allclose(prob.gram, xc.T @ xc, atol=1e-8)
        assert torch.allclose(prob.rhs, xc.T @ yc, atol=1e-8)

    def test_build_problem_no_intercept_uses_raw_gram(self) -> None:
        """Without an intercept the raw (uncentered) Gram is used."""
        torch.manual_seed(1)
        X = torch.randn(80, 6, dtype=torch.float64) + 2.0
        y = torch.randn(80, 3, dtype=torch.float64)

        prob = _solve_utils.build_ridge_problem(
            X, y, fit_intercept=False, use_float64=True, gram_dtype=None
        )

        assert prob.x_mean is None and prob.y_mean is None
        assert torch.allclose(prob.gram, X.T @ X, atol=1e-8)

    def test_recover_intercept_none_without_bias(self) -> None:
        """No intercept is recovered for a bias-free problem."""
        torch.manual_seed(2)
        X = torch.randn(50, 5, dtype=torch.float64)
        y = torch.randn(50, 2, dtype=torch.float64)
        prob = _solve_utils.build_ridge_problem(
            X, y, fit_intercept=False, use_float64=True, gram_dtype=None
        )
        assert _solve_utils.recover_intercept(prob, torch.zeros(5, 2)) is None


# ---------------------------------------------------------------------------
# Instantiation / validation
# ---------------------------------------------------------------------------


class TestInstantiation:
    """Construction, inheritance, and constructor validation."""

    @pytest.mark.parametrize("cls", [RidgeReadoutLayer, SVDReadoutLayer, PinvReadoutLayer])
    def test_is_readout_subclass(self, cls) -> None:
        """All three are ReadoutLayer subclasses (so ESNTrainer accepts them)."""
        readout = cls(in_features=20, out_features=3)
        assert isinstance(readout, ReadoutLayer)
        assert readout.in_features == 20
        assert readout.out_features == 3
        assert not readout.is_fitted

    def test_ridge_rejects_bad_solver(self) -> None:
        """RidgeReadoutLayer validates the solver name."""
        with pytest.raises(ValueError, match="solver must be one of"):
            RidgeReadoutLayer(10, 2, solver="banana")

    def test_pinv_rejects_bad_solver(self) -> None:
        """PinvReadoutLayer validates the solver name."""
        with pytest.raises(ValueError, match="solver must be one of"):
            PinvReadoutLayer(10, 2, solver="banana")

    @pytest.mark.parametrize("cls", [RidgeReadoutLayer, SVDReadoutLayer])
    def test_negative_alpha_rejected(self, cls) -> None:
        """alpha < 0 raises at construction."""
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            cls(10, 2, alpha=-1e-6)

    @pytest.mark.parametrize("cls", [SVDReadoutLayer, PinvReadoutLayer])
    def test_negative_rcond_rejected(self, cls) -> None:
        """rcond < 0 raises at construction."""
        with pytest.raises(ValueError, match="rcond must be non-negative"):
            cls(10, 2, rcond=-1e-9)

    def test_repr_contains_solver(self) -> None:
        """repr surfaces the solver and key hyperparameters."""
        assert "solver='cholesky'" in repr(RidgeReadoutLayer(10, 2, name="r"))
        assert "rcond=" in repr(SVDReadoutLayer(10, 2, name="s"))
        assert "solver='lstsq'" in repr(PinvReadoutLayer(10, 2, name="p"))


# ---------------------------------------------------------------------------
# Acceptance criterion 1: Ridge(cholesky) matches CG to < 1e-8
# ---------------------------------------------------------------------------


class TestRidgeMatchesCG:
    """RidgeReadoutLayer matches CG and the closed form on well-conditioned fits."""

    @pytest.mark.parametrize("solver", ["cholesky", "solve"])
    def test_matches_closed_form(self, solver: str) -> None:
        """Direct ridge solve equals the closed-form solution to < 1e-8."""
        torch.manual_seed(42)
        X = torch.randn(200, 20, dtype=torch.float64)
        y = torch.randn(200, 5, dtype=torch.float64)
        alpha = 1e-3

        readout = RidgeReadoutLayer(20, 5, alpha=alpha, solver=solver)
        coefs, intercept = readout._fit_impl(X, y)

        coefs_cf, intercept_cf = closed_form_ridge(X, y, alpha)
        assert torch.allclose(coefs, coefs_cf, atol=1e-8, rtol=1e-7)
        assert intercept is not None
        assert torch.allclose(intercept, intercept_cf, atol=1e-8, rtol=1e-7)

    def test_matches_cg_within_1e8(self) -> None:
        """Cholesky ridge matches a tightly-converged CG fit to < 1e-8."""
        torch.manual_seed(7)
        X = torch.randn(300, 30, dtype=torch.float64)
        y = torch.randn(300, 4, dtype=torch.float64)
        alpha = 1e-4

        ridge = RidgeReadoutLayer(30, 4, alpha=alpha, solver="cholesky")
        coefs_r, intercept_r = ridge._fit_impl(X, y)

        cg = CGReadoutLayer(30, 4, alpha=alpha, max_iter=5000, tol=1e-14)
        coefs_cg, intercept_cg = cg._solve_ridge_cg(X, y, alpha)

        assert torch.allclose(coefs_r, coefs_cg, atol=1e-8, rtol=1e-7)
        assert intercept_r is not None and intercept_cg is not None
        assert torch.allclose(intercept_r, intercept_cg, atol=1e-8, rtol=1e-7)

    def test_no_bias_solves_raw_normal_equations(self) -> None:
        """bias=False solves the uncentered ridge system (no intercept)."""
        torch.manual_seed(3)
        alpha = 1e-2
        X = torch.randn(300, 10, dtype=torch.float64) + 1.0
        y = torch.randn(300, 4, dtype=torch.float64)

        readout = RidgeReadoutLayer(10, 4, bias=False, alpha=alpha, solver="solve")
        coefs, intercept = readout._fit_impl(X, y)

        assert intercept is None
        gram = X.T @ X + alpha * torch.eye(10, dtype=torch.float64)
        expected = torch.linalg.solve(gram, X.T @ y)
        assert torch.allclose(coefs, expected, atol=1e-8)


# ---------------------------------------------------------------------------
# Acceptance criterion 2: SVD alpha=0 on rank-deficient NG-RC features beats CG
# ---------------------------------------------------------------------------


class TestSVDRankDeficient:
    """SVDReadoutLayer handles alpha=0 and rank-deficient feature matrices."""

    def _exactly_rank_deficient(self) -> tuple[torch.Tensor, torch.Tensor]:
        """An NG-RC-like feature matrix with an *exactly* duplicated column.

        Polynomial powers of a low-dim signal are highly collinear; the
        duplicate column forces true (exact) rank deficiency so a Cholesky /
        normal-equations solve is singular while SVD still recovers a solution.
        """
        torch.manual_seed(11)
        base = torch.randn(150, 3, dtype=torch.float64)
        feats = [base, base**2, base**3, base[:, :1] * base[:, 1:2]]
        X = torch.cat(feats, dim=1)
        X = torch.cat([X, X[:, :1]], dim=1)  # duplicate -> exact rank deficiency
        true_w = torch.randn(X.shape[1], 2, dtype=torch.float64)
        y = X @ true_w
        return X, y

    def _ill_conditioned_ngrc(self) -> tuple[torch.Tensor, torch.Tensor]:
        """A severely ill-conditioned high-degree NG-RC monomial matrix.

        High-degree monomials of a narrow-range signal span a huge dynamic
        range of singular values (``cond(Gram) ~ 1e16`` here), which is exactly
        the regime where a fixed-budget CG solver stalls before converging.
        """
        torch.manual_seed(11)
        base = torch.rand(400, 5, dtype=torch.float64) * 0.4 + 0.3  # narrow range
        feats = [torch.ones(400, 1, dtype=torch.float64)]
        for p in range(1, 8):  # degrees 1..7
            feats.append(base**p)
        X = torch.cat(feats, dim=1)
        true_w = torch.randn(X.shape[1], 2, dtype=torch.float64)
        y = X @ true_w + 0.05 * torch.randn(400, 2, dtype=torch.float64)
        return X, y

    def test_svd_alpha0_fits_rank_deficient(self) -> None:
        """SVD with alpha=0 fits an exactly rank-deficient Gram without NaNs."""
        X, y = self._exactly_rank_deficient()

        readout = SVDReadoutLayer(X.shape[1], 2, bias=False, alpha=0.0)
        coefs, intercept = readout._fit_impl(X, y)

        assert intercept is None
        assert torch.isfinite(coefs).all()
        # Reconstruct the targets despite rank deficiency.
        assert torch.allclose(X @ coefs, y, atol=1e-6)

    def test_svd_lower_train_residual_than_cg(self) -> None:
        """On a high-degree NG-RC feature matrix where CG under-converges,
        SVD (alpha=0) attains a strictly lower train residual."""
        X, y = self._ill_conditioned_ngrc()

        svd = SVDReadoutLayer(X.shape[1], 2, bias=False, alpha=0.0)
        coefs_svd, _ = svd._fit_impl(X, y)
        res_svd = ((X @ coefs_svd - y) ** 2).sum()

        # CG on the catastrophically ill-conditioned normal equations stalls
        # well before its tolerance within a realistic iteration budget.
        cg = CGReadoutLayer(X.shape[1], 2, bias=False, alpha=0.0, max_iter=50, tol=1e-6)
        coefs_cg, _ = cg._solve_ridge_cg(X, y, 0.0)
        res_cg = ((X @ coefs_cg - y) ** 2).sum()

        assert torch.isfinite(coefs_cg).all()
        assert res_svd < res_cg

    def test_svd_alpha_positive_matches_closed_form(self) -> None:
        """With alpha>0 the SVD filter factors equal the closed-form ridge."""
        torch.manual_seed(9)
        X = torch.randn(120, 15, dtype=torch.float64)
        y = torch.randn(120, 3, dtype=torch.float64)
        alpha = 1e-2

        readout = SVDReadoutLayer(15, 3, alpha=alpha)
        coefs, intercept = readout._fit_impl(X, y)

        coefs_cf, intercept_cf = closed_form_ridge(X, y, alpha)
        assert torch.allclose(coefs, coefs_cf, atol=1e-8, rtol=1e-7)
        assert intercept is not None
        assert torch.allclose(intercept, intercept_cf, atol=1e-8, rtol=1e-7)


# ---------------------------------------------------------------------------
# Pinv readout
# ---------------------------------------------------------------------------


class TestPinvReadout:
    """PinvReadoutLayer least-squares behaviour."""

    @pytest.mark.parametrize("solver", ["lstsq", "pinv"])
    def test_recovers_linear_map(self, solver: str) -> None:
        """A noiseless linear map is recovered by least squares.

        Data is float64 here for solver accuracy; predictions are compared in
        float64 since the layer copies the fitted weights into its (float32)
        parameters.
        """
        torch.manual_seed(5)
        X = torch.randn(200, 12, dtype=torch.float64)
        true_w = torch.randn(12, 4, dtype=torch.float64)
        true_b = torch.randn(4, dtype=torch.float64)
        y = X @ true_w + true_b

        readout = PinvReadoutLayer(12, 4, solver=solver)
        coefs, intercept = readout._fit_impl(X, y)

        assert intercept is not None
        assert torch.allclose(X @ coefs + intercept, y, atol=1e-7)

    @pytest.mark.parametrize("solver", ["lstsq", "pinv"])
    def test_rank_deficient_minimum_norm(self, solver: str) -> None:
        """Rank-deficient least squares stays finite and fits the targets."""
        torch.manual_seed(6)
        base = torch.randn(100, 5, dtype=torch.float64)
        X = torch.cat([base, base[:, :2]], dim=1)  # 2 duplicate columns
        true_w = torch.randn(X.shape[1], 3, dtype=torch.float64)
        y = X @ true_w

        readout = PinvReadoutLayer(X.shape[1], 3, bias=False, solver=solver)
        coefs, intercept = readout._fit_impl(X, y)

        assert intercept is None
        assert torch.isfinite(coefs).all()
        assert torch.allclose(X @ coefs, y, atol=1e-6)


# ---------------------------------------------------------------------------
# fit() bookkeeping shared via the base class
# ---------------------------------------------------------------------------


class TestFitBookkeeping:
    """Each layer inherits the base fit() guards and 2D/3D handling."""

    @pytest.mark.parametrize("cls", [RidgeReadoutLayer, SVDReadoutLayer, PinvReadoutLayer])
    def test_fit_3d_and_is_fitted(self, cls) -> None:
        """3D inputs are flattened and is_fitted flips True."""
        torch.manual_seed(0)
        readout = cls(in_features=10, out_features=3)
        X = torch.randn(4, 25, 10)
        y = torch.randn(4, 25, 3)
        readout.fit(X, y)
        assert readout.is_fitted
        assert readout.weight.shape == (3, 10)

    @pytest.mark.parametrize("cls", [RidgeReadoutLayer, SVDReadoutLayer, PinvReadoutLayer])
    def test_fit_wrong_in_features_raises(self, cls) -> None:
        """A wrong in_features surfaces the base-class ValueError."""
        readout = cls(in_features=20, out_features=5)
        with pytest.raises(ValueError, match="state feature dimension"):
            readout.fit(torch.randn(100, 16), torch.randn(100, 5))

    @pytest.mark.parametrize("cls", [RidgeReadoutLayer, SVDReadoutLayer, PinvReadoutLayer])
    def test_fit_sample_mismatch_raises(self, cls) -> None:
        """Mismatched sample counts surface the base-class ValueError."""
        readout = cls(in_features=20, out_features=5)
        with pytest.raises(ValueError, match="sample count mismatch"):
            readout.fit(torch.randn(100, 20), torch.randn(50, 5))


# ---------------------------------------------------------------------------
# Acceptance criterion 4: drop-in compatible with ESNTrainer (no trainer changes)
# ---------------------------------------------------------------------------


class TestESNTrainerCompatibility:
    """Each direct readout trains end-to-end through ESNTrainer, keyed by name."""

    @pytest.mark.parametrize(
        "make_readout",
        [
            lambda n: RidgeReadoutLayer(50, 3, name="output", alpha=1e-6),
            lambda n: SVDReadoutLayer(50, 3, name="output", alpha=1e-6),
            lambda n: PinvReadoutLayer(50, 3, name="output"),
        ],
    )
    def test_trainer_fits_readout(self, make_readout) -> None:
        """ESNTrainer fits the named readout via its pre-hook unchanged."""
        torch.manual_seed(0)
        inp = ps.Input((30, 3))
        reservoir = ESNLayer(50, feedback_size=3, spectral_radius=0.9)(inp)
        readout = make_readout(None)(reservoir)
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

        # The readout was fitted and the model forecasts a finite trajectory.
        readout_layer = next(m for m in model.modules() if isinstance(m, ReadoutLayer))
        assert readout_layer.is_fitted
        pred = model.forecast(warmup, horizon=20)
        assert pred.shape == (1, 20, 3)
        assert torch.isfinite(pred).all()


# ---------------------------------------------------------------------------
# Device coverage
# ---------------------------------------------------------------------------


class TestDevice:
    """Direct readouts fit and predict on every available device."""

    @pytest.mark.parametrize("cls", [RidgeReadoutLayer, SVDReadoutLayer, PinvReadoutLayer])
    def test_fit_on_device(self, cls, device: torch.device) -> None:
        """Fitting and predicting land on the target device."""
        torch.manual_seed(0)
        readout = cls(in_features=20, out_features=5).to(device)
        X = torch.randn(100, 20, device=device)
        y = torch.randn(100, 5, device=device)
        readout.fit(X, y)
        assert readout.is_fitted
        assert readout.weight.device.type == device.type
        pred = readout(torch.randn(10, 20, device=device))
        assert pred.device.type == device.type
        assert pred.shape == (10, 5)


# ---------------------------------------------------------------------------
# Robust ridge solve: the shared solve_ridge helper (Cholesky footgun fixes)
# ---------------------------------------------------------------------------


def _ill_conditioned_design(
    n_features: int,
    cond: float,
    n_samples: int | None = None,
    seed: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A design matrix ``X`` with a prescribed ``cond(X)`` and its targets.

    The singular values are spaced logarithmically from ``1`` to ``1/cond`` so
    ``cond(XᵀX) = cond²`` is controllable — the regime where forming the Gram in
    float32 destroys accuracy a float64 solve cannot recover.
    """
    n = n_samples if n_samples is not None else 4 * n_features
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n_features, dtype=torch.float64, generator=gen)
    u, _, vh = torch.linalg.svd(a, full_matrices=False)
    s = torch.logspace(0, -torch.log10(torch.tensor(cond)), n_features, dtype=torch.float64)
    x = (u * s) @ vh
    true_w = torch.randn(n_features, 3, dtype=torch.float64, generator=gen)
    y = x @ true_w + 1e-3 * torch.randn(n, 3, dtype=torch.float64, generator=gen)
    return x, y


class TestSolveRidgeRobustness:
    """The shared robust solver: no crash on rank deficiency, self-announcing loss."""

    def test_well_conditioned_bit_identical_to_bare_cholesky(self) -> None:
        """On a healthy Gram, solve_ridge is bit-identical to a bare Cholesky solve.

        The robustness wrapper must not perturb the well-conditioned default
        path — the escalated jitter and fallbacks only fire on failure.
        """
        torch.manual_seed(0)
        X = torch.randn(300, 30, dtype=torch.float64)
        y = torch.randn(300, 4, dtype=torch.float64)
        alpha = 1e-4

        prob = _solve_utils.build_ridge_problem(
            X, y, fit_intercept=True, use_float64=True, gram_dtype=None
        )
        eye = torch.eye(prob.n_features, dtype=prob.solve_dtype)
        bare = torch.cholesky_solve(prob.rhs, torch.linalg.cholesky(prob.gram + alpha * eye))

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a warning here would fail the test
            robust = _solve_utils.solve_ridge(
                prob.gram, prob.rhs, alpha, layer_name="ridge", gram_dtype=prob.X.dtype
            )
        assert torch.equal(robust, bare)

    def test_rank_deficient_no_crash_and_warns(self) -> None:
        """A rank-deficient Gram (F > N) is solved finitely with an escalation warning.

        This is the CPU analogue of the CUDA float32 footgun: F=64 features from
        N=40 samples makes the centered Gram singular. A bare Cholesky raises;
        solve_ridge escalates the jitter (or falls back to eigh) and warns.
        """
        torch.manual_seed(0)
        # float32 gram so the singular Gram trips Cholesky the way it does on GPU.
        X = torch.randn(40, 64, dtype=torch.float32)
        y = torch.randn(40, 3, dtype=torch.float32)
        prob = _solve_utils.build_ridge_problem(
            X, y, fit_intercept=True, use_float64=False, gram_dtype=torch.float32
        )
        with pytest.warns(RuntimeWarning, match="rank-deficient|escalated"):
            coefs = _solve_utils.solve_ridge(
                prob.gram, prob.rhs, 1e-6, layer_name="ridge", gram_dtype=torch.float32
            )
        assert torch.isfinite(coefs).all()
        assert coefs.shape == (64, 3)

    def test_indefinite_gram_eigh_fallback(self) -> None:
        """An indefinite Gram (negative eigenvalue > jitter) uses the eigh fallback.

        Both the plain and the jitter-escalated Cholesky fail on a genuinely
        indefinite matrix (a float32-rounding artifact can drive a near-zero
        eigenvalue meaningfully negative); the eigh-clamped fallback must still
        recover a finite solution.
        """
        torch.manual_seed(3)
        n_features = 10
        q, _ = torch.linalg.qr(torch.randn(n_features, n_features, dtype=torch.float64))
        # A spectrum with one eigenvalue that stays negative past the jitter.
        evals = torch.tensor(
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1, 0.01, 1e-4, -0.3], dtype=torch.float64
        )
        gram = q @ torch.diag(evals) @ q.T
        gram = (gram + gram.T) / 2  # enforce exact symmetry
        rhs = torch.randn(n_features, 2, dtype=torch.float64)

        eigh_calls = 0
        original_eigh = torch.linalg.eigh

        def _counting_eigh(*args, **kwargs):
            nonlocal eigh_calls
            eigh_calls += 1
            return original_eigh(*args, **kwargs)

        torch.linalg.eigh = _counting_eigh
        try:
            with pytest.warns(RuntimeWarning, match="rank-deficient|escalated"):
                coefs = _solve_utils.solve_ridge(
                    gram, rhs, 1e-6, layer_name="ridge", gram_dtype=torch.float64
                )
        finally:
            torch.linalg.eigh = original_eigh

        assert eigh_calls == 1  # the fallback genuinely fired
        assert torch.isfinite(coefs).all()

    def test_illconditioned_float32_gram_warns(self) -> None:
        """A cond(X)=1e6 fit with a float32 Gram warns about lost precision.

        cond(XᵀX)=1e12 pushes float32's ~1e-7 entry error past unity, so the
        float64 solve returns a materially different readout than a float64 Gram
        would. The accuracy guard must surface that.
        """
        X, y = _ill_conditioned_design(n_features=400, cond=1e6)
        readout = RidgeReadoutLayer(400, 3, alpha=1e-6, gram_dtype=torch.float32)
        with pytest.warns(RuntimeWarning, match="losing accuracy|precision"):
            readout.fit(X.float(), y.float())

    def test_illconditioned_float64_gram_is_clean(self) -> None:
        """The same cond(X)=1e6 fit with a float64 Gram raises no warning.

        The precision guard keys off the Gram-formation dtype, so a float64 Gram
        (which does *not* lose the precision) must stay quiet.
        """
        X, y = _ill_conditioned_design(n_features=400, cond=1e6)
        readout = RidgeReadoutLayer(400, 3, alpha=1e-6, gram_dtype=torch.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            readout.fit(X, y)

    def test_well_conditioned_fit_does_not_warn(self) -> None:
        """A healthy fit (even with a float32 Gram) is silent."""
        X, y = _ill_conditioned_design(n_features=100, cond=1e2)
        readout = RidgeReadoutLayer(100, 3, alpha=1e-6, gram_dtype=torch.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            readout.fit(X.float(), y.float())

    @pytest.mark.gpu
    def test_rank_deficient_cuda_fit_no_longer_raises(self) -> None:
        """Regression: the library-default CUDA fit on a rank-deficient Gram.

        With F=256 features from N=200 samples on CUDA the auto float32 Gram is
        rank-deficient; a bare Cholesky used to raise a raw ``LinAlgError`` with
        no fallback. solve_ridge must now recover a finite fit.
        """
        if not torch.cuda.is_available():  # pragma: no cover - guarded by marker
            pytest.skip("CUDA not available")
        torch.manual_seed(0)
        X = torch.randn(1, 200, 256, device="cuda")
        y = torch.randn(1, 200, 3, device="cuda")
        readout = RidgeReadoutLayer(256, 3, alpha=1e-6).to("cuda")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            readout.fit(X, y)  # previously raised torch.linalg.LinAlgError
        assert readout.is_fitted
        assert torch.isfinite(readout.weight).all()
