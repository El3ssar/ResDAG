r"""Noise-injection / LMNT regularization for autonomous-rollout robustness.

State-noise (a.k.a. LMNT) regularization is a **standard, tunable tool** for
closed-loop (autonomous-rollout) robustness, not an automatic forecast-quality
win.  Training the readout on noise-perturbed reservoir states trades a little
one-step accuracy for closed-loop stability — the injected noise mimics the
feedback error the model sees when it free-runs on its own output (Jaeger's
state-noise idea; Wikner et al. 2024, *Stabilizing machine learning prediction
of dynamics*, Chaos 34, 043105).  Whether that trade nets out positive on any
given problem is **problem- and σ-dependent: it can help or hurt, and σ must be
tuned per problem.**  These tests deliberately assert **none** of "LMNT beats
plain ridge on VPT" — that claim is not robust across seeds, so asserting it
would only encode an overfit result.

What *is* robust — and what this module verifies — is the **mechanism**:

1. **The math is exact.**  The closed-form **LMNT** term (a single
   :math:`N\sigma^2` / :math:`(N-1)\sigma^2` diagonal add) reproduces the
   expected normal equations for explicitly noise-perturbed states. A single
   LMNT solve is the limit of averaging infinitely many noisy Gram matrices and
   right-hand sides before solving; it is not the mean of separately fitted
   weights. Verified against (a) a hand-built ``(RᵀR + coeff·I)⁻¹RᵀY`` reference,
   (b) Monte-Carlo noisy normal equations, and (c) the Gram diagonal structure.

2. **``state_noise == 0`` is a byte-identical no-op** — the golden-forecast
   contract: the default path must not perturb any existing fit by even a ULP.

3. **Larger σ ⇒ more shrinkage** — a monotone regularization effect, again
   seed-agnostic.

The end-to-end tests train a known periodic signal with both closed-form and
explicit-noise regularization, then check the aligned autonomous continuation.
Finite Monte-Carlo fits need not have the same chaotic forecast horizon as their
expectation: a small coefficient difference can move a threshold crossing by
many steps. Such a seed-dependent VPT comparison is not a correctness invariant.

See Also
--------
resdag.layers.readouts.RidgeReadoutLayer : Readout carrying the ``state_noise`` knob.
resdag.layers.readouts._solve_utils.lmnt_gram_add : The closed-form LMNT term.
resdag.training.ESNTrainer.fit : Trainer plumbing for ``state_noise``.
"""

import pytest
import torch

from resdag.core import ESNModel
from resdag.datasets import lorenz
from resdag.layers.readouts import CholeskyReadoutLayer, ReadoutLayer, RidgeReadoutLayer
from resdag.layers.readouts._solve_utils import (
    apply_lmnt_regularizer,
    build_ridge_problem,
    lmnt_gram_add,
)
from resdag.models import ott_esn
from resdag.training import ESNTrainer


def _readout(model: ESNModel) -> ReadoutLayer:
    """Return the single readout layer of a premade model (by module walk)."""
    readouts = [m for m in model.modules() if isinstance(m, ReadoutLayer)]
    assert len(readouts) == 1, f"expected one readout, found {len(readouts)}"
    return readouts[0]


# --------------------------------------------------------------------------- #
# Closed-form LMNT math: exactness + noise=0 identity. All seed-agnostic.
# --------------------------------------------------------------------------- #


def test_lmnt_gram_add_coefficients() -> None:
    """The diagonal add is ``N*sigma^2`` raw and ``(N-1)*sigma^2`` centered."""
    n, sigma = 500, 0.05
    raw = lmnt_gram_add(n, sigma, fit_intercept=False)
    centered = lmnt_gram_add(n, sigma, fit_intercept=True)
    assert raw == pytest.approx(n * sigma**2)
    assert centered == pytest.approx((n - 1) * sigma**2)
    # sigma == 0 is exactly zero (no floating fuzz) so the fit stays identical.
    assert lmnt_gram_add(n, 0.0, fit_intercept=True) == 0.0
    assert lmnt_gram_add(n, 0.0, fit_intercept=False) == 0.0


def test_lmnt_gram_add_rejects_negative_noise() -> None:
    """A negative noise std is a programming error, not a silent no-op."""
    with pytest.raises(ValueError, match="non-negative"):
        lmnt_gram_add(100, -0.1, fit_intercept=False)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_lmnt_gram_is_plain_plus_diagonal_only(fit_intercept: bool) -> None:
    """LMNT adds *only* a scalar to the Gram diagonal, nothing off-diagonal.

    The whole LMNT claim rests on the noise covariance being ``coeff * I`` — a
    single scalar on the diagonal (``N*sigma^2`` raw, ``(N-1)*sigma^2``
    centered), with the off-diagonal Gram entries untouched. Verified directly on
    the assembled Gram, independent of any seed.
    """
    torch.manual_seed(0)
    n, f, t = 400, 30, 3
    r = torch.randn(n, f, dtype=torch.float64) * 2.0 + 5.0  # non-zero mean
    y = torch.randn(n, t, dtype=torch.float64)
    sigma = 0.1

    plain = build_ridge_problem(
        r, y, fit_intercept=fit_intercept, use_float64=True, gram_dtype=torch.float64
    )
    lmnt = build_ridge_problem(
        r, y, fit_intercept=fit_intercept, use_float64=True, gram_dtype=torch.float64
    )
    apply_lmnt_regularizer(lmnt, sigma)

    coeff = lmnt_gram_add(n, sigma, fit_intercept=fit_intercept)
    expected = (n - 1) * sigma**2 if fit_intercept else n * sigma**2
    assert coeff == pytest.approx(expected)

    delta = lmnt.gram - plain.gram
    diag = delta.diagonal()
    off_diag = delta - torch.diag(diag)
    # Diagonal gains exactly ``coeff``; off-diagonal is untouched.
    assert torch.allclose(diag, torch.full_like(diag, coeff), atol=1e-12)
    assert off_diag.abs().max() == pytest.approx(0.0, abs=1e-12)
    # And the RHS is untouched by LMNT (noise is zero-mean, independent of y).
    assert torch.equal(lmnt.rhs, plain.rhs)


@pytest.mark.parametrize("bias", [True, False])
def test_lmnt_weights_match_hand_built_ridge(bias: bool) -> None:
    """LMNT solved weights == a from-scratch ``(RᵀR + coeff·I)⁻¹ RᵀY`` solve.

    The strongest exactness check: solve the ridge system by hand with the LMNT
    coefficient folded into the diagonal (``alpha + N*sigma^2`` raw / ``alpha +
    (N-1)*sigma^2`` centered) and confirm the fitted readout lands on it to
    machine precision. Seed-agnostic — it is an algebraic identity.
    """
    torch.manual_seed(11)
    n, f, t = 300, 40, 3
    states = torch.randn(1, n, f, dtype=torch.float64) * 1.5 + 2.0
    targets = torch.randn(1, n, t, dtype=torch.float64)
    alpha, sigma = 1e-4, 0.1

    ro = RidgeReadoutLayer(f, t, alpha=alpha, bias=bias, state_noise=sigma).double()
    ro.fit(states, targets)

    # Hand-built reference: (centered when bias) normal equations + LMNT diagonal.
    xd = states.reshape(-1, f).to(torch.float64)
    yd = targets.reshape(-1, t).to(torch.float64)
    coeff = alpha + lmnt_gram_add(n, sigma, fit_intercept=bias)
    if bias:
        xbar, ybar = xd.mean(0, keepdim=True), yd.mean(0, keepdim=True)
        gram = xd.T @ xd - n * (xbar.T @ xbar)
        rhs = xd.T @ yd - n * (xbar.T @ ybar)
    else:
        gram, rhs = xd.T @ xd, xd.T @ yd
    w_hand = torch.linalg.solve(gram + coeff * torch.eye(f, dtype=torch.float64), rhs)

    rel = (ro.weight.T - w_hand).norm() / w_hand.norm()
    assert rel < 1e-10, f"LMNT weights differ from hand-built ridge: rel={rel:.3e}"


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_apply_lmnt_matches_monte_carlo_noisy_gram(fit_intercept: bool) -> None:
    """LMNT's analytic Gram add equals the expectation of the noisy Gram.

    Draw many explicit ``R -> R + xi`` perturbations, form the (centered when
    ``fit_intercept``) Gram for each, average — this Monte-Carlo mean must match
    the plain Gram plus the closed-form LMNT diagonal term.
    """
    torch.manual_seed(0)
    n, f, t = 400, 30, 3
    r = torch.randn(n, f, dtype=torch.float64) * 2.0 + 5.0  # non-zero mean
    y = torch.randn(n, t, dtype=torch.float64)
    sigma = 0.1

    plain = build_ridge_problem(
        r, y, fit_intercept=fit_intercept, use_float64=True, gram_dtype=torch.float64
    )
    lmnt = build_ridge_problem(
        r, y, fit_intercept=fit_intercept, use_float64=True, gram_dtype=torch.float64
    )
    apply_lmnt_regularizer(lmnt, sigma)

    # Monte-Carlo expectation of the noisy Gram.
    gen = torch.Generator().manual_seed(7)
    m = 4000
    gram_mc = torch.zeros_like(plain.gram)
    for _ in range(m):
        rn = r + sigma * torch.randn(n, f, dtype=torch.float64, generator=gen)
        prob = build_ridge_problem(
            rn, y, fit_intercept=fit_intercept, use_float64=True, gram_dtype=torch.float64
        )
        gram_mc += prob.gram
    gram_mc /= m

    # The analytic LMNT Gram matches the MC mean; the residual is pure sampling
    # noise (shrinks like 1/sqrt(m)). Compare relative to the Gram scale.
    rel = (lmnt.gram - gram_mc).norm() / plain.gram.norm()
    assert rel < 5e-3, f"LMNT Gram deviates from MC mean: rel={rel:.3e}"


def test_state_noise_zero_is_byte_identical() -> None:
    """``state_noise=0`` reproduces the plain ridge fit bit-for-bit.

    This is the golden-forecast contract: the default path must not perturb any
    existing fit by even a ULP.
    """
    torch.manual_seed(1)
    states = torch.randn(4, 200, 40, dtype=torch.float64)
    targets = torch.randn(4, 200, 3, dtype=torch.float64)

    for cls in (RidgeReadoutLayer, CholeskyReadoutLayer):
        plain = cls(40, 3, alpha=1e-4).double()
        plain.fit(states, targets)
        noised0 = cls(40, 3, alpha=1e-4, state_noise=0.0).double()
        noised0.fit(states, targets)
        assert torch.equal(plain.weight, noised0.weight), cls.__name__
        assert torch.equal(plain.bias, noised0.bias), cls.__name__


@pytest.mark.parametrize("bias", [True, False])
def test_lmnt_readout_matches_explicit_noise_average(bias: bool) -> None:
    """A closed-form LMNT fit ≈ an explicit many-draw noisy fit (same weights).

    LMNT solves the expected noisy normal equations. The explicit path averages
    many noisy Gram matrices and right-hand sides before its single solve, so
    its fitted weights approach the LMNT solution as sampling error vanishes.
    """
    torch.manual_seed(2)
    states = torch.randn(4, 250, 40, dtype=torch.float64)
    targets = torch.randn(4, 250, 3, dtype=torch.float64)
    sigma = 0.1

    lmnt = RidgeReadoutLayer(40, 3, alpha=1e-4, bias=bias, state_noise=sigma).double()
    lmnt.fit(states, targets)

    explicit = RidgeReadoutLayer(
        40, 3, alpha=1e-4, bias=bias, state_noise=sigma, noise_draws=3000, noise_seed=0
    ).double()
    explicit.fit(states, targets)

    rel = (lmnt.weight - explicit.weight).norm() / lmnt.weight.norm()
    assert rel < 1e-2, f"LMNT vs explicit-noise weights differ: rel={rel:.3e}"


def test_larger_sigma_shrinks_weights_monotonically() -> None:
    """Larger ``state_noise`` ⇒ strictly more shrinkage (monotone regularization).

    The LMNT term adds ``~N*sigma^2`` to the Gram diagonal, a monotonically
    increasing ridge penalty in ``sigma``, so the fitted weight norm must
    decrease as ``sigma`` grows. A seed-agnostic property of the mechanism (not a
    claim about forecast quality).
    """
    torch.manual_seed(3)
    states = torch.randn(4, 200, 40, dtype=torch.float64)
    targets = torch.randn(4, 200, 3, dtype=torch.float64)

    norms = []
    for sigma in (0.0, 0.05, 0.1, 0.2, 0.4):
        ro = RidgeReadoutLayer(40, 3, alpha=1e-6, state_noise=sigma).double()
        ro.fit(states, targets)
        norms.append(float(ro.weight.norm()))
    # Strictly decreasing: each larger sigma shrinks the weights further.
    for prev, cur in zip(norms, norms[1:]):
        assert cur < prev, f"weight norm did not shrink with larger sigma: {norms}"


def test_cholesky_and_ridge_lmnt_agree() -> None:
    """The LMNT path is identical across the two direct ridge readouts."""
    torch.manual_seed(4)
    states = torch.randn(4, 200, 40, dtype=torch.float64)
    targets = torch.randn(4, 200, 3, dtype=torch.float64)
    ridge = RidgeReadoutLayer(40, 3, alpha=1e-4, state_noise=0.1).double()
    chol = CholeskyReadoutLayer(40, 3, alpha=1e-4, state_noise=0.1).double()
    ridge.fit(states, targets)
    chol.fit(states, targets)
    assert torch.allclose(ridge.weight, chol.weight, atol=1e-10)
    assert torch.allclose(ridge.bias, chol.bias, atol=1e-10)


# --------------------------------------------------------------------------- #
# Trainer plumbing.
# --------------------------------------------------------------------------- #


def test_trainer_state_noise_zero_is_identical() -> None:
    """``ESNTrainer.fit(state_noise=0)`` leaves the readout fit unchanged."""
    torch.manual_seed(5)
    series = lorenz(3000, seed=0, dtype=torch.float64, normalize="standard")
    warmup, train_in, train_tgt = series[:, :300], series[:, 300:1300], series[:, 301:1301]

    def build_fit(state_noise: float) -> torch.Tensor:
        model = ott_esn(
            reservoir_size=150, feedback_size=3, output_size=3, readout="ridge", seed=0
        ).double()
        ESNTrainer(model).fit(
            warmup_inputs=(warmup,),
            train_inputs=(train_in,),
            targets={"output": train_tgt},
            state_noise=state_noise,
        )
        return _readout(model).weight.detach().clone()

    assert torch.equal(build_fit(0.0), build_fit(0.0))


def test_trainer_state_noise_changes_the_fit() -> None:
    """A positive ``state_noise`` actually reaches the readout solve."""
    torch.manual_seed(6)
    series = lorenz(3000, seed=0, dtype=torch.float64, normalize="standard")
    warmup, train_in, train_tgt = series[:, :300], series[:, 300:1300], series[:, 301:1301]

    def build_fit(state_noise: float) -> torch.Tensor:
        model = ott_esn(
            reservoir_size=150, feedback_size=3, output_size=3, readout="ridge", seed=0
        ).double()
        ESNTrainer(model).fit(
            warmup_inputs=(warmup,),
            train_inputs=(train_in,),
            targets={"output": train_tgt},
            state_noise=state_noise,
        )
        return _readout(model).weight.detach().clone()

    plain = build_fit(0.0)
    noised = build_fit(1e-2)
    assert not torch.equal(plain, noised)
    # Restored afterwards: the readout's configured state_noise is untouched.
    model = ott_esn(reservoir_size=150, feedback_size=3, output_size=3, readout="ridge", seed=0)
    assert _readout(model).state_noise == 0.0  # type: ignore[attr-defined]


def test_trainer_state_noise_negative_raises() -> None:
    """A negative ``state_noise`` is rejected before any fitting happens."""
    series = lorenz(1500, seed=0, dtype=torch.float64, normalize="standard")
    warmup, train_in, train_tgt = series[:, :200], series[:, 200:700], series[:, 201:701]
    model = ott_esn(
        reservoir_size=100, feedback_size=3, output_size=3, readout="ridge", seed=0
    ).double()
    with pytest.raises(ValueError, match="non-negative"):
        ESNTrainer(model).fit(
            warmup_inputs=(warmup,),
            train_inputs=(train_in,),
            targets={"output": train_tgt},
            state_noise=-0.1,
        )


def test_trainer_state_noise_on_unsupported_readout_raises() -> None:
    """Requesting noise for a readout without a ``state_noise`` knob errors loudly.

    Silently ignoring it would train an unregularised readout while the caller
    believes noise is applied — a correctness trap, so it raises instead.
    """
    series = lorenz(1500, seed=0, dtype=torch.float64, normalize="standard")
    warmup, train_in, train_tgt = series[:, :200], series[:, 200:700], series[:, 201:701]
    # CGReadoutLayer has no state_noise attribute.
    model = ott_esn(
        reservoir_size=100, feedback_size=3, output_size=3, readout="cg", seed=0
    ).double()
    with pytest.raises(TypeError, match="do not support it"):
        ESNTrainer(model).fit(
            warmup_inputs=(warmup,),
            train_inputs=(train_in,),
            targets={"output": train_tgt},
            state_noise=1e-3,
        )


# --------------------------------------------------------------------------- #
# Public trainer -> autonomous continuation on a known periodic signal.
# The analytic/Monte-Carlo identities above verify the regularization mechanism.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 109, 726])
@pytest.mark.parametrize("noise_draws", [None, 20])
def test_noise_regularized_training_preserves_periodic_continuation(
    seed: int, noise_draws: int | None
) -> None:
    """Both regularizers learn an aligned continuation without assuming equal VPT."""
    t = torch.arange(900, dtype=torch.float64) * 0.07
    series = torch.stack((torch.sin(t), torch.cos(t)), dim=-1).unsqueeze(0)
    model = ott_esn(
        80,
        feedback_size=2,
        output_size=2,
        spectral_radius=0.9,
        readout="ridge",
        readout_alpha=1e-6,
        seed=seed,
    ).double()
    ESNTrainer(model).fit(
        (series[:, :100],),
        (series[:, 100:699],),
        {"output": series[:, 101:700]},
        state_noise=1e-3,
        noise_draws=noise_draws,
        noise_seed=seed,
    )
    prediction = model.forecast(
        series[:, 599:699], initial_feedback=series[:, 699:700], horizon=100
    )
    assert prediction.shape == (1, 100, 2)
    assert torch.isfinite(prediction).all()
    assert ((prediction - series[:, 700:800]) ** 2).mean() < 1e-3
