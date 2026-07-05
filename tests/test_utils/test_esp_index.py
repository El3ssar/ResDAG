"""Tests for the cell-agnostic ESP index diagnostic.

These cover the four behaviours fixed in issue #178:

* discovery of every ``BaseReservoirLayer`` (ESN *and* NG-RC), not only
  ``ESNLayer``;
* all state mutation routed through the public ``reset_state`` /
  ``set_random_state`` API (no direct ``res.state = ...``);
* 3-D NG-RC state support;
* an explicit skip for reservoirs whose ESP is undefined.

They also cover the core convergence semantics tracked by issue #252:

* a stable reservoir (``spectral_radius < 1``) drives the ESP index toward
  zero — trajectories from different initial states forget their start;
* an unstable reservoir (``spectral_radius >> 1``) does *not* converge;
* both the ``history=True`` and ``history=False`` return shapes;
* the ``verbose=True`` progress-printing branch;
* every behaviour exercised on the shared ``device`` fixture.
"""

import sys

import pytest
import pytorch_symbolic as ps
import torch

from resdag import ESNModel
from resdag.layers import CGReadoutLayer, Concatenate, ESNLayer, NGReservoirLayer
from resdag.layers.reservoirs.base_reservoir import BaseReservoirLayer
from resdag.utils.states import esp_index

# The ``esp_index`` name re-exported from ``resdag.utils.states`` shadows the
# submodule of the same name as a package attribute, so ``import ... as`` binds
# to the function.  Reach the module object (needed to monkeypatch its private
# helpers) through ``sys.modules`` instead.
_esp_index_module = sys.modules["resdag.utils.states.esp_index"]

# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def _esn_model(
    seq_len: int = 20,
    feedback_dim: int = 3,
    size: int = 40,
    spectral_radius: float = 0.9,
    device: torch.device | str = "cpu",
) -> ESNModel:
    """ESN-only model: Input -> ESNLayer -> CGReadout."""
    inp = ps.Input((seq_len, feedback_dim))
    res = ESNLayer(size, feedback_size=feedback_dim, spectral_radius=spectral_radius)(inp)
    out = CGReadoutLayer(size, feedback_dim, name="output")(res)
    return ESNModel(inp, out).to(device)


def _ngrc_model(seq_len: int = 20, feedback_dim: int = 3, k: int = 2) -> ESNModel:
    """NG-RC-only model: Input -> NGReservoirLayer -> CGReadout."""
    inp = ps.Input((seq_len, feedback_dim))
    res = NGReservoirLayer(input_dim=feedback_dim, k=k, s=1, p=2)(inp)
    out = CGReadoutLayer(res.shape[-1], feedback_dim, name="output")(res)
    return ESNModel(inp, out)


def _mixed_model(seq_len: int = 20, feedback_dim: int = 3) -> ESNModel:
    """Parallel ESN + NG-RC reservoirs concatenated into one readout."""
    inp = ps.Input((seq_len, feedback_dim))
    esn = ESNLayer(40, feedback_size=feedback_dim, spectral_radius=0.9)(inp)
    ng = NGReservoirLayer(input_dim=feedback_dim, k=2, s=1, p=2)(inp)
    feat = Concatenate()(esn, ng)
    out = CGReadoutLayer(feat.shape[-1], feedback_dim, name="output")(feat)
    return ESNModel(inp, out)


def _driven_model(
    seq_len: int = 20,
    feedback_dim: int = 3,
    driver_dim: int = 2,
    size: int = 40,
    spectral_radius: float = 0.9,
    device: torch.device | str = "cpu",
) -> ESNModel:
    """Input-driven ESN model: (feedback, driver) -> ESNLayer -> CGReadout."""
    feedback = ps.Input((seq_len, feedback_dim))
    driver = ps.Input((seq_len, driver_dim))
    res = ESNLayer(
        size,
        feedback_size=feedback_dim,
        input_size=driver_dim,
        spectral_radius=spectral_radius,
    )(feedback, driver)
    out = CGReadoutLayer(size, feedback_dim, name="output")(res)
    return ESNModel([feedback, driver], out).to(device)


def _sequential_esp_reference(
    model: ESNModel,
    inputs: tuple[torch.Tensor, ...],
    *,
    iterations: int,
    transient: int = 0,
    window: int | None = None,
    relative: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute the ESP index the old way: one forward pass per restart.

    This is the pre-batching reference implementation, kept in the test suite so
    the batched :func:`esp_index` can be pinned against it.  Restarts are run one
    at a time at the caller's batch size, so the matmul tiling matches a
    ``chunk_size=1`` batched run exactly.
    """
    reservoirs = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, BaseReservoirLayer)
    ]

    def run_collect() -> dict[str, torch.Tensor]:
        collected: dict[str, torch.Tensor] = {}

        def make_hook(nm: str):  # type: ignore[no-untyped-def]
            def hook(mod: object, i: object, o: torch.Tensor) -> None:
                collected[nm] = o

            return hook

        handles = [r.register_forward_hook(make_hook(n)) for n, r in reservoirs]
        try:
            with torch.no_grad():
                model(*inputs)
        finally:
            for h in handles:
                h.remove()
        return collected

    batch_size = inputs[0].shape[0]
    for _, r in reservoirs:
        r.reset_state(batch_size=batch_size)
    base = run_collect()
    if transient > 0:
        base = {k: v[:, transient:, :] for k, v in base.items()}

    sums = {
        n: torch.zeros((), device=inputs[0].device, dtype=inputs[0].dtype) for n, _ in reservoirs
    }
    for _ in range(iterations):
        for _, r in reservoirs:
            r.set_random_state(batch_size=batch_size)
        rand = run_collect()
        if transient > 0:
            rand = {k: v[:, transient:, :] for k, v in rand.items()}
        for name, b in base.items():
            dist = torch.norm(b - rand[name], dim=-1)
            if relative:
                dist = dist / (torch.norm(b, dim=-1) + 1e-12)
            dist_for_index = dist[:, -window:] if window is not None else dist
            sums[name] += dist_for_index.mean()
    return {n: v / iterations for n, v in sums.items()}


# ---------------------------------------------------------------------------
# ESN path
# ---------------------------------------------------------------------------


def test_esn_returns_index_for_reservoir() -> None:
    model = _esn_model()
    fb = torch.randn(2, 20, 3)

    result = esp_index(model, fb, iterations=3, verbose=False)

    assert isinstance(result, dict)
    assert len(result) == 1
    (only_value,) = result.values()
    assert len(only_value) == 1
    assert only_value[0].ndim == 0  # scalar tensor
    assert torch.isfinite(only_value[0])
    assert only_value[0] >= 0


def test_history_shape() -> None:
    model = _esn_model(seq_len=15)
    fb = torch.randn(2, 15, 3)
    iterations = 4

    indices, history = esp_index(model, fb, iterations=iterations, history=True, verbose=False)

    assert set(indices) == set(history)
    for hist in history.values():
        # (iterations, timesteps, batch)
        assert hist[0].shape == (iterations, 15, 2)


def test_transient_trims_history() -> None:
    model = _esn_model(seq_len=30)
    fb = torch.randn(2, 30, 3)
    transient = 10

    _, history = esp_index(
        model, fb, iterations=2, history=True, transient=transient, verbose=False
    )
    for hist in history.values():
        assert hist[0].shape[1] == 30 - transient


def test_transient_too_large_raises() -> None:
    model = _esn_model(seq_len=10)
    fb = torch.randn(2, 10, 3)
    with pytest.raises(ValueError, match="transient"):
        esp_index(model, fb, transient=10, verbose=False)


# ---------------------------------------------------------------------------
# Tail / window statistic (issue #179, point 2)
# ---------------------------------------------------------------------------


def test_window_focuses_on_asymptotic_regime(device: torch.device) -> None:
    """``window`` averages only the tail, so it lies far below the full-window mean.

    For a healthy-ESP reservoir the trajectories start far apart and converge
    to ~0, so the full-window mean is dominated by the early transient while a
    short trailing window reflects the (near-zero) asymptotic distance.
    """
    torch.manual_seed(0)
    model = _esn_model(seq_len=120, size=60, spectral_radius=0.5, device=device)
    fb = (torch.randn(2, 120, 3, device=device) * 0.5).to(device)

    (full,) = esp_index(model, fb, iterations=4, verbose=False).values()
    (tail,) = esp_index(model, fb, iterations=4, window=20, verbose=False).values()

    assert torch.isfinite(full[0]) and torch.isfinite(tail[0])
    # The asymptotic-window index is much smaller than the transient-dominated
    # full-window index.
    assert tail[0] < full[0]
    assert tail[0] < full[0] * 1e-2


def test_window_does_not_change_history_length(device: torch.device) -> None:
    """``window`` only trims the index average; the history spans the full sequence."""
    torch.manual_seed(0)
    model = _esn_model(seq_len=40, device=device)
    fb = torch.randn(2, 40, 3, device=device)

    _, history = esp_index(model, fb, iterations=2, window=10, history=True, verbose=False)
    for hist in history.values():
        assert hist[0].shape == (2, 40, 2)  # (iterations, full timesteps, batch)


def test_window_combines_with_transient(device: torch.device) -> None:
    """``window`` is measured against post-transient timesteps."""
    torch.manual_seed(0)
    model = _esn_model(seq_len=40, device=device)
    fb = torch.randn(2, 40, 3, device=device)

    # 40 - transient(10) = 30 post-transient steps; window=30 is the maximum.
    (value,) = esp_index(model, fb, iterations=2, transient=10, window=30, verbose=False).values()
    assert torch.isfinite(value[0])


@pytest.mark.parametrize("bad_window", [0, -1, 25])
def test_window_out_of_range_raises(bad_window: int) -> None:
    """A non-positive or too-large window is rejected (21 post-transient steps)."""
    model = _esn_model(seq_len=30)
    fb = torch.randn(2, 30, 3)
    with pytest.raises(ValueError, match="window"):
        esp_index(model, fb, transient=9, window=bad_window, verbose=False)


# ---------------------------------------------------------------------------
# Relative-distance normalization (issue #179, point 3)
# ---------------------------------------------------------------------------


def test_relative_normalization_returns_finite_index(device: torch.device) -> None:
    """``relative=True`` yields a finite, non-negative scale-free index."""
    torch.manual_seed(0)
    model = _esn_model(seq_len=40, size=60, spectral_radius=0.9, device=device)
    fb = (torch.randn(2, 40, 3, device=device) * 0.5).to(device)

    (value,) = esp_index(model, fb, iterations=4, relative=True, verbose=False).values()
    assert torch.isfinite(value[0])
    assert value[0] >= 0


def test_relative_normalization_changes_the_index(device: torch.device) -> None:
    """Relative normalization produces a different number than the absolute index."""
    torch.manual_seed(0)
    model = _esn_model(seq_len=40, size=60, spectral_radius=10.0, device=device)
    fb = (torch.randn(2, 40, 3, device=device) * 0.5).to(device)

    (absolute,) = esp_index(model, fb, iterations=4, relative=False, verbose=False).values()
    (relative,) = esp_index(model, fb, iterations=4, relative=True, verbose=False).values()
    assert not torch.allclose(absolute[0], relative[0])


def test_relative_normalization_applies_to_history(device: torch.device) -> None:
    """The relative flag is reflected in the returned history tensors."""
    torch.manual_seed(0)
    model = _esn_model(seq_len=30, size=60, spectral_radius=10.0, device=device)
    fb = (torch.randn(2, 30, 3, device=device) * 0.5).to(device)

    _, abs_hist = esp_index(model, fb, iterations=2, history=True, verbose=False)
    torch.manual_seed(0)
    _, rel_hist = esp_index(model, fb, iterations=2, relative=True, history=True, verbose=False)

    (abs_h,) = abs_hist.values()
    (rel_h,) = rel_hist.values()
    assert not torch.allclose(abs_h[0], rel_h[0])


# ---------------------------------------------------------------------------
# Acceptance criterion: spectral-radius ranking on the asymptotic window
# ---------------------------------------------------------------------------


def test_spectral_radius_ranks_on_asymptotic_window(device: torch.device) -> None:
    """High spectral radius -> high index, low -> low, on the asymptotic window."""
    torch.manual_seed(0)
    fb = (torch.randn(2, 100, 3, device=device) * 0.5).to(device)

    low = _esn_model(seq_len=100, size=60, spectral_radius=0.5, device=device)
    high = _esn_model(seq_len=100, size=60, spectral_radius=10.0, device=device)

    (low_idx,) = esp_index(low, fb, iterations=4, window=20, verbose=False).values()
    (high_idx,) = esp_index(high, fb, iterations=4, window=20, verbose=False).values()

    assert torch.isfinite(low_idx[0]) and torch.isfinite(high_idx[0])
    # On the asymptotic window the stable reservoir has all but forgotten its
    # start, while the unstable one stays far from the base orbit.
    assert low_idx[0] < high_idx[0]
    assert low_idx[0] < 1e-2
    assert high_idx[0] > 1.0


def test_relative_window_ranks_spectral_radius(device: torch.device) -> None:
    """The relative + windowed index also ranks low below high spectral radius."""
    torch.manual_seed(0)
    fb = (torch.randn(2, 100, 3, device=device) * 0.5).to(device)

    low = _esn_model(seq_len=100, size=60, spectral_radius=0.5, device=device)
    high = _esn_model(seq_len=100, size=60, spectral_radius=10.0, device=device)

    (low_idx,) = esp_index(low, fb, iterations=4, window=20, relative=True, verbose=False).values()
    (high_idx,) = esp_index(
        high, fb, iterations=4, window=20, relative=True, verbose=False
    ).values()

    assert torch.isfinite(low_idx[0]) and torch.isfinite(high_idx[0])
    assert low_idx[0] < high_idx[0]


# ---------------------------------------------------------------------------
# NG-RC path (cell-agnostic discovery + 3-D state)
# ---------------------------------------------------------------------------


def test_ngrc_reservoir_is_discovered_and_scored() -> None:
    """A pure NG-RC model is no longer silently skipped (issue #178)."""
    model = _ngrc_model(k=2)
    fb = torch.randn(2, 20, 3)

    result = esp_index(model, fb, iterations=3, verbose=False)

    assert len(result) == 1  # the NGReservoirLayer was found and scored
    (value,) = result.values()
    assert torch.isfinite(value[0])
    # FIFO delay buffers driven by identical inputs converge exactly once the
    # buffer has refilled, so the ESP index is small but well-defined.
    assert value[0] >= 0


def test_ngrc_k1_is_skipped_with_explicit_message(capsys: pytest.CaptureFixture[str]) -> None:
    """A ``k=1`` NG-RC reservoir has an empty state, so its ESP is undefined."""
    model = _ngrc_model(k=1)
    fb = torch.randn(2, 20, 3)

    with pytest.raises(ValueError, match="well-defined ESP"):
        esp_index(model, fb, iterations=2, verbose=True)

    captured = capsys.readouterr()
    assert "Skipping reservoir" in captured.out
    assert "ESP is undefined" in captured.out


# ---------------------------------------------------------------------------
# Multi-reservoir model (acceptance criterion)
# ---------------------------------------------------------------------------


def test_multi_reservoir_model_scores_all_defined_reservoirs() -> None:
    model = _mixed_model()
    fb = torch.randn(2, 20, 3)

    result = esp_index(model, fb, iterations=3, verbose=False)

    # Both the ESN and the (k=2) NG-RC reservoir are present and scored.
    assert len(result) == 2
    for value in result.values():
        assert torch.isfinite(value[0])


def test_partial_skip_keeps_defined_reservoirs() -> None:
    """Mixed model where one reservoir is undefined: it is skipped, the rest stay."""
    feedback_dim = 3
    inp = ps.Input((15, feedback_dim))
    esn = ESNLayer(40, feedback_size=feedback_dim, spectral_radius=0.9)(inp)
    ng1 = NGReservoirLayer(input_dim=feedback_dim, k=1, p=2)(inp)  # empty state -> skipped
    feat = Concatenate()(esn, ng1)
    out = CGReadoutLayer(feat.shape[-1], feedback_dim, name="output")(feat)
    model = ESNModel(inp, out)
    fb = torch.randn(2, 15, 3)

    result = esp_index(model, fb, iterations=2, verbose=False)

    assert len(result) == 1  # only the ESN survives
    (name,) = result.keys()
    assert "ESN" in name


# ---------------------------------------------------------------------------
# Public state API is used (no direct ``res.state = ...``)
# ---------------------------------------------------------------------------


def test_state_is_set_through_public_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """``esp_index`` must route through reset_state / set_random_state, not assign state."""
    model = _esn_model()
    fb = torch.randn(2, 20, 3)

    reset_calls: list[int | None] = []
    random_calls = {"n": 0}

    orig_reset = BaseReservoirLayer.reset_state
    orig_random = BaseReservoirLayer.set_random_state

    def spy_reset(self: BaseReservoirLayer, batch_size: int | None = None) -> None:  # type: ignore[no-untyped-def]
        reset_calls.append(batch_size)
        orig_reset(self, batch_size=batch_size)

    def spy_random(self: BaseReservoirLayer, *args: object, **kwargs: object) -> None:  # type: ignore[no-untyped-def]
        random_calls["n"] += 1
        orig_random(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(BaseReservoirLayer, "reset_state", spy_reset)
    monkeypatch.setattr(BaseReservoirLayer, "set_random_state", spy_random)

    esp_index(model, fb, iterations=3, verbose=False)

    # Base orbit uses reset_state(batch_size=...); each iteration randomises.
    assert any(bs == 2 for bs in reset_calls)
    assert random_calls["n"] == 3


def test_no_reservoirs_raises() -> None:
    inp = ps.Input((5, 3))
    out = CGReadoutLayer(3, 3, name="output")(inp)
    model = ESNModel(inp, out)
    fb = torch.randn(2, 5, 3)
    with pytest.raises(ValueError, match="No reservoir layers"):
        esp_index(model, fb, verbose=False)


# ---------------------------------------------------------------------------
# Convergence semantics (issue #252) — on the shared device fixture
# ---------------------------------------------------------------------------


def _late_over_early(history: torch.Tensor) -> tuple[float, float]:
    """Return ``(early_mean, late_mean)`` per-timestep distances from a history tensor.

    Parameters
    ----------
    history : torch.Tensor
        ESP distance history of shape ``(iterations, timesteps, batch)``.

    Returns
    -------
    tuple of float
        The mean distance over the first ten timesteps and over the last ten,
        each averaged across iterations and the batch dimension.
    """
    per_timestep = history.mean(dim=(0, 2))  # (timesteps,)
    window = min(10, per_timestep.shape[0])
    early = per_timestep[:window].mean().item()
    late = per_timestep[-window:].mean().item()
    return early, late


def test_stable_reservoir_converges_to_near_zero(device: torch.device) -> None:
    """A ``spectral_radius < 1`` reservoir forgets its initial state: ESP -> 0."""
    torch.manual_seed(0)
    model = _esn_model(seq_len=80, size=60, spectral_radius=0.9, device=device)
    fb = (torch.randn(2, 80, 3, device=device) * 0.5).to(device)

    indices, history = esp_index(model, fb, iterations=4, history=True, verbose=False)

    (value,) = indices.values()
    assert torch.isfinite(value[0])

    (hist,) = history.values()
    early, late = _late_over_early(hist[0])
    # Trajectories converge: the tail distance collapses far below the start.
    assert late < early
    assert late < 1e-3
    assert late / early < 1e-2


def test_unstable_reservoir_does_not_converge(device: torch.device) -> None:
    """A ``spectral_radius >> 1`` reservoir keeps trajectories apart: no ESP."""
    torch.manual_seed(0)
    model = _esn_model(seq_len=80, size=60, spectral_radius=10.0, device=device)
    fb = (torch.randn(2, 80, 3, device=device) * 0.5).to(device)

    indices, history = esp_index(model, fb, iterations=4, history=True, verbose=False)

    (value,) = indices.values()
    assert torch.isfinite(value[0])
    # The averaged index stays well away from zero.
    assert value[0] > 1.0

    (hist,) = history.values()
    early, late = _late_over_early(hist[0])
    # No convergence: the tail distance is the same order as the start.
    assert late > 1.0
    assert late / early > 0.5


def test_stable_beats_unstable_convergence(device: torch.device) -> None:
    """The ESP index ranks a stable reservoir below an unstable one."""
    torch.manual_seed(0)
    fb = (torch.randn(2, 80, 3, device=device) * 0.5).to(device)

    stable = _esn_model(seq_len=80, size=60, spectral_radius=0.9, device=device)
    unstable = _esn_model(seq_len=80, size=60, spectral_radius=10.0, device=device)

    (stable_idx,) = esp_index(stable, fb, iterations=4, verbose=False).values()
    (unstable_idx,) = esp_index(unstable, fb, iterations=4, verbose=False).values()

    assert stable_idx[0] < unstable_idx[0]


def test_history_false_returns_plain_dict(device: torch.device) -> None:
    """``history=False`` returns just the index dict (no history tuple)."""
    torch.manual_seed(0)
    model = _esn_model(seq_len=30, device=device)
    fb = torch.randn(2, 30, 3, device=device)

    result = esp_index(model, fb, iterations=3, history=False, verbose=False)

    assert isinstance(result, dict)
    (value,) = result.values()
    assert value[0].ndim == 0
    assert value[0].device.type == device.type


def test_history_true_returns_index_and_history(device: torch.device) -> None:
    """``history=True`` returns a ``(indices, history)`` tuple with matching keys."""
    torch.manual_seed(0)
    iterations = 3
    model = _esn_model(seq_len=25, device=device)
    fb = torch.randn(2, 25, 3, device=device)

    out = esp_index(model, fb, iterations=iterations, history=True, verbose=False)

    assert isinstance(out, tuple)
    indices, history = out
    assert set(indices) == set(history)
    (hist,) = history.values()
    assert hist[0].shape == (iterations, 25, 2)
    assert hist[0].device.type == device.type


def test_verbose_prints_progress(device: torch.device, capsys: pytest.CaptureFixture[str]) -> None:
    """The ``verbose=True`` branch prints restart-batch progress."""
    torch.manual_seed(0)
    model = _esn_model(seq_len=20, device=device)
    fb = torch.randn(2, 20, 3, device=device)

    esp_index(model, fb, iterations=2, verbose=True)

    captured = capsys.readouterr()
    # Restarts are batched into one forward pass, so progress is reported per
    # chunk of restarts rather than per iteration.
    assert "Restarts 2/2" in captured.out


# ---------------------------------------------------------------------------
# Batched restarts: equivalence to the sequential computation (perf refactor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "transient, window, relative",
    [
        (0, None, False),
        (5, 20, False),
        (0, None, True),
        (0, 10, True),
        (3, 15, False),
    ],
)
def test_chunk_size_one_matches_sequential_bit_for_bit(
    device: torch.device, transient: int, window: int | None, relative: bool
) -> None:
    """``chunk_size=1`` reproduces the pre-batching sequential index bit-for-bit.

    Running each restart at the caller's batch size (chunk of one) uses the same
    matmul tiling as the old one-forward-per-restart loop, so — with the random
    restarts drawn in the same order — the result is exactly equal, not merely
    close.
    """
    torch.manual_seed(7)
    model = _esn_model(seq_len=40, size=60, spectral_radius=10.0, device=device)
    fb = (torch.randn(2, 40, 3, device=device) * 0.5).to(device)

    torch.manual_seed(555)
    reference = _sequential_esp_reference(
        model, (fb,), iterations=8, transient=transient, window=window, relative=relative
    )
    torch.manual_seed(555)
    batched = esp_index(
        model,
        fb,
        iterations=8,
        transient=transient,
        window=window,
        relative=relative,
        chunk_size=1,
        verbose=False,
    )

    assert set(batched) == set(reference)
    for name, ref in reference.items():
        assert torch.equal(batched[name][0], ref)


def test_full_batch_matches_sequential_to_fp_tolerance(device: torch.device) -> None:
    """The fully-batched default matches the sequential index to FP rounding.

    Folding every restart into one wide forward pass reassociates the matmul, so
    the value differs from the sequential one only by floating-point rounding
    (well within ``atol=1e-5``), never by a change in which random states are
    used.
    """
    torch.manual_seed(7)
    model = _esn_model(seq_len=40, size=60, spectral_radius=0.9, device=device)
    fb = (torch.randn(2, 40, 3, device=device) * 0.5).to(device)

    torch.manual_seed(555)
    reference = _sequential_esp_reference(model, (fb,), iterations=10)
    torch.manual_seed(555)
    batched = esp_index(model, fb, iterations=10, verbose=False)

    for name, ref in reference.items():
        assert torch.allclose(batched[name][0], ref, atol=1e-5, rtol=1e-5)


def test_chunk_size_does_not_change_drawn_states(device: torch.device) -> None:
    """The states are pre-drawn, so ``chunk_size`` never changes which are used.

    A run split into chunks of three must equal the single-pass run bit-for-bit,
    because chunking only groups the *forward passes*, not the RNG draws (both
    tile the same number of restarts per matmul).
    """
    torch.manual_seed(7)
    model = _esn_model(seq_len=30, size=50, spectral_radius=0.9, device=device)
    fb = torch.randn(2, 30, 3, device=device)

    torch.manual_seed(555)
    full = esp_index(model, fb, iterations=9, chunk_size=9, verbose=False)
    torch.manual_seed(555)
    chunked = esp_index(model, fb, iterations=9, chunk_size=3, verbose=False)

    for name, value in full.items():
        assert torch.equal(chunked[name][0], value[0])


def test_seeded_result_is_deterministic(device: torch.device) -> None:
    """The same seed yields the same index across independent batched runs."""
    torch.manual_seed(7)
    model = _esn_model(seq_len=40, size=60, spectral_radius=10.0, device=device)
    fb = torch.randn(2, 40, 3, device=device)

    torch.manual_seed(555)
    first = esp_index(model, fb, iterations=10, relative=True, verbose=False)
    torch.manual_seed(555)
    second = esp_index(model, fb, iterations=10, relative=True, verbose=False)

    for name, value in first.items():
        assert torch.equal(second[name][0], value[0])


def test_chunk_size_matches_sequential_for_driven_model(device: torch.device) -> None:
    """Batching also tiles driving inputs, so a driven model matches sequential."""
    torch.manual_seed(7)
    model = _driven_model(seq_len=30, size=50, spectral_radius=0.9, device=device)
    fb = torch.randn(2, 30, 3, device=device)
    driver = torch.randn(2, 30, 2, device=device)

    torch.manual_seed(555)
    reference = _sequential_esp_reference(model, (fb, driver), iterations=6)
    torch.manual_seed(555)
    batched = esp_index(model, fb, driver, iterations=6, chunk_size=1, verbose=False)

    for name, ref in reference.items():
        assert torch.equal(batched[name][0], ref)


def test_multi_reservoir_chunk_one_matches_sequential() -> None:
    """A mixed ESN + NG-RC model matches the sequential reference per reservoir."""
    torch.manual_seed(7)
    model = _mixed_model(seq_len=25)
    fb = torch.randn(2, 25, 3)

    torch.manual_seed(555)
    reference = _sequential_esp_reference(model, (fb,), iterations=5)
    torch.manual_seed(555)
    batched = esp_index(model, fb, iterations=5, chunk_size=1, verbose=False)

    assert set(batched) == set(reference)
    for name, ref in reference.items():
        assert torch.equal(batched[name][0], ref)


# ---------------------------------------------------------------------------
# Caller reservoir state is saved and restored
# ---------------------------------------------------------------------------


def test_warmed_states_are_restored(device: torch.device) -> None:
    """A warmed-up reservoir state is unchanged after ``esp_index`` returns."""
    torch.manual_seed(7)
    model = _esn_model(seq_len=30, size=50, device=device)
    fb = torch.randn(2, 30, 3, device=device)

    model(fb)  # warm up -> non-None state
    before = model.get_reservoir_states()
    assert before  # sanity: at least one warmed reservoir

    esp_index(model, fb, iterations=5, verbose=False)

    after = model.get_reservoir_states()
    assert set(after) == set(before)
    for name, state in before.items():
        assert torch.equal(after[name], state)


def test_none_states_are_restored_to_none(device: torch.device) -> None:
    """A reservoir left at ``None`` (never warmed) is reset back to ``None``."""
    torch.manual_seed(7)
    model = _esn_model(seq_len=30, size=50, device=device)
    fb = torch.randn(2, 30, 3, device=device)

    model.reset_reservoirs()  # all states None
    assert model.get_reservoir_states() == {}

    esp_index(model, fb, iterations=4, verbose=False)

    # Still None: the diagnostic must not leave an internal batched state behind.
    assert model.get_reservoir_states() == {}
    for module in model.modules():
        if isinstance(module, BaseReservoirLayer):
            assert module.state is None


def test_states_restored_even_on_error(device: torch.device) -> None:
    """The saved states are restored even when the computation raises."""
    torch.manual_seed(7)
    model = _esn_model(seq_len=30, size=50, device=device)
    fb = torch.randn(2, 30, 3, device=device)

    model(fb)
    before = model.get_reservoir_states()

    with pytest.raises(ValueError, match="transient"):
        esp_index(model, fb, iterations=5, transient=100, verbose=False)

    after = model.get_reservoir_states()
    assert set(after) == set(before)
    for name, state in before.items():
        assert torch.equal(after[name], state)


# ---------------------------------------------------------------------------
# chunk_size validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_chunk", [0, -1, -5])
def test_non_positive_chunk_size_raises(bad_chunk: int) -> None:
    """A non-positive ``chunk_size`` is rejected."""
    model = _esn_model(seq_len=20)
    fb = torch.randn(2, 20, 3)
    with pytest.raises(ValueError, match="chunk_size"):
        esp_index(model, fb, iterations=3, chunk_size=bad_chunk, verbose=False)


def test_chunk_size_larger_than_iterations_is_clamped(device: torch.device) -> None:
    """A ``chunk_size`` exceeding ``iterations`` behaves like the single-pass run."""
    torch.manual_seed(7)
    model = _esn_model(seq_len=25, size=50, device=device)
    fb = torch.randn(2, 25, 3, device=device)

    torch.manual_seed(555)
    big = esp_index(model, fb, iterations=4, chunk_size=100, verbose=False)
    torch.manual_seed(555)
    default = esp_index(model, fb, iterations=4, verbose=False)

    for name, value in default.items():
        assert torch.equal(big[name][0], value[0])


# ---------------------------------------------------------------------------
# Out-of-memory fallback (chunk_size=None auto-shrinks on CUDA OOM)
# ---------------------------------------------------------------------------


def test_oom_fallback_halves_chunk_and_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``chunk_size=None`` a CUDA OOM shrinks the chunk until it fits.

    The forward batching is simulated to OOM whenever more than two restarts
    share a pass, forcing the fallback to halve the chunk (8 -> 4 -> 2).  Because
    the restart states are pre-drawn, the final result must still match the
    single-pass reference.
    """
    torch.manual_seed(7)
    model = _esn_model(seq_len=25, size=40, spectral_radius=0.9)
    fb = torch.randn(2, 25, 3)

    torch.manual_seed(555)
    reference = _sequential_esp_reference(model, (fb,), iterations=8)

    original_run_chunk = _esp_index_module._run_chunk

    def fault_on_large_chunk(*args: object, **kwargs: object) -> object:
        chunk_states = args[3]  # positional: (model, reservoirs, inputs, chunk_states, ...)
        assert isinstance(chunk_states, list)
        if len(chunk_states) > 2:
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return original_run_chunk(*args, **kwargs)

    monkeypatch.setattr(_esp_index_module, "_run_chunk", fault_on_large_chunk)

    torch.manual_seed(555)
    result = esp_index(model, fb, iterations=8, verbose=False)

    # The pre-drawn states make the result robust to how the passes are chunked,
    # so the OOM-shrunk run still matches the sequential reference to FP rounding.
    for name, ref in reference.items():
        assert torch.allclose(result[name][0], ref, atol=1e-5, rtol=1e-5)


def test_explicit_chunk_size_oom_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``chunk_size`` is honoured: its OOM propagates, not auto-shrunk."""
    torch.manual_seed(7)
    model = _esn_model(seq_len=25, size=40)
    fb = torch.randn(2, 25, 3)

    def always_oom(*args: object, **kwargs: object) -> object:
        raise torch.cuda.OutOfMemoryError("simulated OOM")

    monkeypatch.setattr(_esp_index_module, "_run_chunk", always_oom)

    with pytest.raises(torch.cuda.OutOfMemoryError):
        esp_index(model, fb, iterations=8, chunk_size=4, verbose=False)
