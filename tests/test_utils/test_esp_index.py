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

import pytest
import pytorch_symbolic as ps
import torch

from resdag import ESNModel
from resdag.layers import CGReadoutLayer, Concatenate, ESNLayer, NGReservoir
from resdag.layers.reservoirs.base_reservoir import BaseReservoirLayer
from resdag.utils.states import esp_index

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
    """NG-RC-only model: Input -> NGReservoir -> CGReadout."""
    inp = ps.Input((seq_len, feedback_dim))
    res = NGReservoir(input_dim=feedback_dim, k=k, s=1, p=2)(inp)
    out = CGReadoutLayer(res.shape[-1], feedback_dim, name="output")(res)
    return ESNModel(inp, out)


def _mixed_model(seq_len: int = 20, feedback_dim: int = 3) -> ESNModel:
    """Parallel ESN + NG-RC reservoirs concatenated into one readout."""
    inp = ps.Input((seq_len, feedback_dim))
    esn = ESNLayer(40, feedback_size=feedback_dim, spectral_radius=0.9)(inp)
    ng = NGReservoir(input_dim=feedback_dim, k=2, s=1, p=2)(inp)
    feat = Concatenate()(esn, ng)
    out = CGReadoutLayer(feat.shape[-1], feedback_dim, name="output")(feat)
    return ESNModel(inp, out)


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
# NG-RC path (cell-agnostic discovery + 3-D state)
# ---------------------------------------------------------------------------


def test_ngrc_reservoir_is_discovered_and_scored() -> None:
    """A pure NG-RC model is no longer silently skipped (issue #178)."""
    model = _ngrc_model(k=2)
    fb = torch.randn(2, 20, 3)

    result = esp_index(model, fb, iterations=3, verbose=False)

    assert len(result) == 1  # the NGReservoir was found and scored
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
    ng1 = NGReservoir(input_dim=feedback_dim, k=1, p=2)(inp)  # empty state -> skipped
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
    """The ``verbose=True`` branch prints per-iteration progress."""
    torch.manual_seed(0)
    model = _esn_model(seq_len=20, device=device)
    fb = torch.randn(2, 20, 3, device=device)

    esp_index(model, fb, iterations=2, verbose=True)

    captured = capsys.readouterr()
    assert "Iteration 1/2" in captured.out
    assert "Iteration 2/2" in captured.out
