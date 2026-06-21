"""End-to-end integration tests for NG-RC inside ESNModel.

``NGReservoir`` / ``NGCell`` have thorough *unit* coverage in
``tests/test_layers/test_ngrc.py``, and one test there chains a
``CGReadoutLayer`` directly onto the layer output.  This module instead drops
an ``NGReservoir`` into the ``pytorch_symbolic`` DAG, wraps it in an
:class:`~resdag.core.ESNModel`, trains it with
:class:`~resdag.training.ESNTrainer`, and forecasts — exercising the pieces no
unit test reaches:

- model-level training (``ESNTrainer.fit``) over an NG-RC feature map;
- autoregressive :meth:`~resdag.core.ESNModel.forecast` through the flat
  single-step engine, which threads NG-RC's **3-D** delay buffer
  ``(batch, state_size, input_dim)`` separately while squeezing the per-step
  input to 2-D;
- model-level state management (:meth:`~resdag.core.ESNModel.reset_reservoirs`,
  :meth:`~resdag.core.ESNModel.get_reservoir_states` /
  :meth:`~resdag.core.ESNModel.set_reservoir_states`) over that 3-D buffer,
  which is shaped unlike the ESN's 2-D ``(batch, reservoir_size)`` state;
- the ``(k-1)*s`` warmup-length contract observed at the *model* level.

All tests are seeded and deterministic.  Tracked by #317 (parent #96).
"""

import pytest
import torch

from resdag.core import ESNModel, Input
from resdag.layers import CGReadoutLayer, NGReservoir
from resdag.layers.readouts import ReadoutLayer
from resdag.layers.reservoirs import BaseReservoirLayer
from resdag.training import ESNTrainer

SEED = 1234


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deterministic_series(
    batch: int = 1,
    timesteps: int = 300,
    features: int = 3,
) -> torch.Tensor:
    """Return a smooth, fully deterministic ``(batch, time, features)`` series.

    A small bundle of phase-shifted sinusoids — no randomness — so that a
    seeded run reproduces bit-for-bit and an NG-RC readout has a low-noise
    signal to fit.

    Parameters
    ----------
    batch : int, default=1
        Batch dimension.
    timesteps : int, default=300
        Sequence length.
    features : int, default=3
        Feature dimension (also the NG-RC ``input_dim`` and feedback dim).

    Returns
    -------
    torch.Tensor
        Series of shape ``(batch, timesteps, features)``.
    """
    t = torch.linspace(0, 8 * torch.pi, timesteps)
    base = torch.stack(
        [0.6 * torch.sin(t * (1.0 + 0.15 * i) + 0.5 * i) for i in range(features)],
        dim=-1,
    )
    return base.unsqueeze(0).repeat(batch, 1, 1)


def _build_ngrc_model(
    input_dim: int = 3,
    k: int = 2,
    s: int = 1,
    p: int = 2,
    seq_len: int = 100,
    alpha: float = 1e-6,
) -> tuple[ESNModel, NGReservoir]:
    """Build ``Input -> NGReservoir -> CGReadoutLayer`` wrapped in an ESNModel.

    The readout emits ``input_dim`` features so the first (only) output matches
    the feedback dimension — the precondition for autoregressive
    :meth:`~resdag.core.ESNModel.forecast`.

    Parameters
    ----------
    input_dim : int, default=3
        NG-RC input dimensionality and model feedback dimension.
    k, s, p : int
        NG-RC delay-tap count, tap spacing, and polynomial degree.
    seq_len : int, default=100
        Declared input sequence length (NG-RC is length-agnostic at runtime).
    alpha : float, default=1e-6
        Ridge regularisation strength for the readout.

    Returns
    -------
    model : ESNModel
        The assembled model.
    reservoir : NGReservoir
        The NG-RC layer instance (handy for asserting on its state/warmup).
    """
    torch.manual_seed(SEED)
    inp = Input(shape=(seq_len, input_dim))
    reservoir = NGReservoir(input_dim=input_dim, k=k, s=s, p=p)
    features = reservoir(inp)
    readout = CGReadoutLayer(reservoir.feature_dim, input_dim, alpha=alpha, name="output")(features)
    model = ESNModel(inp, readout)
    return model, reservoir


def _reservoir_modules(model: ESNModel) -> list[BaseReservoirLayer]:
    """Return all reservoir submodules of ``model`` (NG-RC layers here)."""
    return [m for m in model.modules() if isinstance(m, BaseReservoirLayer)]


def _readout_by_name(model: ESNModel, name: str) -> ReadoutLayer:
    """Return the readout submodule whose user-defined ``name`` matches.

    Mirrors how :class:`~resdag.training.ESNTrainer` resolves readouts (it
    walks ``named_modules`` and keys on the user-supplied ``name``), rather than
    relying on attribute access — ESNModel does not expose readouts by name as
    attributes.
    """
    for module in model.modules():
        if isinstance(module, ReadoutLayer) and module.name == name:
            return module
    raise AssertionError(f"no readout named {name!r} found in model")


# ---------------------------------------------------------------------------
# Construction / wiring
# ---------------------------------------------------------------------------


class TestNGRCInESNModelConstruction:
    """An NGReservoir wires cleanly into the symbolic DAG and ESNModel."""

    def test_model_builds_with_ngrc_reservoir(self) -> None:
        """``Input -> NGReservoir -> CGReadoutLayer`` builds an ESNModel."""
        model, reservoir = _build_ngrc_model(input_dim=3, k=2, p=2)
        assert isinstance(model, ESNModel)
        # The NG-RC layer is discoverable as a reservoir submodule.
        reservoirs = _reservoir_modules(model)
        assert len(reservoirs) == 1
        assert isinstance(reservoirs[0], NGReservoir)

    def test_forward_output_shape(self) -> None:
        """A plain forward pass yields ``(batch, time, input_dim)``."""
        model, _ = _build_ngrc_model(input_dim=3, k=2, p=2, seq_len=40)
        x = _deterministic_series(batch=2, timesteps=40, features=3)
        out = model(x)
        assert out.shape == (2, 40, 3)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Training (ESNTrainer) — acceptance: trains with finite, correct-shaped output
# ---------------------------------------------------------------------------


class TestNGRCTrainingViaESNTrainer:
    """ESNTrainer fits the readout sitting on top of an NG-RC feature map."""

    def test_trainer_fits_readout(self) -> None:
        """After ``fit`` the readout is fitted and reconstruction is finite."""
        model, _ = _build_ngrc_model(input_dim=3, k=2, p=2, seq_len=200)
        series = _deterministic_series(batch=1, timesteps=300, features=3)

        warmup = series[:, :50, :]
        # Next-step prediction: inputs are x[t], targets x[t+1].
        train_in = series[:, 50:249, :]
        train_tgt = series[:, 51:250, :]

        trainer = ESNTrainer(model)
        trainer.fit(
            warmup_inputs=(warmup,),
            train_inputs=(train_in,),
            targets={"output": train_tgt},
        )

        readout = _readout_by_name(model, "output")
        assert readout.is_fitted

        # Reconstruction on the training window is finite and correctly shaped.
        model.reset_reservoirs()
        model.warmup(warmup)
        recon = model(train_in)
        assert recon.shape == train_tgt.shape
        assert torch.isfinite(recon).all()

    def test_training_is_deterministic(self) -> None:
        """Two seeded train+reconstruct runs produce identical outputs."""

        def _run() -> torch.Tensor:
            model, _ = _build_ngrc_model(input_dim=2, k=3, s=1, p=2, seq_len=200)
            series = _deterministic_series(batch=1, timesteps=260, features=2)
            warmup = series[:, :40, :]
            train_in = series[:, 40:219, :]
            train_tgt = series[:, 41:220, :]
            ESNTrainer(model).fit(
                warmup_inputs=(warmup,),
                train_inputs=(train_in,),
                targets={"output": train_tgt},
            )
            model.reset_reservoirs()
            model.warmup(warmup)
            out: torch.Tensor = model(train_in)
            return out

        first = _run()
        second = _run()
        assert torch.equal(first, second)

    def test_trained_readout_lowers_reconstruction_error(self) -> None:
        """A fitted NG-RC readout reconstructs the smooth series well.

        Sanity check that the algebraic fit actually learned the next-step map:
        the post-fit reconstruction error on the (deterministic) training window
        is far below the signal scale.
        """
        model, _ = _build_ngrc_model(input_dim=3, k=2, p=2, seq_len=200)
        series = _deterministic_series(batch=1, timesteps=300, features=3)
        warmup = series[:, :50, :]
        train_in = series[:, 50:249, :]
        train_tgt = series[:, 51:250, :]

        ESNTrainer(model).fit(
            warmup_inputs=(warmup,),
            train_inputs=(train_in,),
            targets={"output": train_tgt},
        )
        model.reset_reservoirs()
        model.warmup(warmup)
        recon = model(train_in)

        rmse = (recon - train_tgt).pow(2).mean().sqrt().item()
        signal = train_tgt.pow(2).mean().sqrt().item()
        assert rmse < 0.1 * signal


# ---------------------------------------------------------------------------
# Forecasting — acceptance: forecasts with finite, correct-shaped output
# ---------------------------------------------------------------------------


class TestNGRCForecast:
    """Autoregressive forecast runs end-to-end through the flat step engine."""

    def _fit_model(
        self, input_dim: int = 3, k: int = 2, p: int = 2
    ) -> tuple[ESNModel, torch.Tensor]:
        model, _ = _build_ngrc_model(input_dim=input_dim, k=k, p=p, seq_len=200)
        series = _deterministic_series(batch=1, timesteps=300, features=input_dim)
        warmup = series[:, :50, :]
        train_in = series[:, 50:249, :]
        train_tgt = series[:, 51:250, :]
        ESNTrainer(model).fit(
            warmup_inputs=(warmup,),
            train_inputs=(train_in,),
            targets={"output": train_tgt},
        )
        return model, warmup

    def test_forecast_shape_and_finiteness(self) -> None:
        """``forecast`` returns ``(batch, horizon, input_dim)`` with finite values."""
        model, warmup = self._fit_model(input_dim=3, k=2, p=2)
        horizon = 80
        preds = model.forecast(warmup, horizon=horizon)
        assert isinstance(preds, torch.Tensor)
        assert preds.shape == (1, horizon, 3)
        assert torch.isfinite(preds).all()

    def test_forecast_with_return_warmup(self) -> None:
        """``return_warmup=True`` prepends the warmup outputs with no seam dup."""
        model, warmup = self._fit_model(input_dim=3, k=2, p=2)
        horizon = 40
        warmup_steps = warmup.shape[1]
        full = model.forecast(warmup, horizon=horizon, return_warmup=True)
        assert isinstance(full, torch.Tensor)
        assert full.shape == (1, warmup_steps + horizon, 3)
        assert torch.isfinite(full).all()

    def test_forecast_horizon_one(self) -> None:
        """A single-step forecast is a genuine autoregressive step (finite)."""
        model, warmup = self._fit_model(input_dim=3, k=2, p=2)
        preds = model.forecast(warmup, horizon=1)
        assert isinstance(preds, torch.Tensor)
        assert preds.shape == (1, 1, 3)
        assert torch.isfinite(preds).all()

    def test_forecast_deterministic(self) -> None:
        """Two forecasts from the same fitted model agree exactly."""
        model, warmup = self._fit_model(input_dim=3, k=2, p=2)
        a = model.forecast(warmup, horizon=30)
        b = model.forecast(warmup, horizon=30)
        assert isinstance(a, torch.Tensor)
        assert isinstance(b, torch.Tensor)
        assert torch.equal(a, b)

    def test_forecast_threads_3d_state_with_k3(self) -> None:
        """Forecast works when the 3-D delay buffer has ``state_size > 1`` (k=3).

        With ``k=3, s=1`` the buffer is ``(batch, 2, input_dim)``; the flat
        engine must thread that multi-row 3-D state through every step.
        """
        model, warmup = self._fit_model(input_dim=2, k=3, p=2)
        preds = model.forecast(warmup, horizon=25)
        assert isinstance(preds, torch.Tensor)
        assert preds.shape == (1, 25, 2)
        assert torch.isfinite(preds).all()


# ---------------------------------------------------------------------------
# Model-level state management over the 3-D delay buffer
# ---------------------------------------------------------------------------


class TestNGRCModelStateManagement:
    """reset / get / set reservoir states with NG-RC's 3-D buffer."""

    def test_get_reservoir_states_is_3d_buffer(self) -> None:
        """States returned at the model level carry the NG-RC 3-D shape."""
        model, reservoir = _build_ngrc_model(input_dim=3, k=2, s=1, p=2, seq_len=40)
        x = _deterministic_series(batch=2, timesteps=40, features=3)
        model(x)

        states = model.get_reservoir_states()
        assert len(states) == 1
        (state,) = states.values()
        # (batch, state_size, input_dim) with state_size = (k-1)*s = 1.
        assert state.dim() == 3
        assert state.shape == (2, reservoir.warmup_length, 3)
        assert state.shape == (2, 1, 3)

    def test_get_reservoir_states_returns_clone(self) -> None:
        """Mutating a fetched 3-D state must not touch the model's buffer."""
        model, _ = _build_ngrc_model(input_dim=3, k=2, p=2, seq_len=40)
        model(_deterministic_series(batch=1, timesteps=40, features=3))

        states = model.get_reservoir_states()
        snapshot = {name: s.clone() for name, s in states.items()}
        for s in states.values():
            s.add_(99.0)

        for name, s in model.get_reservoir_states().items():
            assert torch.equal(s, snapshot[name])

    def test_set_reservoir_states_roundtrip(self) -> None:
        """Restoring a saved 3-D buffer reproduces the continued trajectory."""
        model, _ = _build_ngrc_model(input_dim=3, k=3, s=1, p=2, seq_len=40)
        x1 = _deterministic_series(batch=1, timesteps=30, features=3)
        x2 = _deterministic_series(batch=1, timesteps=15, features=3)

        model(x1)
        saved = model.get_reservoir_states()

        out_a = model(x2)  # consumes the saved state
        model.set_reservoir_states(saved)
        out_b = model(x2)  # restored: must reproduce out_a

        assert torch.allclose(out_a, out_b, rtol=1e-5, atol=1e-6)

    def test_set_reservoir_states_rejects_wrong_3d_shape(self) -> None:
        """A buffer with the wrong (state_size, input_dim) is rejected."""
        model, _ = _build_ngrc_model(input_dim=3, k=3, s=1, p=2, seq_len=40)
        model(_deterministic_series(batch=1, timesteps=20, features=3))
        key = next(iter(model.get_reservoir_states()))

        # Correct rank (3-D) but wrong buffer dimensions.
        with pytest.raises(ValueError) as excinfo:
            model.set_reservoir_states({key: torch.zeros(1, 5, 9)})
        message = str(excinfo.value)
        assert "NGCell" in message  # names the offending cell class

    def test_reset_reservoirs_then_lazy_zeros(self) -> None:
        """After reset, the next forward re-inits the buffer to all zeros.

        ``reset_reservoirs`` drops the state to ``None`` (so
        ``get_reservoir_states`` returns ``{}``); the next forward lazily
        re-initialises the delay buffer, and that fresh buffer is all zeros.
        """
        model, reservoir = _build_ngrc_model(input_dim=3, k=3, s=1, p=2, seq_len=40)
        model(_deterministic_series(batch=1, timesteps=40, features=3))
        assert len(model.get_reservoir_states()) > 0  # state populated

        model.reset_reservoirs()
        assert model.get_reservoir_states() == {}  # cleared to None
        assert reservoir.state is None

        # Re-init explicitly with a known batch size; the buffer must be zeros
        # of the NG-RC 3-D shape (batch, (k-1)*s, input_dim) = (1, 2, 3).
        reservoir.reset_state(batch_size=1)
        fresh = reservoir.state
        assert fresh is not None
        assert fresh.shape == (1, 2, 3)
        assert torch.count_nonzero(fresh) == 0


# ---------------------------------------------------------------------------
# (k-1)*s warmup-length behaviour at the model level
# ---------------------------------------------------------------------------


class TestNGRCWarmupLengthAtModelLevel:
    """The ``(k-1)*s`` delay-buffer fill contract holds through the model."""

    @pytest.mark.parametrize(
        ("k", "s", "expected"),
        [(1, 1, 0), (2, 1, 1), (3, 1, 2), (3, 2, 4), (4, 3, 9)],
    )
    def test_warmup_length_equals_state_size(self, k: int, s: int, expected: int) -> None:
        """The reservoir's warmup_length is exactly ``(k-1)*s``."""
        model, reservoir = _build_ngrc_model(input_dim=2, k=k, s=s, p=2, seq_len=40)
        assert reservoir.warmup_length == expected
        assert reservoir.warmup_length == (k - 1) * s
        # And the model exposes that same layer.
        assert _reservoir_modules(model)[0].warmup_length == expected

    def test_state_buffer_rows_match_warmup_length(self) -> None:
        """The populated 3-D buffer has exactly ``(k-1)*s`` rows."""
        k, s = 3, 2
        model, reservoir = _build_ngrc_model(input_dim=2, k=k, s=s, p=2, seq_len=40)
        model(_deterministic_series(batch=1, timesteps=40, features=2))
        (state,) = model.get_reservoir_states().values()
        assert state.shape[1] == reservoir.warmup_length == (k - 1) * s

    def test_first_warmup_length_outputs_use_zero_taps(self) -> None:
        """The first ``(k-1)*s`` model outputs draw delay taps from zeros.

        With a cold buffer the earliest steps have unfilled delay slots, so the
        delay-embedded features (and hence the readout reconstruction) are
        produced from zero taps for the first ``warmup_length`` steps.  We
        verify this at the model level by checking that the model's NG-RC
        features for those early steps match a hand-built reference whose taps
        are zero.

        Concretely: for ``k=2, s=1, include_constant=True, include_linear=True``
        the very first feature row (cold start) must have its delay-tap block
        (columns ``input_dim`` .. ``2*input_dim``) equal to zero, since
        ``X_{-1}`` does not exist.
        """
        input_dim, k, s = 3, 2, 1
        torch.manual_seed(SEED)
        inp = Input(shape=(20, input_dim))
        reservoir = NGReservoir(input_dim=input_dim, k=k, s=s, p=2)
        feats = reservoir(inp)
        feat_model = ESNModel(inp, feats)

        x = _deterministic_series(batch=1, timesteps=20, features=input_dim)
        feat_model.reset_reservoirs()
        out = feat_model(x)

        warmup_length = (k - 1) * s
        assert reservoir.warmup_length == warmup_length == 1

        # Feature layout: [constant(1) | O_lin(k*input_dim) | O_nonlin].
        # O_lin = [X_i | X_{i-s}]; for the first step X_{i-s} taps are zero.
        const = 1  # include_constant=True
        tap_start = const + input_dim  # start of the X_{i-s} block
        tap_end = const + k * input_dim  # end of O_lin
        first_step_taps = out[0, 0, tap_start:tap_end]
        assert torch.count_nonzero(first_step_taps) == 0

        # By the time the buffer has filled (step >= warmup_length), the delay
        # taps reflect real past inputs and are no longer all zero.
        later_taps = out[0, warmup_length, tap_start:tap_end]
        assert torch.count_nonzero(later_taps) > 0
