"""Tests for prepare_esn_data splits and forecast alignment."""

import warnings

import pytest
import torch

from resdag import classic_esn
from resdag.training import ESNTrainer
from resdag.utils.data import prepare_esn_data


def test_target_is_one_step_ahead_of_train() -> None:
    data = torch.arange(200, dtype=torch.float32).view(1, -1, 1)
    warmup_steps, train_steps, val_steps = 20, 80, 30
    warmup, train, target, _, val = prepare_esn_data(
        data, warmup_steps, train_steps, val_steps=val_steps
    )

    train_end = warmup_steps + train_steps
    assert target.shape == train.shape
    assert torch.equal(target[:, 0, :], data[:, warmup_steps + 1, :])
    assert torch.equal(target[:, -1, :], data[:, train_end, :])
    # val starts one step after the train window (the data[train_end] seam is
    # the forecast's bootstrap seed, not a val sample), so train and val are
    # disjoint — no leakage of the last train target into val[0].
    assert torch.equal(val[:, 0, :], data[:, train_end + 1, :])
    assert not torch.equal(target[:, -1, :], val[:, 0, :])


def test_forecast_warmup_is_train_suffix() -> None:
    data = torch.randn(1, 500, 2)
    warmup_steps, train_steps = 50, 200
    _, train, _, f_warmup, _ = prepare_esn_data(data, warmup_steps, train_steps, val_steps=100)

    assert f_warmup.shape[1] == warmup_steps
    assert torch.equal(f_warmup, train[:, -warmup_steps:, :])


def test_val_steps_none_uses_all_remaining() -> None:
    data = torch.arange(120, dtype=torch.float32).view(1, -1, 1)
    warmup_steps, train_steps = 10, 20
    train_end = warmup_steps + train_steps
    _, _, _, _, val = prepare_esn_data(data, warmup_steps, train_steps, val_steps=None)

    # All remaining data after the one-step autoregressive seam (data[train_end]).
    assert val.shape[1] == data.shape[1] - train_end - 1
    assert val[0, -1, 0] == data[0, -1, 0]
    assert val[0, 0, 0] == data[0, train_end + 1, 0]


def test_forecast_predictions_align_with_validation() -> None:
    """``forecast(f_warmup, horizon=val_steps)`` lines up with ``val`` directly.

    The forecast is purely autoregressive (its first step is a genuine
    prediction, not the teacher-forced warmup output), and ``prepare_esn_data``
    starts ``val`` at ``data[train_end + 1]`` to match. So ``pred[:, t]`` and
    ``val[:, t]`` refer to the same timestep — no manual shift needed.
    """
    warmup_steps, train_steps, val_steps = 50, 200, 80
    data = torch.arange(1000, dtype=torch.float32).view(1, -1, 1)

    warmup, train, target, f_warmup, val = prepare_esn_data(
        data, warmup_steps, train_steps, val_steps=val_steps
    )

    model = classic_esn(
        reservoir_size=80,
        feedback_size=1,
        output_size=1,
        spectral_radius=0.9,
    )
    ESNTrainer(model).fit(
        warmup_inputs=(warmup,),
        train_inputs=(train,),
        targets={"output": target},
    )

    model.reset_reservoirs()
    pred = model.forecast(f_warmup, horizon=val_steps)

    assert pred.shape[1] == val.shape[1]
    assert torch.allclose(pred, val, atol=1e-3)


# ---------------------------------------------------------------------------
# Input validation: negative / zero / empty-split-producing step counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("warmup_steps", [-5, -1, 0])
def test_nonpositive_warmup_steps_raises(warmup_steps: int) -> None:
    data = torch.arange(100, dtype=torch.float32).view(1, -1, 1)
    with pytest.raises(ValueError, match=rf"warmup_steps.*positive.*{warmup_steps}"):
        prepare_esn_data(data, warmup_steps, train_steps=20)


@pytest.mark.parametrize("train_steps", [-5, -1, 0])
def test_nonpositive_train_steps_raises(train_steps: int) -> None:
    data = torch.arange(100, dtype=torch.float32).view(1, -1, 1)
    with pytest.raises(ValueError, match=rf"train_steps.*positive.*{train_steps}"):
        prepare_esn_data(data, warmup_steps=10, train_steps=train_steps)


@pytest.mark.parametrize("val_steps", [-5, -1])
def test_negative_val_steps_raises(val_steps: int) -> None:
    data = torch.arange(100, dtype=torch.float32).view(1, -1, 1)
    with pytest.raises(ValueError, match=rf"val_steps.*non-negative.*{val_steps}"):
        prepare_esn_data(data, warmup_steps=10, train_steps=20, val_steps=val_steps)


def test_negative_discard_steps_raises() -> None:
    data = torch.arange(100, dtype=torch.float32).view(1, -1, 1)
    with pytest.raises(ValueError, match=r"discard_steps.*non-negative.*-3"):
        prepare_esn_data(data, warmup_steps=10, train_steps=20, discard_steps=-3)


def test_zero_val_steps_warns_and_returns_empty_val() -> None:
    data = torch.arange(100, dtype=torch.float32).view(1, -1, 1)
    with pytest.warns(UserWarning, match="empty validation split"):
        _, _, _, _, val = prepare_esn_data(data, warmup_steps=10, train_steps=20, val_steps=0)
    assert val.shape[1] == 0


def test_exhausted_data_val_none_warns_empty_val() -> None:
    # warmup + train + seam == timesteps exactly -> no remaining val data.
    data = torch.arange(31, dtype=torch.float32).view(1, -1, 1)
    with pytest.warns(UserWarning, match="Validation window is empty"):
        _, _, _, _, val = prepare_esn_data(data, warmup_steps=10, train_steps=20, val_steps=None)
    assert val.shape[1] == 0


def test_valid_call_does_not_warn() -> None:
    data = torch.arange(200, dtype=torch.float32).view(1, -1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warmup, train, _, _, val = prepare_esn_data(
            data, warmup_steps=20, train_steps=80, val_steps=30
        )
    assert warmup.shape[1] == 20
    assert train.shape[1] == 80
    assert val.shape[1] == 30
