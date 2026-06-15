"""Tests for prepare_esn_data splits and forecast alignment."""

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
    assert torch.equal(target[:, -1, :], val[:, 0, :])


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

    assert val.shape[1] == data.shape[1] - train_end
    assert val[0, -1, 0] == data[0, -1, 0]


def test_forecast_predictions_continue_validation_series() -> None:
    """The purely-autoregressive forecast continues the series one step ahead.

    ``forecast()`` emits ``horizon`` genuine autoregressive steps with no
    teacher-forced frame, so the first step is the model's prediction for the
    timestep *after* the warmup's final prediction.  Under the
    ``prepare_esn_data`` layout (``val[0] == data[train_end]``) that means
    ``pred[:, t]`` predicts ``data[train_end + 1 + t]`` — i.e. ``pred[:, :-1]``
    lines up with ``val[:, 1:]``.  (The forecast/validation index offset is
    reconciled in the data-prep layer; see issue #19.)
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
    # Slot 0 is a real autoregressive step (not an echo of the warmup output),
    # so the forecast is shifted one step ahead of the validation window.
    assert torch.allclose(pred[:, :-1, :], val[:, 1:, :], atol=1e-3)
