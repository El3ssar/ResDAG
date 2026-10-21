"""Device-local HPO scores must preserve the public NumPy loss definitions."""

from contextlib import nullcontext
from functools import partial

import pytest
import torch

from resdag.hpo import get_loss
from resdag.hpo._torch_losses import score_builtin_tensor_loss
from resdag.hpo.trial_runner import TrialRunner


@pytest.mark.parametrize("key", ["efh", "forecast_horizon", "standard", "lyapunov", "soft_horizon"])
@pytest.mark.parametrize("metric", ["nrmse", "rmse", "mse", "mae"])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
            ],
        ),
    ],
)
def test_tensor_loss_matches_numpy(key, metric, batch, dtype, device):
    generator = torch.Generator().manual_seed(17)
    truth = torch.randn(batch, 32, 3, dtype=dtype, generator=generator)
    prediction = truth + torch.randn(truth.shape, dtype=dtype, generator=generator) * 0.1
    truth[:, :, -1] = 2.0  # population std and zero-scale channel convention
    prediction[:, :, -1] = 2.05
    prediction[0] = truth[0]  # zero-safe geometric mean, even-batch median
    fn = partial(get_loss(key), metric=metric)
    with (
        pytest.warns(UserWarning)
        if metric != "nrmse" and key in {"efh", "forecast_horizon", "soft_horizon"}
        else nullcontext()
    ):
        expected = fn(truth.numpy(), prediction.numpy())
    # The adapter intentionally emits the same raw-scale warning.
    with (
        pytest.warns(UserWarning)
        if metric != "nrmse" and key in {"efh", "forecast_horizon", "soft_horizon"}
        else nullcontext()
    ):
        actual = score_builtin_tensor_loss(fn, truth.to(device), prediction.to(device))
    assert actual is not None
    assert actual.device.type == device
    assert float(actual) == pytest.approx(expected, rel=3e-5, abs=1e-6)


def test_builtin_runner_never_converts_trajectory_to_numpy(monkeypatch):
    runner = TrialRunner(
        model_creator=lambda: None,
        search_space=lambda trial: {},
        data_loader=lambda trial: {},
        loss_fn=partial(get_loss("efh"), threshold=0.4, softness=0.1),
    )

    def forbid_numpy(self):
        raise AssertionError("trajectory transferred to NumPy")

    monkeypatch.setattr(torch.Tensor, "numpy", forbid_numpy)
    truth = torch.ones(4, 20, 2)
    score = runner._score(truth, truth)
    assert score < -1


def test_custom_function_with_builtin_name_is_not_replaced():
    def expected_forecast_horizon(y_true, y_pred):
        return 123.0

    truth = torch.ones(1, 5, 1)
    assert score_builtin_tensor_loss(expected_forecast_horizon, truth, truth) is None
