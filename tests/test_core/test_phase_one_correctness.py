"""Inference conveniences must preserve modes, output precision and placement."""

import pytest
import torch
from torch import nn

import resdag as rd


class PreciseAuxiliary(nn.Module):
    supports_framewise = True

    def forward(self, x):
        return x.double() + 1e-10


@pytest.mark.parametrize("alignment", ["next", "legacy"])
@pytest.mark.parametrize("return_warmup", [False, True])
def test_forecast_preserves_auxiliary_precision_in_every_slot(alignment, return_warmup):
    x = rd.reservoir_input(1)
    model = rd.ESNModel(x, (nn.Identity()(x), PreciseAuxiliary()(x)))
    history = torch.ones(1, 2, 1)
    actual = model.forecast(history, horizon=4, alignment=alignment, return_warmup=return_warmup)
    reference = model.forecast(
        history, horizon=4, alignment=alignment, return_warmup=return_warmup, no_grad=False
    )
    assert actual[1].dtype == torch.float64
    torch.testing.assert_close(actual[1], reference[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[1] - 1, torch.full_like(actual[1], 1e-10), rtol=1e-6, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_forecast_preserves_auxiliary_output_device():
    class CPUAuxiliary(nn.Module):
        supports_framewise = True

        def forward(self, x):
            return x.cpu()

    x = rd.reservoir_input(1, device="mps")
    model = rd.ESNModel(x, (nn.Identity()(x), CPUAuxiliary()(x)))
    primary, auxiliary = model.forecast(torch.ones(1, 2, 1, device="mps"), horizon=4)
    assert primary.device.type == "mps"
    assert auxiliary.device.type == "cpu"
    torch.testing.assert_close(auxiliary, primary.cpu(), rtol=0, atol=0)


def test_facade_disables_training_noise_only_during_forecasting():
    series = torch.sin(torch.arange(120, dtype=torch.float64) * 0.1).unsqueeze(-1)
    esn = rd.ESN(reservoir_size=10, washout=10, noise=0.02, seed=3).fit(series)
    assert esn.graph.training
    modes = [module.training for module in esn.graph.modules()]
    first, second = esn.forecast(horizon=8), esn.forecast(horizon=8)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert [module.training for module in esn.graph.modules()] == modes


@pytest.mark.parametrize("fail", [False, True])
def test_facade_restores_mixed_module_modes_even_after_error(monkeypatch, fail):
    series = torch.arange(30, dtype=torch.float64).sin().unsqueeze(-1)
    esn = rd.ESN(reservoir_size=5, washout=3, seed=1).fit(series)
    reservoir = next(module for module in esn.graph.modules() if isinstance(module, rd.ESNLayer))
    reservoir.eval()
    modes = [(module, module.training) for module in esn.graph.modules()]
    original = esn.graph.forecast

    def checking_forecast(*args, **kwargs):
        assert all(not module.training for module, _ in modes)
        if fail:
            raise RuntimeError("intentional forecast failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(esn.graph, "forecast", checking_forecast)
    if fail:
        with pytest.raises(RuntimeError, match="intentional forecast failure"):
            esn.forecast(horizon=3)
    else:
        esn.forecast(horizon=3)
    assert all(module.training == training for module, training in modes)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_parallel_factory_places_geometry_with_reservoir():
    model = rd.parallel_esn(4, 2, 1, reservoir_size=4, device="mps")
    assert model.patch_index.device == model.reservoir.weight_hh.device
    assert model(torch.ones(1, 3, 4, device="mps")).device.type == "mps"
