"""Shared fixtures for the ResDAG test suite.

Conventions
-----------
- Tests that should run on every available device take the ``device``
  fixture; the CUDA variant carries the ``gpu`` marker and auto-skips on
  machines without a GPU.
- Data factories return ``(batch, time, features)`` tensors, seeded — use
  them instead of ad-hoc ``torch.randn`` so failures reproduce.
- Performance assertions live in ``tests/test_performance/`` behind the
  ``benchmark`` marker (deselected by default; run with ``pytest -m benchmark``).
"""

import pytest
import torch

import resdag as rd
from resdag.core import ESNModel, reservoir_input
from resdag.layers import CGReadoutLayer, ESNLayer


def _cuda_unavailable() -> bool:
    return not torch.cuda.is_available()


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(_cuda_unavailable(), reason="CUDA not available"),
            ],
        ),
    ]
)
def device(request: pytest.FixtureRequest) -> torch.device:
    """Parametrize a test over all available devices."""
    return torch.device(request.param)


@pytest.fixture(autouse=False)
def seeded() -> None:
    """Deterministic global torch seed for tests that build random weights."""
    torch.manual_seed(42)


@pytest.fixture
def make_sine_data():
    """Factory for a clean (batch, time, features) sine series."""

    def _make(
        batch: int = 1,
        timesteps: int = 600,
        features: int = 3,
        noise: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        torch.manual_seed(7)
        t = torch.linspace(0, 12 * torch.pi, timesteps)
        base = torch.stack([torch.sin(t * (1 + 0.1 * i)) for i in range(features)], dim=-1)
        data = base.unsqueeze(0).repeat(batch, 1, 1)
        if noise > 0:
            data = data + noise * torch.randn_like(data)
        return data.to(device)

    return _make


@pytest.fixture
def make_esn_splits(make_sine_data):
    """Factory for ready-to-train ESN data splits on a sine series."""

    def _make(
        batch: int = 1,
        warmup_steps: int = 50,
        train_steps: int = 300,
        val_steps: int = 100,
        features: int = 3,
        device: torch.device | str = "cpu",
    ):
        data = make_sine_data(
            batch=batch,
            timesteps=warmup_steps + train_steps + val_steps + 1,
            features=features,
            device=device,
        )
        return rd.utils.prepare_esn_data(
            data,
            warmup_steps=warmup_steps,
            train_steps=train_steps,
            val_steps=val_steps,
        )

    return _make


@pytest.fixture
def make_tiny_model():
    """Factory for a minimal hand-wired ESN model (reservoir -> readout)."""

    def _make(
        reservoir_size: int = 32,
        feedback_size: int = 3,
        output_size: int | None = None,
        device: torch.device | str = "cpu",
        **layer_kwargs,
    ) -> ESNModel:
        torch.manual_seed(42)
        output_size = feedback_size if output_size is None else output_size
        inp = reservoir_input(feedback_size)
        states = ESNLayer(
            reservoir_size=reservoir_size,
            feedback_size=feedback_size,
            spectral_radius=layer_kwargs.pop("spectral_radius", 0.9),
            **layer_kwargs,
        )(inp)
        out = CGReadoutLayer(reservoir_size, output_size, name="output")(states)
        return ESNModel(inp, out).to(device)

    return _make
