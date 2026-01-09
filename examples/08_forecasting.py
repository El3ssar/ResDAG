"""Example usage of forecasting functionality.

This script demonstrates the two-step forecasting process:
1. Teacher-forced warmup to initialize reservoir states
2. Closed-loop autoregressive prediction
"""

import torch

from torch_rc.composition import ModelBuilder
from torch_rc.layers import ReservoirLayer
from torch_rc.layers.readouts import CGReadoutLayer
from torch_rc.models import classic_esn


def generate_sine_data(n_samples=200, freq=0.1):
    """Generate simple sine wave data for demonstration."""
    t = torch.linspace(0, n_samples * freq, n_samples)
    y = torch.sin(2 * torch.pi * t)
    return y.unsqueeze(0).unsqueeze(-1)  # (1, n_samples, 1)


def example_basic_forecasting():
    """Example: Basic forecasting without driving inputs."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Forecasting")
    print("=" * 60)

    # Create model
    model = classic_esn(100, 1, 1)

    # Generate data
    data = generate_sine_data(200)
    warmup = data[:, :100, :]  # First 100 steps
    ground_truth = data[:, 100:150, :]  # Next 50 steps

    # Forecast
    predictions = model.forecast(warmup_feedback=warmup, forecast_steps=50)

    print(f"Warmup shape: {warmup.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")

    # Compute error
    mse = torch.mean((predictions - ground_truth) ** 2).item()
    print(f"MSE: {mse:.6f}")


def example_with_warmup_return():
    """Example: Return warmup predictions for visualization."""
    print("\n" + "=" * 60)
    print("Example 2: Forecasting with Warmup Return")
    print("=" * 60)

    model = classic_esn(100, 1, 1)
    data = generate_sine_data(200)

    warmup = data[:, :100, :]

    # Get full trajectory (warmup + forecast)
    full_predictions = model.forecast(warmup_feedback=warmup, forecast_steps=50, return_warmup=True)

    print(f"Full trajectory shape: {full_predictions.shape}")  # (1, 150, 1)
    print(f"  Warmup: {full_predictions[:, :100, :].shape}")
    print(f"  Forecast: {full_predictions[:, 100:, :].shape}")


def example_with_driving_inputs():
    """Example: Forecasting with exogenous driving inputs."""
    print("\n" + "=" * 60)
    print("Example 3: Forecasting with Driving Inputs")
    print("=" * 60)

    # Build model with driving input
    builder = ModelBuilder()
    feedback = builder.input("input")
    driving = builder.input("reservoir_driving")
    reservoir = builder.add(ReservoirLayer(100, 1, 5), inputs=[feedback, driving], name="reservoir")
    readout = builder.add(CGReadoutLayer(100, 1), inputs=reservoir, name="output")
    model = builder.build(outputs=readout)

    # Generate data
    warmup_feedback = torch.randn(2, 50, 1)
    warmup_driving = {"reservoir": torch.randn(2, 50, 5)}  # Exogenous variables

    # Known future driving inputs (e.g., weather forecast)
    forecast_driving = {"reservoir": torch.randn(2, 30, 5)}

    # Forecast
    predictions = model.forecast(
        warmup_feedback=warmup_feedback,
        warmup_driving=warmup_driving,
        forecast_steps=30,
        forecast_driving=forecast_driving,
    )

    print(f"Predictions shape: {predictions.shape}")
    print("✓ Successfully forecasted with driving inputs!")


def example_state_history():
    """Example: Track reservoir state evolution."""
    print("\n" + "=" * 60)
    print("Example 4: State History Tracking")
    print("=" * 60)

    model = classic_esn(100, 1, 1)
    warmup = generate_sine_data(100)

    # Forecast with state history
    predictions, states = model.forecast(
        warmup_feedback=warmup, forecast_steps=50, return_state_history=True
    )

    print(f"Predictions shape: {predictions.shape}")
    print(f"State history keys: {list(states.keys())}")
    
    # Get reservoir name (auto-generated with pytorch_symbolic)
    reservoir_name = list(states.keys())[0]
    print(f"Reservoir states shape: {states[reservoir_name].shape}")

    # Analyze state evolution
    state_trajectory = states[reservoir_name][0]  # (150, 100)
    state_norms = torch.norm(state_trajectory, dim=1)

    print("\nState norm statistics:")
    print(f"  Mean: {state_norms.mean():.4f}")
    print(f"  Std: {state_norms.std():.4f}")
    print(f"  Min: {state_norms.min():.4f}")
    print(f"  Max: {state_norms.max():.4f}")


def example_custom_initial_feedback():
    """Example: Provide custom initial feedback for forecast."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Initial Feedback")
    print("=" * 60)

    model = classic_esn(100, 1, 1)
    warmup = generate_sine_data(100)

    # Normal forecast (uses last warmup prediction)
    predictions_normal = model.forecast(warmup_feedback=warmup, forecast_steps=20)

    # Reset model
    model.reset_reservoirs()

    # Forecast with custom initial feedback
    custom_initial = torch.tensor([[[0.5]]])  # Start from specific value
    predictions_custom = model.forecast(
        warmup_feedback=warmup,
        forecast_steps=20,
        forecast_initial_feedback=custom_initial,
    )

    print(f"Normal forecast first value: {predictions_normal[0, 0, 0]:.4f}")
    print(f"Custom forecast first value: {predictions_custom[0, 0, 0]:.4f}")
    print("✓ Different initial conditions lead to different forecasts")


def example_multi_reservoir():
    """Example: Forecasting with multiple reservoirs and driving inputs."""
    print("\n" + "=" * 60)
    print("Example 6: Multiple Reservoirs with Different Drivers")
    print("=" * 60)

    # Build model with two reservoirs, each with different driving inputs
    builder = ModelBuilder()
    feedback = builder.input("input")
    driving1 = builder.input("reservoir1_driving")
    driving2 = builder.input("reservoir2_driving")

    res1 = builder.add(ReservoirLayer(50, 1, 3), inputs=[feedback, driving1], name="reservoir1")
    res2 = builder.add(ReservoirLayer(60, 1, 5), inputs=[feedback, driving2], name="reservoir2")

    # Concatenate reservoirs
    import torch.nn as nn

    class ConcatLayer(nn.Module):
        def forward(self, *inputs):
            return torch.cat(inputs, dim=-1)

    concat = builder.add(ConcatLayer(), inputs=[res1, res2], name="concat")
    readout = builder.add(CGReadoutLayer(110, 1), inputs=concat, name="output")
    model = builder.build(outputs=readout)

    # Forecast with different driving inputs for each reservoir
    warmup_feedback = torch.randn(2, 30, 1)
    warmup_driving = {
        "reservoir1": torch.randn(2, 30, 3),  # Local features
        "reservoir2": torch.randn(2, 30, 5),  # Global features
    }
    forecast_driving = {
        "reservoir1": torch.randn(2, 20, 3),
        "reservoir2": torch.randn(2, 20, 5),
    }

    predictions, states = model.forecast(
        warmup_feedback=warmup_feedback,
        warmup_driving=warmup_driving,
        forecast_steps=20,
        forecast_driving=forecast_driving,
        return_state_history=True,
    )

    print(f"Predictions shape: {predictions.shape}")
    print(f"State history keys: {list(states.keys())}")
    print(f"Reservoir1 states: {states['reservoir1'].shape}")
    print(f"Reservoir2 states: {states['reservoir2'].shape}")
    print("✓ Multi-reservoir forecasting successful!")


def example_long_horizon_forecast():
    """Example: Long-horizon forecasting."""
    print("\n" + "=" * 60)
    print("Example 7: Long-Horizon Forecasting")
    print("=" * 60)

    model = classic_esn(100, 1, 1)
    data = generate_sine_data(500)

    warmup = data[:, :100, :]

    # Forecast different horizons
    for horizon in [10, 50, 100, 200, 1000, 2000, 10000]:
        model.reset_reservoirs()
        predictions = model.forecast(warmup_feedback=warmup, forecast_steps=horizon)
        print(f"Horizon {horizon:3d}: predictions shape {predictions.shape}")


def example_batch_forecasting():
    """Example: Batch forecasting for multiple sequences."""
    print("\n" + "=" * 60)
    print("Example 8: Batch Forecasting")
    print("=" * 60)

    model = classic_esn(100, 1, 1)

    # Multiple sequences with different frequencies
    batch_size = 8
    warmup_data = []
    for i in range(batch_size):
        freq = 0.05 + i * 0.01  # Different frequencies
        data = generate_sine_data(150, freq=freq)
        warmup_data.append(data)

    warmup = torch.cat(warmup_data, dim=0)  # (8, 150, 1)

    # Forecast all sequences in parallel
    predictions = model.forecast(warmup_feedback=warmup, forecast_steps=50)

    print(f"Batch size: {batch_size}")
    print(f"Warmup shape: {warmup.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print("✓ Batch forecasting successful!")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run all examples
    example_basic_forecasting()
    example_with_warmup_return()
    example_with_driving_inputs()
    example_state_history()
    example_custom_initial_feedback()
    example_multi_reservoir()
    example_long_horizon_forecast()
    example_batch_forecasting()

    print("\n" + "=" * 60)
    print("All forecasting examples completed successfully!")
    print("=" * 60)
