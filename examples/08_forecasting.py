"""Example usage of forecasting functionality.

This script demonstrates the two-step forecasting process:
1. Teacher-forced warmup to initialize reservoir states (Echo State Property)
2. Closed-loop autoregressive prediction

Convention: First input is always feedback, remaining inputs are drivers.
"""

import torch

import resdag as trc
from resdag.composition import ESNModel, Input
from resdag.models import classic_esn


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

    # Forecast: warmup is the feedback input
    predictions = model.forecast(warmup, horizon=50)

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
    full_predictions = model.forecast(warmup, horizon=50, return_warmup=True)

    print(f"Full trajectory shape: {full_predictions.shape}")  # (1, 150, 1)
    print(f"  Warmup: {full_predictions[:, :100, :].shape}")
    print(f"  Forecast: {full_predictions[:, 100:, :].shape}")


def example_with_driving_inputs():
    """Example: Forecasting with exogenous driving inputs."""
    print("\n" + "=" * 60)
    print("Example 3: Forecasting with Driving Inputs")
    print("=" * 60)

    # Build model with driving input using symbolic API
    feedback = Input(shape=(50, 1))
    driving = Input(shape=(50, 5))
    reservoir = trc.ReservoirLayer(100, 1, input_size=5)(feedback, driving)
    readout = trc.CGReadoutLayer(100, 1)(reservoir)
    model = ESNModel([feedback, driving], readout)

    # Generate warmup data
    warmup_feedback = torch.randn(2, 50, 1)
    warmup_driving = torch.randn(2, 50, 5)

    # Known future driving inputs (e.g., weather forecast)
    forecast_driving = torch.randn(2, 30, 5)

    # Forecast: (feedback, driving) as warmup, forecast_drivers for future
    predictions = model.forecast(
        warmup_feedback,
        warmup_driving,
        horizon=30,
        forecast_drivers=(forecast_driving,),
    )

    print(f"Predictions shape: {predictions.shape}")
    print("✓ Successfully forecasted with driving inputs!")


def example_warmup_only():
    """Example: Using warmup separately for state synchronization."""
    print("\n" + "=" * 60)
    print("Example 4: Warmup for State Synchronization")
    print("=" * 60)

    model = classic_esn(100, 1, 1)
    warmup_data = generate_sine_data(100)

    # Warmup without returning outputs (just sync states)
    model.warmup(warmup_data)
    print("States synchronized (no output)")

    # Warmup with outputs for visualization
    model.reset_reservoirs()
    outputs = model.warmup(warmup_data, return_outputs=True)
    print(f"Warmup outputs shape: {outputs.shape}")

    # Check reservoir state is updated
    states = model.get_reservoir_states()
    print(f"Reservoir states: {list(states.keys())}")


def example_custom_initial_feedback():
    """Example: Provide custom initial feedback for forecast."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Initial Feedback")
    print("=" * 60)

    model = classic_esn(100, 1, 1)
    warmup = generate_sine_data(100)

    # Normal forecast (uses last warmup prediction)
    predictions_normal = model.forecast(warmup, horizon=20)

    # Reset model
    model.reset_reservoirs()

    # Forecast with custom initial feedback
    custom_initial = torch.tensor([[[0.5]]])  # Start from specific value
    predictions_custom = model.forecast(
        warmup,
        horizon=20,
        initial_feedback=custom_initial,
    )

    print(f"Normal forecast first value: {predictions_normal[0, 0, 0]:.4f}")
    print(f"Custom forecast first value: {predictions_custom[0, 0, 0]:.4f}")
    print("✓ Different initial conditions lead to different forecasts")


def example_multi_reservoir():
    """Example: Forecasting with multiple reservoirs and driving inputs."""
    print("\n" + "=" * 60)
    print("Example 6: Multiple Reservoirs with Stacking")
    print("=" * 60)

    # Build deep ESN: stacked reservoirs
    feedback = Input(shape=(30, 1))
    res1 = trc.ReservoirLayer(50, 1)(feedback)
    res2 = trc.ReservoirLayer(60, 50)(res1)  # Takes res1 output as feedback
    readout = trc.CGReadoutLayer(60, 1)(res2)
    model = ESNModel(feedback, readout)

    # Forecast
    warmup = torch.randn(2, 30, 1)
    predictions = model.forecast(warmup, horizon=20)

    print(f"Predictions shape: {predictions.shape}")

    # Get all reservoir states
    states = model.get_reservoir_states()
    print(f"Reservoir states: {list(states.keys())}")
    for name, state in states.items():
        print(f"  {name}: {state.shape}")
    print("✓ Multi-reservoir forecasting successful!")


def example_complex_topology():
    """Example: Complex model with multiple inputs, reservoirs and outputs."""
    print("\n" + "=" * 60)
    print("Example 7: Complex Topology (Multi-Input, Multi-Reservoir, Multi-Output)")
    print("=" * 60)

    # Model: feedback + 2 driving inputs, 2 reservoirs, outputs reservoir state
    feedback = Input(shape=(1, 1))
    driver1 = Input(shape=(1, 1))
    driver2 = Input(shape=(1, 2))

    # First reservoir: feedback + driver1
    res1 = trc.ReservoirLayer(100, 1, input_size=1)(feedback, driver1)
    readout1 = trc.CGReadoutLayer(100, 2)(res1)

    # Second reservoir: readout1 output + driver2
    res2 = trc.ReservoirLayer(80, 2, input_size=2)(readout1, driver2)
    readout2 = trc.CGReadoutLayer(80, 1)(res2)

    readout3 = trc.CGReadoutLayer(80, 15)(res2)

    model = ESNModel(inputs=(feedback, driver1, driver2), outputs=(readout2, readout3))

    print(f"Model input shapes: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    # Create data
    batch_size = 4
    warmup_len = 50
    forecast_len = 30

    warmup_fb = torch.randn(batch_size, warmup_len, 1)
    warmup_d1 = torch.randn(batch_size, warmup_len, 1)
    warmup_d2 = torch.randn(batch_size, warmup_len, 2)

    forecast_d1 = torch.randn(batch_size, forecast_len, 1)
    forecast_d2 = torch.randn(batch_size, forecast_len, 2)

    # Forecast
    predictions = model.forecast(
        warmup_fb,
        warmup_d1,
        warmup_d2,
        horizon=forecast_len,
        forecast_drivers=(forecast_d1, forecast_d2),
    )

    print(f"Predictions shape: {[pred.shape for pred in predictions]}")
    print("✓ Complex topology forecasting successful!")


def example_long_horizon_forecast():
    """Example: Long-horizon forecasting."""
    print("\n" + "=" * 60)
    print("Example 8: Long-Horizon Forecasting")
    print("=" * 60)

    model = classic_esn(100, 1, 1)
    data = generate_sine_data(500)

    warmup = data[:, :100, :]

    # Forecast different horizons
    for horizon in [10, 50, 100, 200, 1000, 10000]:
        model.reset_reservoirs()
        predictions = model.forecast(warmup, horizon=horizon)
        print(f"Horizon {horizon:4d}: predictions shape {predictions.shape}")


def example_batch_forecasting():
    """Example: Batch forecasting for multiple sequences."""
    print("\n" + "=" * 60)
    print("Example 9: Batch Forecasting")
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
    predictions = model.forecast(warmup, horizon=50)

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
    example_warmup_only()
    example_custom_initial_feedback()
    example_multi_reservoir()
    example_complex_topology()
    example_long_horizon_forecast()
    example_batch_forecasting()

    print("\n" + "=" * 60)
    print("All forecasting examples completed successfully!")
    print("=" * 60)
