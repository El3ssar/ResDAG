"""Example demonstrating ESN training with ESNTrainer.

This example shows how to train ESN models using algebraic readout fitting.
Each ReadoutLayer is fitted via ridge regression, not SGD.

Key concepts:
1. Warmup phase: Synchronize reservoir states with input dynamics
2. Training phase: Fit each readout layer in topological order
3. Each readout stores its own hyperparameters (e.g., alpha for ridge)
"""

import torch

import resdag as trc
from resdag.composition import ESNModel, Input
from resdag.layers.readouts import CGReadoutLayer
from resdag.training import ESNTrainer


def generate_sine_data(n_samples: int = 500, freq: float = 0.1, noise: float = 0.1):
    """Generate sine wave data with noise."""
    t = torch.linspace(0, n_samples * freq, n_samples)
    y = torch.sin(2 * torch.pi * t) + noise * torch.randn(n_samples)
    return y.unsqueeze(0).unsqueeze(-1)  # (1, n_samples, 1)


def example_simple_training():
    """Example: Train a simple single-readout ESN."""
    print("\n" + "=" * 60)
    print("Example 1: Simple Single-Readout Training")
    print("=" * 60)

    # Build model with named readout
    feedback = Input(shape=(100, 1))
    reservoir = trc.ReservoirLayer(100, 1)(feedback)
    readout = CGReadoutLayer(100, 1, name="output", alpha=1e-6)(reservoir)
    model = ESNModel(feedback, readout)

    print("Model structure:")
    model.summary()

    # Generate training data (separate warmup and train)
    warmup_data = generate_sine_data(100)  # 100 steps for warmup
    train_data = generate_sine_data(400)  # 400 steps for training
    train_target = train_data.clone()  # Target same as train data

    print(f"\nWarmup data shape: {warmup_data.shape}")
    print(f"Training data shape: {train_data.shape}")
    print(f"Target shape: {train_target.shape}")

    # Train
    trainer = ESNTrainer(model)
    trainer.fit(
        warmup_inputs=(warmup_data,),
        train_inputs=(train_data,),
        targets={"output": train_target},
    )

    print(f"\nReadout fitted: {model.CGReadoutLayer_1.is_fitted}")

    # Test the trained model
    model.reset_reservoirs()
    test_data = generate_sine_data(200)
    output = model(test_data)
    print(f"Test output shape: {output.shape}")
    print("✓ Training successful!")


def example_multi_readout_training():
    """Example: Train a stacked multi-readout ESN."""
    print("\n" + "=" * 60)
    print("Example 2: Stacked Multi-Readout Training")
    print("=" * 60)

    # Build stacked model: reservoir1 -> readout1 -> reservoir2 -> readout2
    feedback = Input(shape=(100, 1))

    reservoir1 = trc.ReservoirLayer(100, 1)(feedback)
    readout1 = CGReadoutLayer(100, 3, name="intermediate", alpha=1e-5)(reservoir1)

    reservoir2 = trc.ReservoirLayer(50, 3)(readout1)
    readout2 = CGReadoutLayer(50, 1, name="output", alpha=1e-6)(reservoir2)

    model = ESNModel(feedback, readout2)

    print("Model structure:")
    model.summary()

    # Generate training data (separate warmup and train)
    batch_size = 4
    warmup_steps = 100
    train_steps = 400

    warmup_data = torch.randn(batch_size, warmup_steps, 1)
    train_data = torch.randn(batch_size, train_steps, 1)

    # Each readout needs its own target!
    intermediate_targets = torch.randn(batch_size, train_steps, 3)
    output_targets = torch.randn(batch_size, train_steps, 1)

    print(f"\nWarmup data shape: {warmup_data.shape}")
    print(f"Training data shape: {train_data.shape}")
    print(f"Intermediate target shape: {intermediate_targets.shape}")
    print(f"Output target shape: {output_targets.shape}")

    # Train
    trainer = ESNTrainer(model)
    trainer.fit(
        warmup_inputs=(warmup_data,),
        train_inputs=(train_data,),
        targets={
            "intermediate": intermediate_targets,
            "output": output_targets,
        },
    )

    print(f"\nIntermediate readout fitted: {model.CGReadoutLayer_1.is_fitted}")
    print(f"Output readout fitted: {model.CGReadoutLayer_2.is_fitted}")

    # Test
    model.reset_reservoirs()
    output = model(torch.randn(2, 200, 1))
    print(f"Test output shape: {output.shape}")
    print("✓ Multi-readout training successful!")


def example_training_with_drivers():
    """Example: Train model with driving inputs."""
    print("\n" + "=" * 60)
    print("Example 3: Training with Driving Inputs")
    print("=" * 60)

    # Build model with feedback + driving input
    feedback = Input(shape=(100, 1))
    driver = Input(shape=(100, 5))

    reservoir = trc.ReservoirLayer(100, 1, input_size=5)(feedback, driver)
    readout = CGReadoutLayer(100, 1, name="output", alpha=1e-6)(reservoir)

    model = ESNModel(inputs=[feedback, driver], outputs=readout)

    print("Model structure:")
    model.summary()

    # Generate training data (separate warmup and train)
    batch_size = 4
    warmup_steps = 100
    train_steps = 400

    warmup_feedback = torch.randn(batch_size, warmup_steps, 1)
    warmup_driver = torch.randn(batch_size, warmup_steps, 5)
    train_feedback = torch.randn(batch_size, train_steps, 1)
    train_driver = torch.randn(batch_size, train_steps, 5)
    train_target = torch.randn(batch_size, train_steps, 1)

    print(f"\nWarmup feedback shape: {warmup_feedback.shape}")
    print(f"Warmup driver shape: {warmup_driver.shape}")
    print(f"Training feedback shape: {train_feedback.shape}")
    print(f"Training driver shape: {train_driver.shape}")
    print(f"Target shape: {train_target.shape}")

    # Train with multiple inputs
    trainer = ESNTrainer(model)
    trainer.fit(
        warmup_inputs=(warmup_feedback, warmup_driver),
        train_inputs=(train_feedback, train_driver),
        targets={"output": train_target},
    )

    print(f"\nReadout fitted: {model.CGReadoutLayer_1.is_fitted}")

    # Test
    model.reset_reservoirs()
    output = model(torch.randn(2, 200, 1), torch.randn(2, 200, 5))
    print(f"Test output shape: {output.shape}")
    print("✓ Training with drivers successful!")


def example_forecasting_after_training():
    """Example: Train model then use it for forecasting."""
    print("\n" + "=" * 60)
    print("Example 4: Training + Forecasting")
    print("=" * 60)

    # Build model
    feedback = Input(shape=(100, 1))
    reservoir = trc.ReservoirLayer(100, 1)(feedback)
    readout = CGReadoutLayer(100, 1, name="output", alpha=1e-6)(reservoir)
    model = ESNModel(feedback, readout)

    # Generate sine data
    data = generate_sine_data(600, freq=0.05)
    warmup_data = data[:, :100, :]  # First 100 for warmup
    train_data = data[:, 100:500, :]  # 400 steps for training
    train_target = train_data.clone()  # Target same as train data
    forecast_warmup = data[:, 400:500, :]  # Last 100 of training for forecast warmup

    print(f"Warmup data shape: {warmup_data.shape}")
    print(f"Training data shape: {train_data.shape}")
    print(f"Target shape: {train_target.shape}")

    # Train
    trainer = ESNTrainer(model)
    trainer.fit(
        warmup_inputs=(warmup_data,),
        train_inputs=(train_data,),
        targets={"output": train_target},
    )
    print("Model trained!")

    # Forecast
    model.reset_reservoirs()
    predictions = model.forecast(forecast_warmup, horizon=100)

    print(f"Forecast warmup shape: {forecast_warmup.shape}")
    print(f"Forecast shape: {predictions.shape}")

    # Compare with ground truth
    ground_truth = data[:, 500:600, :]
    mse = torch.mean((predictions - ground_truth) ** 2).item()
    print(f"Forecast MSE: {mse:.6f}")
    print("✓ Forecasting after training successful!")


def example_different_alphas():
    """Example: Different regularization per readout."""
    print("\n" + "=" * 60)
    print("Example 5: Different Regularization per Readout")
    print("=" * 60)

    feedback = Input(shape=(100, 1))
    reservoir = trc.ReservoirLayer(100, 1)(feedback)

    # Different alpha for each readout
    readout1 = CGReadoutLayer(100, 2, name="weak_reg", alpha=1e-3)(reservoir)
    reservoir2 = trc.ReservoirLayer(50, 2)(readout1)
    readout2 = CGReadoutLayer(50, 1, name="strong_reg", alpha=1e-8)(reservoir2)

    model = ESNModel(feedback, readout2)

    print("Model with different regularization:")
    print(f"  weak_reg: alpha={model.CGReadoutLayer_1.alpha}")
    print(f"  strong_reg: alpha={model.CGReadoutLayer_2.alpha}")

    # Train
    batch_size = 4
    warmup_steps = 100
    train_steps = 400

    warmup_data = torch.randn(batch_size, warmup_steps, 1)
    train_data = torch.randn(batch_size, train_steps, 1)

    trainer = ESNTrainer(model)
    trainer.fit(
        warmup_inputs=(warmup_data,),
        train_inputs=(train_data,),
        targets={
            "weak_reg": torch.randn(batch_size, train_steps, 2),
            "strong_reg": torch.randn(batch_size, train_steps, 1),
        },
    )

    print("✓ Different regularization training successful!")


if __name__ == "__main__":
    torch.manual_seed(42)

    example_simple_training()
    example_multi_readout_training()
    example_training_with_drivers()
    example_forecasting_after_training()
    example_different_alphas()

    print("\n" + "=" * 60)
    print("All training examples completed successfully!")
    print("=" * 60)
