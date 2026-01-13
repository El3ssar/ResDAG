"""Example usage of model save/load functionality.

This script demonstrates how to save and load trained ESN models using
PyTorch's standard save/load mechanisms.
"""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from resdag.models import classic_esn, headless_esn


def example_basic_save_load():
    """Example: Basic save and load."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Save and Load")
    print("=" * 60)

    # Create model
    model = classic_esn(reservoir_size=100, feedback_size=1, output_size=1)

    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model.pt"

        # Save model
        model.save(save_path)
        print(f"Model saved to: {save_path}")

        # Create new model and load
        new_model = classic_esn(reservoir_size=100, feedback_size=1, output_size=1)
        new_model.load(save_path)
        print("Model loaded successfully!")

        # Verify parameters match
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert torch.allclose(param1, param2)
        print("✓ All parameters match!")


def example_training_workflow():
    """Example: Complete training workflow with save/load."""
    print("\n" + "=" * 60)
    print("Example 2: Training Workflow")
    print("=" * 60)

    # Create model with trainable readout
    model = classic_esn(
        reservoir_size=100,
        feedback_size=1,
        output_size=1,
        trainable=False,  # Frozen reservoir
    )

    # Generate synthetic training data
    torch.manual_seed(42)
    x_train = torch.randn(32, 50, 1)
    y_train = torch.sin(x_train * 2)  # Simple target

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    print("Training...")
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        print(f"  Epoch {epoch + 1}/5, Loss: {loss.item():.4f}")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "trained_model.pt"

        # Save trained model
        model.save(save_path)
        print(f"\nTrained model saved to: {save_path}")

        # Load for inference
        inference_model = classic_esn(
            reservoir_size=100,
            feedback_size=1,
            output_size=1,
            trainable=False,
        )
        inference_model.load(save_path)
        inference_model.eval()
        print("Model loaded for inference")

        # Test inference
        x_test = torch.randn(4, 20, 1)
        with torch.no_grad():
            predictions = inference_model(x_test)
        print(f"Inference output shape: {predictions.shape}")
        print("✓ Training workflow complete!")


def example_save_with_reservoir_states():
    """Example: Saving and loading reservoir states."""
    print("\n" + "=" * 60)
    print("Example 3: Saving Reservoir States")
    print("=" * 60)

    model = headless_esn(reservoir_size=100, feedback_size=2)

    # Run forward to initialize states
    x = torch.randn(4, 50, 2)
    states_before = model(x)
    print(f"Reservoir states shape: {states_before.shape}")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model_with_states.pt"

        # Save WITH reservoir states
        model.save(save_path, include_states=True)
        print("Model saved with reservoir states")

        # Reset states
        model.reset_reservoirs()
        print("Reservoir states reset")

        # Load WITHOUT states (default)
        model.load(save_path, load_states=False)
        states_after_load_no_states = model.get_reservoir_states()
        print("Loaded without states - reservoir states are reset")

        # Load WITH states
        model.load(save_path, load_states=True)
        states_after_load_with_states = model.get_reservoir_states()
        print("Loaded with states - reservoir states restored")

        # Continue processing from saved state
        x_continue = torch.randn(4, 20, 2)
        output = model(x_continue)
        print(f"Continued processing: output shape {output.shape}")
        print("✓ Reservoir state management complete!")


def example_cross_device_save_load():
    """Example: Saving on one device, loading on another."""
    print("\n" + "=" * 60)
    print("Example 4: Cross-Device Save/Load")
    print("=" * 60)

    # Create model on CPU
    model_cpu = classic_esn(50, 1, 1)
    print("Created model on CPU")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model.pt"

        # Save CPU model
        model_cpu.save(save_path)
        print("Saved CPU model")

        if torch.cuda.is_available():
            # Load on GPU
            model_gpu = classic_esn(50, 1, 1).cuda()
            model_gpu.load(save_path)
            print("Loaded on GPU")

            # Test on GPU
            x_gpu = torch.randn(2, 10, 1).cuda()
            output_gpu = model_gpu(x_gpu)
            print(f"GPU inference output shape: {output_gpu.shape}")
            print("✓ Cross-device loading successful!")
        else:
            print("CUDA not available - skipping GPU test")


def example_checkpoint_system():
    """Example: Checkpoint system for long training."""
    print("\n" + "=" * 60)
    print("Example 5: Checkpoint System")
    print("=" * 60)

    model = classic_esn(100, 1, 1, trainable=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir()

        # Simulate training with checkpoints
        for epoch in range(3):
            # Simulate training...
            print(f"Epoch {epoch + 1}/3")

            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            model.save(checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path.name}")

        # List checkpoints
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
        print(f"\nFound {len(checkpoints)} checkpoints:")
        for cp in checkpoints:
            print(f"  - {cp.name}")

        # Load best checkpoint (last one in this example)
        best_checkpoint = checkpoints[-1]
        model.load(best_checkpoint)
        print(f"\nLoaded best checkpoint: {best_checkpoint.name}")
        print("✓ Checkpoint system complete!")


def example_model_versioning():
    """Example: Model versioning and metadata."""
    print("\n" + "=" * 60)
    print("Example 6: Model Versioning")
    print("=" * 60)

    model = classic_esn(100, 1, 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model_v1.pt"

        # Save with custom metadata (using torch.save kwargs)
        model.save(save_path)

        # You can also manually add metadata to the checkpoint
        checkpoint = torch.load(save_path, weights_only=False)
        checkpoint["metadata"] = {
            "version": "1.0",
            "reservoir_size": 100,
            "trained_on": "2024-01-01",
            "notes": "Initial model",
        }
        torch.save(checkpoint, save_path)
        print("Saved model with metadata")

        # Load and inspect metadata
        checkpoint = torch.load(save_path, weights_only=False)
        if "metadata" in checkpoint:
            print("\nModel metadata:")
            for key, value in checkpoint["metadata"].items():
                print(f"  {key}: {value}")

        # Load model
        model.load(save_path)
        print("\n✓ Model with metadata loaded!")


if __name__ == "__main__":
    # Run all examples
    example_basic_save_load()
    example_training_workflow()
    example_save_with_reservoir_states()
    example_cross_device_save_load()
    example_checkpoint_system()
    example_model_versioning()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
