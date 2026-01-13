"""Tests for model save/load functionality."""

import tempfile
from pathlib import Path

import pytest
import pytorch_symbolic as ps
import torch

from resdag.composition import ESNModel
from resdag.layers import ReservoirLayer
from resdag.layers.readouts import CGReadoutLayer
from resdag.models import classic_esn, headless_esn


class TestBasicSaveLoad:
    """Test basic save and load functionality."""

    def test_save_and_load_simple_model(self):
        """Test saving and loading a simple model."""
        # Create model
        model = classic_esn(50, 1, 1)

        # Get initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            # Save
            model.save(path)
            assert path.exists()

            # Modify parameters to verify loading works
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(1.0)

            # Load
            model.load(path)

            # Verify parameters match initial
            for name, param in model.named_parameters():
                assert torch.allclose(param, initial_params[name])

    def test_save_creates_parent_directories(self):
        """Test that save creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "model.pt"

            model = classic_esn(50, 1, 1)
            model.save(path)

            assert path.exists()
            assert path.parent.exists()

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading from nonexistent file raises FileNotFoundError."""
        model = classic_esn(50, 1, 1)

        with pytest.raises(FileNotFoundError):
            model.load("nonexistent_model.pt")

    def test_save_and_load_with_string_path(self):
        """Test save/load with string paths (not Path objects)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_str = str(Path(tmpdir) / "model.pt")

            model = classic_esn(50, 1, 1)
            initial_params = {name: param.clone() for name, param in model.named_parameters()}

            # Save with string path
            model.save(path_str)

            # Modify
            with torch.no_grad():
                for param in model.parameters():
                    param.mul_(2.0)

            # Load with string path
            model.load(path_str)

            # Verify
            for name, param in model.named_parameters():
                assert torch.allclose(param, initial_params[name])


class TestReservoirStates:
    """Test saving and loading reservoir states."""

    def test_save_without_states_by_default(self):
        """Test that reservoir states are not saved by default."""
        model = headless_esn(50, 1)

        # Run forward to initialize states
        x = torch.randn(2, 10, 1)
        model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            # Save without states
            model.save(path)

            # Load checkpoint
            checkpoint = torch.load(path, weights_only=False)

            # Verify no reservoir states
            assert "reservoir_states" not in checkpoint

    def test_save_with_states(self):
        """Test saving with reservoir states."""
        model = headless_esn(50, 1)

        # Run forward to initialize states
        x = torch.randn(2, 10, 1)
        model(x)

        # Get states
        states_before = model.get_reservoir_states()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            # Save with states
            model.save(path, include_states=True)

            # Load checkpoint
            checkpoint = torch.load(path, weights_only=False)

            # Verify states are present
            assert "reservoir_states" in checkpoint
            assert len(checkpoint["reservoir_states"]) > 0

    def test_load_states(self):
        """Test loading reservoir states."""
        model = headless_esn(50, 1)

        # Run forward to initialize states
        x = torch.randn(2, 10, 1)
        model(x)

        # Get states
        states_before = model.get_reservoir_states()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            # Save with states
            model.save(path, include_states=True)

            # Reset states
            model.reset_reservoirs()
            states_after_reset = model.get_reservoir_states()

            # Verify states were reset (dict should be empty since states are None)
            assert len(states_after_reset) == 0

            # Load with states
            model.load(path, load_states=True)
            states_after_load = model.get_reservoir_states()

            # Verify states match original
            for key in states_before:
                assert torch.allclose(states_before[key], states_after_load[key])

    def test_load_states_warning_when_not_present(self):
        """Test warning when trying to load states that weren't saved."""
        model = headless_esn(50, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            # Save without states
            model.save(path, include_states=False)

            # Load with load_states=True should warn
            with pytest.warns(UserWarning, match="no reservoir states found"):
                model.load(path, load_states=True)


class TestModelArchitecture:
    """Test save/load with different model architectures."""

    def test_manually_built_model(self):
        """Test save/load with manually built pytorch_symbolic model."""
        # Build model manually with pytorch_symbolic
        inp = ps.Input((20, 1))
        res = ReservoirLayer(reservoir_size=50, feedback_size=1, input_size=0)(inp)
        out = CGReadoutLayer(in_features=50, out_features=1)(res)
        model = ESNModel(inp, out)

        # Get initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            # Save
            model.save(path)

            # Modify
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(0.5)

            # Load
            model.load(path)

            # Verify
            for name, param in model.named_parameters():
                assert torch.allclose(param, initial_params[name])

    def test_different_premade_models(self):
        """Test save/load with different premade architectures."""
        from resdag.models import ott_esn

        models = [
            classic_esn(50, 1, 1),
            ott_esn(50, 1, 1),
            headless_esn(50, 1),
        ]

        for i, model in enumerate(models):
            initial_params = {name: param.clone() for name, param in model.named_parameters()}

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / f"model_{i}.pt"

                # Save
                model.save(path)

                # Modify
                with torch.no_grad():
                    for param in model.parameters():
                        param.mul_(1.5)

                # Load
                model.load(path)

                # Verify
                for name, param in model.named_parameters():
                    assert torch.allclose(param, initial_params[name])


class TestStrictLoading:
    """Test strict parameter matching during loading."""

    def test_strict_loading_mismatch_raises_error(self):
        """Test that strict loading raises error on architecture mismatch."""
        # Create and save model with size 50
        model1 = classic_esn(50, 1, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            model1.save(path)

            # Try to load into model with different size
            model2 = classic_esn(100, 1, 1)  # Different reservoir size

            with pytest.raises(RuntimeError):
                model2.load(path, strict=True)

    def test_non_strict_loading_allows_mismatch(self):
        """Test that non-strict loading allows partial parameter loading."""
        # Create and save model
        model1 = classic_esn(50, 1, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            model1.save(path)

            # Load into different architecture with strict=False
            model2 = classic_esn(100, 1, 1)

            # PyTorch's load_state_dict with strict=False still raises errors
            # for size mismatches, it only ignores missing/unexpected keys
            # So this test should actually expect an error
            with pytest.raises(RuntimeError, match="size mismatch"):
                model2.load(path, strict=False)


class TestLoadFromFile:
    """Test class method load_from_file."""

    def test_load_from_file_with_model(self):
        """Test load_from_file class method."""
        model1 = classic_esn(50, 1, 1)
        initial_params = {name: param.clone() for name, param in model1.named_parameters()}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            model1.save(path)

            # Create new model
            model2 = classic_esn(50, 1, 1)

            # Load using class method
            loaded_model = ESNModel.load_from_file(path, model=model2)

            # Verify it's the same instance
            assert loaded_model is model2

            # Verify parameters match
            for name, param in loaded_model.named_parameters():
                assert torch.allclose(param, initial_params[name])

    def test_load_from_file_without_model_raises_error(self):
        """Test that load_from_file without model raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            with pytest.raises(ValueError, match="model argument is required"):
                ESNModel.load_from_file(path, model=None)


class TestTrainingWorkflow:
    """Test realistic training workflow with save/load."""

    def test_train_save_load_inference(self):
        """Test complete workflow: train, save, load, inference."""
        # Create model
        model = classic_esn(50, 1, 1)

        # Simulate training data
        x_train = torch.randn(4, 20, 1)
        y_train = torch.randn(4, 20, 1)

        # Simple "training" (just forward pass to initialize)
        model.train()
        output = model(x_train)

        # Get trained parameters
        trained_params = {name: param.clone() for name, param in model.named_parameters()}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trained_model.pt"

            # Save trained model
            model.save(path)

            # Create new model for inference
            inference_model = classic_esn(50, 1, 1)
            inference_model.load(path)
            inference_model.eval()

            # Verify parameters match
            for name, param in inference_model.named_parameters():
                assert torch.allclose(param, trained_params[name])

            # Run inference
            x_test = torch.randn(2, 10, 1)
            with torch.no_grad():
                output = inference_model(x_test)

            assert output.shape == (2, 10, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUSaveLoad:
    """Test save/load with GPU models."""

    def test_save_gpu_load_cpu(self):
        """Test saving GPU model and loading on CPU."""
        model_gpu = classic_esn(50, 1, 1).cuda()

        # Get parameters
        initial_params = {name: param.cpu().clone() for name, param in model_gpu.named_parameters()}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            # Save GPU model
            model_gpu.save(path)

            # Load on CPU
            model_cpu = classic_esn(50, 1, 1)
            model_cpu.load(path)

            # Verify parameters match
            for name, param in model_cpu.named_parameters():
                assert torch.allclose(param, initial_params[name])

    def test_save_cpu_load_gpu(self):
        """Test saving CPU model and loading on GPU."""
        model_cpu = classic_esn(50, 1, 1)

        # Get parameters
        initial_params = {name: param.clone() for name, param in model_cpu.named_parameters()}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            # Save CPU model
            model_cpu.save(path)

            # Load on GPU
            model_gpu = classic_esn(50, 1, 1).cuda()
            model_gpu.load(path)

            # Verify parameters match (compare on CPU)
            for name, param in model_gpu.named_parameters():
                assert torch.allclose(param.cpu(), initial_params[name])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
