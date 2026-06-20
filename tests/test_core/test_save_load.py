"""ESNModel persistence contracts (save / load / load_from_file).

Pins down: state-dict round-trips, parent-directory creation, optional
reservoir-state checkpointing, strict/non-strict loading on architecture
mismatch, the full train-save-load-infer workflow, and cross-device
(GPU<->CPU) checkpoint portability.
"""

import tempfile
from pathlib import Path

import pytest
import pytorch_symbolic as ps
import torch

from resdag.core import ESNModel
from resdag.layers import ESNLayer
from resdag.layers.readouts import CGReadoutLayer
from resdag.models import classic_esn, headless_esn, linear_esn
from resdag.training import ESNTrainer


class TestBasicSaveLoad:
    """Test basic save and load functionality."""

    def test_save_and_load_simple_model(self) -> None:
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

    def test_save_creates_parent_directories(self) -> None:
        """Test that save creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "model.pt"

            model = classic_esn(50, 1, 1)
            model.save(path)

            assert path.exists()
            assert path.parent.exists()

    def test_load_nonexistent_file_raises_error(self) -> None:
        """Test that loading from nonexistent file raises FileNotFoundError."""
        model = classic_esn(50, 1, 1)

        with pytest.raises(FileNotFoundError):
            model.load("nonexistent_model.pt")

    def test_save_and_load_with_string_path(self) -> None:
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

    def test_save_without_states_by_default(self) -> None:
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

    def test_save_with_states(self) -> None:
        """Test saving with reservoir states."""
        model = headless_esn(50, 1)

        # Run forward to initialize states
        x = torch.randn(2, 10, 1)
        model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            # Save with states
            model.save(path, include_states=True)

            # Load checkpoint
            checkpoint = torch.load(path, weights_only=False)

            # Verify states are present
            assert "reservoir_states" in checkpoint
            assert len(checkpoint["reservoir_states"]) > 0

    def test_load_states(self) -> None:
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

    def test_load_states_warning_when_not_present(self) -> None:
        """Test warning when trying to load states that weren't saved."""
        model = headless_esn(50, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            # Save without states
            model.save(path, include_states=False)

            # Load with load_states=True should warn
            with pytest.warns(UserWarning, match="no reservoir states found"):
                model.load(path, load_states=True)

    def test_load_states_coerces_dtype(self) -> None:
        """load(load_states=True) coerces a mismatched-dtype state, preserving values.

        Exercises the save-as-one-dtype / load-as-another path: a float64 state
        in the checkpoint must be cast back to the model's float32 dtype instead
        of being silently re-zeroed on the next forward pass.
        """
        model = headless_esn(50, 1)

        x = torch.randn(2, 10, 1)
        model(x)
        states_before = model.get_reservoir_states()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            model.save(path, include_states=True)

            # Rewrite the checkpoint so the stored states are float64.
            checkpoint = torch.load(path, weights_only=False)
            checkpoint["reservoir_states"] = {
                k: v.double() for k, v in checkpoint["reservoir_states"].items()
            }
            torch.save(checkpoint, path)

            model.reset_reservoirs()
            with pytest.warns(UserWarning, match="coerced"):
                model.load(path, load_states=True)

            states_after = model.get_reservoir_states()
            for key in states_before:
                assert states_after[key].dtype == torch.float32
                assert torch.allclose(states_before[key], states_after[key], rtol=1e-5, atol=1e-6)


class TestModelArchitecture:
    """Test save/load with different model architectures."""

    def test_manually_built_model(self) -> None:
        """Test save/load with manually built pytorch_symbolic model."""
        # Build model manually with pytorch_symbolic
        inp = ps.Input((20, 1))
        res = ESNLayer(reservoir_size=50, feedback_size=1, input_size=0)(inp)
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

    def test_different_premade_models(self) -> None:
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

    def test_strict_loading_mismatch_raises_error(self) -> None:
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

    def test_non_strict_loading_allows_mismatch(self) -> None:
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

    def test_load_from_file_with_model(self) -> None:
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

    def test_load_from_file_without_model_raises_error(self) -> None:
        """Test that load_from_file without model raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            with pytest.raises(ValueError, match="model argument is required"):
                ESNModel.load_from_file(path, model=None)


class TestTrainingWorkflow:
    """Test realistic training workflow with save/load."""

    def test_train_save_load_inference(self) -> None:
        """Test complete workflow: train, save, load, inference."""
        # Create model
        model = classic_esn(50, 1, 1)

        # Simulate training data
        x_train = torch.randn(4, 20, 1)

        # Simple "training" (just forward pass to initialize)
        model.train()
        model(x_train)

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


class TestFullSaveLoad:
    """Full-model serialization (save_full / load_full) via pytorch-symbolic 1.2 pickling.

    Unlike state-dict save/load, these round-trip the whole architecture, so no
    pre-built model is needed on the loading side.
    """

    def test_save_full_load_full_roundtrip_no_rebuild(self) -> None:
        """load_full reconstructs a working model without re-creating it first."""
        model = classic_esn(50, 3, 3)
        ESNTrainer(model).fit(
            warmup_inputs=(torch.randn(1, 20, 3),),
            train_inputs=(torch.randn(1, 40, 3),),
            targets={"output": torch.randn(1, 40, 3)},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "full.pt"
            model.save_full(path)

            restored = ESNModel.load_full(path)  # no build_model() needed
            assert isinstance(restored, ESNModel)

            x = torch.randn(2, 20, 3)
            model.reset_reservoirs()
            restored.reset_reservoirs()
            assert torch.allclose(model(x), restored(x), atol=1e-5)

    def test_save_full_preserves_subclass_and_methods(self) -> None:
        """The reconstructed object keeps the ESNModel subclass and its API."""
        model = classic_esn(40, 3, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "full.pt"
            model.save_full(path)
            restored = ESNModel.load_full(path)
            assert type(restored).__name__ == "ESNModel"
            restored.reset_reservoirs()  # ESN-specific method survives
            out = restored.forecast(torch.randn(1, 20, 3), horizon=10)
            assert out.shape == (1, 10, 3)

    def test_save_full_preserves_reservoir_states(self) -> None:
        """Live reservoir states are captured by the full save."""
        model = headless_esn(50, 3)
        model.warmup(torch.randn(1, 20, 3))
        states_before = model.get_reservoir_states()
        assert states_before  # non-empty

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "full.pt"
            model.save_full(path)
            restored = ESNModel.load_full(path)
            states_after = restored.get_reservoir_states()
            assert set(states_after) == set(states_before)
            for key in states_before:
                assert torch.allclose(states_before[key], states_after[key])

    def test_save_full_metadata_roundtrip(self) -> None:
        """Metadata kwargs are stored and returned with return_metadata=True."""
        model = classic_esn(40, 3, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "full.pt"
            model.save_full(path, epoch=10, val_mse=0.012)

            restored, meta = ESNModel.load_full(path, return_metadata=True)
            assert isinstance(restored, ESNModel)
            assert meta == {"epoch": 10, "val_mse": 0.012}

    def test_save_full_creates_parent_directories(self) -> None:
        """save_full creates missing parent directories like save does."""
        model = classic_esn(40, 3, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "full.pt"
            model.save_full(path)
            assert path.exists()

    def test_linear_esn_identity_activation_is_picklable(self) -> None:
        """Regression: identity activation must be a module-level fn, not a lambda.

        ``linear_esn`` uses ``activation="identity"``; a local lambda there
        would raise ``PicklingError`` on save_full / torch.save(model).
        """
        model = linear_esn(40, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "linear_full.pt"
            model.save_full(path)  # must not raise
            restored = ESNModel.load_full(path)
            assert isinstance(restored, ESNModel)
            x = torch.randn(2, 20, 3)
            model.reset_reservoirs()
            restored.reset_reservoirs()
            assert torch.allclose(model(x), restored(x), atol=1e-5)

    def test_load_full_rejects_state_dict_checkpoint(self) -> None:
        """Pointing load_full at a save() file gives a clear error."""
        model = classic_esn(40, 3, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state_dict.pt"
            model.save(path)  # state-dict format, not full
            with pytest.raises(ValueError, match="does not contain a full ESNModel"):
                ESNModel.load_full(path)

    def test_save_full_with_string_path(self) -> None:
        """save_full / load_full accept string paths."""
        model = classic_esn(40, 3, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path_str = str(Path(tmpdir) / "full.pt")
            model.save_full(path_str)
            restored = ESNModel.load_full(path_str)
            assert isinstance(restored, ESNModel)

    def test_load_rejects_full_checkpoint(self) -> None:
        """Pointing the state-dict load() at a save_full() file gives a clear error."""
        model = classic_esn(40, 3, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "full.pt"
            model.save_full(path)
            with pytest.raises(ValueError, match="load_full"):
                classic_esn(40, 3, 3).load(path)

    def test_load_full_accepts_bare_torch_save(self) -> None:
        """load_full also reads a plain torch.save(model) file (no metadata wrapper)."""
        model = classic_esn(40, 3, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bare.pt"
            torch.save(model, path)  # no save_full wrapper
            restored = ESNModel.load_full(path)
            assert isinstance(restored, ESNModel)
            restored2, meta = ESNModel.load_full(path, return_metadata=True)
            assert meta == {}  # bare files carry no metadata
            x = torch.randn(2, 20, 3)
            model.reset_reservoirs()
            restored2.reset_reservoirs()
            assert torch.allclose(model(x), restored2(x), atol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUSaveLoad:
    """Save/load with GPU models (cross-device checkpoint portability)."""

    def test_save_gpu_load_cpu(self) -> None:
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

    def test_save_cpu_load_gpu(self) -> None:
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
