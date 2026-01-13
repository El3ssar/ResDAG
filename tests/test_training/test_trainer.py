"""Tests for ESNTrainer."""

import pytest
import torch

import resdag as trc
from resdag.composition import ESNModel, Input
from resdag.layers.readouts import CGReadoutLayer
from resdag.training import ESNTrainer


class TestESNTrainerBasic:
    """Basic trainer tests with single readout."""

    def test_trainer_init(self):
        """Test trainer initialization."""
        feedback = Input(shape=(10, 1))
        reservoir = trc.ReservoirLayer(50, 1)(feedback)
        readout = CGReadoutLayer(50, 1, name="output")(reservoir)
        model = ESNModel(feedback, readout)

        trainer = ESNTrainer(model)
        assert trainer.model is model

    def test_simple_training(self):
        """Test training a simple single-readout model."""
        # Build model
        feedback = Input(shape=(10, 1))
        reservoir = trc.ReservoirLayer(50, 1)(feedback)
        readout = CGReadoutLayer(50, 1, name="output")(reservoir)
        model = ESNModel(feedback, readout)

        # Create training data
        batch_size = 4
        warmup_steps = 50
        train_steps = 150

        warmup_data = torch.randn(batch_size, warmup_steps, 1)
        train_data = torch.randn(batch_size, train_steps, 1)
        train_targets = torch.randn(batch_size, train_steps, 1)

        # Train
        trainer = ESNTrainer(model)
        trainer.fit(
            warmup_inputs=(warmup_data,),
            train_inputs=(train_data,),
            targets={"output": train_targets},
        )

        # Check readout is fitted
        readout_layer = model.CGReadoutLayer_1
        assert readout_layer.is_fitted

    def test_unnamed_readout(self):
        """Test training with auto-generated readout names."""
        feedback = Input(shape=(10, 1))
        reservoir = trc.ReservoirLayer(50, 1)(feedback)
        # No name provided - uses auto-generated "CGReadoutLayer_1"
        readout = CGReadoutLayer(50, 1)(reservoir)
        model = ESNModel(feedback, readout)

        batch_size = 4
        warmup_steps = 50
        train_steps = 150

        warmup_data = torch.randn(batch_size, warmup_steps, 1)
        train_data = torch.randn(batch_size, train_steps, 1)
        train_targets = torch.randn(batch_size, train_steps, 1)

        trainer = ESNTrainer(model)
        trainer.fit(
            warmup_inputs=(warmup_data,),
            train_inputs=(train_data,),
            targets={"CGReadoutLayer_1": train_targets},
        )

        assert model.CGReadoutLayer_1.is_fitted

    def test_model_produces_output_after_training(self):
        """Test model works after training."""
        feedback = Input(shape=(10, 1))
        reservoir = trc.ReservoirLayer(50, 1)(feedback)
        readout = CGReadoutLayer(50, 1, name="output")(reservoir)
        model = ESNModel(feedback, readout)

        # Train
        batch_size = 4
        warmup_data = torch.randn(batch_size, 50, 1)
        train_data = torch.randn(batch_size, 150, 1)
        train_targets = torch.randn(batch_size, 150, 1)

        trainer = ESNTrainer(model)
        trainer.fit(
            warmup_inputs=(warmup_data,),
            train_inputs=(train_data,),
            targets={"output": train_targets},
        )

        # Model should work
        model.reset_reservoirs()
        test_input = torch.randn(2, 100, 1)
        output = model(test_input)
        assert output.shape == (2, 100, 1)


class TestESNTrainerMultiReadout:
    """Tests for multi-readout models."""

    def test_stacked_readouts(self):
        """Test training stacked readouts (readout1 -> reservoir2 -> readout2)."""
        feedback = Input(shape=(10, 1))
        reservoir1 = trc.ReservoirLayer(50, 1)(feedback)
        readout1 = CGReadoutLayer(50, 2, name="intermediate")(reservoir1)

        reservoir2 = trc.ReservoirLayer(30, 2)(readout1)
        readout2 = CGReadoutLayer(30, 1, name="output")(reservoir2)

        model = ESNModel(feedback, readout2)

        batch_size = 4
        warmup_steps = 50
        train_steps = 150

        warmup_data = torch.randn(batch_size, warmup_steps, 1)
        train_data = torch.randn(batch_size, train_steps, 1)
        intermediate_targets = torch.randn(batch_size, train_steps, 2)
        output_targets = torch.randn(batch_size, train_steps, 1)

        trainer = ESNTrainer(model)
        trainer.fit(
            warmup_inputs=(warmup_data,),
            train_inputs=(train_data,),
            targets={
                "intermediate": intermediate_targets,
                "output": output_targets,
            },
        )

        # Both readouts should be fitted
        assert model.CGReadoutLayer_1.is_fitted  # intermediate
        assert model.CGReadoutLayer_2.is_fitted  # output

    def test_parallel_readouts_from_same_reservoir(self):
        """Test training parallel readouts from the same reservoir."""
        feedback = Input(shape=(10, 1))
        reservoir = trc.ReservoirLayer(50, 1)(feedback)

        # Two readouts from same reservoir
        readout1 = CGReadoutLayer(50, 2, name="branch1")(reservoir)
        readout2 = CGReadoutLayer(50, 3, name="branch2")(reservoir)

        # Concatenate and output
        from resdag.layers.custom import Concatenate

        concat = Concatenate()(readout1, readout2)

        reservoir2 = trc.ReservoirLayer(30, 5)(concat)
        readout_final = CGReadoutLayer(30, 1, name="output")(reservoir2)

        model = ESNModel(feedback, readout_final)

        batch_size = 4
        warmup_steps = 50
        train_steps = 150

        warmup_data = torch.randn(batch_size, warmup_steps, 1)
        train_data = torch.randn(batch_size, train_steps, 1)

        trainer = ESNTrainer(model)
        trainer.fit(
            warmup_inputs=(warmup_data,),
            train_inputs=(train_data,),
            targets={
                "branch1": torch.randn(batch_size, train_steps, 2),
                "branch2": torch.randn(batch_size, train_steps, 3),
                "output": torch.randn(batch_size, train_steps, 1),
            },
        )

        # All readouts should be fitted
        assert model.CGReadoutLayer_1.is_fitted
        assert model.CGReadoutLayer_2.is_fitted
        assert model.CGReadoutLayer_3.is_fitted


class TestESNTrainerWithDrivers:
    """Tests for models with driving inputs."""

    def test_training_with_driving_input(self):
        """Test training model with feedback + driving input."""
        feedback = Input(shape=(10, 1))
        driver = Input(shape=(10, 3))

        reservoir = trc.ReservoirLayer(50, 1, input_size=3)(feedback, driver)
        readout = CGReadoutLayer(50, 1, name="output")(reservoir)

        model = ESNModel(inputs=[feedback, driver], outputs=readout)

        batch_size = 4
        warmup_steps = 50
        train_steps = 150

        warmup_feedback = torch.randn(batch_size, warmup_steps, 1)
        warmup_driver = torch.randn(batch_size, warmup_steps, 3)
        train_feedback = torch.randn(batch_size, train_steps, 1)
        train_driver = torch.randn(batch_size, train_steps, 3)
        train_targets = torch.randn(batch_size, train_steps, 1)

        trainer = ESNTrainer(model)
        trainer.fit(
            warmup_inputs=(warmup_feedback, warmup_driver),
            train_inputs=(train_feedback, train_driver),
            targets={"output": train_targets},
        )

        assert model.CGReadoutLayer_1.is_fitted


class TestESNTrainerValidation:
    """Tests for validation and error handling."""

    def test_missing_target_raises_error(self):
        """Test that missing target raises ValueError."""
        feedback = Input(shape=(10, 1))
        reservoir = trc.ReservoirLayer(50, 1)(feedback)
        readout = CGReadoutLayer(50, 1, name="output")(reservoir)
        model = ESNModel(feedback, readout)

        trainer = ESNTrainer(model)

        with pytest.raises(ValueError, match="Missing targets"):
            trainer.fit(
                warmup_inputs=(torch.randn(4, 50, 1),),
                train_inputs=(torch.randn(4, 150, 1),),
                targets={},  # No targets!
            )

    def test_wrong_target_name_raises_error(self):
        """Test that wrong target name raises ValueError."""
        feedback = Input(shape=(10, 1))
        reservoir = trc.ReservoirLayer(50, 1)(feedback)
        readout = CGReadoutLayer(50, 1, name="output")(reservoir)
        model = ESNModel(feedback, readout)

        trainer = ESNTrainer(model)

        with pytest.raises(ValueError, match="Missing targets"):
            trainer.fit(
                warmup_inputs=(torch.randn(4, 50, 1),),
                train_inputs=(torch.randn(4, 150, 1),),
                targets={"wrong_name": torch.randn(4, 150, 1)},
            )

    def test_input_count_mismatch_raises_error(self):
        """Test that mismatched warmup/train input counts raise error."""
        feedback = Input(shape=(10, 1))
        reservoir = trc.ReservoirLayer(50, 1)(feedback)
        readout = CGReadoutLayer(50, 1, name="output")(reservoir)
        model = ESNModel(feedback, readout)

        trainer = ESNTrainer(model)

        with pytest.raises(ValueError, match="Must match"):
            trainer.fit(
                warmup_inputs=(torch.randn(4, 50, 1), torch.randn(4, 50, 3)),  # 2 inputs
                train_inputs=(torch.randn(4, 150, 1),),  # 1 input
                targets={"output": torch.randn(4, 150, 1)},
            )

    def test_target_timesteps_mismatch_raises_error(self):
        """Test that target with wrong timesteps raises error."""
        feedback = Input(shape=(10, 1))
        reservoir = trc.ReservoirLayer(50, 1)(feedback)
        readout = CGReadoutLayer(50, 1, name="output")(reservoir)
        model = ESNModel(feedback, readout)

        trainer = ESNTrainer(model)

        with pytest.raises(ValueError, match="timesteps"):
            trainer.fit(
                warmup_inputs=(torch.randn(4, 50, 1),),
                train_inputs=(torch.randn(4, 150, 1),),
                targets={"output": torch.randn(4, 100, 1)},  # Wrong! Should be 150
            )

    def test_extra_targets_warning(self):
        """Test that extra targets raise a warning."""
        feedback = Input(shape=(10, 1))
        reservoir = trc.ReservoirLayer(50, 1)(feedback)
        readout = CGReadoutLayer(50, 1, name="output")(reservoir)
        model = ESNModel(feedback, readout)

        trainer = ESNTrainer(model)

        with pytest.warns(UserWarning, match="non-existent"):
            trainer.fit(
                warmup_inputs=(torch.randn(4, 50, 1),),
                train_inputs=(torch.randn(4, 150, 1),),
                targets={
                    "output": torch.randn(4, 150, 1),
                    "extra": torch.randn(4, 150, 1),  # Extra!
                },
            )

    def test_no_warmup_inputs_raises_error(self):
        """Test that no warmup inputs raises error."""
        feedback = Input(shape=(10, 1))
        reservoir = trc.ReservoirLayer(50, 1)(feedback)
        readout = CGReadoutLayer(50, 1, name="output")(reservoir)
        model = ESNModel(feedback, readout)

        trainer = ESNTrainer(model)

        with pytest.raises(ValueError, match="warmup"):
            trainer.fit(
                warmup_inputs=(),
                train_inputs=(torch.randn(4, 150, 1),),
                targets={"output": torch.randn(4, 150, 1)},
            )

    def test_no_train_inputs_raises_error(self):
        """Test that no train inputs raises error."""
        feedback = Input(shape=(10, 1))
        reservoir = trc.ReservoirLayer(50, 1)(feedback)
        readout = CGReadoutLayer(50, 1, name="output")(reservoir)
        model = ESNModel(feedback, readout)

        trainer = ESNTrainer(model)

        with pytest.raises(ValueError, match="training"):
            trainer.fit(
                warmup_inputs=(torch.randn(4, 50, 1),),
                train_inputs=(),
                targets={"output": torch.randn(4, 150, 1)},
            )


class TestESNTrainerGPU:
    """GPU tests (if available)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_training_on_gpu(self):
        """Test training works on GPU."""
        device = torch.device("cuda")

        feedback = Input(shape=(10, 1))
        reservoir = trc.ReservoirLayer(50, 1)(feedback)
        readout = CGReadoutLayer(50, 1, name="output")(reservoir)
        model = ESNModel(feedback, readout).to(device)

        batch_size = 4
        warmup_data = torch.randn(batch_size, 50, 1, device=device)
        train_data = torch.randn(batch_size, 150, 1, device=device)
        train_targets = torch.randn(batch_size, 150, 1, device=device)

        trainer = ESNTrainer(model)
        trainer.fit(
            warmup_inputs=(warmup_data,),
            train_inputs=(train_data,),
            targets={"output": train_targets},
        )

        assert model.CGReadoutLayer_1.is_fitted
