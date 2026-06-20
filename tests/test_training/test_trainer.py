"""Tests for ESNTrainer.

Every test seeds ``torch.manual_seed`` so reservoir topology, input/feedback
weights, and any random data are reproducible — failures here must be
deterministic, not flaky.

Beyond the structural ``is_fitted`` / output-shape checks, this module contains
*correctness* tests that fit a **learnable, deterministic** target (a known
affine map of the reservoir states) and assert the trained model reconstructs
it to a tight ``atol``.  Fitting pure ``torch.randn`` targets — as the original
suite did — only proves the solver ran, not that it found the right weights: a
no-op or a wrong solve would still flip ``is_fitted`` to ``True``.  Anchoring on
a target the readout can represent exactly turns the trainer into a falsifiable
check.

See Also
--------
resdag.training.ESNTrainer : Trainer under test.
resdag.layers.readouts.CGReadoutLayer : Readout fitted by the trainer.
"""

import pytest
import torch

import resdag as rd
from resdag.core import ESNModel, Input
from resdag.layers import ESNLayer
from resdag.layers.readouts import CGReadoutLayer
from resdag.training import ESNTrainer

# Tight reconstruction tolerance for the correctness tests.  The readout solves
# ridge regression with a tiny ``alpha`` on a system it can represent exactly
# (target = affine map of the reservoir states), so the residual is governed by
# float32 accumulation, not by the regularizer.
RECON_ATOL = 1e-3


def _capture_reservoir_states(
    reservoir: ESNLayer,
    warmup_inputs: tuple[torch.Tensor, ...],
    train_inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Replay the trainer's warmup + single forward pass on a reservoir layer.

    The :class:`ESNTrainer` warms the model up (resetting state first) and then
    runs exactly one teacher-forced forward pass over ``train_inputs``; the
    readout is fitted on the reservoir states produced by *that* pass.  This
    helper reproduces the same two phases on the ``reservoir`` layer in
    isolation so a test can build a target that is an exact function of the
    states the trainer will actually see.

    Parameters
    ----------
    reservoir : ESNLayer
        The reservoir layer instance used inside the model under test.
    warmup_inputs : tuple of torch.Tensor
        Warmup sequences, ``(feedback, driver1, ...)``.
    train_inputs : tuple of torch.Tensor
        Training sequences, ``(feedback, driver1, ...)``.

    Returns
    -------
    torch.Tensor
        Reservoir states for the training pass of shape
        ``(batch, train_steps, reservoir_size)``.
    """
    with torch.no_grad():
        reservoir.reset_state()
        reservoir(*warmup_inputs)
        states: torch.Tensor = reservoir(*train_inputs)
    return states


class TestESNTrainerBasic:
    """Basic trainer tests with single readout."""

    def test_trainer_init(self):
        """Test trainer initialization."""
        torch.manual_seed(0)
        feedback = Input(shape=(10, 1))
        reservoir = rd.ESNLayer(50, 1)(feedback)
        readout = CGReadoutLayer(50, 1, name="output")(reservoir)
        model = ESNModel(feedback, readout)

        trainer = ESNTrainer(model)
        assert trainer.model is model

    def test_simple_training(self):
        """Test training a simple single-readout model."""
        torch.manual_seed(0)
        # Build model
        feedback = Input(shape=(10, 1))
        reservoir = rd.ESNLayer(50, 1)(feedback)
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

    def test_reconstructs_learnable_linear_target(self) -> None:
        """Trainer recovers a known affine map of the reservoir states.

        Targets are constructed as ``states @ W_true.T + b_true`` for the exact
        states the trainer fits on, so the readout (ridge regression with a tiny
        ``alpha``) must reconstruct them to within ``RECON_ATOL``.  This is the
        correctness signal absent when targets are pure noise.
        """
        torch.manual_seed(0)

        reservoir_size = 60
        out_features = 2
        batch_size = 3
        warmup_steps = 60
        train_steps = 200

        # Shared reservoir instance so the captured states match the model's.
        reservoir_layer = ESNLayer(reservoir_size, 1)

        feedback = Input(shape=(10, 1))
        reservoir = reservoir_layer(feedback)
        readout = CGReadoutLayer(reservoir_size, out_features, name="output")(reservoir)
        model = ESNModel(feedback, readout)

        warmup_data = torch.randn(batch_size, warmup_steps, 1)
        train_data = torch.randn(batch_size, train_steps, 1)

        # Capture the exact states the readout will be fitted on, then build a
        # deterministic, learnable target from them.
        states = _capture_reservoir_states(reservoir_layer, (warmup_data,), (train_data,))
        w_true = torch.randn(out_features, reservoir_size)
        b_true = torch.randn(out_features)
        train_targets = states @ w_true.T + b_true

        trainer = ESNTrainer(model)
        trainer.fit(
            warmup_inputs=(warmup_data,),
            train_inputs=(train_data,),
            targets={"output": train_targets},
        )

        assert model.CGReadoutLayer_1.is_fitted

        # Re-run the model exactly as the trainer did and compare to the target.
        model.reset_reservoirs()
        model.warmup(warmup_data)
        with torch.no_grad():
            predictions = model(train_data)

        assert predictions.shape == train_targets.shape
        assert torch.allclose(predictions, train_targets, atol=RECON_ATOL), (
            "max abs error "
            f"{(predictions - train_targets).abs().max().item():.3e} exceeds {RECON_ATOL}"
        )

    def test_reconstructs_target_from_seeded_sine_drive(self) -> None:
        """Trainer reconstructs a learnable target driven by a seeded sine.

        Uses a deterministic ``sin`` input (not ``torch.randn``) to drive the
        reservoir, then builds the target as a known affine map of the resulting
        states.  Asserts reconstruction to a tight ``atol`` — exercising the
        smooth-input regime while keeping the correctness check exact.
        """
        torch.manual_seed(0)

        reservoir_size = 120
        batch_size = 1
        warmup_steps = 100
        train_steps = 400

        reservoir_layer = ESNLayer(reservoir_size, 1)

        feedback = Input(shape=(10, 1))
        reservoir = reservoir_layer(feedback)
        readout = CGReadoutLayer(reservoir_size, 1, name="output")(reservoir)
        model = ESNModel(feedback, readout)

        t_warmup = torch.linspace(0.0, 10.0, warmup_steps).reshape(1, warmup_steps, 1)
        t_train = torch.linspace(10.0, 50.0, train_steps).reshape(1, train_steps, 1)
        warmup_data = torch.sin(t_warmup).expand(batch_size, -1, -1).contiguous()
        train_data = torch.sin(t_train).expand(batch_size, -1, -1).contiguous()

        states = _capture_reservoir_states(reservoir_layer, (warmup_data,), (train_data,))
        w_true = torch.randn(1, reservoir_size)
        b_true = torch.randn(1)
        train_targets = states @ w_true.T + b_true

        trainer = ESNTrainer(model)
        trainer.fit(
            warmup_inputs=(warmup_data,),
            train_inputs=(train_data,),
            targets={"output": train_targets},
        )

        assert model.CGReadoutLayer_1.is_fitted

        model.reset_reservoirs()
        model.warmup(warmup_data)
        with torch.no_grad():
            predictions = model(train_data)

        assert predictions.shape == train_targets.shape
        assert torch.allclose(predictions, train_targets, atol=RECON_ATOL), (
            "max abs error "
            f"{(predictions - train_targets).abs().max().item():.3e} exceeds {RECON_ATOL}"
        )

    def test_unnamed_readout(self):
        """Test training with auto-generated readout names."""
        torch.manual_seed(0)
        feedback = Input(shape=(10, 1))
        reservoir = rd.ESNLayer(50, 1)(feedback)
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
        torch.manual_seed(0)
        feedback = Input(shape=(10, 1))
        reservoir = rd.ESNLayer(50, 1)(feedback)
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
        torch.manual_seed(0)
        feedback = Input(shape=(10, 1))
        reservoir1 = rd.ESNLayer(50, 1)(feedback)
        readout1 = CGReadoutLayer(50, 2, name="intermediate")(reservoir1)

        reservoir2 = rd.ESNLayer(30, 2)(readout1)
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

    def test_stacked_readouts_reconstruct_known_targets(self) -> None:
        """Stacked DAG fitting reconstructs both readouts to a tight ``atol``.

        Exercises the topological-order fitting in ``ESNTrainer.fit`` with a
        *correctness* check rather than ``is_fitted``.  The DAG is
        ``feedback -> res1 -> intermediate -> res2 -> output``, so the states
        feeding ``output`` depend on ``intermediate`` having been fitted first —
        if the trainer fitted out of order, the final target would not match.

        Targets are built as exact affine maps of the states each readout sees,
        replaying the model's true two-phase data flow (warmup then a single
        forward pass *without* an intervening reset on ``res2``, matching how the
        trainer drives the graph).  Both readouts must then reconstruct their
        targets to within ``RECON_ATOL``.
        """
        torch.manual_seed(0)

        res1_size = 50
        res2_size = 40
        intermediate_dim = 2
        batch_size = 3
        warmup_steps = 60
        train_steps = 220

        res1_layer = ESNLayer(res1_size, 1)
        intermediate_layer = CGReadoutLayer(res1_size, intermediate_dim, name="intermediate")
        res2_layer = ESNLayer(res2_size, intermediate_dim)
        output_layer = CGReadoutLayer(res2_size, 1, name="output")

        feedback = Input(shape=(10, 1))
        reservoir1 = res1_layer(feedback)
        readout1 = intermediate_layer(reservoir1)
        reservoir2 = res2_layer(readout1)
        readout2 = output_layer(reservoir2)
        model = ESNModel(feedback, readout2)

        warmup_data = torch.randn(batch_size, warmup_steps, 1)
        train_data = torch.randn(batch_size, train_steps, 1)

        # Stage 1: states feeding the intermediate readout -> intermediate target.
        states1 = _capture_reservoir_states(res1_layer, (warmup_data,), (train_data,))
        w1 = torch.randn(intermediate_dim, res1_size)
        b1 = torch.randn(intermediate_dim)
        intermediate_targets = states1 @ w1.T + b1

        # Pre-fit the intermediate readout so we can drive ``res2`` with the
        # *actual* output it will emit (not the idealized target — they differ
        # by the solver residual, which the nonlinear res2 dynamics amplify).
        intermediate_layer.fit(states1, intermediate_targets)

        # Stage 2: replay the model's true warmup -> forward flow to capture the
        # ``res2`` states feeding ``output``.  No reset between the two phases,
        # exactly as the model executes them.
        with torch.no_grad():
            res1_layer.reset_state()
            res2_layer.reset_state()
            warmup_intermediate = intermediate_layer(res1_layer(warmup_data))
            res2_layer(warmup_intermediate)
            train_intermediate = intermediate_layer(res1_layer(train_data))
            states2 = res2_layer(train_intermediate)
        w2 = torch.randn(1, res2_size)
        b2 = torch.randn(1)
        output_targets = states2 @ w2.T + b2

        # Re-fit everything through the trainer (it discovers and fits both
        # readouts in topological order during a single forward pass).
        trainer = ESNTrainer(model)
        trainer.fit(
            warmup_inputs=(warmup_data,),
            train_inputs=(train_data,),
            targets={
                "intermediate": intermediate_targets,
                "output": output_targets,
            },
        )

        assert model.CGReadoutLayer_1.is_fitted
        assert model.CGReadoutLayer_2.is_fitted

        # Intermediate readout: exact affine map of its states, recovered tightly.
        model.reset_reservoirs()
        res1_layer.reset_state()
        res1_layer(warmup_data)
        with torch.no_grad():
            intermediate_pred = intermediate_layer(res1_layer(train_data))
        assert torch.allclose(intermediate_pred, intermediate_targets, atol=RECON_ATOL), (
            "intermediate max abs error "
            f"{(intermediate_pred - intermediate_targets).abs().max().item():.3e}"
        )

        # Final output: run the whole DAG end to end and compare to the target.
        model.reset_reservoirs()
        model.warmup(warmup_data)
        with torch.no_grad():
            output_pred = model(train_data)
        assert output_pred.shape == output_targets.shape
        assert torch.allclose(output_pred, output_targets, atol=RECON_ATOL), (
            "output max abs error " f"{(output_pred - output_targets).abs().max().item():.3e}"
        )

    def test_parallel_readouts_from_same_reservoir(self):
        """Test training parallel readouts from the same reservoir."""
        torch.manual_seed(0)
        feedback = Input(shape=(10, 1))
        reservoir = rd.ESNLayer(50, 1)(feedback)

        # Two readouts from same reservoir
        readout1 = CGReadoutLayer(50, 2, name="branch1")(reservoir)
        readout2 = CGReadoutLayer(50, 3, name="branch2")(reservoir)

        # Concatenate and output
        from resdag.layers.transforms import Concatenate

        concat = Concatenate()(readout1, readout2)

        reservoir2 = rd.ESNLayer(30, 5)(concat)
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
        torch.manual_seed(0)
        feedback = Input(shape=(10, 1))
        driver = Input(shape=(10, 3))

        reservoir = rd.ESNLayer(50, 1, input_size=3)(feedback, driver)
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

    def test_driven_model_reconstructs_learnable_target(self) -> None:
        """Driven model recovers a known affine map of its states.

        Same correctness contract as the feedback-only case, but exercising the
        multi-input warmup/forward path: targets are an exact affine function of
        the states produced by the feedback+driver pass, so the fit must
        reconstruct them within ``RECON_ATOL``.
        """
        torch.manual_seed(0)

        reservoir_size = 60
        batch_size = 3
        warmup_steps = 60
        train_steps = 200

        reservoir_layer = ESNLayer(reservoir_size, 1, input_size=3)

        feedback = Input(shape=(10, 1))
        driver = Input(shape=(10, 3))
        reservoir = reservoir_layer(feedback, driver)
        readout = CGReadoutLayer(reservoir_size, 1, name="output")(reservoir)
        model = ESNModel(inputs=[feedback, driver], outputs=readout)

        warmup_feedback = torch.randn(batch_size, warmup_steps, 1)
        warmup_driver = torch.randn(batch_size, warmup_steps, 3)
        train_feedback = torch.randn(batch_size, train_steps, 1)
        train_driver = torch.randn(batch_size, train_steps, 3)

        states = _capture_reservoir_states(
            reservoir_layer,
            (warmup_feedback, warmup_driver),
            (train_feedback, train_driver),
        )
        w_true = torch.randn(1, reservoir_size)
        b_true = torch.randn(1)
        train_targets = states @ w_true.T + b_true

        trainer = ESNTrainer(model)
        trainer.fit(
            warmup_inputs=(warmup_feedback, warmup_driver),
            train_inputs=(train_feedback, train_driver),
            targets={"output": train_targets},
        )

        assert model.CGReadoutLayer_1.is_fitted

        model.reset_reservoirs()
        model.warmup(warmup_feedback, warmup_driver)
        with torch.no_grad():
            predictions = model(train_feedback, train_driver)

        assert predictions.shape == train_targets.shape
        assert torch.allclose(predictions, train_targets, atol=RECON_ATOL), (
            "max abs error "
            f"{(predictions - train_targets).abs().max().item():.3e} exceeds {RECON_ATOL}"
        )


class TestESNTrainerValidation:
    """Tests for validation and error handling."""

    def test_missing_target_raises_error(self):
        """Test that missing target raises ValueError."""
        torch.manual_seed(0)
        feedback = Input(shape=(10, 1))
        reservoir = rd.ESNLayer(50, 1)(feedback)
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
        torch.manual_seed(0)
        feedback = Input(shape=(10, 1))
        reservoir = rd.ESNLayer(50, 1)(feedback)
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
        torch.manual_seed(0)
        feedback = Input(shape=(10, 1))
        reservoir = rd.ESNLayer(50, 1)(feedback)
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
        torch.manual_seed(0)
        feedback = Input(shape=(10, 1))
        reservoir = rd.ESNLayer(50, 1)(feedback)
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
        torch.manual_seed(0)
        feedback = Input(shape=(10, 1))
        reservoir = rd.ESNLayer(50, 1)(feedback)
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
        torch.manual_seed(0)
        feedback = Input(shape=(10, 1))
        reservoir = rd.ESNLayer(50, 1)(feedback)
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
        torch.manual_seed(0)
        feedback = Input(shape=(10, 1))
        reservoir = rd.ESNLayer(50, 1)(feedback)
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
        torch.manual_seed(0)
        device = torch.device("cuda")

        feedback = Input(shape=(10, 1))
        reservoir = rd.ESNLayer(50, 1)(feedback)
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


class TestESNTrainerInternals:
    """Regression tests covering the trainer's reliance on pytorch_symbolic."""

    def test_trainer_does_not_reach_into_pytorch_symbolic_privates(self):
        """The trainer must only use stable PyTorch APIs (named_modules)
        to discover readouts (Phase 3.1).

        Concretely: ESNTrainer must not access ``_execution_order_nodes``,
        ``_execution_order_layers``, or ``_node_to_layer_name`` on the
        underlying SymbolicModel.  We assert by replacing those attributes
        with sentinels that explode on access and confirming training
        still works.
        """
        torch.manual_seed(0)

        class _Forbidden:
            def __get__(self, *_args, **_kwargs):
                raise AssertionError("ESNTrainer reached into a private pytorch_symbolic attribute")

        feedback = Input(shape=(10, 1))
        reservoir = rd.ESNLayer(40, 1)(feedback)
        readout_layer = CGReadoutLayer(40, 1, name="output")
        readout = readout_layer(reservoir)
        model = ESNModel(feedback, readout)

        # Stash the originals so we can restore them and avoid leaking
        # sentinels into other tests sharing the symbolic graph.
        cls = type(model)
        originals = {}
        for name in (
            "_execution_order_nodes",
            "_execution_order_layers",
            "_node_to_layer_name",
        ):
            if hasattr(cls, name):
                originals[name] = getattr(cls, name)
            setattr(cls, name, _Forbidden())

        try:
            warmup = torch.randn(2, 50, 1)
            train = torch.randn(2, 100, 1)
            target = torch.randn(2, 100, 1)
            ESNTrainer(model).fit(
                warmup_inputs=(warmup,),
                train_inputs=(train,),
                targets={"output": target},
            )
            assert readout_layer.is_fitted
        finally:
            for name, original in originals.items():
                setattr(cls, name, original)
            # Use ``cls.__dict__`` lookup rather than ``hasattr`` so we don't
            # re-trigger the descriptor we planted.
            for name in (
                "_execution_order_nodes",
                "_execution_order_layers",
                "_node_to_layer_name",
            ):
                if name not in originals and name in cls.__dict__:
                    delattr(cls, name)

    def test_branching_dag_multi_readout(self):
        """Trainer fits all readouts in a branching DAG."""
        torch.manual_seed(0)
        feedback = Input(shape=(10, 1))
        res = rd.ESNLayer(30, 1)(feedback)

        from resdag.layers.transforms import Concatenate

        head_a_layer = CGReadoutLayer(30, 2, name="head_a")
        head_b_layer = CGReadoutLayer(30, 3, name="head_b")
        out_layer = CGReadoutLayer(20, 1, name="out")

        head_a = head_a_layer(res)
        head_b = head_b_layer(res)
        merged = Concatenate()(head_a, head_b)
        res2 = rd.ESNLayer(20, 5)(merged)
        out = out_layer(res2)

        model = ESNModel(feedback, out)

        warmup = torch.randn(2, 40, 1)
        train = torch.randn(2, 80, 1)
        targets = {
            "head_a": torch.randn(2, 80, 2),
            "head_b": torch.randn(2, 80, 3),
            "out": torch.randn(2, 80, 1),
        }

        ESNTrainer(model).fit(
            warmup_inputs=(warmup,),
            train_inputs=(train,),
            targets=targets,
        )

        # Confirm each named readout layer was fitted.
        for readout_layer in (head_a_layer, head_b_layer, out_layer):
            assert readout_layer.is_fitted

        fitted_readouts = [m for _, m in model.named_modules() if isinstance(m, CGReadoutLayer)]
        assert len(fitted_readouts) == 3
