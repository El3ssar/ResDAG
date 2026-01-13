#!/usr/bin/env python3
"""Example 10: Hyperparameter Optimization for ESN Models.

This example demonstrates how to use Optuna-based hyperparameter optimization
to find optimal ESN configurations for chaotic time series prediction.

The HPO system uses three user-defined callbacks:
    - model_creator: Creates a fresh model with given hyperparameters
    - search_space: Defines what hyperparameters to optimize
    - data_loader: Provides training and validation data

Key Features:
    - Multiple specialized loss functions for chaotic systems
    - Study persistence for resumable optimization
    - Parallel execution support
    - Flexible interface for any ESN architecture
"""

import numpy as np
import torch

from resdag.hpo import LOSSES, get_study_summary, run_hpo
from resdag.models import ott_esn

# =============================================================================
# Example 1: Basic HPO with Synthetic Data
# =============================================================================


def example_basic_hpo():
    """Basic hyperparameter optimization example.

    This is the simplest possible HPO setup using synthetic data.
    Good for understanding the core concepts.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic HPO with Synthetic Data")
    print("=" * 60)

    # ---------------------------
    # 1. Define model_creator
    # ---------------------------
    # This function creates a fresh model for each trial.
    # It must accept ALL hyperparameters from search_space as kwargs.

    def model_creator(reservoir_size: int, spectral_radius: float, leak_rate: float):
        """Create an ESN with the given hyperparameters."""
        return ott_esn(
            reservoir_size=reservoir_size,
            feedback_size=3,
            output_size=3,
            spectral_radius=spectral_radius,
            leak_rate=leak_rate,
            topology=("random", {"density": 0.1}),
        )

    # ---------------------------
    # 2. Define search_space
    # ---------------------------
    # Uses Optuna's trial.suggest_* methods to define the search space.
    # Returns a dict that maps to model_creator's parameters.

    def search_space(trial):
        """Define hyperparameter search space."""
        return {
            "reservoir_size": trial.suggest_int("reservoir_size", 50, 200, step=50),
            "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.5),
            "leak_rate": trial.suggest_float("leak_rate", 0.1, 1.0),
        }

    # ---------------------------
    # 3. Define data_loader
    # ---------------------------
    # Returns a dict with required keys: warmup, train, target, f_warmup, val
    # Data shapes: (B, T, D) where B=batch, T=timesteps, D=features

    def data_loader(trial):
        """Load and prepare data for training/evaluation."""
        # Generate synthetic Lorenz-like data
        torch.manual_seed(42)
        data = torch.randn(1, 300, 3).cumsum(dim=1) * 0.1

        return {
            "warmup": data[:, :50, :],  # Warmup: first 50 steps
            "train": data[:, 50:150, :],  # Training input: next 100 steps
            "target": data[:, 51:151, :],  # Target: shifted by 1 (next-step pred)
            "f_warmup": data[:, 150:200, :],  # Forecast warmup: 50 steps
            "val": data[:, 200:250, :],  # Validation: 50 steps
        }

    # ---------------------------
    # 4. Run optimization
    # ---------------------------
    study = run_hpo(
        model_creator=model_creator,
        search_space=search_space,
        data_loader=data_loader,
        n_trials=5,  # Number of trials (increase for real use!)
        loss="efh",  # Expected Forecast Horizon (default)
        verbosity=1,
    )

    # ---------------------------
    # 5. Analyze results
    # ---------------------------
    print("\n" + get_study_summary(study))
    print("\nBest parameters found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


# =============================================================================
# Example 2: Different Loss Functions
# =============================================================================


def example_loss_functions():
    """Compare different loss functions.

    Available losses:
        - "efh": Expected Forecast Horizon (smooth, differentiable)
        - "horizon": Forecast Horizon (contiguous valid steps)
        - "lyap": Lyapunov-weighted (accounts for exponential divergence)
        - "standard": Simple mean error
        - "discounted": Half-life weighted RMSE
    """
    print("\n" + "=" * 60)
    print("Example 2: Different Loss Functions")
    print("=" * 60)

    print(f"\nAvailable loss functions: {list(LOSSES.keys())}")

    def model_creator(reservoir_size):
        return ott_esn(reservoir_size=reservoir_size, feedback_size=3, output_size=3)

    def search_space(trial):
        return {"reservoir_size": trial.suggest_int("reservoir_size", 50, 150, step=50)}

    def data_loader(trial):
        torch.manual_seed(42)
        data = torch.randn(1, 200, 3).cumsum(dim=1) * 0.1
        return {
            "warmup": data[:, :30, :],
            "train": data[:, 30:100, :],
            "target": data[:, 31:101, :],
            "f_warmup": data[:, 100:130, :],
            "val": data[:, 130:180, :],
        }

    # Run with Lyapunov loss (good for chaotic systems)
    print("\nRunning with Lyapunov-weighted loss...")
    study = run_hpo(
        model_creator=model_creator,
        search_space=search_space,
        data_loader=data_loader,
        n_trials=3,
        loss="lyap",
        loss_params={"lle": 0.9, "dt": 0.02},  # System-specific params
        verbosity=0,
    )
    print(f"Best value (lyap): {study.best_value:.4f}")

    # Run with discounted RMSE (emphasizes early predictions)
    print("\nRunning with discounted RMSE...")
    study = run_hpo(
        model_creator=model_creator,
        search_space=search_space,
        data_loader=data_loader,
        n_trials=3,
        loss="discounted",
        loss_params={"half_life": 32},
        verbosity=0,
    )
    print(f"Best value (discounted): {study.best_value:.4f}")


# =============================================================================
# Example 3: Study Persistence
# =============================================================================


def example_persistence():
    """Persist and resume optimization studies.

    Studies can be saved to SQLite databases, allowing:
        - Interrupting and resuming long optimizations
        - Sharing results across machines
        - Analyzing results offline
    """
    print("\n" + "=" * 60)
    print("Example 3: Study Persistence")
    print("=" * 60)

    import tempfile
    from pathlib import Path

    def model_creator(reservoir_size):
        return ott_esn(reservoir_size=reservoir_size, feedback_size=3, output_size=3)

    def search_space(trial):
        return {"reservoir_size": trial.suggest_int("reservoir_size", 50, 150, step=50)}

    def data_loader(trial):
        torch.manual_seed(42)
        data = torch.randn(1, 200, 3).cumsum(dim=1) * 0.1
        return {
            "warmup": data[:, :30, :],
            "train": data[:, 30:100, :],
            "target": data[:, 31:101, :],
            "f_warmup": data[:, 100:130, :],
            "val": data[:, 130:180, :],
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "my_study.db"
        storage_url = f"sqlite:///{db_path}"

        # Run first 3 trials
        print("\nRunning first 3 trials...")
        study = run_hpo(
            model_creator=model_creator,
            search_space=search_space,
            data_loader=data_loader,
            n_trials=3,
            storage=storage_url,
            study_name="persistent_study",
            verbosity=0,
        )
        print(f"Completed {len(study.trials)} trials")

        # Simulate resuming (e.g., after restart)
        print("\nResuming with 2 more trials...")
        study = run_hpo(
            model_creator=model_creator,
            search_space=search_space,
            data_loader=data_loader,
            n_trials=5,  # Target 5 total
            storage=storage_url,
            study_name="persistent_study",
            verbosity=0,
        )
        print(f"Total trials: {len(study.trials)}")
        print(f"Database size: {db_path.stat().st_size / 1024:.1f} KB")


# =============================================================================
# Example 4: Parallel Execution
# =============================================================================


def example_parallel():
    """Run optimization trials in parallel.

    Uses Optuna's built-in n_jobs for parallel execution.
    Note: Each worker needs independent data/model creation.
    """
    print("\n" + "=" * 60)
    print("Example 4: Parallel Execution")
    print("=" * 60)

    import time

    def model_creator(reservoir_size):
        return ott_esn(reservoir_size=reservoir_size, feedback_size=3, output_size=3)

    def search_space(trial):
        return {"reservoir_size": trial.suggest_int("reservoir_size", 50, 200, step=50)}

    def data_loader(trial):
        torch.manual_seed(42)
        data = torch.randn(1, 200, 3).cumsum(dim=1) * 0.1
        return {
            "warmup": data[:, :30, :],
            "train": data[:, 30:100, :],
            "target": data[:, 31:101, :],
            "f_warmup": data[:, 100:130, :],
            "val": data[:, 130:180, :],
        }

    # Sequential execution
    print("\nSequential execution (n_workers=1)...")
    start = time.time()
    study = run_hpo(
        model_creator=model_creator,
        search_space=search_space,
        data_loader=data_loader,
        n_trials=4,
        n_workers=1,
        verbosity=0,
    )
    seq_time = time.time() - start
    print(f"Time: {seq_time:.2f}s")

    # Parallel execution
    print("\nParallel execution (n_workers=2)...")
    start = time.time()
    study = run_hpo(
        model_creator=model_creator,
        search_space=search_space,
        data_loader=data_loader,
        n_trials=4,
        n_workers=2,
        verbosity=0,
    )
    par_time = time.time() - start
    print(f"Time: {par_time:.2f}s")
    print(f"Speedup: {seq_time / par_time:.2f}x")


# =============================================================================
# Example 5: Advanced Search Space
# =============================================================================


def example_advanced_search():
    """Advanced search space with conditional parameters.

    Optuna supports conditional hyperparameters where some params
    depend on others. Useful for exploring architecture choices.
    """
    print("\n" + "=" * 60)
    print("Example 5: Advanced Search Space")
    print("=" * 60)

    def model_creator(
        reservoir_size: int,
        spectral_radius: float,
        topology_type: str,
        topology_param: float,
    ):
        """Create model with variable topology."""
        # Build topology config based on type
        if topology_type == "random":
            topology = ("random", {"density": topology_param})
        elif topology_type == "erdos_renyi":
            topology = ("erdos_renyi", {"p": topology_param})
        else:
            topology = ("watts_strogatz", {"p": topology_param, "k": 4})

        return ott_esn(
            reservoir_size=reservoir_size,
            feedback_size=3,
            output_size=3,
            spectral_radius=spectral_radius,
            topology=topology,
        )

    def search_space(trial):
        """Search space with conditional topology parameters."""
        # Choose topology type
        topology_type = trial.suggest_categorical(
            "topology_type", ["random", "erdos_renyi", "watts_strogatz"]
        )

        # Topology parameter depends on type
        if topology_type == "random":
            # Density: fraction of non-zero weights
            topology_param = trial.suggest_float("topology_param", 0.05, 0.3)
        elif topology_type == "erdos_renyi":
            # Connection probability
            topology_param = trial.suggest_float("topology_param", 0.1, 0.5)
        else:
            # Rewiring probability for watts-strogatz
            topology_param = trial.suggest_float("topology_param", 0.1, 0.5)

        return {
            "reservoir_size": trial.suggest_int("reservoir_size", 100, 300, step=50),
            "spectral_radius": trial.suggest_float("spectral_radius", 0.7, 1.3),
            "topology_type": topology_type,
            "topology_param": topology_param,
        }

    def data_loader(trial):
        torch.manual_seed(42)
        data = torch.randn(1, 200, 3).cumsum(dim=1) * 0.1
        return {
            "warmup": data[:, :30, :],
            "train": data[:, 30:100, :],
            "target": data[:, 31:101, :],
            "f_warmup": data[:, 100:130, :],
            "val": data[:, 130:180, :],
        }

    study = run_hpo(
        model_creator=model_creator,
        search_space=search_space,
        data_loader=data_loader,
        n_trials=5,
        verbosity=0,
    )

    print("\nBest configuration found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


# =============================================================================
# Example 6: Using Real Data
# =============================================================================


def example_real_data():
    """HPO with real chaotic time series data.

    Uses resdag's data loading utilities to load and prepare
    real trajectory data for optimization.
    """
    print("\n" + "=" * 60)
    print("Example 6: Using Real Data (if available)")
    print("=" * 60)

    from pathlib import Path

    # Check if Lorenz data exists
    data_path = Path("/data/Pincha/MachineLearning/ESN/Research/Data/continuous/Lorenz")
    if not data_path.exists():
        print("Lorenz data not found. Skipping this example.")
        print("To run: place .npy files in the expected path.")
        return

    from resdag.utils.data import load_and_prepare

    def model_creator(reservoir_size, spectral_radius, alpha):
        return ott_esn(
            reservoir_size=reservoir_size,
            feedback_size=3,
            output_size=3,
            spectral_radius=spectral_radius,
            readout_alpha=alpha,
            topology=("random", {"density": 0.1}),
        )

    def search_space(trial):
        return {
            "reservoir_size": trial.suggest_int("reservoir_size", 200, 500, step=100),
            "spectral_radius": trial.suggest_float("spectral_radius", 0.7, 1.2),
            "alpha": trial.suggest_float("alpha", 1e-8, 1e-4, log=True),
        }

    def data_loader(trial):
        # Load multiple trajectories for robust training
        files = [str(data_path / f"Lorenz_{i}.npy") for i in range(1, 4)]
        warmup, train, target, f_warmup, val = load_and_prepare(
            files,
            warmup_steps=500,
            train_steps=5000,
            val_steps=500,
            discard_steps=500,
            normalize=True,
            norm_method="minmax",
        )
        return {
            "warmup": warmup,
            "train": train,
            "target": target,
            "f_warmup": f_warmup,
            "val": val,
        }

    study = run_hpo(
        model_creator=model_creator,
        search_space=search_space,
        data_loader=data_loader,
        n_trials=10,
        loss="efh",
        loss_params={"threshold": 0.2},
        verbosity=1,
    )

    print("\n" + get_study_summary(study))


# =============================================================================
# Example 7: Custom Loss Function
# =============================================================================


def example_custom_loss():
    """Using a custom loss function.

    You can define any loss function that takes (y_true, y_pred) arrays
    of shape (B, T, D) and returns a float to minimize.
    """
    print("\n" + "=" * 60)
    print("Example 7: Custom Loss Function")
    print("=" * 60)

    def my_custom_loss(y_true, y_pred, early_weight=2.0):
        """Custom loss emphasizing early predictions.

        Applies higher weight to first few timesteps.
        """
        T = y_true.shape[1]
        weights = np.exp(-np.arange(T) / (T / early_weight))
        weights = weights / weights.sum()

        errors = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 2)))
        return float(np.sum(weights * errors))

    def model_creator(reservoir_size):
        return ott_esn(reservoir_size=reservoir_size, feedback_size=3, output_size=3)

    def search_space(trial):
        return {"reservoir_size": trial.suggest_int("reservoir_size", 50, 150, step=50)}

    def data_loader(trial):
        torch.manual_seed(42)
        data = torch.randn(1, 200, 3).cumsum(dim=1) * 0.1
        return {
            "warmup": data[:, :30, :],
            "train": data[:, 30:100, :],
            "target": data[:, 31:101, :],
            "f_warmup": data[:, 100:130, :],
            "val": data[:, 130:180, :],
        }

    # Pass custom loss directly
    study = run_hpo(
        model_creator=model_creator,
        search_space=search_space,
        data_loader=data_loader,
        n_trials=3,
        loss=my_custom_loss,  # Custom callable
        verbosity=0,
    )

    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")


# =============================================================================
# Example 8: Input-Driven Multi-Reservoir Model
# =============================================================================


def example_input_driven():
    """HPO for custom multi-reservoir model with driving inputs.

    Demonstrates:
        - Custom model architecture (not premade)
        - Multiple reservoir layers, each with its own driver
        - Driving inputs (external forcing) with feedback_size + input_size
        - Proper data handling for multiple drivers

    Architecture:
        feedback ──┬──────────────────────────────────> Reservoir0 ──┐
                   │                                         ↑       │
        driver0 ───┴─────────────────────────────────────────┘       │
                                                                     │
        feedback ──┬──────────────────────────────────> Reservoir1 ──┼──> Concat ──> Readout
                   │                                         ↑       │
        driver1 ───┴─────────────────────────────────────────┘       │
                                                                     │
        feedback ────────────────────────────────────────────────────┘
    """
    print("\n" + "=" * 60)
    print("Example 8: Input-Driven Multi-Reservoir Model")
    print("=" * 60)

    import pytorch_symbolic as ps

    from resdag.composition import ESNModel
    from resdag.init.topology import get_topology
    from resdag.layers import ReservoirLayer
    from resdag.layers.custom import Concatenate
    from resdag.layers.readouts import CGReadoutLayer

    # Dimensions
    FEEDBACK_DIM = 3
    DRIVER0_DIM = 2
    DRIVER1_DIM = 4
    OUTPUT_DIM = 3

    def model_creator(
        reservoir0_size: int,
        reservoir1_size: int,
        spectral_radius: float,
        leak_rate: float,
    ):
        """Create a multi-reservoir model with driving inputs.

        Each reservoir receives:
        - feedback: the main signal (used for autoregressive forecasting)
        - driver: an external forcing signal specific to that reservoir

        The readout combines feedback + both reservoir outputs.
        """
        # Input layers: feedback + 2 drivers
        feedback = ps.Input((100, FEEDBACK_DIM))
        driver0 = ps.Input((100, DRIVER0_DIM))
        driver1 = ps.Input((100, DRIVER1_DIM))

        # Create topology initializer
        topology = get_topology("random", density=0.1)

        # Reservoir0: receives feedback + driver0
        reservoir0 = ReservoirLayer(
            reservoir_size=reservoir0_size,
            feedback_size=FEEDBACK_DIM,
            input_size=DRIVER0_DIM,  # Driver input
            spectral_radius=spectral_radius,
            leak_rate=leak_rate,
            topology=topology,
        )(feedback, driver0)  # Two inputs: (feedback, driver)

        # Reservoir1: receives feedback + driver1
        reservoir1 = ReservoirLayer(
            reservoir_size=reservoir1_size,
            feedback_size=FEEDBACK_DIM,
            input_size=DRIVER1_DIM,  # Different driver
            spectral_radius=spectral_radius,
            leak_rate=leak_rate,
            topology=topology,
        )(feedback, driver1)  # Two inputs: (feedback, driver)

        # Readout: combines feedback + both reservoir outputs
        concat_out = Concatenate()(feedback, reservoir0, reservoir1)
        readout = CGReadoutLayer(
            in_features=FEEDBACK_DIM + reservoir0_size + reservoir1_size,
            out_features=OUTPUT_DIM,
            name="output",
        )(concat_out)

        return ESNModel([feedback, driver0, driver1], readout)

    def search_space(trial):
        """Search over reservoir sizes and dynamics."""
        return {
            "reservoir0_size": trial.suggest_int("reservoir0_size", 30, 100, step=10),
            "reservoir1_size": trial.suggest_int("reservoir1_size", 50, 150, step=25),
            "spectral_radius": trial.suggest_float("spectral_radius", 0.7, 1.2),
            "leak_rate": trial.suggest_float("leak_rate", 0.3, 1.0),
        }

    def data_loader(trial):
        """Load data with multiple driving inputs.

        For input-driven models with multiple drivers, we need:
        - warmup_{driver}: driver during training warmup
        - train_{driver}: driver during training
        - f_warmup_{driver}: driver during forecast warmup
        - forecast_{driver}: driver during autoregressive forecasting

        Each driver needs its own set of data with these prefixes.
        """
        torch.manual_seed(42)

        # Generate synthetic data
        # feedback: the main signal we're predicting
        feedback = torch.randn(1, 300, FEEDBACK_DIM).cumsum(dim=1) * 0.1

        # driver0: sinusoidal forcing signal
        driver0 = torch.sin(torch.linspace(0, 20 * np.pi, 300)).unsqueeze(0).unsqueeze(-1)
        driver0 = driver0.expand(1, 300, DRIVER0_DIM).clone()

        # driver1: different frequency sinusoidal forcing
        driver1 = torch.cos(torch.linspace(0, 15 * np.pi, 300)).unsqueeze(0).unsqueeze(-1)
        driver1 = driver1.expand(1, 300, DRIVER1_DIM).clone()

        return {
            # Feedback data (first input, used for autoregressive forecasting)
            "warmup": feedback[:, :50, :],
            "train": feedback[:, 50:150, :],
            "target": feedback[:, 51:151, :],
            "f_warmup": feedback[:, 150:200, :],
            "val": feedback[:, 200:250, :],
            # Driver0 data (second input)
            "warmup_driver0": driver0[:, :50, :],
            "train_driver0": driver0[:, 50:150, :],
            "f_warmup_driver0": driver0[:, 150:200, :],
            "forecast_driver0": driver0[:, 200:250, :],
            # Driver1 data (third input)
            "warmup_driver1": driver1[:, :50, :],
            "train_driver1": driver1[:, 50:150, :],
            "f_warmup_driver1": driver1[:, 150:200, :],
            "forecast_driver1": driver1[:, 200:250, :],
        }

    study = run_hpo(
        model_creator=model_creator,
        search_space=search_space,
        data_loader=data_loader,
        n_trials=3,
        loss="efh",
        drivers_keys=["driver0", "driver1"],  # Both drivers
        verbosity=0,
    )

    print(f"Best value: {study.best_value:.4f}")
    print("\nBest configuration:")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("resdag Hyperparameter Optimization Examples")
    print("=" * 60)

    # Run all examples
    example_basic_hpo()
    example_loss_functions()
    example_persistence()
    example_parallel()
    example_advanced_search()
    example_custom_loss()
    example_input_driven()

    # Optional: real data example (requires data files)
    # example_real_data()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
