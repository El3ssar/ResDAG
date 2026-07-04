"""
Reservoir-computing metrics
===========================

Forecast-quality and dynamical metrics for reservoir models, with first-class
support for reporting horizons in **Lyapunov times** — the natural clock of a
chaotic attractor — instead of raw integration steps.

Lyapunov bookkeeping
--------------------
lyapunov_time, steps_per_lyapunov_time, steps_to_lyapunov, lyapunov_to_steps
    Pure, dependency-free conversions between step counts and Lyapunov times.
CANONICAL_LAMBDA_MAX, MACKEY_GLASS_LAMBDA
    Literature largest-Lyapunov-exponent tables for resdag's built-in
    generators.
largest_lyapunov
    Estimate the largest Lyapunov exponent of a generator name (canonical),
    a measured series, or a ``tsdynamics`` System.
reservoir_lyapunov
    Native-torch Benettin/QR estimate of a trained reservoir's own maximal
    Lyapunov exponent (an edge-of-chaos / Echo-State-Property diagnostic).

Forecast accuracy
-----------------
valid_prediction_time
    Contiguous valid forecast horizon (NRMSE below a threshold), reportable in
    steps or Lyapunov times.

Optional dependency
-------------------
The series- and System-based Lyapunov estimators require the optional
``tsdynamics`` package (``pip install resdag[dynamics]``, Python >= 3.12);
every ``tsdynamics`` import is lazy and guarded.  Everything else — the
conversions, the canonical constants, the reservoir Jacobian estimate, and the
forecast metrics — is pure PyTorch/NumPy and always available.

Examples
--------
>>> from resdag.metrics import steps_to_lyapunov, valid_prediction_time
>>> round(steps_to_lyapunov(357, dt=0.02, lambda_max=0.9056), 2)
6.47
"""

from .forecast import valid_prediction_time
from .lyapunov import (
    CANONICAL_LAMBDA_MAX,
    MACKEY_GLASS_LAMBDA,
    largest_lyapunov,
    lyapunov_time,
    lyapunov_to_steps,
    reservoir_lyapunov,
    steps_per_lyapunov_time,
    steps_to_lyapunov,
)

__all__ = [
    # Lyapunov constants
    "CANONICAL_LAMBDA_MAX",
    "MACKEY_GLASS_LAMBDA",
    # Pure conversions
    "lyapunov_time",
    "steps_per_lyapunov_time",
    "steps_to_lyapunov",
    "lyapunov_to_steps",
    # Estimators
    "largest_lyapunov",
    "reservoir_lyapunov",
    # Forecast accuracy
    "valid_prediction_time",
]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
