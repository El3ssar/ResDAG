"""
ESN Training Utilities
======================

This module provides trainers for ESN models that fit readout layers
algebraically using ridge regression, rather than stochastic gradient descent.

Classes
-------
ESNTrainer
    Trainer for fitting readout layers in ESN models.

Examples
--------
>>> from resdag.training import ESNTrainer
>>> trainer = ESNTrainer(model)
>>> trainer.fit(
...     warmup_inputs=(warmup,),
...     train_inputs=(train,),
...     targets={"output": target},
... )

See Also
--------
resdag.layers.readouts.CGReadoutLayer : Readout with CG solver.
resdag.core.ESNModel : ESN model class.
"""

from .trainer import ESNTrainer

__all__ = ["ESNTrainer"]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
