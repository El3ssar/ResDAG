"""
Model Composition
=================

This module provides tools for building ESN models using the
``pytorch_symbolic`` library for symbolic tensor computation.

Classes
-------
ESNModel
    Extended SymbolicModel with ESN-specific methods for forecasting
    and reservoir state management.
Input
    Alias for ``pytorch_symbolic.Input`` for defining model inputs.

Examples
--------
Building a simple ESN:

>>> import pytorch_symbolic as ps
>>> from resdag.composition import ESNModel
>>> from resdag.layers import ReservoirLayer
>>> from resdag.layers.readouts import CGReadoutLayer
>>>
>>> inp = ps.Input((100, 3))
>>> reservoir = ReservoirLayer(200, feedback_size=3)(inp)
>>> readout = CGReadoutLayer(200, 3)(reservoir)
>>> model = ESNModel(inp, readout)

Multi-input model:

>>> feedback = ps.Input((100, 3))
>>> driver = ps.Input((100, 5))
>>> reservoir = ReservoirLayer(200, feedback_size=3, input_size=5)(feedback, driver)
>>> readout = CGReadoutLayer(200, 3)(reservoir)
>>> model = ESNModel([feedback, driver], readout)

See Also
--------
resdag.models : Premade ESN architectures.
resdag.training.ESNTrainer : Trainer for fitting readouts.
"""

import pytorch_symbolic as ps

from .symbolic import ESNModel, Input

__all__ = ["ESNModel", "Input", "ps"]
