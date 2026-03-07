# Adding a New Premade Model

Premade models are factory functions that return a configured `ESNModel`. They make it easy to instantiate common architectures with a single call.

---

## Step 1: Create the Model File

Create `src/resdag/models/my_model.py`:

```python
"""
My Custom ESN Architecture
==========================
Description and reference paper if applicable.
"""

from typing import Any

import pytorch_symbolic as ps
import torch

from resdag.composition import ESNModel
from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.layers import CGReadoutLayer, ESNLayer


def my_model(
    reservoir_size: int,
    feedback_size: int,
    output_size: int,
    # Reservoir params
    topology: TopologySpec | None = None,
    spectral_radius: float = 0.9,
    leak_rate: float = 1.0,
    feedback_initializer: InitializerSpec | None = None,
    activation: str = "tanh",
    bias: bool = True,
    trainable: bool = False,
    # Readout params
    readout_alpha: float = 1e-6,
    readout_name: str = "output",
    # Extra kwargs
    **reservoir_kwargs: Any,
) -> ESNModel:
    """
    Build my custom ESN model.

    Description of the architecture and what makes it special.

    Architecture::

        Input -> Reservoir -> [custom transform] -> Readout

    Parameters
    ----------
    reservoir_size : int
        Number of reservoir units.
    feedback_size : int
        Dimension of feedback signal (input features).
    output_size : int
        Dimension of output signal.
    topology : str, tuple, or TopologyInitializer, optional
        Topology for recurrent weights.
    spectral_radius : float, default=0.9
        Target spectral radius for recurrent weights.
    leak_rate : float, default=1.0
        Leaky integration rate.
    feedback_initializer : str, tuple, or InputFeedbackInitializer, optional
        Initializer for feedback weights.
    activation : str, default='tanh'
        Activation function.
    bias : bool, default=True
        Whether to include bias in reservoir.
    trainable : bool, default=False
        Whether reservoir weights are trainable.
    readout_alpha : float, default=1e-6
        L2 regularization for readout.
    readout_name : str, default='output'
        Readout layer name (key in targets dict).
    **reservoir_kwargs
        Additional kwargs passed to ESNLayer.

    Returns
    -------
    ESNModel
        Configured model ready for training.

    Examples
    --------
    >>> from resdag.models import my_model
    >>> model = my_model(reservoir_size=500, feedback_size=3, output_size=3)
    """
    inp = ps.Input((100, feedback_size), dtype=torch.get_default_dtype())

    reservoir = ESNLayer(
        reservoir_size=reservoir_size,
        feedback_size=feedback_size,
        topology=topology,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        feedback_initializer=feedback_initializer,
        activation=activation,
        bias=bias,
        trainable=trainable,
        **reservoir_kwargs,
    )(inp)

    # [Apply custom transformations here]
    # e.g., augmented = MyTransform(...)(reservoir)

    readout = CGReadoutLayer(
        in_features=reservoir_size,
        out_features=output_size,
        alpha=readout_alpha,
        name=readout_name,
    )(reservoir)

    return ESNModel(inp, readout)
```

---

## Step 2: Export in `__init__.py`

Add to `src/resdag/models/__init__.py`:

```python
from .my_model import my_model
```

For the top-level public API, add to `src/resdag/__init__.py`:

```python
from .models import my_model

__all__ = [
    # ... existing ...
    "my_model",
]
```

---

## Step 3: Write Tests

Add `tests/test_models/test_my_model.py`:

```python
import torch
import pytest
from resdag.models import my_model
from resdag.training import ESNTrainer


def test_my_model_shapes():
    model = my_model(reservoir_size=50, feedback_size=3, output_size=3)
    x = torch.randn(2, 10, 3)
    out = model(x)
    assert out.shape == (2, 10, 3)


def test_my_model_training():
    model = my_model(reservoir_size=50, feedback_size=3, output_size=3)
    warmup = torch.randn(1, 20, 3)
    train  = torch.randn(1, 50, 3)
    target = torch.randn(1, 50, 3)

    trainer = ESNTrainer(model)
    trainer.fit(
        warmup_inputs=(warmup,),
        train_inputs=(train,),
        targets={"output": target},
    )


def test_my_model_forecast():
    model = my_model(reservoir_size=50, feedback_size=3, output_size=3)
    warmup = torch.randn(1, 20, 3)
    trainer = ESNTrainer(model)
    trainer.fit(
        warmup_inputs=(warmup,),
        train_inputs=(torch.randn(1, 50, 3),),
        targets={"output": torch.randn(1, 50, 3)},
    )
    preds = model.forecast(warmup, horizon=10)
    assert preds.shape == (1, 10, 3)
```

---

## Step 4: Document It

Add a section to [Premade Models](../guide/premade-models.md) following the existing pattern.

---

## Design Checklist

- [ ] Factory function returns `ESNModel`
- [ ] Accepts `topology` and `feedback_initializer` in TopologySpec / InitializerSpec format
- [ ] Accepts `readout_name` so users can customize the targets key
- [ ] Has NumPy-style docstring with `Parameters`, `Returns`, `Examples`
- [ ] Tests cover shape, training, and forecasting
- [ ] Exported from `src/resdag/models/__init__.py`
- [ ] Added to public API in `src/resdag/__init__.py` and `__all__`
