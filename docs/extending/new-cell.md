# Adding a New Cell & Layer

Reservoir cells implement the **single-step dynamics** of a reservoir. A layer wraps the cell and handles the **sequence loop** and **state management**.

---

## Cell Interface

All cells extend `ReservoirCell`:

```python
# src/resdag/layers/cells/base_cell.py (simplified)
class ReservoirCell(nn.Module):
    @property
    def state_size(self) -> int:
        """Size of the state (first dim after batch)."""
        ...

    @property
    def output_size(self) -> int:
        """Output feature dimension."""
        ...

    def forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step update.

        Parameters
        ----------
        inputs : tuple of Tensor
            Input tensors at current timestep: (feedback_t, driver_t, ...).
            Each has shape (batch, feat).
        state : Tensor
            Current state. Shape: (batch, state_size, ...).

        Returns
        -------
        output : Tensor
            Output at current step. Shape: (batch, output_size).
        new_state : Tensor
            Updated state.
        """
        ...

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create zero initial state."""
        ...
```

---

## Step 1: Implement the Cell

Create `src/resdag/layers/cells/my_cell.py`:

```python
"""
My Custom Reservoir Cell
========================
"""

import torch
import torch.nn as nn

from .base_cell import ReservoirCell


class MyCell(ReservoirCell):
    """
    Custom reservoir cell with [your dynamics here].

    Parameters
    ----------
    reservoir_size : int
        Number of reservoir units.
    feedback_size : int
        Feedback input dimension.
    my_param : float
        A custom parameter controlling [something].
    """

    def __init__(
        self,
        reservoir_size: int,
        feedback_size: int,
        my_param: float = 0.5,
    ) -> None:
        super().__init__()
        self.reservoir_size = reservoir_size
        self.feedback_size = feedback_size
        self.my_param = my_param

        # Define weights
        self.weight_hh = nn.Parameter(
            torch.randn(reservoir_size, reservoir_size) * 0.1,
            requires_grad=False,   # frozen in standard ESN
        )
        self.weight_fh = nn.Parameter(
            torch.randn(reservoir_size, feedback_size) * 0.1,
            requires_grad=False,
        )

    @property
    def state_size(self) -> int:
        return self.reservoir_size

    @property
    def output_size(self) -> int:
        return self.reservoir_size

    def forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feedback = inputs[0]   # (batch, feedback_size)

        # Custom dynamics
        pre_activation = (
            state @ self.weight_hh.T
            + feedback @ self.weight_fh.T
        )
        new_state = torch.tanh(pre_activation) * self.my_param + state * (1 - self.my_param)

        return new_state, new_state

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.state_size, device=device, dtype=dtype)
```

---

## Step 2: Implement the Layer

Create `src/resdag/layers/reservoirs/my_layer.py`:

```python
"""
My Reservoir Layer
==================
"""

from .base_reservoir import BaseReservoirLayer
from resdag.layers.cells.my_cell import MyCell


class MyReservoir(BaseReservoirLayer):
    """
    Stateful sequence layer wrapping MyCell.

    Parameters
    ----------
    reservoir_size : int
        Number of reservoir units.
    feedback_size : int
        Feedback input dimension.
    my_param : float
        A custom parameter.
    """

    def __init__(
        self,
        reservoir_size: int,
        feedback_size: int,
        my_param: float = 0.5,
    ) -> None:
        cell = MyCell(
            reservoir_size=reservoir_size,
            feedback_size=feedback_size,
            my_param=my_param,
        )
        super().__init__(cell)

    def __getattr__(self, name: str) -> object:
        """Delegate unknown lookups to the wrapped cell."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        modules = self.__dict__.get("_modules")
        if modules is not None and "cell" in modules:
            try:
                return getattr(modules["cell"], name)
            except AttributeError:
                pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return repr(self.cell)
```

---

## Step 3: Register in `__init__.py`

Add to `src/resdag/layers/cells/__init__.py`:
```python
from .my_cell import MyCell
```

Add to `src/resdag/layers/reservoirs/__init__.py`:
```python
from .my_layer import MyReservoir
```

If it belongs in the public API, add to `src/resdag/layers/__init__.py`:
```python
from .reservoirs.my_layer import MyReservoir
```

And to `src/resdag/__init__.py`:
```python
from .layers import MyReservoir
# ... add to __all__
```

---

## Step 4: Use It

```python
from resdag.layers import MyReservoir
import pytorch_symbolic as ps
from resdag import ESNModel, CGReadoutLayer

inp       = ps.Input((100, 3))
reservoir = MyReservoir(reservoir_size=200, feedback_size=3, my_param=0.7)(inp)
readout   = CGReadoutLayer(200, 3, name="output")(reservoir)
model     = ESNModel(inp, readout)
```

---

## BaseReservoirLayer Provides

By extending `BaseReservoirLayer`, your layer automatically gets:

- Full sequence loop (scans over time axis, feeds state forward)
- `reset_state()` — reset to None or zeros
- `get_state()` — return clone or None
- `set_state()` — restore saved state
- `set_random_state()` — random initialization
- `state` attribute

You can override `set_state()` to add custom shape validation (as `NGReservoir` does for its 3D buffer).
