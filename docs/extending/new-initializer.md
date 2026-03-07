# Adding a New Initializer

Input and feedback initializers generate the weight matrices \(W_{in}\) and \(W_{fb}\) that connect external signals to the reservoir.

---

## Step 1: Create the initializer file

Create `src/resdag/init/input_feedback/my_initializer.py`:

```python
"""
My Custom Weight Initializer
============================
Description and use cases.
"""

import torch

from resdag.init.input_feedback import (
    InputFeedbackInitializer,
    register_input_feedback,
)


@register_input_feedback("my_init")
class MyInitializer(InputFeedbackInitializer):
    """
    My custom initialization strategy.

    Parameters
    ----------
    input_scaling : float
        Global weight scaling factor.
    sparsity : float
        Fraction of weights set to zero.
    """

    def __init__(
        self,
        input_scaling: float = 1.0,
        sparsity: float = 0.0,
    ) -> None:
        self.input_scaling = input_scaling
        self.sparsity = sparsity

    def __call__(
        self,
        reservoir_size: int,
        input_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate the weight matrix.

        Parameters
        ----------
        reservoir_size : int
            Number of reservoir neurons (rows of weight matrix).
        input_size : int
            Input/feedback dimension (columns of weight matrix).
        dtype : torch.dtype
            Desired tensor dtype.
        device : torch.device
            Target device.

        Returns
        -------
        torch.Tensor
            Weight matrix of shape (reservoir_size, input_size).
        """
        W = torch.randn(reservoir_size, input_size, dtype=dtype, device=device)
        W = W * self.input_scaling

        if self.sparsity > 0:
            mask = torch.rand_like(W) < self.sparsity
            W[mask] = 0.0

        return W
```

!!! important "Required interface"
    - Must extend `InputFeedbackInitializer`
    - Must implement `__call__(reservoir_size, input_size, dtype, device) → Tensor`
    - Return shape must be `(reservoir_size, input_size)`

---

## Step 2: Import in `__init__.py`

Add to `src/resdag/init/input_feedback/__init__.py`:

```python
# ... existing imports ...
from . import my_initializer  # noqa: F401 — triggers registration
```

---

## Step 3: Use it

```python
from resdag.init.input_feedback import my_initializer  # trigger registration
from resdag.layers import ESNLayer

# By string
reservoir = ESNLayer(
    500, feedback_size=3,
    feedback_initializer="my_init",
)

# With custom params
reservoir = ESNLayer(
    500, feedback_size=3,
    feedback_initializer=("my_init", {"input_scaling": 0.5, "sparsity": 0.1}),
)

# As object
from resdag.init.input_feedback import get_input_feedback
init = get_input_feedback("my_init", input_scaling=0.8)
reservoir = ESNLayer(500, feedback_size=3, feedback_initializer=init)
```

---

## Example: Pseudo-Orthogonal Initialization

```python
@register_input_feedback("pseudo_orthogonal")
class PseudoOrthogonalInit(InputFeedbackInitializer):
    """
    Initialize weights with a pseudo-orthogonal structure.
    Generates columns that are approximately orthogonal.
    """

    def __init__(self, input_scaling: float = 1.0) -> None:
        self.input_scaling = input_scaling

    def __call__(
        self,
        reservoir_size: int,
        input_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        # Generate via QR decomposition of a random tall matrix
        if reservoir_size >= input_size:
            A = torch.randn(reservoir_size, input_size, dtype=dtype, device=device)
            Q, _ = torch.linalg.qr(A)
            W = Q[:reservoir_size, :input_size]
        else:
            A = torch.randn(input_size, reservoir_size, dtype=dtype, device=device)
            Q, _ = torch.linalg.qr(A)
            W = Q[:input_size, :reservoir_size].T

        return W * self.input_scaling
```
