# Input & Feedback Initializers

Initializers control how the **input weight matrix** \(W_{in}\) and **feedback weight matrix** \(W_{fb}\) are generated. Different strategies create different coupling patterns between the input/output signals and the reservoir.

---

## Specifying Initializers

Same three-way format as [topologies](topologies.md):

=== "String"
    ```python
    reservoir = ESNLayer(
        500, feedback_size=3,
        feedback_initializer="chebyshev",
    )
    ```

=== "Tuple"
    ```python
    reservoir = ESNLayer(
        500, feedback_size=3,
        feedback_initializer=("random", {"input_scaling": 0.5}),
        input_initializer=("pseudo_diagonal", {"input_scaling": 1.0}),
    )
    ```

=== "Object"
    ```python
    from resdag.init.input_feedback import get_input_feedback

    fb_init = get_input_feedback("chebyshev", input_scaling=0.8)
    in_init = get_input_feedback("random", input_scaling=0.5)

    reservoir = ESNLayer(
        500, feedback_size=3, input_size=5,
        feedback_initializer=fb_init,
        input_initializer=in_init,
    )
    ```

When `None` (the default), a uniform random initialization is used.

---

## Registry API

```python
from resdag.init.input_feedback import get_input_feedback, show_input_initializers

# List all available initializers
show_input_initializers()

# Get details for a specific one
show_input_initializers("chebyshev")

# Create an initializer
init = get_input_feedback("random", input_scaling=0.5)
```

---

## Available Initializers

<div class="rd-topology-grid">
<div class="rd-topology-item">random</div>
<div class="rd-topology-item">random_binary</div>
<div class="rd-topology-item">binary_balanced</div>
<div class="rd-topology-item">chebyshev</div>
<div class="rd-topology-item">pseudo_diagonal</div>
<div class="rd-topology-item">chessboard</div>
<div class="rd-topology-item">ring_window</div>
<div class="rd-topology-item">opposite_anchors</div>
<div class="rd-topology-item">chain_of_neurons_input</div>
<div class="rd-topology-item">dendrocycle_input</div>
<div class="rd-topology-item">zero</div>
</div>

---

## Initializer Reference

### `random`
Standard random initialization from a uniform or normal distribution, scaled by `input_scaling`.

```python
init = get_input_feedback("random", input_scaling=1.0)
```

| Param | Default | Description |
|---|---|---|
| `input_scaling` | `1.0` | Weight scale factor |

**Best for**: General-purpose baseline.

---

### `random_binary`
Weights are randomly ±1 (or ±`input_scaling`), sparse with density `density`.

```python
init = get_input_feedback("random_binary", input_scaling=1.0, density=0.1)
```

---

### `binary_balanced`
Like `random_binary` but ensures exact balance between positive and negative weights per reservoir unit.

```python
init = get_input_feedback("binary_balanced", input_scaling=1.0)
```

**Best for**: When balanced excitatory/inhibitory inputs are desired.

---

### `chebyshev`
Weights derived from Chebyshev polynomial spacing. Creates a non-uniform, mathematically structured input projection.

```python
init = get_input_feedback("chebyshev", input_scaling=1.0)
```

**Best for**: Structured input coupling; sometimes improves performance on smooth signals.

---

### `pseudo_diagonal`
Near-diagonal weight matrix — each input feature connects primarily to a small subset of reservoir neurons, with structured coupling.

```python
init = get_input_feedback("pseudo_diagonal", input_scaling=1.0)
```

**Best for**: When input features should map to separate "regions" of the reservoir.

---

### `chessboard`
Alternating ±1 pattern in a grid layout.

```python
init = get_input_feedback("chessboard", input_scaling=1.0)
```

---

### `ring_window`
Each input dimension connects to a contiguous window of reservoir neurons arranged in a ring.

```python
init = get_input_feedback("ring_window", input_scaling=1.0)
```

**Best for**: Spatially organized input coupling; works well with ring or cycle topologies.

---

### `opposite_anchors`
Each input connects to two "anchor" neurons on opposite sides of the reservoir, with decay away from them.

```python
init = get_input_feedback("opposite_anchors", input_scaling=1.0)
```

---

### `chain_of_neurons_input`
Chain-structured input: each input dimension drives a linear chain of reservoir neurons.

```python
init = get_input_feedback("chain_of_neurons_input", input_scaling=1.0)
```

**Best for**: Pairing with dendrocycle or chain-like reservoir topologies.

---

### `dendrocycle_input`
Dendritic input initialization that complements the `dendrocycle` topology.

```python
init = get_input_feedback("dendrocycle_input", input_scaling=1.0)
```

---

### `zero`
All-zero weight matrix. Removes input coupling completely.

```python
init = get_input_feedback("zero")
```

**Best for**: Ablation studies; feedback-only models where input coupling should be zero.

---

## Input Scaling

The `input_scaling` parameter is a global multiplier on the weight matrix. It controls how strongly the input signal drives the reservoir relative to the internal dynamics:

- **Low scaling** (< 0.5): reservoir dynamics dominated by recurrent connections
- **Medium scaling** (0.5–2.0): balanced input and recurrent drive
- **High scaling** (> 2.0): reservoir closely tracks input signal

```python
# Common pattern: tune input_scaling separately for feedback vs driver
reservoir = ESNLayer(
    500,
    feedback_size=3,
    input_size=5,
    feedback_initializer=("random", {"input_scaling": 1.0}),
    input_initializer=("random", {"input_scaling": 0.3}),  # lighter driver coupling
    spectral_radius=0.9,
)
```

---

## Adding Custom Initializers

See [Extending resdag → New Initializer](../extending/new-initializer.md) for a step-by-step guide.

```python
from resdag.init.input_feedback import register_input_feedback, InputFeedbackInitializer
import torch

@register_input_feedback("my_init")
class MyInitializer(InputFeedbackInitializer):
    def __init__(self, input_scaling: float = 1.0):
        self.input_scaling = input_scaling

    def __call__(
        self,
        reservoir_size: int,
        input_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        # Return (reservoir_size, input_size) weight matrix
        W = torch.randn(reservoir_size, input_size, dtype=dtype, device=device)
        return W * self.input_scaling
```
