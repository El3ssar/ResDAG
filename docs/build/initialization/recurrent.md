---
description: Choose recurrent weights by their memory, spectrum and cost, independently of graph connectivity.
---

# Choose recurrent weights

`recurrent_initializer=` builds the square matrix that carries information
between timesteps. It accepts graph builders, direct matrix builders, your own
functions, and registered names. `topology=` remains an equivalent spelling;
pass one or the other.

```python
import torch
from resdag import ESNLayer

reservoir = ESNLayer(
    128, feedback_size=3,
    recurrent_initializer=("signed_permutation", {"gain": 0.9}),
    feedback_initializer=("normal", {"scale": 0.2}),
    seed=7, dtype=torch.float64,
)
```

## Pick a family

| Family | Choose it when | Main control |
| --- | --- | --- |
| Graph connectivity | Connections have a meaningful structure | Existing [graph catalog](topologies/index.md) |
| Delay line / cycle | You want simple, interpretable memory | Connection weights |
| Haar orthogonal | You want random rotations with uniform singular values | `gain` |
| `signed_permutation` | You want a cheap, sparse orthogonal matrix | `gain` |
| `householder_orthogonal` | You want orthogonal mixing without a QR decomposition | Reflector count and `gain` |
| `singular_spectrum` | You want to control amplification in each direction | Prescribed singular values |
| `eigen_spectrum` | You want prescribed real modes and oscillatory pairs | Prescribed eigenvalues |
| `fast_spectral_initialization` | You want approximate radius initialization without eigenvalue computation | Native `spectral_radius`, explicit `rescale=False` |

A singular value controls how much one update can amplify a direction. An
eigenvalue describes a repeated linear mode. They answer different questions:
a non-normal matrix can transiently amplify inputs even with all eigenvalues
inside the unit circle. Neither statistic alone guarantees useful nonlinear
reservoir dynamics or accurate autonomous forecasts.

The structured builders return ordinary dense tensors. Their construction can
be cheaper than QR; dense recurrence still costs a matrix multiplication at
each timestep. `signed_permutation` does not automatically install a specialized
permutation kernel. On `ESNLayer`, `sparse=True` explicitly selects the existing
frozen CSR recurrence.

## Decide who controls scaling

Most historical builders are rescaled by the layer's `spectral_radius=` after
construction. Builders registered with `prescaled=True` own their scale; the
layer leaves their prescribed spectrum or `gain` intact and warns if its
radius would otherwise be ignored.

For an explicit policy, configure the initializer object:

```python
from resdag.init import get_recurrent_initializer

native = get_recurrent_initializer(
    "fast_spectral_initialization", spectral_radius=0.8, rescale=False, seed=7,
)
reservoir = ESNLayer(128, feedback_size=3, recurrent_initializer=native, seed=7)

# Explicitly permit an outer rescale, including a prescaled builder.
scaled = get_recurrent_initializer("orthogonal", gain=1.0, rescale=True, seed=7)
reservoir = ESNLayer(
    128, feedback_size=3, recurrent_initializer=scaled, spectral_radius=0.8,
)
```

`rescale=False` avoids the outer eigenvalue calculation. FSI's radius is an
asymptotic random-matrix approximation, not an exact finite-size guarantee.
`rescale=True` applies the layer's radius when supplied; it changes the absolute
values of a prescribed spectrum. Leaving the policy as `None` keeps historical
behavior. This explicit object form also avoids reserving a new keyword in
arbitrary custom builder tuples.

## Precision and placement

Pass `dtype=` and `device=` when constructing a reservoir to initialize its
parameters in the requested precision and location. Direct matrix builders
accept these arguments too. A builder used through the registry receives the
target tensor's dtype/device when it supports them. Explicit values bound in
a spec take precedence.

```python
from resdag.init.matrices import orthogonal_matrix

matrix = orthogonal_matrix(64, seed=4, dtype=torch.float64, device="cpu")
```

Graph construction and third-party callables may still run on the CPU or in
their own precision, then copy to the target. Converting an already initialized
float32 layer with `.double()` does not recover the precision lost during its
original draws. Reproducibility is per device/backend; do not assume CPU and
GPU random draws are identical.

Continue with [custom registration](custom.md), the
[complete builder catalog](topologies/index.md), or
[training-data calibration](../../workflows/calibration.md).

## Spectral measurement and initialization cost

Automatic scaling measures dense matrices with `torch.linalg.eigvals`. For
large matrices with at most 10% nonzero entries it uses SciPy's sparse Arnoldi
solver on CPU, with a dense fallback if that solve fails. Arnoldi is a numerical
approximation; requesting several leading eigenvalues reduces sensitivity to
clustered spectra. MPS measurements also use CPU. Dense eigendecomposition can
make construction expensive for a large reservoir; it does not run at every
forecast step.

A power estimate that stops changing can still have missed the dominant mode.
It is therefore insufficient to validate the radius used for scaling. When
construction cost matters, prefer a builder whose spectrum is known by design
and explicitly keep its native scale with `prescaled=True` or `rescale=False`.
Benchmark both construction time and subsequent forecasting for your workload.

## Migration: corrected graph weights and degree

Two graph builders now follow their stated mathematical definitions. Reservoirs
using them can produce different weights and forecasts with the same old seed;
retrain readouts and reevaluate previously selected hyperparameters.

- `regular` now connects each node to exactly `k` distinct neighbours, excluding
  optional self-loops. Previously odd `k` silently produced `k - 1` neighbours.
  Odd `k` uses an antipodal matching, so `n` must be even. Valid configurations
  satisfy `0 <= k < n` and `n * k` even. Directed mode has `k` incoming and `k`
  outgoing neighbours. To preserve the old odd-degree connectivity, request
  `k - 1`; matching an old random realization exactly requires saved weights.
- `simple_cycle_jumps` now adds the cycle and jump contributions on shared edges.
  With `jump_length=1`, forward edges have weight `r_c + r_l` and reverse edges
  have weight `r_l`. Previously jumps overwrote the cycle weight, so changing
  `r_c` could have no effect. Each bidirectional jump pair is counted once.
  Saved weights are the reliable way to retain an existing realization.
- `spectral_cascade(..., spectral_radius=0)` now produces zero clique weights;
  it previously retained nonzero weights.

```python
# True degree three, including one opposite neighbour on an even ring.
reservoir = ESNLayer(
    128, feedback_size=3,
    recurrent_initializer=("regular", {"k": 3}), seed=7,
)

# Cycle and jump contributions are both preserved where their edges overlap.
reservoir = ESNLayer(
    128, feedback_size=3, spectral_radius=None,
    recurrent_initializer=("simple_cycle_jumps", {
        "jump_length": 1, "r_c": 0.6, "r_l": 0.1,
    }), seed=7,
)
```
