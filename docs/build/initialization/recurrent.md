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
