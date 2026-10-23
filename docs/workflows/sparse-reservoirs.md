---
description: Compact recurrent weights, low-memory readouts and sparse-native custom initialization.
---

# Keep large models within memory

Three allocations grow for different reasons: recurrent weights with reservoir
size, collected features with sequence length, and solver workspaces with the
number of features. Choose a remedy for the allocation that limits your model.

## Compressed reservoir weights

For supported frozen ESN reservoirs, `sparse=True` keeps only compressed
recurrent weights. It no longer retains a full square matrix alongside them.
The same fit, forecast and save commands work:

```python
import torch
import resdag as rd

t = torch.arange(400, dtype=torch.float64) * 0.05
training = torch.stack((t.sin(), t.cos()), dim=-1)
model = rd.ESN(
    64, input_size=2, washout=30, seed=7, dtype=torch.float64,
    topology=("erdos_renyi", {"p": 6 / 64}), sparse=True,
    readout="lsmr", alpha=1e-3,
    readout_kwargs={"max_iter": 1000, "chunk_size": 128},
).fit(training)
prediction = model.forecast(horizon=40)
assert prediction.shape == (40, 2)

cell = next(layer.cell for layer in model.graph.modules()
            if isinstance(layer, rd.ESNLayer))
assert cell.weight_hh.layout == torch.sparse_csr
model.save("compact.pt")
restored = rd.load("compact.pt")
expected = model.forecast(history=training[-60:], horizon=40)
actual = restored.forecast(history=training[-60:], horizon=40)
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

The `weight_hh` attribute is a CSR Tensor view over ordinary persistent index
and value buffers. It is no longer a dense Parameter for a sparse reservoir.
`cell.weight_hh.to_dense()` explicitly makes an inspection copy. Change existing
connection values with `cell.weight_hh.values()` under `torch.no_grad()`.
The view shares those values; there is no stale operator cache to refresh.

The readout remains a normal Torch layer. Gradients can pass through the frozen
reservoir to earlier layers. Setting `cell.trainable = True` explicitly expands
the recurrent weights into a dense Parameter and warns about that allocation;
create an optimizer after changing the set of trainable parameters. Freezing
again restores the original sparse request.
`model.requires_grad_(True)` affects registered Parameters; compressed recurrent
buffers remain fixed. Set `cell.trainable = True` first when the recurrent
connections themselves must become trainable.

CSR is available on CPU and supported CUDA devices. MPS uses dense weights;
moving a model to MPS can therefore increase its memory requirement. Returning
to CPU restores an explicit `sparse=True` request. No CUDA throughput was
measured for this work.

## Compact does not always mean faster

Dense remains the default. `sparse="auto"` currently uses size/density heuristics,
not a hardware calibration. Dense multiplication beat CSR on the measured
Apple Silicon CPU workloads at 400–2,000 units; other hardware and larger sizes
have different crossovers. Choose explicitly and measure your actual batch,
precision and reservoir size. Sparse storage can still be valuable when memory
is the binding constraint.

Initialization also matters. The built-in Erdős–Rényi sparse route uses bounded
random blocks and produces the same unscaled edges and signs as its dense
definition. Edge-native graph builders avoid a dense adjacency as well. Some
other graph functions allocate dense NumPy arrays internally, so this is not a
universal constant-memory initialization claim.

Sparse spectral scaling uses SciPy Arnoldi or an analytic triangular spectrum.
Only matrices of size one or two use a bounded dense fallback. An unsuccessful
large sparse estimate raises instead of allocating a hidden dense fallback.
Its scaling may differ in the last digits from dense eigvals, especially when
initializing in float32. New sparse runs can therefore differ from older seeded
experiments. Dense initialization and execution keep their existing path.

## Register a sparse-native initializer

Ordinary registered matrix builders can return a Torch CSR Tensor or a SciPy
sparse matrix. No full dense destination is allocated for these build-style
callables. Placement and dtype are passed when declared in the signature:

```python
@rd.init.register_recurrent_initializer("my_sparse_cycle", prescaled=True, weight=0.8)
def sparse_cycle(n, weight=0.8, *, dtype=None, device=None):
    rows = torch.arange(n + 1, dtype=torch.int64, device=device)
    columns = (torch.arange(n, dtype=torch.int64, device=device) + 1) % n
    values = torch.full((n,), weight, dtype=dtype, device=device)
    return torch.sparse_csr_tensor(rows, columns, values, size=(n, n))

layer = rd.ESNLayer(
    64, feedback_size=2, recurrent_initializer="my_sparse_cycle",
    spectral_radius=None, sparse=True, dtype=torch.float64,
)
assert layer.weight_hh._nnz() == 64
```

Existing in-place builders remain supported through a temporary dense scratch
matrix. A custom `TopologyInitializer` subclass can override
`build_sparse(n, *, dtype, device, spectral_radius)` to construct CSR directly.
Its normal `initialize(...)` method still supports dense models. Returned
storage is copied into model-owned components; later edits to the initializer's
own matrix cannot change an already built model.

## Readout and sequence memory

The example opts into [LSMR](../reference/lsmr-readout.md), which avoids a full
feature-by-feature Gram matrix. It solves on CPU and reports convergence;
automatic readout selection is unchanged. This is a separate choice from
compressed recurrence.

Batch fitting still collects feature sequences. Reducing a dense `N × N`
recurrent allocation does not remove a `time × features` history. For long
sequences, use the [streaming workflow](streaming.md) with a compatible
incremental readout. Explicit-state DAG evaluation releases intermediate
feature tensors after their last consumer, while keeping requested outputs,
final states and tensors needed by autograd alive.

Tuning's optional memory-budget planner remains conservative: its default
estimate reserves dense preset weights and general direct-solver workspaces,
even when a candidate selects sparse storage or LSMR. It can therefore reject
a model that would actually fit. Supply a workload-specific `memory_estimator`
when you need tighter admission; see [tuning resources](../advanced/tuning-resources.md).
This estimate does not change the model's actual storage or solver.
