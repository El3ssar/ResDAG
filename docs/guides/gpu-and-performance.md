# GPU & performance tips

## Device placement

ResDAG follows PyTorch — move model and tensors to the same device:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ott_esn(1000, feedback_size=3, output_size=3).to(device)
warmup = warmup.to(device)
```

## Batch size

Reservoir forward is `(batch, time, features)`. Larger batch improves GPU utilization
but multiplies state memory linearly.

## Readout `float64`

`CGReadoutLayer` solves ridge in `float64` by default (`use_float64=True`) for stability.
For very large reservoirs ($N > 5000$), set `use_float64=False` if memory is the bottleneck.

## `torch.compile`

Works on `ESNModel` like any `nn.Module` — test on your PyTorch version; dynamic sequence
lengths may reduce gains.

```python
model = torch.compile(model)
```

## HPO parallelism

`run_hpo(..., n_workers=4)` forks processes. The runner throttles BLAS/OpenMP threads
per worker to avoid CPU oversubscription. Use `storage="study.log"` for multi-worker studies.

## Stateful inference

Avoid calling `forward` from multiple threads on the same model — reservoir `state` is
not thread-safe.

## See also

- [Installation](../getting-started/installation.md)
- [`run_hpo` runners](../reference/hpo/internals.md)
