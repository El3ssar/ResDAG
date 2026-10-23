# Single-pass training benchmark

At a fixed CPU workload, fitting eight cascaded reservoir/readout stages took
236.195 ms with the staged trainer and 30.915 ms with the single-pass trainer.
The median paired speedup was **7.64×**. A single stage took essentially the same
time. Fitted coefficients and final recurrent states were bit-identical in every
comparison.

These are measurements from one Apple Silicon machine. The
[reproduction script](https://github.com/El3ssar/ResDAG/blob/main/benchmarks/single_pass_training.py)
and [raw measurements](https://github.com/El3ssar/ResDAG/blob/main/benchmarks/results/single-pass-training-2026-10-23.json)
record the workload, every timed repeat, structural counts and process memory.
They introduce no performance thresholds.

## Workload and method

The baseline is the independently imported trainer from Git revision
`c0c61d09a22759f2d5c3f24c78d53803cd658778`. Both trainers use the same reservoir
and readout implementations. Each pair starts from fresh copies of identical
model weights and uses the same data and targets.

- CPU, `float64`, batch size 2, reservoir width 64, input/output width 4.
- Warmup length 100 and supervised training length 500 at depths 1, 2, 4 and 8.
- Reservoir spectral radius 0.7, leak rate 0.8, zero reservoir noise; ridge
  regularization 0.001.
- macOS 27.0.1 on Apple Silicon, Python 3.14.8, PyTorch 2.10.0.
- One intra-op thread and one inter-op thread.
- Two discarded warmup pairs followed by seven measured pairs. Baseline/current
  execution order alternates. Model copying and data preparation are excluded
  from fit timing; lightweight instrumentation is included.

## Fit time

| Depth | Staged median ms (min–max) | Single-pass median ms (min–max) | Median paired speedup (min–max) |
| ---: | ---: | ---: | ---: |
| 1 | 4.018 (3.968–4.229) | 4.009 (3.970–4.119) | 0.990× (0.976–1.059) |
| 2 | 15.108 (14.960–15.461) | 7.881 (7.687–8.009) | 1.943× (1.868–1.968) |
| 4 | 60.081 (59.363–60.445) | 15.732 (15.419–16.554) | 3.841× (3.626–3.877) |
| 8 | 236.195 (235.554–236.796) | 30.915 (30.402–31.131) | 7.641× (7.566–7.769) |

The depth-one paired ratio differs from 1 by about 1%, within the observed
variation. Paired speedups are computed per repeat, so their median can differ
from the ratio of the two separate time medians.

At depth eight, the component medians were:

| Component | Staged ms | Single-pass ms |
| --- | ---: | ---: |
| Reservoir propagation | 231.699 | 28.827 |
| Readout fitting | 1.879 | 1.719 |
| Other fit overhead | 2.630 | 0.412 |
| Complete fit | 236.195 | 30.915 |

Propagation sums reservoir forward durations. Readout fitting includes feature
validation, the algebraic solve and coefficient installation. Other overhead is
the per-repeat residual. Component medians need not sum to the total median.

## Computation counts

Each readout fits once in both implementations. The difference is repeated
propagation: staged fitting reruns the whole graph after every recurrent
readout dependency. Single-pass fitting propagates each reservoir through the
complete warmup and training prefix once.

| Depth | Reservoir calls: staged → single-pass | Processed batch samples: staged → single-pass |
| ---: | ---: | ---: |
| 1 | 2 → 1 | 1,200 → 1,200 |
| 2 | 8 → 2 | 4,800 → 2,400 |
| 4 | 32 → 4 | 19,200 → 4,800 |
| 8 | 128 → 8 | 76,800 → 9,600 |

A processed batch sample is one trajectory at one timestep. At depth eight,
the new traversal performs eight times less recurrent sample work. Shared
branches reuse computed features, and intermediate values are released after
their last consumer.

## Process memory

Memory was measured in a fresh process for each depth/mode, using the process
high-water mark from `resource.getrusage`. The figures include imports, model
construction, data and allocator state. Each cell contains one fit, and small
process-to-process differences are visible before fitting.

| Depth | Staged absolute peak MiB | Single-pass absolute peak MiB |
| ---: | ---: | ---: |
| 1 | 226.516 | 227.234 |
| 2 | 225.609 | 227.469 |
| 4 | 228.531 | 226.031 |
| 8 | 230.734 | 227.641 |

At depth eight, the pre-fit high-water marks were 223.828 and 223.813 MiB,
respectively. The measured increases were 6.906 and 3.828 MiB. These are process
measurements for this workload; allocator behavior and different graph shapes
can change the result.

## Reproduce

From the repository root:

```bash
uv run python benchmarks/single_pass_training.py \
  --baseline-ref c0c61d0 --rss \
  --output benchmarks/results/single-pass-training-rerun.json
```

Omit `--rss` to collect timing and parity checks alone. `--baseline-ref` reads
the historical trainer with `git show` into a temporary import; the working
checkout stays in place. Use `--help` for repeat and depth controls.

The recorded working-trainer SHA-256 before and after the measured run is
`bb7e2ca68416f3ed5bca3817b74d102930beb19442e8dac6ec097dfb06bb3d7d`.
Those matching hashes establish that the working source stayed unchanged
during measurement. The baseline has the separate Git identity recorded above.
