# Fixed-weight CPU workflow protocol

`cpu_workflows.py` compares two ResDAG implementations on synthetic data and
identical numerical weights. It measures implementation costs and checks exact
numerical agreement. It does not search for better models or change timing gates.

Run from the repository with its development and benchmark dependencies installed:

The following command reproduces the historical `dc695a6` candidate comparison.
Its timestep-view and driver-slicing rewrites were rejected after inconclusive
measurements. The adopted changes retain AutoReadout residual reuse and analytic
CSR correctness fixes; the original sequence-loop and driver slicing are kept.
For a new adopted-runtime measurement, supply its exact git ref or `WORKTREE`.

```sh
uv run --no-sync python benchmarks/cpu_workflows.py \
  --before 18f78c83 --after dc695a6 \
  --reservoir-forward \
  --artifacts-dir /tmp/resdag-cpu-18f78c83-dc695a6
```

Use a new artifact directory for every run. Existing directories and report files
are rejected. `--output` optionally selects a separate, new JSON report file;
otherwise the report is `ARTIFACTS/results.json`. Interrupted runs retain their
source snapshots, input artifacts and completed process outputs for inspection.

## Default matrix

All rows use CPU float64, three input/output features, leak 0.8, requested spectral radius
0.7, a ridge readout with penalty `1e-3`, standard public preprocessing and no
state noise. Classic models concatenate inputs with reservoir states; OTT also
augments the states. Sparse rows use the classic architecture.
Dense classic/OTT rows use fully connected uniform recurrent weights; ER rows
use signed graph weights. These are different recipes from the Gaussian,
state-only equal-time comparison, so their raw timings are not interchangeable.

| Workflow | Units | Batch | Input steps | Washout | History | Forecast |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense classic | 400 | 1 | 1,501 | 100 | 100 | 257 |
| Dense OTT | 400 | 1 | 1,501 | 100 | 100 | 257 |
| Sparse simple cycle | 400 | 1 | 1,501 | 100 | 100 | 257 |
| Sparse Erdős–Rényi, expected degree 6 | 400 | 1 | 1,501 | 100 | 100 | 257 |
| Dense classic | 1,000 | 1 / 4 / 16 | 501 | 100 | 100 | 129 |

Input steps include washout. Each artifact contains one additional observation
for implicit next-sample targets. The three scaling rows share the same weight
artifact and differ in batch size, with distinct synthetic batch phases.

`--only` selects named rows. `--train-steps`, `--horizon`, `--washout` and
`--history-steps` change the corresponding counts; `--scaling-train-steps` and
`--scaling-horizon` control the 1,000-unit rows separately. `--profile smoke`
uses 32/64 units, shorter sequences and one block with two repetitions. Every
resolved setting is recorded explicitly; smoke results are not full-size results.

`--profile storage` is an optional six-row diagnostic: Erdős–Rényi reservoirs
with expected degree 6 at 400/1,000/2,000 units, each represented as dense and CSR.
All use batch 1, 501 input steps, washout/history 100 and horizon 129; the scaling
length options also configure this profile. Names such as `er-1000-b1-dense`
and `er-1000-b1-csr` identify the representations. Each width shares one baseline
dense weight artifact. This does not alter automatic operator selection.

## What the clocks include

- **Fit:** `rd.Model.fit(observations)`, including transactional graph staging,
  learned preprocessing, washout, feature generation and the readout solve.
- **History plus forecast:** `rd.Model.forecast(history=..., horizon=...,
  compile=False)`, including original-unit transforms, reset/history processing,
  generated steps and returned prediction assembly.
- **Optional reservoir forward:** public reservoir `reset_state()` and forward
  over the complete input prefix. This uses raw synthetic inputs and omits the
  public workflow's preprocessing, readout and forecast. It is a diagnostic.

Imports, graph construction, baseline-weight loading, input loading, outer model
deepcopy, hashing and thread-policy verification are outside these clocks. Each
fit starts from a fresh copy of the same unfitted template. Public fit's own
internal staging remains timed. No profiling hooks are installed.

## Scheduling and native threads

The supervisor runs serialized fresh processes in **before / after / after /
before** order, repeated for two blocks by default. Each process executes every
selected row with one discarded warmup and seven timed repetitions. This yields
four processes and 28 timed repetitions per revision per row. `--blocks`,
`--repeats` and `--warmups` override those counts; `--timeout` bounds each process.
Together, the default seven-row matrix and six-row storage profile contain
**728 timed executions of each operation**, plus **104 discarded warmups of each
operation** when the optional reservoir forward is enabled in both runs.

The existing benchmark thread policy restarts the CLI before numerical imports,
overriding inherited vendor limits. It requests and verifies one Torch intra-op
and inter-op thread, native BLAS/OpenMP limits, and Accelerate's calling-thread
policy where applicable. Checks surround operations and abort on violations.
Recorded native API settings do not measure operating-system thread occupancy.

## Artifacts and agreement checks

Git refs are archived independently. `--after WORKTREE` copies current saved
`src/resdag` source, including uncommitted source changes. File manifests and
import origins identify the actual implementations; no mixed-revision imports
are accepted. Frozen runner/thread-policy sources and dependency-file hashes are
also retained. Run both revisions in the same installed dependency environment.

Preparation generates baseline dense weight checkpoints and synthetic sine/cosine
observations once. Both variants explicitly load those artifacts. Sparse storage
is compared through canonical numerical recurrent weights, preserving the actual
operator rather than assuming identical seeds imply identical initialization.
Only reference preparation applies spectral normalization. Measurement workers
construct placeholders with `spectral_radius=None`, then strictly load the saved
weights and verify their logical equality before any timed operation. This avoids
irrelevant eigensolver work or failures on weights that would be discarded.
For initial logical recurrent-weight equality only, signed zeros are normalized
to positive zero: dense ER masks can retain `-0.0`, whereas CSR drops structural
zeros. Saved artifact bytes retain their original hash. Fitted tensor weights,
general tensors, complete predictions and final states keep exact zero signs.
An initial mismatch retains both complete fingerprints in a failure receipt.

Every repetition, including discarded warmups, hashes fitted parameters/buffers,
learned scaling, canonical checkpoint metadata, complete predictions and final
states. Optional forward outputs/states are also checked. Predictions must have
the complete requested shape and remain finite. Disagreement produces a report
and a nonzero exit status. All captured warnings and process stdout/stderr remain
available, alongside raw timings and exact provenance.

The storage profile still requires exact before/after agreement **within each
case**. Dense-versus-CSR arithmetic may differ. Separate `storage_diagnostics`
report logical initial-weight equality, cross-storage result-hash agreement and
the maximum absolute difference of complete forecasts from the last repetition.
Those cross-storage forecast differences do not fail the revision comparison.
Storage pairs run in their declared order; their timings are exploratory rather
than an independently balanced dense/CSR crossover study.

## Reading the statistics

The report retains each process's median, then pairs the nearest AB and BA process
medians. Ratios are **before time / after time**: above one means the new version
is faster. It reports every paired ratio, their median and spread, and each
block's median ratio. Repetitions within one process are not independent process
observations; the descriptive spread is not a confidence interval.
The historical candidate run experienced a large mid-run latency increase across
both revisions despite unchanged observed dependencies, imports and native thread
settings. All runs remain included; its median ratios do not establish a small
whole-workflow improvement or regression. They are rejected-candidate evidence.

The published within-process loop diagnostic is separate from this harness:
it swaps only the original/candidate `forward_stateless` functions in one shared
runtime, using ABBA rounds within two processes. Its repeated-pair spread remains
broad. It does not compare isolated revisions, test the driver rewrite separately
or establish the throughput of the final adopted implementation. Raw samples are
published in `docs/assets/data/phase-five-cpu-interleaved-2026-10-23.json`.
The separate exploratory compile probe also used a then-live candidate checkout
before slicing was restored, without an isolated overall manifest. Its cold/warm
times are historical observations rather than final adopted-runtime speed evidence.

Batch scaling trades latency for throughput. Compare total operation latency and
`batch × input_steps / fit_seconds`, or `batch × horizon / history_forecast_seconds`.
The forecast rate includes its history cost. Use the same row lengths when
comparing batches; the shorter scaling rows cannot establish a width-only effect.

Use synthetic/development measurements to guide implementation choices. Saved
accuracy models and the completed final test bank remain regression evidence,
without retuning. This protocol makes no compiler, cold-start, process-memory,
cross-library superiority or universal speed claim.
