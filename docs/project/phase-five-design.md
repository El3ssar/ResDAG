---
description: Accuracy preservation, held-out forecasting and fair speed–accuracy measurements for phase five.
---

# Phase five: measure before optimizing

The goal is fast, accurate autonomous forecasts with the same customizable
model interface. A faster fit does not establish an accurate forecast. Keep
training speed, forecasting speed, numerical agreement and forecast skill as
separate measurements.

!!! note "Current measurements"
    The first-increment results below precede the correction to independent
    initialization streams. The [continuation report](phase-five-results.md)
    records fresh tuning on the corrected streams, saved-model parity and
    measured execution costs. Newly seeded models can differ from the
    historical results on this page.

## Agreed accuracy target

The maintainer approved these choices on 23 October 2026:

- Aim for a **median 7–8 Lyapunov times** over several reservoir seeds and
  previously unseen forecast starting points. Publish every observation and
  its spread; an unusually successful forecast does not establish success.
- Count the uninterrupted prefix whose per-step normalized RMS error stays
  strictly below **0.3**. Also publish the result at **0.5**.
- Estimate each component's variation from training observations only. Do
  not change the denominator when extending the forecast horizon.
- Compute a horizon for each seed and starting point, then aggregate those
  horizons. Taking the first crossing of a median error curve can conceal
  failures in individual forecasts.

For a forecast `prediction[t, j]`, reference `truth[t, j]` and fixed training
standard deviations `scale[j]`, the per-step error is
`sqrt(mean_j(((prediction - truth) / scale)**2))`.
The first invalid or nonfinite step ends the valid prefix. Convert its length
to Lyapunov times with `steps * actual_dt * lambda_max`. Record the system
parameters, integration method, time step and exponent source. If the whole
evaluated horizon remains valid, report that result as a lower bound.

Seven or eight Lyapunov times is a target for the experiment program. The
library does not guarantee that score for an arbitrary model or dataset.

## First increment: preparation and regression evidence

Repair the existing comparison harness before publishing new rankings:

1. Generate one raw trajectory shared by the adapters. Fit scaling on the
   admitted training observations for each workload, never the entire series.
2. Use inputs at indices `0..T-1` and next-sample targets at `1..T`. Synchronize
   forecasting on observed history ending at `T`; the first unseen target is
   at `T+1`. Score only future observations.
3. Check the first emitted sample with an independent time-index oracle.
   Every adapter must emit the same physical time, including its first step.
4. Pass all declared hyperparameters through the adapters and honor requested
   timing repeats. Keep unavailable optional libraries as explicit skips;
   an execution failure in an available adapter must make the runner fail.
5. Freeze classical and state-augmented model recipes. Compare the phase-three
   and phase-four implementations using the same raw data, initial weights,
   scaling, precision, solver, seeds, histories and forecast horizons.

The existing Lorenz golden tests cover short prefixes and limited quality
floors. They remain useful regression tests, but do not prove the 7–8 target
or Rössler performance. Do not regenerate fixtures to make a loss disappear.

The public `valid_prediction_time` currently normalizes using the supplied
truth window and reduces batch errors before taking the first crossing.
Keep that documented behavior available. The phase-five experiment protocol
uses fixed training scales and individual horizons explicitly; changing the
public metric requires a separate API decision.

### Frozen model regression results

On 23 October 2026, source-isolated runs compared phase three (`00ca741`) with
merged phase four (`0b546d6`). All **36 complete forecasts** and fitted tensor
fingerprints matched. These are deterministic float64 CPU workloads, with no
stochastic training noise. They do not establish unchanged results for every
architecture or an earlier major version.

| System / model | Seeds × origins | Median at 0.3 | Range at 0.3 | Median at 0.5 |
| --- | --- | --- | --- | --- |
| Lorenz / vanilla | 3 × 3 | 3.659 | 0.561–5.379 | 3.713 |
| Lorenz / OTT | 3 × 3 | 3.151 | 0.561–5.252 | 3.622 |
| Rössler / vanilla | 6 × 3 | 5.855 | 3.396–8.376 | 5.891 |

Horizons are in Lyapunov times; none reached the ten-Lyapunov-time scoring
limit. The starting points come from one held-out portion of each trajectory;
some Rössler forecast windows overlap. The spread is descriptive, not a
confidence interval for independent trajectories. These frozen recipes
establish an implementation baseline, **not the best achievable accuracy**.

Reproduce the matched comparison from the repository with the HPO and dynamics
extras installed on Python 3.12 or newer:

```sh
python benchmarks/accuracy_regression.py --before 00ca741 --after 0b546d6 --artifacts-dir /tmp/resdag-frozen-accuracy --output frozen-accuracy.json
```

Use a new artifact directory for each run. The
[committed measurements](https://github.com/El3ssar/ResDAG/blob/main/benchmarks/results/phase-five-accuracy-2026-10-23.json)
record data hashes, recipes, per-seed/origin scores, source import paths and
artifact hashes. Fit/forecast wall times there are exploratory observations,
not paired timing evidence.

## Then: improve models without tuning on the test set

Use chronological training, validation and final-test data. Select time step,
reservoir size, initialization, leak, input/bias scales and readout settings
with validation data only. Freeze the selected recipe before scoring the final
test trajectories. Use the existing Optuna workflow, its seed/origin controls
and its complete model artifacts rather than building a second tuning engine.

Distinguish these two questions:

- **Accuracy regression:** did the same model and data lose skill after an
  implementation change? Match weights and settings; compare coefficients,
  states, full forecasts and per-origin horizons.
- **Best practical result:** how good can each library get with the same
  search budget? Let each use its ordinary APIs, record actual time and full
  configurations, and evaluate the selected recipes on untouched data.

For implementation timing, use paired/interleaved runs on fixed workloads,
record thread counts and synchronize accelerators. Report construction and
compilation separately from fit and forecast. A setup cost must not disappear
inside an untimed preparation step. Memory results must identify what was
measured and whether inputs, solver workspaces and compilation are included.

## Historical first tuning measurements

Four bounded searches used the existing `Model.tune` workflow and Optuna TPE:
vanilla and OTT models on both systems. Each attempted **24 configurations**,
using three initialization seeds and three validation starting points. Pruning
was explicitly disabled. The successful/failed counts were **19/5**, **24/0**,
**15/9** and **12/12**, respectively; invalid candidates were never eligible to
win. Every successful candidate completed all nine full validation forecasts.

The search varied width (300, 400, 600 or 1,000), spectral radius, leak,
input/bias scale, ridge penalty and recurrent degree. All candidates used tanh,
float64 CPU execution and an SVD readout. Lorenz used 10,001 training
observations at `dt=0.02`; Rössler used 8,501 at `dt=0.145`. Both discarded
500 training inputs as washout. Validation and final-test periods each covered
three nonoverlapping windows of at least ten Lyapunov times.

Winning settings were frozen before evaluating the reserved seeds 201–205.
The first engineering smoke had used seed 101; its separate 101–105 bank is
retained as a supplementary check. No test scores selected or changed the
winning settings. This experiment uses explicit validation, so fits retain
only the original training observations. Histories can include later observed
values, but forecasts never receive future truth after their starting point.

### Test results at the 0.3 threshold

All numbers below are Lyapunov times. Each primary median uses five reserved
seeds and three starting points. `≥` means the evaluated horizon ended before
the forecast failed; it is a lower bound.

| System / model | Untuned median | Tuned primary median | Primary range | Valid through the full window | Pooled ten-seed median |
| --- | --- | --- | --- | --- | --- |
| Lorenz / vanilla | 5.252 | 9.110 | 3.351–≥10.016 | 7 / 15 | ≥10.016 |
| Lorenz / OTT | 5.271 | ≥10.016 | 8.440–≥10.016 | 8 / 15 | ≥10.016 |
| Rössler / vanilla | 0.352 | 7.578 | 5.456–≥10.001 | 1 / 15 | 6.761 |
| Rössler / OTT | 0.362 | ≥10.001 | 7.547–≥10.001 | 9 / 15 | ≥10.001 |

The supplementary vanilla Rössler median was **6.326**, so its pooled result
is below the 7–8 target. That seed sensitivity remains work to improve, even
though the primary bank reaches the target. These results describe one
canonical trajectory per system, with several seeds and origins. They are not
a guarantee across systems, sampling intervals, trajectories or hardware.

Replaying all selected settings and controls before and after phase four gave
**150 matched cases** (300 rollouts) with byte-identical full forecasts,
fitted tensors and scores. Together with the frozen comparison, this provides
measured nonregression evidence for the tested deterministic classical models.
It does not establish an unchanged numerical trajectory for every earlier
major release, stochastic fit or custom temporal operation.

The [complete report](https://github.com/El3ssar/ResDAG/blob/main/benchmarks/results/phase-five-tuned-accuracy-2026-10-23.json)
includes both thresholds, all individual scores, the supplementary bank,
failed-trial reasons, complete winning recipes and exact execution snapshots.
The final reporting follow-up and original tuning execution have separately
recorded source hashes. Its timings are not controlled speed comparisons.

These measurements used explicit model-building recipes and search spaces.
They are not measurements of the automatic default search. The subsequent
approved API increment brings input strength, bias and recurrent density into
built-in vanilla and OTT tuning, while retaining the existing
quick/balanced/thorough budgets and custom search controls. It preserves
custom components and current workflow settings. Edited graphs keep fixed-DAG
tuning until given an explicit rebuilding recipe. See the
[default ranges and component policies](../reference/hpo.md#search-and-repetitions).

That increment also fixed sparse workflow staging: at that version, derived
CSR weights were rebuilt from the copied dense parameter, avoiding a PyTorch
sparse-storage deepcopy failure. The continuation replaces that duplicate
representation with [compressed-only storage](../workflows/sparse-reservoirs.md).
The recorded experiment recipes and their original execution hashes remain
available separately.

Run the full bounded experiment with:

```sh
python benchmarks/tuned_accuracy.py --before 00ca741 --after 0b546d6 --artifacts-dir /tmp/resdag-tuned-accuracy --output tuned-accuracy.json
```

`--quick --n-trials 1` is an interface smoke check with a separate seed bank;
it does not reproduce the reported search budget. Saved winning `.pt` models
are written to the artifact directory and can be loaded with `rd.load`.
Supply observed history up to the desired test origin to reproduce that
forecast; their cached training tail ends earlier.

## Decisions still requiring consultation

- Any change to public metrics or the model/state/solver contracts.
- Sparse/dense selection policies or a new compilation default after profiling.
- Changes to the approved cross-library budget or model family. The first
  comparison uses vanilla ESNs on both systems, **600 tuning seconds per
  library/system**, then five reserved seeds and three starts. Different
  budgets or broader scale regimes need agreement before another long run.
  Neither equal candidate counts nor equal elapsed time proves universal
  superiority.

Use standard Torch/SciPy operators, Optuna and existing benchmark timing tools.
The first increment makes no CUDA performance or cross-library ranking claim.
See [the tracker](roadmap.md#phase-5-speed-memory-and-comparisons) for status.
