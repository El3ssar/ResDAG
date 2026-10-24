---
description: Validation-only recipe selection, controlled feature comparison and complete held-out Lorenz and Rössler results.
---

# Vanilla forecasting robustness

The previous independent-trajectory study found no prediction regression against
the accepted implementation. It also found that the friendly vanilla recipes
fell short of the typical seven-to-eight Lyapunov-time goal. This continuation
tests stronger recipes without choosing settings on those inspected test runs.

## What changes in the model

`rd.ESN(..., input_to_readout=False)` fits the prediction head on reservoir
memory alone. The default `True` also supplies the current input. The same
setting is available on `classic_esn`; ordinary construction keeps its default.
Automatic vanilla tuning searches both choices with its existing budgets.
OTT's augmented features are unchanged. See the
[model workflow](../start/model-workflow.md#use-a-preset-graph) for usage and
the [tuning reference](../reference/hpo.md#search-and-repetitions) for controls.

Changing this switch does not change initialized reservoir weights. The first
two trials below compare identical settings and reservoir weights, differing
only in the head's features. Subsequent trials vary several settings, so their
results describe whole recipes rather than the effect of this switch alone.

## Validation before final testing

The checked-in runner is `benchmarks/vanilla_robustness.py`. It uses standard
Optuna TPE and public Torch `fit`/`forecast` calls. Each candidate is fitted
separately on three new trajectories per system, with three reservoir seeds and
three nonoverlapping forecast starting points. A batch fit would share one
readout across trajectories and would answer a different question.

Both systems use canonical Lorenz/Rössler parameters and TSDynamics DOP853
integration at `rtol=1e-10`, `atol=1e-12`. The protocol records exact initial
conditions, sampling intervals and solver settings before integration. The
search attempts at most 24 configurations per system with 600 seconds of soft
search time. An admitted configuration finishes all its seed/origin checks;
neither pruning nor optional screening is enabled. Both controlled baseline
trials must complete, even if the time budget expires before additional trials.

Search controls are width, spectral radius, leak, input strength, bias strength,
recurrent degree, ridge strength and the input-to-readout choice. The numerical
policy is fixed: float64 Torch SVD with an unpenalized intercept. Training-only
standard scaling and full observed history are used throughout.

The objective is the median of the three trajectory medians at the 0.3 error
threshold. Each trajectory median contains nine seed/origin forecasts.
Per-step error is RMS error divided componentwise by training standard deviation,
measured on `raw[washout + 1 : train_observations]`, matching the preceding
primary protocol. A forecast ends at its first error greater than or equal to
the threshold, or its first nonfinite error. Both 0.3 and 0.5 results are retained.

Both winners are frozen before generating any final observations. Final tests
compare the preceding primary vanilla recipe with the selected recipe on five
fresh trajectories per system, three fresh seeds and three starting points.
Each window spans ten Lyapunov times. Results at that limit are right-censored:
they establish accuracy through the window, without estimating its later end.
The independent unit is a trajectory, not each nested forecast.

## Reproduce the experiment

Install the repository's development, HPO and dynamics extras, and the pinned
benchmark requirements, in an isolated environment. From the repository root:

```bash
python benchmarks/vanilla_robustness.py --output /tmp/resdag-vanilla-study
```

The output directory must not exist. `--quick` runs a small execution smoke;
its short synthetic protocol is not forecast-quality evidence. `--trials` and
`--timeout` allow explicit budgets; changing them defines a different experiment.

The runner freezes the runtime source and records source/helper hashes, dependency
versions, IC and reservoir seed banks, generated observations, warning messages,
completed and failed trials, controlled reservoir fingerprints, frozen winners,
full error traces and prediction hashes. Partial final failures retain a failure
receipt. Dataset hashes are checked before loading. Its saved `runner.py` is a
provenance copy: rerun it from the original repository layout with the recorded
runtime, helpers and dependencies. It is not a standalone installed application.

Timing observations in this study are exploratory. They do not establish a
controlled CPU ranking or any CUDA performance claim. See the
[phase-five measurements](phase-five-results.md) for the separate speed evidence.

## Held-out results

The run completed all **48 validation trials**: 24 per system, with **432 fits
and 1,296 forecasts**. Both selected recipes were frozen before the final bank
was generated. Final evaluation used **60 independent fits and 180 forecasts**,
including each incumbent control, with no exclusions or failed fits.

The main statistic is the median of five independent trajectory medians. Each
trajectory contributes nine forecasts from three initialization seeds and three
starting points. The individual range shows the nested forecasts' spread; it
is not a confidence interval. A ≥ sign means the forecast remained below the
threshold through the ten-Lyapunov-time observation window.

| System / recipe | Median at 0.3 (LT) | Median at 0.5 (LT) | Individual range at 0.3 (LT) | Censored at 0.3 |
| --- | ---: | ---: | --- | ---: |
| Lorenz incumbent and selected recipe | ≥10.016 | ≥10.016 | 2.155–≥10.016 | 27 / 45 |
| Rössler incumbent | 3.013 | 3.810 | 0.849–9.225 | 0 / 45 |
| Rössler selected recipe | **6.750** | **7.154** | 1.677–≥10.001 | 1 / 45 |

All five trajectory medians at the main 0.3 threshold, in their recorded order:

- Lorenz: **[≥10.016, 5.995, ≥10.016, 9.654, ≥10.016]**.
- Rössler incumbent: **[3.013, 2.899, 3.810, 3.800, 2.485]**.
- Rössler selected: **[7.144, 6.253, 6.750, 8.779, 6.233]**.

Lorenz retained the existing primary recipe. Rössler improved from 3.01 to
6.75 LT on the fresh test bank, but **the typical 7–8 LT goal at 0.3 remains
unmet**. The more forgiving 0.5 result does not replace that criterion.
These are bounded vanilla-model accuracy results, not a comparison against
other libraries or a guarantee for other systems.

Both winners use `input_to_readout=True`. In the controlled validation pair,
Lorenz scored 10.016 LT with the input and 4.800 LT without it; Rössler scored
4.576 and 4.638 LT respectively. All other settings and initialized reservoirs
were identical within each pair. The later Rössler gain comes from selecting
a whole recipe; it cannot be attributed to the new switch alone. The default
therefore remains unchanged, and tuning can still select either configuration.

The selected Rössler recipe is:

```python
import torch
import resdag as rd

model = rd.ESN(
    input_size=3,
    reservoir_size=400,
    spectral_radius=0.316473400694296,
    leak_rate=0.6988018952693786,
    feedback_initializer=("random", {"input_scaling": 0.3162866539497749}),
    bias_scaling=0.029171927900014934,
    alpha=2.8843872485074098e-11,
    topology=("erdos_renyi", {"p": 6 / 400, "directed": True, "self_loops": True}),
    washout=500,
    seed=7849201,
    input_to_readout=True,
    readout="svd",
    dtype=torch.float64,
)
```

Use the recorded protocol's training length, washout, sampling interval and
training-only scaling when reproducing these measurements. A parameter list
alone does not reproduce an experiment; the download below records the full
configuration and exact source.

### Failures and integrity

All validation fits completed, but **104 of the 648 Rössler validation
forecasts became nonfinite**. They remain in the report and are scored at their
first invalid error, rather than being discarded. Lorenz validation and all
180 final forecasts were finite. No dataset-integrity failure or fit failure
was observed. These are different conditions: a completed fit can still produce
an unstable autonomous forecast.

After the API and persistence cleanup, a separate replay refitted all 60 final
models. All **180 complete prediction arrays matched their recorded SHA-256
hashes byte for byte** on this host. An independent NumPy calculation recovered
every threshold crossing and error curve. This checks the adopted cleanup
against the recorded experiment; it does not establish cross-device bitwise
reproducibility.

[Download the complete protocol, source manifest, frozen winners, every trial,
error trace and prediction hash](../assets/data/phase-five-vanilla-robustness-2026-10-24.json).
The [forecasting and animation guide](../workflows/visualize.md) uses the first
predeclared held-out case for each system. Those display cases are fixed,
include the complete recorded window and are not the headline aggregate.

## Verification status

The scientific evaluation and exact replay are complete. The release-readiness
record in the [roadmap](roadmap.md) distinguishes local, installed-package and
hosted verification. Phase five remains open for typical Rössler accuracy,
comparative CPU speed and broader model/hardware scales.
