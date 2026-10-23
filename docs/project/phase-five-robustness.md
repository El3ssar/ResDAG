---
description: A controlled vanilla feature comparison and a separate validation search before unseen trajectory tests.
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

## Status

Implementation and bounded CLI replay are verified locally. The full fresh
validation/final study and hosted acceptance are in progress. Results will be
reported with the median across trajectories, all trajectory medians and the
individual range, including failures and censoring. No headline success is
claimed before that evaluation.
