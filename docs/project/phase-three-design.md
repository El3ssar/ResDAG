---
description: Agreed tuning experience, numerical safeguards and implementation progress for phase three.
---

# Phase three: reliable tuning

**Status: Done — merged and released in 3.0.0.** The
[roadmap](roadmap.md#phase-3-tuning-and-automatic-solver-policy) records the
implementation and verification. The common `model.tune(...)` lifecycle,
joblib execution and automatic readout are released in 2.1.0. The continuation
adds forecast-generation pruning, optional screening and memory planning, and
safe raw-runner reuse. All local and locked/latest hosted acceptance gates pass
in [PR #462](https://github.com/El3ssar/ResDAG/pull/462), merged as `f8d7637`.
Release publication, post-merge CI and installed-package verification passed.

## Approved experience

Tuning builds candidate models, trains them, compares their autonomous
forecasts and keeps the winning configuration. Basic ESNs and custom DAGs
use the same command. Quick, balanced and thorough presets control the amount
of work; balanced is the default. Advanced users retain their own Optuna
samplers, search spaces and objectives.

The maintainer approved:

- `model.tune(data)` reserves the last **20%** chronologically for validation,
  then refits the winner on **all supplied development data**. Keep the final
  test data outside this call. Explicit splits remain available.
- A custom DAG may supply a Python construction recipe for each candidate.
  Changing reservoir sizes can require changing the downstream readout width;
  the recipe rebuilds those connections together. Preset ESNs do this
  automatically.
- The same outer model object receives the fitted winner. Its internal graph
  and parameters may change. Tune before creating a PyTorch optimizer.
- The existing automatic-solver goal remains: choose a safe method, allow
  explicit overrides, and disclose precision, device and fallback decisions.
- Balanced tuning checks every candidate with its full seed/origin bank by
  default. Optional screening first evaluates a smaller bank, then reevaluates
  the promoted candidates using the full bank before choosing a winner.
- Transfer to another task reuses settings and scores them again. Copying
  completed scores requires an exact match of the recorded evaluation protocol.
- An optional memory budget reduces concurrency and skips an oversized
  candidate before construction. Estimates never change its architecture or
  precision and are not operating-system allocation limits. Custom recipes
  provide their own estimates.

The fluent interface returns the same model, like `fit`.
`model.tuning_report` exposes the full Optuna study, resolved winning settings,
fixed seeds, validation windows, errors and timings. Candidate evaluation, final
refitting and owned-storage cleanup finish before replacing the current model.
Changing internal parameters requires creating any PyTorch optimizer afterward.

## Data and timing safeguards

Trials use the common `Model.fit` and `Model.forecast` behavior, including
training-only preprocessing, named ports, prediction leads, feedback delays,
device placement and trajectory timestamps. The older raw-graph HPO runner
does not provide this common workflow by itself.

Validation boundaries must account for prediction distance. If training
observations stop at sample 800 and an output predicts two samples ahead,
the example originating at sample 799 has its label at 801. That example
must not enter the training solve. All observations through 800 can still
warm the reservoir. Final refitting must keep the last supplied observation
as the forecast endpoint.

For trajectories, split the raw observations before interpolation. Otherwise
PCHIP or cubic coefficients can depend on the validation suffix. Tensor
targets need equally explicit alignment: each target row labels its input
origin plus the declared prediction lead.

Every configuration faces the same initialization seeds and forecast
starting points. Scoring scales stay fixed across forecast prefixes;
recomputing them from a growing validation suffix changes the meaning of an
early pruning score. Named outputs require explicit aggregation rather than
silently scoring the first head.

The private streaming executor retains delayed-feedback queues and emits
aligned forecast chunks. Scoring accumulates on the tensor device; only scalar
checkpoint summaries cross to the parent for an Optuna pruning decision.
Stopping closes the stream before more predictions are generated. Ordinary
public `forecast()` retains its established optimized execution path.

Whole-forecast custom objectives still receive the complete prediction. They
can prune between completed seed/origin runs, or supply a separate
`pruning_objective` that explicitly accepts prefixes. A pruned or screened-only
candidate cannot become the winner. History replay and custom objectives can
retain growing histories or prediction prefixes; those memory costs are explicit.

## Winner and parallel policy

- The default reward is the length of the contiguous accurate forecast, using
  a normalized error threshold of **0.2** and fixed training-only scales.
- Within an origin, batch cases use their median horizon; multiple heads use
  the weakest horizon. The trial uses the median across seeds and origins.
  Mean head aggregation and custom objectives remain available.
- `n_jobs=1` runs in the caller. More workers use joblib/loky with notebook
  recipe support and bounded submission windows. Failed attempts consume the
  configuration budget. A timeout stops new windows; already running work and
  the final refit finish normally.
- Reusing a study requires the same recorded data/task, seeds, objective and
  construction recipe. Foreign evaluated scores are rejected. Standard Optuna
  queued configurations remain supported.
- The parent owns sampling, storage and pruning decisions in both workflows.
  Workers receive prepared numerical jobs through joblib/loky. The raw runner
  no longer launches or monitors its own worker processes.

| Preset | New configurations | Initialization seeds | Forecast origins |
| --- | ---: | ---: | ---: |
| Quick | 8 | 1 | 1 |
| Balanced | 24 | 3 | 3 |
| Thorough | 80 | 5 | 5 |

Each budget can be overridden. The default horizon is half the validation span
and stays the same across presets. Origins can overlap and are not independent
replicates. A custom DAG without a construction recipe retains its feature
weights and uses one initialization; it does not pretend that copying the same
weights produces independent reservoirs. Explicit validation remains outside
the final refit. See the [tuning guide](../workflows/tune.md).

The friendly `ESN` now defaults to `readout="auto"`, with guarded primal/dual
Cholesky, augmented QR and SVD routes. Forced methods keep their algorithm and
requested alpha. Reports disclose precision, conditioning and any CPU bridge;
finite precision still limits coefficient accuracy. The lower-level factories
retain their explicitly documented readout defaults. See
[readout solvers](../guides/readout-solvers.md).

## First correctness increment

The initial work repairs independent defects before adding the new lifecycle:

- SVD ridge filter factors avoid squaring large finite singular values.
  The requested regularization is preserved. Non-finite or negative `alpha`
  and `rcond` are rejected at construction and before solving.
- Study transfer skips missing/non-finite completed scores, including trials
  produced outside ResDAG. Matching multi-objective values remain intact.
- Study summaries order their top trials in the actual objective direction.
- Owned RDB sessions and persistent-database connection pools are released
  after `run_hpo`, even when setup or trial evaluation fails. A returned SQLite
  memory study retains its backing pool so its data survives. Path-based
  transfer releases its own source connections; caller-supplied studies retain
  their ownership. A cleanup failure cannot hide the original operation error.

Both tuning workflows now use managed finite joblib windows and parent-owned
Optuna trials. Raw task binding also fingerprints materialized data and rechecks
historical data on resume; opaque external resources need an explicit task
context version. Active Python custom-layer implementations are included in
provenance. Hashes guard recorded configuration, rather than proving arbitrary
Python or native code semantically equivalent. Verification is recorded in the
roadmap as each increment passes its checks.
