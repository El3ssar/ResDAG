---
description: Common Model/ESN tuning options, validation and scoring contracts, Optuna reports, and advanced raw HPO tools.
---

<span class="nb-kicker">Reference</span>

# Hyperparameter tuning

[`model.tune(data)`](../workflows/tune.md) is the common entry point for
`rd.ESN` and `rd.Model`. It compares autonomous forecasts, installs a refitted
winner, and returns the same outer PyTorch module. The standalone
`resdag.hpo.tune_model(model, data, ...)` accepts the same advanced options.

Install `resdag[hpo]` for Optuna and joblib. Add the `dynamics` extra to accept
TSDynamics trajectories on Python 3.12 or newer.

## Common tuning API

::: resdag.hpo.tuning
    options:
      members:
        - tune_model
        - ScoreContext
        - PrefixScoreContext
        - TuningReport

### Data and validation

| Argument | Default | Contract |
| --- | --- | --- |
| `training` | required | Floating `(time, features)` or `(batch, time, features)` tensors, a named tensor mapping, a bare trajectory, or named trajectory inputs. |
| `targets` | `None` | Explicit named output targets, or implicit future observations for one input/output of equal width. Tensor targets are already aligned to input origins; trajectory targets are sampled at each head's physical prediction lead. |
| `validation` | `None` | `(inputs, targets_or_None)` for an explicit later segment. It is excluded from both candidate fitting and final refitting. |
| `validation_fraction` | `0.2` | Chronological held-out fraction when validation is automatic. |
| `horizon` | `None` | Complete scored length per origin. Default: half the available validation span, at least one sample. It stays the same across presets and candidates. |
| `n_origins` | preset | Number of distinct, evenly spread supported forecast starts; reduced if fewer complete windows exist. |

Automatic validation fits candidates on the earlier observations, then refits
the winning configuration on all supplied development observations. Keep a
separate final test set. Explicit validation stays held out during refitting;
supply its later observations as forecast history when continuing after it.

Every origin uses the candidate's fixed readout and preprocessing. A later
origin can synchronize on earlier validation observations without fitting on
them. Complete windows can overlap. No unsupported horizon is silently shortened.

For an observed Tensor fit window of length `C` and largest head lead `L`,
shared supervised origins stop before `C - L`. This prevents future labels
crossing the fit boundary. The complete observed inputs are still retained for
forecast history. The same physical-support rule applies during final refitting;
`validation["refit_discarded_target_rows"]` reports discarded rows by head.
Standalone `fit(inputs, tensor_targets)` trusts the caller's aligned rows;
tuning deliberately caps them to the observed development window.

With explicit origin-aligned Tensor validation, the first scored forecast starts
after `max(prediction_leads)` validation observations. For example, leads one
and three skip the first three validation observations before selecting scored
origins. `first_validation_index` includes the preceding training length.
Trajectory truth is queried at physical dates and uses its actual source support
instead. Inputs and validation targets must preserve component names/order.

Trajectory interpolation is prepared once per fixed view from raw source slices.
Training and origin histories cannot use later source values in interpolation
coefficients. A named clock can require unscored preroll to reach the requested
validation date; reports preserve its actual endpoint, first scored date and
skip count. Forecasts still receive only declared non-feedback future drivers.

### Search and repetitions

| Argument | Default | Contract |
| --- | --- | --- |
| `preset` | `"balanced"` | `quick`: 8 trials/1 seed/1 origin; `balanced`: 24/3/3; `thorough`: 80/5/5. |
| `n_trials` | preset | Positive budget of **new attempts in this call**; failures, pruning, memory rejection and screening rejection count. Timeout can stop early. |
| `search_space` | built-in | `search_space(optuna_trial) -> mapping`; ordinary Optuna suggestions and conditional branches. Return fixed settings alongside sampled settings. |
| `factory` | built-in | `factory(params, seed) -> Model or ESN`, rebuilding custom candidates. Accepted on a `Model`; the ESN preset rebuilds itself. |
| `seed` | `0` | Sampler seed and start of the default initialization seed bank. |
| `seeds` | preset | Explicit nonempty seed sequence; duplicates are removed. The same bank evaluates every configuration. |

Built-in `rd.ESN` and unmodified classic/OTT graphs wrapped in `rd.Model` use
the following space. All presets use the same ranges; presets change the
number of trials, seeds and starting points.

| Setting | Default range | Sampling |
| --- | --- | --- |
| <code style="white-space: nowrap">reservoir_size</code> | Half to twice the initial width, each bound at least eight | Integer |
| <code style="white-space: nowrap">spectral_radius</code> | 0.05–1.5 | Uniform |
| <code style="white-space: nowrap">leak_rate</code> | 0.05–1.0 | Uniform |
| <code style="white-space: nowrap">alpha</code> | `1e-12`–`1e-2` | Logarithmic, when the readout exposes this setting |
| <code style="white-space: nowrap">input_scaling</code> | `1e-3`–2.0 | Logarithmic, when supported by the input initializer |
| <code style="white-space: nowrap">bias_scaling</code> | `1e-3`–1.0 | Logarithmic, when bias uses its default initializer |
| <code style="white-space: nowrap">mean_degree</code> | Three to twelve, capped by available recurrent inputs | Integer, for the default or configured Erdős–Rényi topology |
| <code style="white-space: nowrap">input_to_readout</code> | `True`, `False` | Categorical, vanilla recipes only; input plus states or states alone |

`mean_degree` is a tuning control for the automatic recipe. It controls the
expected number of recurrent matrix entries per
unit, including one for a permitted self-connection. It determines the
Erdős–Rényi probability for the candidate's sampled width; the actual sampled
connection count varies. Existing topology flags and initializer settings are
preserved. A custom non-Erdős–Rényi topology, disabled bias or custom bias
initializer keeps its own construction rather than being replaced merely to
expose another search dimension.

Named input initializers and `InputFeedbackInitializer` objects support the
shared scaling control without losing their other options. Arbitrary matrix
callables and `FunctionInitializer` objects keep their original construction;
the tuner cannot assume that they accept or honor a scaling keyword. Supply
an explicit factory when you want to search their own settings.

`search_space` replaces the default space completely. Return only the settings
you want to vary or fix, using ordinary Optuna suggestions for any ranges.
Current workflow settings and custom transforms are copied into candidates;
the winner is refitted rather than inheriting fitted transform statistics.

Without a factory, a custom `Model` keeps its DAG and reservoir weights. This
also applies when the structure or reservoir weights of a built-in graph have
been edited after construction: automatic rebuilding must not discard edits. The
default search varies `alpha`; supplied settings may contain only `alpha` and
`washout`. It evaluates one initialization seed, since copying frozen weights
does not create a different reservoir. Its readouts must support the chosen
setting. A factory enables structural changes and multiple initialization seeds.

Ordinary fitting, recurrent memory, device/dtype changes, copying and portable
save/load retain an eligible built-in recipe. A portable graph whose initializer
recipe contains unsupported local Python code still saves its materialized
weights; its automatic rebuilding recipe is omitted. After loading, use a
custom factory to tune its architecture, or a trusted bundle when you need to
retain that local Python recipe. See [save and load](../workflows/save-load.md).

Recipes can vary hidden widths, layer families, initialization and preprocessing.
They must retain declared input/output names and widths, prediction leads,
feedback routes/delays, and resampling settings. Returned settings must be finite
JSON-compatible Python data; tuples are retained during reconstruction. Capture
modules, initializer objects and callables in the factory instead of returning
them as trial parameters. Forward the supplied seed when constructing reservoirs.

### Scoring

| Argument | Default | Contract |
| --- | --- | --- |
| `threshold` | `0.2` | Positive threshold for the RMS of per-feature normalized errors. Accuracy requires error **strictly below** this value. |
| `aggregation` | `"weakest"` | Minimum head horizon; `"mean"` averages head horizons. |
| `objective` | `None` | `objective(ScoreContext) -> finite scalar` replaces the ranking. |
| `pruning_objective` | `None` | Optional `pruning_objective(PrefixScoreContext) -> finite scalar` enables custom within-forecast pruning while preserving the complete final objective. |
| `direction` | `"maximize"` | Built-in horizon uses maximize. Set `"minimize"` for a custom error objective. |

The default counts each batch item's contiguous accurate prefix, stopping at
the first threshold failure. Each head uses the batch median, averaging the
two middle values for an even batch. `"weakest"` selects the shortest head
horizon. Thus horizons of 400 and 100 yield 100. The trial score is the median
across its initialization-seed/origin evaluations. Scores are positive sample
counts; zero is possible. The winner is refitted with the first seed in the bank.

Scoring divides each feature's original-unit error by a fixed training-label
population standard deviation. The scale uses the prototype's absolute washout,
not a trial's tuned washout or any validation prefix. Constant features use scale
one; nonconstant scales have a dtype-epsilon floor. Half-precision scale
calculation uses float32. This scoring scale is separate from model preprocessing.

`ScoreContext` fields are named `predictions`, `truth`, `scales`, plus `seed` and
`origin`. Predictions/truth retain the data's 2-D/3-D layout and original units.
Truth and scale tensors are cloned for custom objectives. A callback receives
one full seed/origin forecast and returns one finite scalar; the tuner takes
its median across runs. Built-in diagnostics remain available with a custom
objective. Nonfinite forecasts, scales or objective values fail the evaluation.

`PrefixScoreContext` adds `completed_steps` and the complete requested `horizon`.
Its prediction/truth tensors contain only the completed prefix. A prefix callback
controls pruning, not final ranking, and must return comparable finite values
at matching resource steps with the study's direction. Without this explicit
callback, a custom complete objective is evaluated only on full forecasts;
pruning can then skip later origins/seeds but not shorten the current forecast.

`nrmse` is a diagnostic: each head's full-window RMS normalized error, averaged
across heads with feature-count weights, then aggregated by the median across
runs. It does not break ties in the default horizon objective. Trajectory
reports include `dt`; multiply a horizon by it for physical duration.

### Execution and budgets

| Argument | Default | Contract |
| --- | --- | --- |
| `n_jobs` | `1` | Caller process by default; other nonzero integers use joblib workers (`-1` selects available CPUs). |
| `inner_max_num_threads` | `1` | Positive worker BLAS-thread limit. |
| `timeout` | `None` | Stops dispatching new windows after elapsed seconds; preparation counts toward this time. Running windows and final refitting finish. |
| `catch_exceptions` | `True` | Failed candidate evaluations become Optuna `FAIL` trials; `False` raises the original exception. Setup/search errors still raise. |
| `execution` | `"auto"` | Select fast step execution when supported, history replay for temporal DAGs; `"fast"` and `"history"` force a mode. |
| `pruner` | `None` | Registry key or configured Optuna pruner; the parent owns pruning decisions. A supplied `study` owns its pruner instead. |
| `checkpoint_interval` | `64` | Positive scored-sample chunk length. Rejection at a checkpoint prevents later forecast chunks/evaluations. |
| `screening` | `False` | `True` uses `Screening()`; a policy customizes cheap repetition counts, promotion fraction and cohort size. Full evaluation remains the default. |
| `memory_budget` | `None` | Integer bytes, a size string or `MemoryBudget`; estimated admission/concurrency policy, never a process/RSS/GPU allocator cap. |
| `memory_estimator` | `None` | `estimator(params, MemoryContext) -> MemoryEstimate or int`; required with a custom factory and budget. Returns complete totals, floored by known resident requirements. |

The parent samples a finite candidate window before workers evaluate it. Failed
candidates consume attempts, and no window dispatches beyond the remaining
trial budget. A timeout is a dispatch boundary, not cancellation of an ongoing
fit/forecast and not a hard total deadline. Preparation and bookkeeping also
consume elapsed time. A run with no valid completed candidate raises.

Workers receive prepared numerical jobs, without a study or storage connection.
The parent owns sampling, real Optuna trial reports and pruning. Joblib/loky
owns process creation, cancellation and reuse; there is no fork-based worker
launcher. Errors/interruptions cancel outstanding work and finalize this call's
unfinished reservations without replacing the original model.

All candidates use the same fixed data/seed protocol, but an adaptive sampler
can propose a different sequence at a different worker count. Caller Python,
NumPy and Torch RNG states are restored around search/evaluation callbacks.
Fit/forecast timings synchronize accelerators before recording elapsed time.
Fit time is recorded once per seed, on its first origin; later origins record
zero fit time. Outer module identity, mode and hooks survive successful tuning;
parameters can be replaced. Create optimizers afterward. A failed final refit
or owned-storage cleanup leaves the original model unchanged.

Built-in scoring accumulates fixed-scale statistics while forecasts stream.
Resource steps are `completed_seed_origin_count * horizon + completed_prefix`,
so intermediate values from matching resources are comparable across trials.
The reported intermediate value is the median of completed evaluations and the
current prefix score. No pruner is used during the optional screening stage.

The common `pruner="threshold"` preset uses `ThresholdPruner(lower=0.1,
n_warmup_steps=2 * checkpoint_interval)` for positive horizon scores. It rejects
that string preset with a custom objective; pass a configured Optuna instance
with appropriate bounds instead. Other shared keys are `"none"`, `"median"`,
`"asha"` and `"hyperband"`. The sample-accuracy `threshold=0.2` is a separate setting.

Screening uses the same complete horizon with fewer seeds/origins, then ranks
each cohort and fully evaluates the promoted configurations. With balanced
defaults, all 24 candidates use one seed/origin first; the best eight receive
all three seeds and three origins. Only full evaluations become eligible
`COMPLETE` winners. Nonpromoted, oversized and timeout-unsubmitted candidates
are recorded as `PRUNED`, with reasons. A later full evaluation can still be
pruned by Optuna. See [tuning resources](../advanced/tuning-resources.md).

Memory admission accounts for complete winner refits, known long warmup traces,
retained custom-objective predictions and history replay. A custom factory
changing execution capabilities must estimate its own temporary/replay needs.
Individually oversized candidates are skipped before construction; estimated
concurrency is reduced for candidates that fit individually. The source model's
architecture and precision are never silently changed to satisfy a budget.

### Optuna studies and resumption

| Argument | Default | Contract |
| --- | --- | --- |
| `study` | `None` | Caller-owned single-objective Optuna study with a matching direction and protocol. Queued trials are supported. |
| `sampler` | seeded `TPESampler` | Used when ResDAG creates the study. |
| `storage` | memory | Journal path, SQLite `.db` path or supported storage URL. |
| `study_name` | generated | Persistent study name; provide it with storage to resume a named study. |
| `warm_start` | `None` | Source Optuna study whose complete configurations are queued for reevaluation by default. |
| `transfer_mode` | `"reevaluate"` | `"scores"` explicitly imports evaluated values only for identical bound protocols and objective directions. |

Pass `study=` **or** storage/study-name/sampler/pruner settings. A supplied study owns
its sampler, pruner and connections. An unbound study may contain queued `WAITING`
configurations; unverified evaluated scores cannot enter the new protocol.

The protocol binds prepared data, splits/origins/scales, task timing, seed bank,
model configuration, search/objective/recipe configuration and relevant library
environment. Changed data, callables or model configuration require a fresh
study. Keep callbacks deterministic from their arguments and captured settings.
The common tuner checks protocol compatibility before appending attempts.
Resuming the same fitted model reuses its original candidate template, rather
than treating the fitted winner as a new starting experiment.

Settings transfer preserves fixed returned values, suggestions and source
study/trial provenance. Reevaluation queues single-objective results best-first
(trial number breaks ties). Queued configurations consume attempts when run;
those beyond this call's budget remain `WAITING`. Imported scores do not consume
new attempts, but only exact matching protocols/directions allow them to compete.

Owned RDB sessions/pools are released before installing the winner. Returned
studies remain usable and reconnect when required; SQLite memory pools remain
alive while needed to retain the study. Caller-owned studies retain ownership.

### Report fields

| Field | Meaning |
| --- | --- |
| `study` | Full, usable Optuna study. |
| `best_params`, `best_trial` | Resolved winning settings, including fixed values, and trial number. |
| `score`, `nrmse` | Winning objective and full-window normalized-error diagnostic. |
| `seeds`, `refit_seed` | Evaluation bank and seed used for final refitting. |
| `validation` | Split/refit policy, supervised windows, discarded rows, horizon, clock and actual origins. |
| `protocol_hash` | Fingerprint checked when resuming. |
| `attempts` | New attempts made by this call. |
| `elapsed_seconds`, `refit_seconds` | Total elapsed time and final fitting time. |
| `runs` | Winning configuration's per-seed/origin scores, head metrics and timings. |
| `resources` | Requested/effective jobs, maximum running jobs, screening/promotion counts, memory skips, pruning and timeout outcome. |

Optuna trial attributes also retain resolved parameters, seeds, run diagnostics
and evaluation error messages. `screening_runs`/`screening_score` describe cheap
evaluations; partial pruned runs identify `complete=False` and completed steps.
`memory_estimate` records totals and any requested-versus-known-minimum floor.
Optuna's ordinary `best_params` contains sampled
parameters; use the report's `best_params` when fixed returned settings matter.

---

## Advanced raw runner

`run_hpo` consumes manually prepared five-array data loaders and raw `ESNModel`
creators. It returns a study; callers reconstruct the winner themselves. See the
[raw HPO guide](../advanced/raw-hpo.md) for an end-to-end example.

| Contract | Common `model.tune` | Raw `run_hpo` |
| --- | --- | --- |
| Data | Chronological/named/trajectory preparation | Explicit five-array loader and driver companions |
| Result | Same module with a refitted winner | Optuna study |
| Parallel count | `n_jobs`, joblib conventions | Positive `n_workers` |
| Default objective | Positive accurate horizon, maximized | Negated expected horizon, minimized |
| Checkpoint step | Cumulative scored samples across seed/origin cases | 1-based checkpoint index; actual prefix length is recorded |
| Prefix normalization | Fixed training scales | Existing raw loss normalization, potentially prefix-dependent |
| Threshold-pruner preset | `lower=0.1`, warmup `2 * checkpoint_interval` | `upper=-0.1`, two checkpoint-index warmup steps |

Both use parent-owned Optuna bookkeeping and managed joblib/loky numerical
workers. Search/data callbacks, sampler, pruner, user trial callbacks and storage
stay in the parent. In-memory studies work with multiple workers. Finite windows
respect the number of **new attempts** requested, including `FAIL` and `PRUNED`.
Queued transfer configurations count when evaluated; imported old scores do not.

::: resdag.hpo
    options:
      members:
        - run_hpo

### Raw execution and pruning

Raw forecasts are generated in chunks. Each growing-prefix report reaches the
parent pruner before later chunks are generated. A rejection stops future
forecast generation. `n_checkpoints=5` requests five checkpoints by default;
short horizons can yield fewer distinct checkpoints.

The raw losses retain their existing prefix-dependent normalization, so raw
scoring can retain a complete prediction prefix: **O(horizon)** storage. Custom
and monitor losses receive full-horizon tensors when an evaluation completes.
With `torch_scoring=True`, supported losses stay on the tensor device until their
final scalar; custom losses may use the NumPy fallback. Attractor-dimension
scoring uses NumPy.

Registry keys are `"none"`, `"median"`, `"asha"`, `"hyperband"` and `"threshold"`.
Raw defaults are minimization losses; choose a configured pruner with suitable
bounds for a custom metric. `clip_value`/`prune_on_clip` is a separate raw-loss
ceiling and can be combined with a pruner. The common tuner's positive threshold
preset is not the raw preset.

`memory_budget` accepts the same `MemoryBudget`, size string or byte count.
An arbitrary raw factory requires `memory_estimator(params, context)` when a
budget is set. Known prepared data, transfers and retained output storage form
a minimum; oversized candidates are `PRUNED` before factory construction.
Parent data preparation happens before estimation, and a pending configuration
may remain resident while an admitted window runs. This is planning, not a hard
process/RSS/device limit. See [tuning resources](../advanced/tuning-resources.md).

`inner_max_num_threads=1` limits native threads in numerical worker processes.
The parent environment, thread settings, logging and Python/NumPy/Torch RNG
streams are preserved/restored. `seed` initializes the sampler and each trial's
`seed + trial.number` RNG scope. Make the factory accept and forward a `seed`
keyword; an explicit model seed in resolved settings takes precedence. Forwarding
it also controls initializers with their own local generators. Adaptive proposals
can change with worker count and scheduling; exact reproducibility is not a
cross-device or cross-schedule guarantee.

### Reuse and transfer

`run_hpo(..., warm_start=[config])` queues explicit settings.
`run_hpo(..., transfer_from=source)` queues complete valid settings from a study
or storage path for **reevaluation** by default. Data-only suggestions and fixed
model values are preserved. Single-objective configurations are ordered best
first, then by trial number; multi-objective sources retain source order.

Use `export_best_config(study)` when fixed returned values matter;
`study.best_params` contains only Optuna suggestions. Incompatible model settings
must be adapted explicitly. Source study/trial/protocol provenance is retained
under the trial's `resdag_transfer` attribute.

`transfer_mode="scores"` requires exactly matching bound task protocols and
objective directions. Raw resumption and score import replay historical data
preparation with Optuna `FixedTrial` and check the actual prepared tensor
contents. A file changed at the same path is therefore a changed task. Callbacks
can run again during validation; keep preparation free of side effects.

When historical preparation cannot run without live-study resources, supply
`task_context="dataset-v3"` or portable metadata. This explicitly asserts the
identity of opaque resources, so update it when they change. Successful replay
still checks tensor contents. New attempts receive a real Optuna trial.
Use settings reevaluation for changed experiments rather than copying scores.

::: resdag.hpo.transfer
    options:
      members:
        - transfer_trials
        - apply_warm_start
        - export_best_config

### Study ownership and inspection

Owned RDB sessions and persistent connection pools are released after success
or failure, including setup failures. Returned studies remain usable and
reconnect when needed. SQLite memory pools remain alive while their returned
study needs them. Caller-owned studies retain their connections/ownership.
Cleanup errors preserve an already-active original error and add a note;
otherwise the cleanup error is raised.

A source storage path must contain exactly one study; pass an explicit source
study to choose among several. Transfer excludes incomplete, invalid and
nonfinite evaluations; exact-score imports preserve matching finite objective
vectors. `get_study_summary` orders valid results according to study direction.

## Resource policies and helpers

`MemoryContext` is the immutable public input to a custom `memory_estimator`.
Tuning supplies shapes, placement, retained storage and recognized execution
layout; the callback returns a `MemoryEstimate` or a byte count. For separate
planning, construct a context from known facts and call `estimate_memory`.
The preparation-specific context builders are implementation details.
Estimates guide admission and concurrency; they are not process or allocator
limits and do not change model precision or device policy.

::: resdag.hpo.resources
    options:
      members:
        - Screening
        - MemoryBudget
        - MemoryEstimate
        - MemoryContext
        - estimate_memory

## Raw losses

::: resdag.hpo.losses
    options:
      members:
        - LOSSES
        - LossProtocol
        - get_loss
        - expected_forecast_horizon
        - forecast_horizon
        - lyapunov_weighted
        - standard_loss
        - soft_valid_horizon
        - attractor_dimension_error

## Study utilities

::: resdag.hpo.utils
    options:
      members:
        - get_study_summary
        - make_study_name
        - get_best_params
