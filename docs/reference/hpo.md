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
| `n_trials` | preset | Positive number of **new attempts in this call**; failures count. A resumed study adds this many attempts. |
| `search_space` | built-in | `search_space(optuna_trial) -> mapping`; ordinary Optuna suggestions and conditional branches. Return fixed settings alongside sampled settings. |
| `factory` | built-in | `factory(params, seed) -> Model or ESN`, rebuilding custom candidates. Accepted on a `Model`; the ESN preset rebuilds itself. |
| `seed` | `0` | Sampler seed and start of the default initialization seed bank. |
| `seeds` | preset | Explicit nonempty seed sequence; duplicates are removed. The same bank evaluates every configuration. |

The ESN default space samples `reservoir_size` between half and twice its
initial size (with a minimum bound of eight), `spectral_radius` from 0.2 to 1.5,
`leak_rate` from 0.05 to 1.0, and `alpha` from `1e-10` to `1e-2` logarithmically.
Use a custom search to choose other ranges and settings.

Without a factory, a custom `Model` keeps its DAG and reservoir weights. The
default search varies `alpha`; supplied settings may contain only `alpha` and
`washout`. It evaluates one initialization seed, since copying frozen weights
does not create a different reservoir. Its readouts must support the chosen
setting. A factory enables structural changes and multiple initialization seeds.

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

The parent samples a finite candidate window before workers evaluate it. Failed
candidates consume attempts, and no window dispatches beyond the remaining
trial budget. A timeout is a dispatch boundary, not cancellation of an ongoing
fit/forecast and not a hard total deadline. Preparation and bookkeeping also
consume elapsed time. A run with no valid completed candidate raises.

All candidates use the same fixed data/seed protocol, but an adaptive sampler
can propose a different sequence at a different worker count. Caller Python,
NumPy and Torch RNG states are restored around search/evaluation callbacks.
Fit/forecast timings synchronize accelerators before recording elapsed time.
Fit time is recorded once per seed, on its first origin; later origins record
zero fit time. Outer module identity, mode and hooks survive successful tuning;
parameters can be replaced. Create optimizers afterward. A failed final refit
or owned-storage cleanup leaves the original model unchanged.

**Forecast-generation pruning is not implemented in this common tuner.** It
does not accept `pruner=` and evaluates complete forecasts. The raw runner's
scoring-checkpoint pruning below is a different protocol.

### Optuna studies and resumption

| Argument | Default | Contract |
| --- | --- | --- |
| `study` | `None` | Caller-owned single-objective Optuna study with a matching direction and protocol. Queued trials are supported. |
| `sampler` | seeded `TPESampler` | Used when ResDAG creates the study. |
| `storage` | memory | Journal path, SQLite `.db` path or supported storage URL. |
| `study_name` | generated | Persistent study name; provide it with storage to resume a named study. |

Pass `study=` **or** storage/study-name/sampler settings. A supplied study owns
its sampler and connections. An unbound study may contain queued `WAITING`
configurations; unverified evaluated scores cannot enter the new protocol.

The protocol binds prepared data, splits/origins/scales, task timing, seed bank,
model configuration, search/objective/recipe configuration and relevant library
environment. Changed data, callables or model configuration require a fresh
study. Keep callbacks deterministic from their arguments and captured settings.
The common tuner checks protocol compatibility before appending attempts.
Resuming the same fitted model reuses its original candidate template, rather
than treating the fitted winner as a new starting experiment.

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

Optuna trial attributes also retain resolved parameters, seeds, run diagnostics
and evaluation error messages. Optuna's ordinary `best_params` contains sampled
parameters; use the report's `best_params` when fixed returned settings matter.

---

## Advanced raw runner

`run_hpo` consumes manually prepared five-array data loaders and raw `ESNModel`
creators. It returns a study rather than installing a ready common-workflow
model. Its default losses are minimized (horizon losses are negated), unlike
the positive maximized horizon of `model.tune`. Its manual multiprocessing,
seed and checkpoint-scoring behavior are unchanged by the common tuner.
See the [raw HPO guide](../advanced/raw-hpo.md) for its complete workflow.

::: resdag.hpo
    options:
      members:
        - run_hpo

### Study ownership and inspection

`run_hpo` releases the RDB sessions and persistent-database connection pools
it opens after optimization or a failure, including errors during setup.
A returned Optuna study remains usable: later reads or writes reconnect as
needed. SQLite memory studies retain their backing pool while the returned
study owns it, because closing the last connection would erase their data.
If cleanup also fails during an existing error, the original error is raised
with a note describing the cleanup failure. Cleanup errors after otherwise
successful work are raised normally.

Passing an existing `optuna.Study` to `transfer_trials` retains caller ownership.
Passing a storage path lets ResDAG open and release the source connections for
that transfer. A path must contain exactly one study; pass an explicit study
when choosing between several.

Transfer ignores completed trials with missing or non-finite objective values,
even when an external producer did not attach an error flag. Matching
multi-objective studies preserve the complete vector of finite values.
Previously evaluated scores are meaningful only for the same data,
preprocessing and objective; use `warm_start` to re-evaluate configurations
for a different experiment.

`get_study_summary` orders its top trials according to the study direction:
lower scores first for minimization and higher scores first for maximization.

---

### Raw-runner reproducibility

Pass `seed=` to `run_hpo` to make a study reproducible end to end. The seed is
applied in two places:

1. **The Optuna sampler** — so the *sequence of sampled hyperparameters* is
   identical across runs.
2. **Each trial**, where the per-trial seed `seed + trial.number` seeds PyTorch,
   NumPy, and Python's `random`, *and* is threaded into your `model_creator`.

That last step is the one that matters for the reservoir. The recurrent
**topology** and the **input/feedback initializers** build their own RNGs
(`np.random.default_rng(...)` / `torch.Generator`) and do **not** read NumPy's
legacy global state — so seeding `np.random` alone leaves the reservoir weights
varying run-to-run. To close the gap, make your creator accept a `seed` keyword
and forward it to the reservoir (the premade factories forward `**kwargs`
straight to `ESNLayer`, which fixes the whole reservoir):

```python
from resdag.models import ott_esn

def model_creator(reservoir_size, spectral_radius, seed=None):
    # `seed` reaches ESNLayer/ESNCell, fixing the topology, the
    # feedback/input initializers, and the random bias.
    return ott_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
        seed=seed,
    )

study_a = run_hpo(model_creator, search_space, data_loader, n_trials=20, seed=42)
study_b = run_hpo(model_creator, search_space, data_loader, n_trials=20, seed=42)
assert study_a.best_value == study_b.best_value  # identical
```

`run_hpo` only injects `seed` when `model_creator` declares a `seed` parameter
(or `**kwargs`); creators without one are called unchanged, and an explicit
`seed` returned by `search_space` always wins. Reproducibility is exact for
single-worker (`n_workers=1`) studies; with `n_workers > 1`, trial *scheduling*
across processes is non-deterministic, so both the sampled configurations and
`best_value` may differ across runs.

---

### Raw-runner scoring-checkpoint pruning

Each trial forecasts **once** and is scored at a handful of growing
forecast-horizon checkpoints, reporting the prefix loss at every one. A diverging
configuration already shows a large loss at an early checkpoint — so an Optuna
**pruner** can skip its remaining scoring checkpoints. The full trajectory has
already been generated, so pruning does not currently reduce forecast-generation
work. Pass `pruner=` to turn this on:

```python
from resdag.hpo import run_hpo

study = run_hpo(
    model_creator, search_space, data_loader,
    n_trials=100,
    pruner="median",   # stop trials worse than the running median at each checkpoint
)

# Pruned trials are recorded but excluded from `best_value`:
print(get_study_summary(study))   # the summary lists the Pruned count
```

`pruner` accepts a registry key, a fully-configured
[`optuna.pruners.BasePruner`](https://optuna.readthedocs.io/en/stable/reference/pruners.html)
instance, or `None`:

| Key | Pruner | When to reach for it |
|---|---|---|
| `"none"` (default) | `NopPruner` | No pruning. |
| `"median"` | `MedianPruner` | Good default — prunes trials worse than the running median. |
| `"asha"` | `SuccessiveHalvingPruner` (ASHA) | Aggressive, scales well with many parallel workers. |
| `"hyperband"` | `HyperbandPruner` | ASHA across several budgets; strong when the right budget is unknown. |
| `"threshold"` | `ThresholdPruner` | Prune collapsed negated-horizon losses (`upper=-0.1`, two warmup checkpoints); pass a configured instance for positive error losses or another bound. |

Need to tune a pruner's knobs? Build the instance and pass it in:

```python
import optuna

study = run_hpo(
    model_creator, search_space, data_loader, n_trials=100,
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
)
```

Pruning is honored in both single- and multi-process (`n_workers > 1`) runs — the
resolved pruner is attached to every worker's study. It also composes with the
`clip_value` / `prune_on_clip` bound: the clip path prunes on the *raw*
loss exceeding a ceiling, while the pruner reacts to the *intermediate* horizon
reports; both can be active at once.

---

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

---

::: resdag.hpo.utils
    options:
      members:
        - get_study_summary
        - make_study_name
        - get_best_params
