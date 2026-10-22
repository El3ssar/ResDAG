---
description: Tune a preset or custom DAG, compare autonomous forecasts, and keep a fitted winner ready to use.
---

<span class="nb-kicker">Work · Tune</span>

# Tune a model

Call `model.tune(data)` to choose a configuration using autonomous forecasts.
It returns the same model, fitted with the winning settings and ready for
`forecast()`. `rd.ESN` and custom `rd.Model` networks use the same commands.

Install the optional tuning tools:

```sh
python -m pip install "resdag[hpo]"
```

## Start with a preset

Keep a final test segment separate from tuning. This small example uses a
sine wave so the complete workflow is easy to run:

```python
import torch
import resdag as rd

series = torch.sin(torch.arange(600, dtype=torch.float64) * 0.05)[:, None]
development, test = series[:500], series[500:540]

model = rd.ESN(reservoir_size=64, washout=30, readout="svd", dtype=torch.float64)
model.tune(development, preset="quick", horizon=40, seed=7)
prediction = model.forecast(horizon=len(test))

print(model.tuning_report.best_params)
print("Test RMSE:", (prediction - test).square().mean().sqrt().item())
assert prediction.shape == test.shape
```

The tuner first fits on the earliest 80% of `development` and checks forecasts
against its final 20%. After choosing a winner, it refits on **all** supplied
development observations. `forecast()` then starts after their final sample.
Scaling is learned during each fit; predictions remain in the original units.
The untouched `test` segment gives a separate final evaluation.

`model.tuning_report.study` is the full Optuna study.
`model.tuning_report.best_params` includes fixed values returned by your search
function as well as values sampled by Optuna.

### Choose the amount of work

| Preset | New trial attempts | Initialization seeds | Forecast starting points |
| --- | ---: | ---: | ---: |
| <code style="white-space: nowrap">"quick"</code> | 8 | 1 | 1 |
| <code style="white-space: nowrap">"balanced"</code> (default) | 24 | 3 | 3 |
| <code style="white-space: nowrap">"thorough"</code> | 80 | 5 | 5 |

Seeds and starting points are reused across configurations. A starting point
is a later moment in the held-out segment: the model sees observations before
that moment to synchronize its state, then forecasts without seeing future
feedback values. Its readout and preprocessing stay fixed during these checks.
Windows may overlap; they are several checks of one series, not independent
experiments.

Set `n_trials`, `seeds` and `n_origins` to override those counts. There can be
fewer starting points when the data cannot support more distinct full windows.
All use the same `horizon`. Omitting it uses half the available validation span,
with a minimum of one sample; changing presets does not change that horizon.
An unsupported explicit horizon raises before any trial starts.

For `rd.ESN`, the default search varies reservoir size, spectral radius, leak
rate and readout regularization (`alpha`). Custom DAGs have the options below.

### Optionally screen candidates first

Full evaluation remains the default. To shortlist candidates before spending
the complete repetition budget, enable screening:

```python
screened = rd.ESN(reservoir_size=32, washout=20, readout="svd", dtype=torch.float64)
screened.tune(
    development, preset="balanced", horizon=40, seed=7,
    screening=True, memory_budget="8GiB",
)
print(screened.tuning_report.resources)
```

This checks all 24 candidates on one seed and one starting point, using the
**complete** 40-sample forecast. The best eight then receive the full three-seed,
three-origin evaluation. Only a fully evaluated candidate can win. Candidates
outside the shortlist are recorded as `PRUNED`; they still consume attempts.
Screening may discard a configuration that would perform better across other
seeds or starting points.

`Screening(...)` lets you choose repetition counts, promotion fraction and cohort
size. Memory planning can reduce concurrency or skip an individually oversized
candidate; it does not change its architecture or precision. A budget is an
estimate, not an OS/RSS/GPU allocation limit. See
[tuning resources](../advanced/tuning-resources.md) for the controls and custom
factory estimates.

## Tune a custom DAG

The functional construction stays the same: connect layers with `layer(x)` and
wrap named inputs and outputs in `rd.Model`. Here two reservoirs process the
same signal at different update rates:

```python
def make_model(width=64, alpha=1e-6, seed=0):
    x = rd.reservoir_input(1, dtype=torch.float64)
    slow = rd.ESNLayer(
        width, feedback_size=1, leak_rate=0.2,
        spectral_radius=0.9, seed=seed, dtype=torch.float64,
    )(x)
    fast = rd.ESNLayer(
        width, feedback_size=1, leak_rate=1.0,
        spectral_radius=0.7, seed=seed + 1, dtype=torch.float64,
    )(x)
    joined = rd.Concatenate()(slow, fast)
    y = rd.SVDReadoutLayer(2 * width, 1, alpha=alpha).double()(joined)
    return rd.Model(inputs={"state": x}, outputs={"next": y}, washout=30)

custom = make_model()
custom.tune(development, preset="quick", horizon=40, seed=7)
custom_prediction = custom.forecast(horizon=40)
assert custom_prediction.shape == (40, 1)
```

Without a recipe, a custom DAG keeps its wiring and reservoir weights. Its
default search varies `alpha`; a custom search may vary `alpha` and `washout`.
Only one initialization seed is evaluated, since copying the same weights
would not create new initializations. Readouts must support the selected
settings; supply your own search or recipe for other readout types.

### Change sizes, layers or initializers

Provide an ordinary Python recipe, `factory(params, seed)`, when candidates need
new structure or weights. It rebuilds the whole candidate, so changing a
reservoir's width also updates its readout automatically:

```python
def search_space(trial):
    return {
        "width": trial.suggest_int("width", 32, 96, step=32),
        "alpha": trial.suggest_float("alpha", 1e-9, 1e-3, log=True),
    }

def factory(params, seed):
    return make_model(width=params["width"], alpha=params["alpha"], seed=seed)

custom.tune(
    development,
    factory=factory,
    search_space=search_space,
    preset="quick",
    horizon=40,
    seed=7,
)
custom_prediction = custom.forecast(horizon=40)
assert custom_prediction.shape == (40, 1)
```

Use the recipe to choose any supported layers, topology, registered initializer
or custom module. Return a built `rd.Model` (or a compatible `rd.ESN`). Candidates
must preserve the task: input/output names and widths, prediction leads,
feedback routes/delays, and resampling settings. Their hidden structure,
preprocessing and trainable settings can differ.

`search_space(trial)` uses standard Optuna suggestions, including conditional
branches. Return a mapping of finite JSON-compatible settings (tuples are retained). Capture Python
objects such as custom initializers in the recipe instead. `rd.ESN` rebuilds
itself and therefore uses its own recipe; use `rd.Model` for a custom factory.

## Tune a TSDynamics trajectory

Install `resdag[hpo,dynamics]` on Python 3.12 or newer. The same tuning call
accepts a trajectory and returns forecasts with future times and signal names:

```python
import numpy as np
import tsdynamics as ts

trajectory = ts.systems.Lorenz().run(final_time=6.0, dt=0.02)
dynamics_development, dynamics_test = trajectory[:241], trajectory[241:261]
dynamics_model = rd.ESN(
    reservoir_size=48, washout=20, readout="svd", dtype=torch.float64,
)
dynamics_model.tune(dynamics_development, preset="quick", horizon=20, seed=7)
trajectory_prediction = dynamics_model.forecast(horizon=len(dynamics_test))
tensor_prediction = dynamics_model.forecast(horizon=20, return_type="tensor")

assert isinstance(trajectory_prediction, ts.Trajectory)
assert trajectory_prediction.variables == dynamics_test.variables
np.testing.assert_allclose(trajectory_prediction.t, dynamics_test.t, atol=1e-12)
assert tensor_prediction.shape == (20, 3)
```

This is an interface example, not a forecast-quality benchmark. The
[named trajectory guide](../start/named-trajectories.md) explains multiple
inputs and targets. Tuning uses the same physical alignment, component names
and configurable resampling. Raw source observations are split before
interpolation, so held-out values cannot influence training interpolation.
Scoring stays in tensors; trajectory output converts to CPU only when requested.

## Supply your own validation segment

Use `validation=(inputs, targets_or_None)` when a particular held-out segment
must decide the winner. It is excluded from **both** candidate fits and final
refitting:

```python
held_out = rd.ESN(reservoir_size=64, washout=30, readout="svd", dtype=torch.float64)
held_out.tune(
    series[:400],
    validation=(series[400:500], None),
    preset="quick",
    horizon=40,
    seed=7,
)
# Use the later observations as history without fitting on them.
held_out_prediction = held_out.forecast(history=series[400:500], horizon=40)
assert held_out_prediction.shape == (40, 1)
```

Without that explicit history, this model's cached forecast begins after its
training segment at sample 399. Tensor validation is treated as the continuation
of training, with the same batch and feature layout. Explicit trajectory
validation must occur later in time and preserve component names/order.

For separate input/target tensors, targets are already aligned to each input
origin, as in `fit(inputs, targets)`. A head with lead three expects target row
zero to describe the signal three samples after input row zero. Tuning withholds
labels beyond each observed fit boundary; the shared supervision window stops
at the largest head lead while retaining the complete observed history.
Explicit Tensor validation starts scoring after its first `max(prediction_leads)`
observations, so every head has labels originating within validation.
Trajectory targets instead supply physical timestamps and are queried directly.
Actual origins and discarded target rows are recorded in `tuning_report.validation`.

Named input/target mappings and `(batch, time, features)` tensors work as in
`fit`. Inputs without a feedback route become known future drivers during
validation. Future observed feedback signals are withheld from the rollout.
See [feedback connections](../start/feedback.md) for leads and independent delays.

## Understand the score

The default winner has the longest **contiguous accurate forecast**, measured
in samples. At each sample, errors are divided by fixed training channel
scales, then combined with an RMS across features. A sample counts while this
error is strictly below `threshold=0.2`; counting stops at the first failure.
Scales come from training labels after the original model's washout, and stay
fixed across trials, seeds, starting points and forecast prefixes.

For batches, each output uses the median horizon across batch items. With
several outputs, the weakest output determines the score: position lasting
400 samples and velocity lasting 100 gives a score of **100**. Set
`aggregation="mean"` to average outputs instead. Each trial then takes the
median score across its initialization seeds and starting points. The winner
is refitted using the first seed in that fixed bank.

The report also includes full-window normalized error (`nrmse`), per-output
horizons/errors and fit/forecast timings. These diagnostics do not break ties
in the default horizon objective. For trajectory data, multiply sample counts
by the recorded `validation["dt"]` to obtain physical duration.

### Choose an error objective

A callback can replace the ranking while retaining the normal fit, forecast
and reporting workflow. `ScoreContext` contains named predictions and truth in
original units, fixed training scales, and the current seed/origin. Truth and
scales are copied for the callback so it cannot alter later evaluations.

```python
from resdag.hpo import ScoreContext

def normalized_error(context: ScoreContext):
    errors = [
        ((context.predictions[name] - context.truth[name]) / context.scales[name])
        .square().mean()
        for name in context.predictions
    ]
    return torch.stack(errors).mean().sqrt().item()

error_model = rd.ESN(reservoir_size=64, washout=30, readout="svd", dtype=torch.float64)
error_model.tune(
    development, preset="quick", horizon=40, seed=7,
    objective=normalized_error, direction="minimize",
)
```

Return one finite scalar per seed/origin. Scores are aggregated by their median.
`direction="minimize"` chooses the smallest error; the built-in horizon objective
requires `"maximize"`. Custom metrics can remain on the tensor device until
the final scalar conversion.

### Stop unpromising forecasts early

An Optuna pruner can stop future forecast generation at a checkpoint:

```python
pruned_model = rd.ESN(reservoir_size=32, washout=20, readout="svd", dtype=torch.float64)
pruned_model.tune(
    development, preset="quick", horizon=40, seed=7,
    pruner="median", checkpoint_interval=10,
)
```

Forecasting proceeds in chunks of at most ten scored samples here. The parent
process owns the real Optuna trial and makes pruning decisions, including when
numerical work runs in joblib workers. A rejected checkpoint prevents later
chunks, origins and seeds from running. Fitting and work before that checkpoint
have already happened. Pruning is disabled by default, and the median pruner
also has its own startup/warmup rules.

For the built-in objective, intermediate values are positive accurate-horizon
scores with the same fixed training scales. `pruner="threshold"` uses a lower
score bound of 0.1 after a warmup of `2 * checkpoint_interval` resource steps.
This is separate from `threshold=0.2`, which defines an accurate forecast sample.
The [raw runner](../advanced/raw-hpo.md) uses negated horizon losses and a
different threshold-pruner preset; its bounds are not interchangeable.

Custom `objective` callbacks still receive a **complete** forecast. Without an
explicit prefix callback, pruning is checked only after each complete
seed/origin forecast and can skip later evaluations. Opt in to within-forecast
pruning when your metric makes sense on partial predictions:

```python
from resdag.hpo import PrefixScoreContext

def prefix_error(context: PrefixScoreContext):
    # Predictions/truth contain only the completed prefix. The same fixed
    # training scales used by normalized_error are available here.
    return normalized_error(context)

prefix_model = rd.ESN(reservoir_size=32, washout=20, readout="svd", dtype=torch.float64)
prefix_model.tune(
    development, preset="quick", horizon=40, seed=7,
    objective=normalized_error, pruning_objective=prefix_error,
    direction="minimize", pruner="median", checkpoint_interval=10,
)
```

`PrefixScoreContext.completed_steps` is the admitted prefix length;
`horizon` is the complete requested scored length. Return a finite scalar with
the same direction and comparable meaning at matching resource steps. The
complete `objective` still decides the final ranking. Custom callbacks retain
prediction prefixes; the built-in scorer accumulates its statistics without
retaining a full output trajectory.

## Run and resume a study

Optuna provides samplers, queued starting configurations and study inspection.
For example, enqueue a known configuration before asking for more candidates:

```python
import optuna

study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=7),
)
study.enqueue_trial({
    "reservoir_size": 64, "spectral_radius": 0.9,
    "leak_rate": 0.5, "alpha": 1e-6,
})
resumable = rd.ESN(reservoir_size=64, washout=30, readout="svd", dtype=torch.float64)
resumable.tune(
    development, study=study, preset="quick", horizon=40, seed=7,
    n_trials=4, n_jobs=2,
)
resumable.tune(
    development, study=study, preset="quick", horizon=40, seed=7,
    n_trials=2, n_jobs=2,
)
assert len(study.trials) == 6
```

`n_trials` counts **new attempts per call**, including failed, pruned,
memory-rejected and screened-out candidates. A timeout can stop a call before
it uses the whole budget.
`catch_exceptions=True` records evaluation failures as `FAIL` trials, with
messages in trial attributes; they cannot win. Set it to `False` while debugging
to raise the original error. If no valid candidate completes, tuning raises.
The source model changes only after a winner refits successfully. Construct
optimizers after tuning, since the winning configuration can replace parameters.

The default `n_jobs=1` runs in the caller. Other values use joblib/loky process workers
and its standard worker-count conventions, including `-1` for available CPUs.
Workers evaluate parent-sampled windows of candidates; `inner_max_num_threads=1`
limits worker BLAS threads. Search suggestions, Optuna storage and pruning stay
in the parent; workers receive prepared numerical jobs. Submission uses finite
windows bounded by remaining attempts and estimated memory, so a worker queue
cannot exceed the new-attempt budget. `timeout` stops dispatching new windows, while an
in-flight window and the final winner fit finish normally. It is not a hard
wall-clock deadline. Changing worker count can change an adaptive sampler's
sequence, even though candidates reuse the same evaluation seeds and origins.

On errors or interruption, joblib cancels outstanding work and this call's
unfinished trial reservations are finalized. An error during final refitting
or owned-storage cleanup leaves the original model unchanged.

Use `storage="tuning.log", study_name="my-experiment"` for a persistent journal,
or a `.db` path for SQLite. Pass those options instead of `study=`; a caller-owned
study also owns its sampler and pruner. Resuming checks data, validation, task, seed bank,
objective and captured configuration against the stored protocol. Keep callback
configuration stable, and use a fresh study when the experiment changes.
Unverified prior evaluated scores are rejected; queued trials are supported.
Caller-owned studies remain usable, and returned owned studies reconnect after
storage cleanup when needed.

Forecast execution defaults to `execution="auto"`: compatible DAGs use the fast
step loop, while temporal custom layers use history replay. Force `"fast"` or
`"history"` to select a mode explicitly. See [forecast capabilities](../advanced/forecast-capabilities.md).

### Reuse settings from another experiment

```python
reused = rd.ESN(reservoir_size=64, washout=30, readout="svd", dtype=torch.float64)
reused.tune(
    development, warm_start=model.tuning_report.study,
    preset="quick", horizon=40, seed=7, n_trials=2,
)
```

The default `transfer_mode="reevaluate"` queues complete configurations from
the source, best first, then evaluates them on the destination task. Old scores
are not copied. Fixed returned settings and source-trial provenance are kept.
Queued configurations consume new attempts when evaluated; any beyond the
budget remain `WAITING`. This is the useful choice when data, architecture or
the objective changes. Adapt incompatible settings explicitly.

`transfer_mode="scores"` is an explicit import of prior evaluated scores. It
requires identical bound task protocols and objective directions, including
the actual prepared data and model/objective configuration. Imported scores
do not consume the budget for new attempts. Use reevaluation when the experiment
changes; matching parameter names alone do not make scores comparable.

## Continue

- [HPO reference](../reference/hpo.md) — every option and report field
- [Tuning resources](../advanced/tuning-resources.md) — optional screening and memory planning
- [Common model workflow](../start/model-workflow.md) — build, fit and use a custom network
- [Raw HPO runner](../advanced/raw-hpo.md) — manually prepared five-array studies
