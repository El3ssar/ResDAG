---
description: Report forecast horizons in Lyapunov times — the hardware- and system-independent clock of a chaotic attractor — plus the reservoir's own edge-of-chaos exponent.
---

<span class="nb-kicker">Work · Lyapunov time</span>

# Lyapunov time

A forecast horizon of "347 steps" means nothing on its own: it depends on the
integration step, the system, and the sampling rate. `resdag.metrics` reports
horizons in **Lyapunov times** instead — the natural clock of a chaotic
attractor — so a number is comparable across systems, integration steps, and
papers.

## Why Lyapunov times

On a chaotic system, two trajectories that start infinitesimally close separate
on average like $e^{\lambda_\mathrm{max} t}$, where $\lambda_\mathrm{max}$ is the
**largest Lyapunov exponent**. One Lyapunov time

$$
t_\lambda = \frac{1}{\lambda_\mathrm{max}}
$$

is the timescale over which that separation grows by a factor of $e$. It is the
system's own unit of "how far ahead is hard", so it makes forecast horizons
directly comparable: a valid prediction of $6\,t_\lambda$ on the Lorenz system
and $6\,t_\lambda$ on Rössler span the same multiple of the intrinsic instability
timescale. Comparing predictive skill still requires matching error thresholds,
normalization, observation noise, and evaluation conditions.

A horizon in steps converts to Lyapunov times by the headline formula

$$
\text{VPT}_{\Lambda t} = \text{steps} \cdot dt \cdot \lambda_\mathrm{max},
$$

where $dt$ is the integration step. This is
[`steps_to_lyapunov`][resdag.metrics.steps_to_lyapunov]; its inverse is
[`lyapunov_to_steps`][resdag.metrics.lyapunov_to_steps], and
[`steps_per_lyapunov_time`][resdag.metrics.steps_per_lyapunov_time] gives
$1/(dt\,\lambda_\mathrm{max})$ — how many samples span one $t_\lambda$.

## Canonical exponents

For resdag's built-in chaotic generators the literature $\lambda_\mathrm{max}$
values are tabulated in
[`CANONICAL_LAMBDA_MAX`][resdag.metrics.CANONICAL_LAMBDA_MAX] (and, keyed by the
Mackey-Glass delay $\tau$, in
[`MACKEY_GLASS_LAMBDA`][resdag.metrics.MACKEY_GLASS_LAMBDA]):

| Generator | $\lambda_\mathrm{max}$ (per unit $t$) | $dt$ |
| --- | --- | --- |
| `lorenz` (σ, ρ, β = 10, 28, 8/3) | 0.9056 | 0.02 |
| `rossler` (a, b, c = 0.2, 0.2, 5.7) | 0.0714 | 0.05 |
| `mackey_glass` (τ = 17) | 0.0086 | 1.0 |

The exponent is in inverse natural-time units, so multiply by $dt$ to turn a
step count into Lyapunov times. Passing a `dataset=` name to the metrics below
looks these up for you.

## Measuring forecast quality

The **Valid Prediction Time** (VPT) is the standard forecast-skill metric on
chaotic systems: the number of contiguous steps a closed-loop forecast tracks
the truth before its normalised error (NRMSE) first crosses a threshold.
[`valid_prediction_time`][resdag.metrics.valid_prediction_time] returns it in
raw steps by default, or in Lyapunov times with `units="lyapunov"` — pass `dt`
and `lambda_max` explicitly, or a known `dataset=` name to infer both.

<div class="nb-specimen" data-label="lyapunov_vpt.py" markdown>

```python
import torch

import resdag as rd
from resdag.datasets import lorenz
from resdag.metrics import reservoir_lyapunov, steps_to_lyapunov, valid_prediction_time
from resdag.utils import prepare_esn_data

torch.manual_seed(0)

# Fit scaling on training history only; validation must not set its statistics.
series = lorenz(6000, normalize=None)  # (1, 6000, 3), dt = 0.02
history = series[:, :500 + 4000 + 1]
mean = history.mean(dim=(0, 1), keepdim=True)
scale = history.std(dim=(0, 1), keepdim=True, correction=0).clamp_min(1e-8)
series = (series - mean) / scale
warmup, train, target, f_warmup, val = prepare_esn_data(
    series, warmup_steps=500, train_steps=4000, val_steps=1000
)

# Train an Ott-augmented ESN in one algebraic pass.
model = rd.ott_esn(reservoir_size=400, feedback_size=3, output_size=3, seed=0)
rd.ESNTrainer(model).fit((warmup,), (train,), {"output": target})

# Forecast aligns one-to-one with the validation split.
preds = model.forecast(f_warmup, horizon=val.shape[1])

# VPT in raw steps and in Lyapunov times (dt / lambda_max inferred from the name).
vpt_steps = valid_prediction_time(preds, val)
vpt_lyap = valid_prediction_time(preds, val, units="lyapunov", dataset="lorenz")
print(f"VPT = {vpt_steps:.0f} steps  =  {vpt_lyap:.2f} Lyapunov times")

# The same conversion done by hand from the canonical exponent.
print("check:", round(steps_to_lyapunov(vpt_steps, dt=0.02, lambda_max=0.9056), 2))

# The reservoir's own maximal Lyapunov exponent (edge-of-chaos diagnostic).
lam = reservoir_lyapunov(model, f_warmup, horizon=500)
print(f"reservoir lambda_max = {lam:.4f} per step  (< 0 => sampled contraction)")
```

</div>

Results depend on reservoir weights, the data split, preprocessing and the
numerical backend. The example deliberately computes scaling before preparing
the split, using only the history that belongs to training. Built-in generators
return raw data by default. An explicit `normalize="standard"` or
`normalize="minmax"` acts on the entire returned window; keep it disabled
when fitting scaling separately on a training prefix.

Report the normalized horizon alongside the threshold, data scaling, and
evaluation conditions. Lyapunov-time conversion removes the arbitrary step
unit; it does not make different forecasting tasks identical.

## The reservoir's own exponent

[`reservoir_lyapunov`][resdag.metrics.reservoir_lyapunov] estimates the maximal
Lyapunov exponent of the *trained reservoir's own* driven dynamics — a distinct
quantity from the attractor's $\lambda_\mathrm{max}$ above. It runs Benettin's
algorithm with QR reorthonormalisation of the analytic tangent map along an
input-driven trajectory, using Torch on a local CPU float64 copy with no
optional dependency. It preserves the caller's weights, device, cached state
and training/evaluation mode.

It is the standard **edge-of-chaos** diagnostic. The Echo State Property
requires trajectories from different initial states to converge under the same
drive. A negative measured exponent supports contraction along the sampled
trajectory; it does not establish the global ESP for every input. A positive
value indicates local sensitivity and motivates further stability checks.
The returned exponent is in inverse-**step** units (per reservoir update);
multiply by $1/dt$ to compare it with a natural-time attractor exponent.

```python
lam = reservoir_lyapunov(model, f_warmup, horizon=500)   # scalar, k=1
spectrum = reservoir_lyapunov(model, f_warmup, horizon=500, k=4)   # top-4 as (4,) array
```

Pass a single `ESNLayer` or a model containing exactly one ESN reservoir; the
reservoir must use the standard tanh `ESNCell`. The diagnostic includes scalar
or per-neuron leakage, recurrent calibration gains and intrinsic gains/biases.
Alternative cell update equations raise an explicit error until their tangent
maps are validated. Training-only state noise is disabled during measurement.
Inputs are used for an initial warmup, then repeated for `transient + horizon`
additional steps; `transient` growth is discarded and exactly `horizon` steps
are averaged. Repeating a short drive can alter the measured dynamics, so use
a representative input window.

## Arbitrary systems — the `dynamics` extra

For systems that are *not* in the canonical tables,
[`largest_lyapunov`][resdag.metrics.largest_lyapunov] dispatches on its
argument:

- a **generator name** (`"lorenz"`, `"rossler"`, `"mackey_glass"`) returns the
  canonical constant — **no dependency**;
- a **measured series** gives a noisy, data-driven diagnostic estimate (opt in
  with `from_="data"`, and pass `dt`);
- a **`tsdynamics` System** computes a numerical Benettin spectrum estimate;
  check convergence as you increase its averaging time.

```python
from resdag.metrics import largest_lyapunov

largest_lyapunov("lorenz")                 # 0.9056  — canonical, no dependency
largest_lyapunov("mackey_glass", tau=17)   # 0.0086  — canonical, no dependency
```

!!! note "Optional dependency: `resdag[dynamics]`"
    The series- and System-based paths of `largest_lyapunov` require the
    optional [`tsdynamics`](https://pypi.org/project/tsdynamics/) package, which
    needs **Python ≥ 3.12**:

    ```bash
    pip install "resdag[dynamics]"   # or: uv sync --extra dynamics
    ```

    Every `tsdynamics` import is lazy and guarded, so this extra is only needed
    to estimate an exponent for an arbitrary system. The Lyapunov-time
    conversions, the canonical constants, `valid_prediction_time`, and
    `reservoir_lyapunov` all work with **no extra installed**.

## Memory capacity and effective spectral radius

These diagnose different aspects of a reservoir; evaluate autonomous forecast
accuracy separately on chronological held-out data.

```python
from resdag import ESNLayer
from resdag.metrics import memory_capacity, effective_spectral_radius

scalar_reservoir = ESNLayer(40, feedback_size=1, bias=False, seed=7)
capacity, curve = memory_capacity(
    scalar_reservoir, max_delay=80, n_samples=2000,
    test_samples=4000, seed=3, return_curve=True,
)

# Use consecutive states from each trajectory: leakage inversion needs neighbours.
representative = torch.sin(torch.arange(300).float()).reshape(1, -1, 1)
scalar_reservoir.reset_state(batch_size=1)
states = scalar_reservoir(representative)
radius = effective_spectral_radius(scalar_reservoir, states, max_states=64)
```

Memory readouts fit one random input stream and score a separate, independently
warmed stream. Ridge fitting uses an augmented least-squares solve, so
correlated states do not lose the regularizer through normal equations.
Increasing `test_samples` and repeating seeds helps assess
sampling noise. The theoretical population capacity bound is the state count;
a finite-sample sum of squared correlations can exceed it. Capacity is not
guaranteed to increase with spectral radius. Earlier versions scored fitting
samples and could substantially overstate capacity; old numbers must be
remeasured.

Effective radius averages local Jacobian eigenvalue radii. It includes the same
calibration and per-neuron leakage as the actual standard tanh ESN update and
preserves model state and mode. It does not bound transient amplification for
non-normal matrices and is not a global stability guarantee.

## Next

- [**Forecast**](forecast.md) — the two-phase loop these horizons measure, and
  how far you can push it
- [Theory · Timing](../theory/timing.md) — the one-step-ahead contract and the
  forecast index map
- [Reference · Metrics](../reference/metrics.md) — the full `resdag.metrics` API
