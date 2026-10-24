---
description: API reference for resdag.metrics — valid prediction time and Lyapunov-time bookkeeping for chaotic-system forecasts.
---

<span class="nb-kicker">Reference</span>

# Metrics

Forecast-quality and dynamical metrics for reservoir models, with first-class
support for reporting horizons in **Lyapunov times** — the natural clock of a
chaotic attractor — rather than raw integration steps. A valid-prediction time
in Lyapunov times ($\text{VPT}_{\Lambda t} = \text{steps} \cdot dt \cdot
\lambda_\mathrm{max}$) expresses a physical horizon relative to the system's
instability rate. Comparisons still require matching error definitions,
thresholds, normalization, forecast origins and censoring conventions. Supply
the actual sampling interval and a suitable exponent when changing system
parameters; canonical table values are reference values for their stated systems.

The conversions, the canonical $\lambda_\mathrm{max}$ tables, `valid_prediction_time`,
and `reservoir_lyapunov` are pure PyTorch/NumPy and always available. The
series- and System-based paths of `largest_lyapunov` need the optional
`resdag[dynamics]` extra (`tsdynamics`, Python ≥ 3.12) — see
[Work · Lyapunov time](../workflows/lyapunov.md) for the prose walkthrough.

## Forecast accuracy

`valid_prediction_time` counts the contiguous prefix strictly below its
threshold. Its NRMSE scales come from the supplied truth tensor, and per-step
batch errors are reduced with the median. The common tuning workflow instead
uses fixed training scales; record that distinction when comparing results.
`rmse`, `nrmse` and `r2` return aggregate scalar errors rather than a valid prefix.

::: resdag.metrics.valid_prediction_time

::: resdag.metrics.rmse

::: resdag.metrics.nrmse

::: resdag.metrics.r2

---

## Lyapunov-time conversions

::: resdag.metrics.lyapunov_time

::: resdag.metrics.steps_per_lyapunov_time

::: resdag.metrics.steps_to_lyapunov

::: resdag.metrics.lyapunov_to_steps

---

## Canonical exponents

::: resdag.metrics.CANONICAL_LAMBDA_MAX

::: resdag.metrics.MACKEY_GLASS_LAMBDA

---

## Estimators

::: resdag.metrics.largest_lyapunov

::: resdag.metrics.reservoir_lyapunov

## Reservoir and attractor observables

::: resdag.metrics.memory_capacity

::: resdag.metrics.effective_spectral_radius

::: resdag.metrics.correlation_dimension
