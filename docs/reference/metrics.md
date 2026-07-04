---
description: API reference for resdag.metrics — valid prediction time and Lyapunov-time bookkeeping for chaotic-system forecasts.
---

<span class="nb-kicker">Reference</span>

# Metrics

Forecast-quality and dynamical metrics for reservoir models, with first-class
support for reporting horizons in **Lyapunov times** — the natural clock of a
chaotic attractor — rather than raw integration steps. A valid-prediction time
in Lyapunov times ($\text{VPT}_{\Lambda t} = \text{steps} \cdot dt \cdot
\lambda_\mathrm{max}$) is hardware- and integration-step-independent, so it is
comparable across systems and papers.

The conversions, the canonical $\lambda_\mathrm{max}$ tables, `valid_prediction_time`,
and `reservoir_lyapunov` are pure PyTorch/NumPy and always available. The
series- and System-based paths of `largest_lyapunov` need the optional
`resdag[dynamics]` extra (`tsdynamics`, Python ≥ 3.12) — see
[Work · Lyapunov time](../workflows/lyapunov.md) for the prose walkthrough.

## Forecast accuracy

::: resdag.metrics
    options:
      members:
        - valid_prediction_time

---

## Lyapunov-time conversions

::: resdag.metrics
    options:
      members:
        - lyapunov_time
        - steps_per_lyapunov_time
        - steps_to_lyapunov
        - lyapunov_to_steps

---

## Canonical exponents

::: resdag.metrics
    options:
      members:
        - CANONICAL_LAMBDA_MAX
        - MACKEY_GLASS_LAMBDA

---

## Estimators

::: resdag.metrics
    options:
      members:
        - largest_lyapunov
        - reservoir_lyapunov
