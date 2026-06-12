---
description: API reference for resdag.hpo — run_hpo, the five forecast-quality loss functions, and study utilities.
---

<span class="nb-kicker">Reference</span>

# HPO

Optuna-backed hyperparameter optimization: the `run_hpo` entry point, five
loss functions built for judging forecasts of (possibly chaotic) dynamics,
and helpers for inspecting finished studies.

!!! note "Optional dependency"
    `run_hpo` and the study utilities require `optuna`
    (`pip install "resdag[hpo]"`). The loss functions import without it.

::: resdag.hpo
    options:
      members:
        - run_hpo

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

---

::: resdag.hpo.utils
    options:
      members:
        - get_study_summary
        - make_study_name
        - get_best_params
