# HPO API

!!! warning "Optional dependency"
    The HPO module requires `optuna`. Install with `pip install "resdag[hpo]"`.

::: resdag.hpo.run.run_hpo
    options:
      show_root_heading: true

::: resdag.hpo.utils.get_best_params
    options:
      show_root_heading: true

::: resdag.hpo.utils.get_study_summary
    options:
      show_root_heading: true

::: resdag.hpo.utils.make_study_name
    options:
      show_root_heading: true

## Loss Functions

::: resdag.hpo.losses.expected_forecast_horizon
    options:
      show_root_heading: true

::: resdag.hpo.losses.forecast_horizon
    options:
      show_root_heading: true

::: resdag.hpo.losses.lyapunov_weighted
    options:
      show_root_heading: true

::: resdag.hpo.losses.standard_loss
    options:
      show_root_heading: true

::: resdag.hpo.losses.soft_valid_horizon
    options:
      show_root_heading: true

::: resdag.hpo.losses.get_loss
    options:
      show_root_heading: true
