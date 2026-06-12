---
description: API reference for resdag.models and resdag.ensemble — premade architectures, coupled ensembles, and aggregators.
---

<span class="nb-kicker">Reference</span>

# Models & ensembles

Five factory functions that return ready-to-use `ESNModel` instances
(two of them — `linear_esn` and `headless_esn` — deliberately ship without
a readout), plus `coupled_ensemble_esn`, which builds a
`CoupledEnsembleESNModel`: the coupled-ensemble wrapper that runs N
sub-models against a shared feedback signal, with the aggregators that
merge their forecasts.

::: resdag.models
    options:
      members: false

::: resdag.models.classic_esn.classic_esn
    options:
      heading_level: 3

::: resdag.models.ott_esn.ott_esn
    options:
      heading_level: 3

::: resdag.models.power_augmented.power_augmented
    options:
      heading_level: 3

::: resdag.models.linear_esn.linear_esn
    options:
      heading_level: 3

::: resdag.models.headless_esn.headless_esn
    options:
      heading_level: 3

::: resdag.models.coupled_ensemble_esn.coupled_ensemble_esn
    options:
      heading_level: 3

---

::: resdag.ensemble
    options:
      members:
        - CoupledEnsembleESNModel

::: resdag.ensemble.aggregators
    options:
      members:
        - OutliersFilteredMean
