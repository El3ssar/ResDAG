---
description: API reference for resdag.models and resdag.ensemble — premade architectures, coupled ensembles, and aggregators.
---

<span class="nb-kicker">Reference</span>

# Models & ensembles

Architecture factories return ready-to-use reservoir models. `headless_esn`
and `linear_esn` omit the readout; `parallel_esn` returns a `ParallelESNModel`
for spatial patching, while `coupled_ensemble_esn` builds the
`CoupledEnsembleESNModel` wrapper. The other factories return `ESNModel`
instances with a configured readout. `KnowledgeModel` adapts a physics prior
for `hybrid_esn`.

::: resdag.models
    options:
      members: false

::: resdag.models.classic_esn.classic_esn
    options:
      heading_level: 3

::: resdag.models.coupled_ensemble_esn.coupled_ensemble_esn
    options:
      heading_level: 3

::: resdag.models.deep_esn.deep_esn
    options:
      heading_level: 3

::: resdag.models.es2n.es2n
    options:
      heading_level: 3

::: resdag.models.eusn.eusn
    options:
      heading_level: 3

::: resdag.models.gated_esn.gated_esn
    options:
      heading_level: 3

::: resdag.models.headless_esn.headless_esn
    options:
      heading_level: 3

::: resdag.models.hybrid_esn.KnowledgeModel
    options:
      heading_level: 3

::: resdag.models.hybrid_esn.hybrid_esn
    options:
      heading_level: 3

::: resdag.models.linear_esn.linear_esn
    options:
      heading_level: 3

::: resdag.models.lsm.lsm
    options:
      heading_level: 3

::: resdag.models.ott_esn.ott_esn
    options:
      heading_level: 3

::: resdag.models.parallel_esn.ParallelESNModel
    options:
      heading_level: 3

::: resdag.models.parallel_esn.parallel_esn
    options:
      heading_level: 3

::: resdag.models.power_augmented.power_augmented
    options:
      heading_level: 3

::: resdag.models.resesn.resesn
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
