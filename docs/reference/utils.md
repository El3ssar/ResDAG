---
description: API reference for resdag.utils — data loading and splitting, random number generation, and the ESP index.
---

<span class="nb-kicker">Reference</span>

# Utilities

Data loading and ESN-shaped splitting, seeded RNG construction, and the
echo-state-property diagnostic. `load_file` and `prepare_esn_data` are
re-exported at `resdag.utils` for convenience; their canonical home is
`resdag.data`. The old `resdag.utils.data` namespace remains a compatibility
import. The ESP index, `esp_index`, is likewise re-exported at
`resdag.utils` and at the top level, so all three of these work:

```python
from resdag import esp_index
from resdag.utils import esp_index
from resdag.utils.states import esp_index  # canonical home
```

::: resdag.utils
    options:
      members:
        - create_rng
        - create_torch_generator
        - coerce_seed_to_int

---

::: resdag.data
    options:
      members:
        - load_file
        - load_csv
        - load_npy
        - load_npz
        - load_nc
        - save_csv
        - save_npy
        - save_npz
        - save_nc
        - list_files
        - prepare_esn_data
        - normalize_data
        - load_and_prepare

---

::: resdag.utils.states
    options:
      members: false

::: resdag.utils.states.esp_index.esp_index
    options:
      heading_level: 3

---

## DataLoader path

`TimeSeriesWindowDataset` and `make_dataloader` (both in `resdag.data`, and
re-exported at the top level) are the windowed-`DataLoader` training path — see
[Work · Streaming & DataLoaders](../workflows/streaming.md) for the worked
loops. The time-series generators in `resdag.datasets` (`lorenz`, `rossler`,
`henon`, `mackey_glass`, `narma`, `sine`) each return a `(1, n_timesteps,
features)` tensor and take `n_timesteps` as their first positional argument.

::: resdag.data
    options:
      members:
        - TimeSeriesWindowDataset
        - make_dataloader
