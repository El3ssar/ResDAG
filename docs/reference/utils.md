---
description: Random number generation, device selection, reproducible component seeds and the ESP index.
---

<span class="nb-kicker">Reference</span>

# Utilities

Use `rd.SeedStreams(master_seed)["branch_name"]` for stable, independent
initialization seeds. Adding or reordering named branches leaves their existing
seeds unchanged. See the [public component recipes](../advanced/custom-components.md).

Use [data and datasets](data.md) for file I/O, preparation, streaming windows
and benchmark generators. Existing convenience imports from `resdag.utils`
and `resdag.utils.data` resolve to the same canonical functions.

The ESP index, `esp_index`, is re-exported at
`resdag.utils` and at the top level, so all three of these work:

```python
from resdag import esp_index
from resdag.utils import esp_index
from resdag.utils.states import esp_index  # canonical home
```

::: resdag.utils
    options:
      members:
        - DeviceLike
        - SeedLike
        - SeedStreams
        - create_rng
        - create_torch_generator
        - coerce_seed_to_int
        - seed_everything
        - resolve_device

---

::: resdag.utils.states
    options:
      members: false

::: resdag.utils.states.esp_index.esp_index
    options:
      heading_level: 3

---

`seed_everything` explicitly seeds process-wide Python, NumPy and Torch RNGs.
Prefer local generators or `SeedStreams` when independent components should
not share or advance one global stream. `resolve_device` resolves the requested
device; it does not move model parameters or data.
