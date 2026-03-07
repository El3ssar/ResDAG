# API Reference

Auto-generated API documentation from source docstrings.

| Module | Description |
|---|---|
| [Layers](layers.md) | ESNLayer, NGReservoir, CGReadoutLayer, and custom layers |
| [Models](models.md) | Premade model factory functions |
| [Training](training.md) | ESNTrainer |
| [Initialization](init.md) | Topology and input/feedback initializer systems |
| [Composition](composition.md) | ESNModel |
| [HPO](hpo.md) | run_hpo and HPO utilities |

---

## Quick Import Reference

```python
# Core
from resdag import ESNModel, ESNLayer, NGReservoir, CGReadoutLayer, ESNTrainer

# Premade models
from resdag.models import ott_esn, classic_esn, headless_esn, linear_esn, power_augmented

# Custom layers
from resdag.layers import Concatenate, SelectiveExponentiation, OutliersFilteredMean

# Init systems
from resdag.init.topology import get_topology, show_topologies
from resdag.init.input_feedback import get_input_feedback, show_input_initializers

# HPO (requires optuna)
from resdag.hpo import run_hpo, get_best_params, get_study_summary

# Utils
from resdag.utils.data import load_file, prepare_esn_data
from resdag.utils.states import esp_index
```
