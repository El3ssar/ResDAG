# Extend ResDAG

Every major subsystem uses a **registry**: implement a small interface, decorate once,
use by name in layer constructors or HPO.

<div class="grid cards" markdown>

-   [Custom topology](custom-topology.md) — `@register_graph_topology`
-   [Custom initializer](custom-initializer.md) — `@register_input_feedback`
-   [Custom cell](custom-cell.md) — `ReservoirCell` + `BaseReservoirLayer`
-   [Custom readout](custom-readout.md) — `ReadoutLayer._fit_impl`
-   [Custom loss](custom-loss.md) — `LossProtocol` for `run_hpo`
-   [Custom model](custom-model.md) — premade factory pattern
-   [Custom aggregator](custom-aggregator.md) — ensemble `(N,B,T,F)` modules


## Testing your extension

```python
# After decorating, import the package submodule so registration runs
import resdag.init.graphs.my_topology  # noqa: F401

from resdag.init.topology import show_topologies
assert "my_topology" in show_topologies()
```

Run `pytest` under `tests/` mirroring the subsystem you touched.
