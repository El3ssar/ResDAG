# Extend ResDAG

Extension points use registries or base classes:

- **[Custom topology](custom-topology.md)** — `@register_graph_topology` on a NetworkX generator.
- **[Custom initializer](custom-initializer.md)** — `@register_input_feedback` on `InputFeedbackInitializer`.
- **[Custom cell](custom-cell.md)** — `ReservoirCell` wrapped by `BaseReservoirLayer`.
- **[Custom readout](custom-readout.md)** — subclass `ReadoutLayer`, implement `_fit_impl`.
- **[Custom loss](custom-loss.md)** — callable for `run_hpo` (`LossProtocol`).
- **[Custom model](custom-model.md)** — factory returning `ESNModel`.
- **[Custom aggregator](custom-aggregator.md)** — ensemble combine `(N, B, T, F) → (B, T, F)`.

After registering a topology or initializer, import the module so the decorator runs
at load time, then verify with `show_topologies()` or `show_input_initializers()`.
