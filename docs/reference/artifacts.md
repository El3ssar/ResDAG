---
description: Complete model artifacts, generic loading and custom component codecs.
---

<span class="nb-kicker">Reference</span>

# Model artifacts

Start with [save, load and diagrams](../workflows/save-load.md).
The common `Model.save` / `ESN.save` methods select complete checkpoints or
image formats by suffix. `save_model` also accepts supported raw graph families.

Portable artifacts use registered declarative configuration and tensor state.
Loading defaults to Torch's restricted loader (`weights_only=True`) and does
not import Python paths selected by the file. `register_component` and
`custom_objects` supply application-trusted `ComponentCodec` constructors;
register custom component types explicitly before portable loading.

An optional Python bundle requires `trusted=True` at both save and load. It
can execute Python when loaded and needs compatible Python/package versions.
Restricted loading reduces code-execution exposure but does not provide a
resource sandbox for malformed or oversized files.

`include_state=True` retains recurrent memory. Setting it to `False` resets
ordinary recurrent memory on reload while retaining fitted preprocessing and
observed forecast history. Artifacts do not retain an autonomous forecast
cursor, optimizer or autograd graph. `return_metadata=True` returns
`(model, metadata)` from `load`; ordinary `state_dict` files still require a
matching model and `load_state_dict`.

::: resdag.artifacts
    options:
      members:
        - load
        - save_model
        - register_component
        - ComponentCodec
        - ArtifactError
