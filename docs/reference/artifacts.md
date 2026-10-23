---
description: Complete model artifacts, generic loading and custom component codecs.
---

<span class="nb-kicker">Reference</span>

# Model artifacts

Start with [save, load and diagrams](../workflows/save-load.md).
The common `Model.save` / `ESN.save` methods select complete checkpoints or
image formats by suffix. `save_model` also accepts supported raw graph families.

::: resdag.artifacts
    options:
      members:
        - load
        - save_model
        - register_component
        - ComponentCodec
        - ArtifactError
