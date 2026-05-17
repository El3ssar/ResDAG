# Get started

Welcome to ResDAG. This section takes you from installation to a working forecast
in a few short pages.

## What you should know first

- Basic **PyTorch** (`torch.Tensor`, `nn.Module`, shapes `(batch, time, features)`).
- The idea that a reservoir is a **fixed random recurrent map** and only the
  **readout** is trained.
- Optional: skimming [Mental model](mental-model.md) before coding.

## Path through this section

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    `pip install resdag`, optional `[hpo]` and `[docs]` extras, GPU notes.

    [:octicons-arrow-right-24: Install](installation.md)

-   :material-brain:{ .lg .middle } **Mental model**

    ---

    Ten-minute picture of reservoir computing and two-phase training.

    [:octicons-arrow-right-24: Read](mental-model.md)

-   :material-code-tags:{ .lg .middle } **Your first ESN**

    ---

    Build, train, and forecast a sine wave in ~30 lines.

    [:octicons-arrow-right-24: Code](your-first-esn.md)

-   :material-chart-timeline-variant:{ .lg .middle } **Lorenz walkthrough**

    ---

    End-to-end chaotic forecast with `ott_esn` and `ESNTrainer`.

    [:octicons-arrow-right-24: Walkthrough](lorenz-walkthrough.md)


## Next steps

After this section, continue with [Learn](../learn/index.md) for theory or
[Guides](../guides/index.md) for task recipes. API details live in
[Reference](../reference/index.md).
