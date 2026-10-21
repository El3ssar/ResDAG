---
description: Install ResDAG with pip or uv, with optional HPO extras, and set expectations for GPU use.
---

<span class="nb-kicker">Start · 01</span>

# Install

!!! note "Documentation version"
    These docs are built from the repository source, version
    **{{ resdag_version }}**. See [GitHub Releases](https://github.com/El3ssar/ResDAG/releases)
    for published versions. To use the latest development API, install from git:
    `pip install "git+https://github.com/El3ssar/ResDAG"`.

=== "uv"

    ```bash
    uv add resdag            # core
    uv add "resdag[hpo]"     # + Optuna hyperparameter optimization
    ```

=== "pip"

    ```bash
    pip install resdag            # core
    pip install "resdag[hpo]"     # + Optuna hyperparameter optimization
    ```

Requirements: Python 3.11–3.14 and PyTorch ≥ 2.10. Verify:

```bash
python -c "import resdag; print(resdag.__version__)"
```

## On GPUs

Models, data, training, and forecasting all run on CUDA via `.to("cuda")`.
Performance depends on scale:

- **Large reservoirs and batches.** Reservoirs of ~2,000+ units or batches of
  multiple trajectories train and forecast up to an order of magnitude faster
  than on CPU. Run `examples/11_gpu_benchmark.py` to measure the crossover
  point on your hardware.
- **Small models.** A single trajectory through a few hundred neurons is
  bound by kernel-launch overhead rather than arithmetic, so expect
  performance comparable to CPU.

[Scale & deploy](../workflows/deploy.md) covers device placement patterns in
detail.

## Development install

```bash
git clone https://github.com/El3ssar/ResDAG && cd ResDAG
uv sync --extra dev
uv run pytest --no-cov -q
```

## Next

[**02 · First forecast**](first-forecast.md) — train a forecaster on a
chaotic attractor and run an autoregressive forecast.
