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

Move the model and its floating inputs to CUDA with `.to("cuda")`. Forward
calls preserve gradients and device placement inside larger Torch networks.
Readout fitting may use a reported CPU fallback when the selected solver cannot
complete a safe solve on the accelerator; configure its solver policy explicitly
when that fallback is unsuitable.

Speed depends on reservoir width, batch size, sequence length, sparsity,
precision and the surrounding network. Small sequential forecasts can spend
more time launching kernels than multiplying matrices. Larger batches can
amortize that overhead, but there is no universal reservoir-size crossover or
guaranteed speedup. Run `examples/11_gpu_benchmark.py` on the target hardware,
and measure both startup and repeated execution. CUDA throughput is not
established by the current CPU/MPS correctness checks.

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
