---
description: Install ResDAG with pip or uv, with optional HPO extras, and set expectations for GPU use.
---

<span class="nb-kicker">Start · 01</span>

# Install

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

Requirements: Python ≥ 3.11 and PyTorch ≥ 2.10. Verify:

```bash
python -c "import resdag; print(resdag.__version__)"
```

## On GPUs

Everything runs on CUDA with a single `.to("cuda")` — models, data, training,
forecasting. Two honest expectations:

- **GPUs win at scale.** Reservoirs of ~2 000+ units or batches of multiple
  trajectories train and forecast up to an order of magnitude faster than
  CPU. Run `examples/11_gpu_benchmark.py` to see the crossover on your
  hardware.
- **Tiny models don't benefit.** A single trajectory through a few hundred
  neurons is bound by kernel-launch overhead, not arithmetic — expect rough
  parity with the CPU there, and don't read it as a defect.

[Scale & deploy](../workflows/deploy.md) covers device placement patterns in
detail.

## Development install

```bash
git clone https://github.com/El3ssar/resdag && cd resdag
uv sync --extra dev
uv run pytest --no-cov -q
```

## Next

[**02 · First forecast**](first-forecast.md) — a trained chaotic-attractor
forecaster in five minutes.
