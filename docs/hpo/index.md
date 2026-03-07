# Hyperparameter Optimization

resdag includes a first-class HPO integration built on [Optuna](https://optuna.org/). The HPO module is an **optional dependency** — install it separately.

<div class="rd-features">

<div class="rd-feature-card">
<span class="rd-feature-card__icon">🎛️</span>
<p class="rd-feature-card__title"><a href="overview/">Overview</a></p>
<p class="rd-feature-card__body">The run_hpo() API, search spaces, data loaders, and study results.</p>
</div>

<div class="rd-feature-card">
<span class="rd-feature-card__icon">📊</span>
<p class="rd-feature-card__title"><a href="loss-functions/">Loss Functions</a></p>
<p class="rd-feature-card__body">EFH, Horizon, Lyapunov, Standard, and Discounted loss functions explained.</p>
</div>

<div class="rd-feature-card">
<span class="rd-feature-card__icon">⚡</span>
<p class="rd-feature-card__title"><a href="advanced/">Advanced Usage</a></p>
<p class="rd-feature-card__body">Multi-worker parallel optimization, storage backends, custom samplers.</p>
</div>

</div>

## Installation

```bash
pip install "resdag[hpo]"
# or
uv add "resdag[hpo]"
```

HPO functions are lazily imported — `from resdag.hpo import run_hpo` fails gracefully if Optuna is not installed.
