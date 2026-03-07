---
template: home.html
title: resdag — Reservoir Computing for PyTorch
hide:
  - navigation
  - toc
---

<div class="rd-stats">
  <div>
    <span class="rd-stat__value">17</span>
    <span class="rd-stat__label">Graph Topologies</span>
  </div>
  <div>
    <span class="rd-stat__value">11</span>
    <span class="rd-stat__label">Weight Initializers</span>
  </div>
  <div>
    <span class="rd-stat__value">5</span>
    <span class="rd-stat__label">HPO Loss Functions</span>
  </div>
  <div>
    <span class="rd-stat__value">GPU</span>
    <span class="rd-stat__label">Accelerated</span>
  </div>
  <div>
    <span class="rd-stat__value">v0.3</span>
    <span class="rd-stat__label">Current Version</span>
  </div>
</div>

---

## Why resdag?

<div class="rd-features">

<div class="rd-feature-card">
<span class="rd-feature-card__icon">⚡</span>
<p class="rd-feature-card__title">GPU-Native</p>
<p class="rd-feature-card__body">Full GPU support throughout — reservoir dynamics, readout fitting, and forecasting all run on-device without CPU round-trips.</p>
</div>

<div class="rd-feature-card">
<span class="rd-feature-card__icon">🧩</span>
<p class="rd-feature-card__title">Modular by Design</p>
<p class="rd-feature-card__body">Cells, layers, readouts, and topologies compose cleanly via <code>pytorch_symbolic</code>'s functional API — any DAG you can draw, you can build.</p>
</div>

<div class="rd-feature-card">
<span class="rd-feature-card__icon">🔬</span>
<p class="rd-feature-card__title">Algebraic Training</p>
<p class="rd-feature-card__body">Readouts are fitted with Conjugate Gradient ridge regression — exact, fast, and immune to learning-rate tuning. No SGD needed.</p>
</div>

<div class="rd-feature-card">
<span class="rd-feature-card__icon">🌐</span>
<p class="rd-feature-card__title">Rich Topology Library</p>
<p class="rd-feature-card__body">17 graph topologies (Erdős–Rényi, Watts–Strogatz, Barabási–Albert, dendrocycles…) plus an extensible registry for custom graphs.</p>
</div>

<div class="rd-feature-card">
<span class="rd-feature-card__icon">🚀</span>
<p class="rd-feature-card__title">NG-RC Support</p>
<p class="rd-feature-card__body">Next Generation Reservoir Computing with delay-embedded polynomial features — no weights, pure math, state-of-the-art on low-dimensional chaos.</p>
</div>

<div class="rd-feature-card">
<span class="rd-feature-card__icon">🎛️</span>
<p class="rd-feature-card__title">HPO Integration</p>
<p class="rd-feature-card__body">Plug-and-play Optuna HPO with 5 specialist loss functions including Expected Forecast Horizon — optimized for chaotic time series.</p>
</div>

</div>

---

## The resdag Workflow

<div class="rd-pipeline">

<div class="rd-pipeline__step">
  <div class="rd-pipeline__indicator">
    <div class="rd-pipeline__dot">1</div>
    <div class="rd-pipeline__line"></div>
  </div>
  <div class="rd-pipeline__content">
    <h4>Define Architecture</h4>
    <p>Compose reservoir layers, readouts, and custom transformations into any DAG using the <code>pytorch_symbolic</code> functional API or use a premade factory model.</p>
  </div>
</div>

<div class="rd-pipeline__step">
  <div class="rd-pipeline__indicator">
    <div class="rd-pipeline__dot">2</div>
    <div class="rd-pipeline__line"></div>
  </div>
  <div class="rd-pipeline__content">
    <h4>Warmup &amp; Train</h4>
    <p><code>ESNTrainer.fit()</code> synchronizes reservoir states with a teacher-forced warmup pass, then fits all readouts algebraically in a single forward pass.</p>
  </div>
</div>

<div class="rd-pipeline__step">
  <div class="rd-pipeline__indicator">
    <div class="rd-pipeline__dot">3</div>
    <div class="rd-pipeline__line"></div>
  </div>
  <div class="rd-pipeline__content">
    <h4>Forecast</h4>
    <p><code>model.forecast()</code> drives the reservoir with warmup data, then rolls out autonomously (or with known future drivers) for any horizon.</p>
  </div>
</div>

<div class="rd-pipeline__step">
  <div class="rd-pipeline__indicator">
    <div class="rd-pipeline__dot">4</div>
  </div>
  <div class="rd-pipeline__content">
    <h4>Optimize (optional)</h4>
    <p><code>run_hpo()</code> wraps the whole pipeline in an Optuna study, parallelizes across workers, and selects hyperparameters by forecast horizon or Lyapunov exponent.</p>
  </div>
</div>

</div>

---

## 30-Second Example

```python
import torch
import pytorch_symbolic as ps
from resdag import ESNModel, ESNLayer, CGReadoutLayer, ESNTrainer

# --- 1. Build model ---
inp = ps.Input((100, 3))                          # (seq_len, features)
reservoir = ESNLayer(
    reservoir_size=500,
    feedback_size=3,
    spectral_radius=0.9,
    topology="erdos_renyi",
)(inp)
readout = CGReadoutLayer(500, 3, alpha=1e-6, name="output")(reservoir)
model = ESNModel(inp, readout)

# --- 2. Train (algebraic — no SGD!) ---
trainer = ESNTrainer(model)
trainer.fit(
    warmup_inputs=(warmup_data,),
    train_inputs=(train_data,),
    targets={"output": train_targets},
)

# --- 3. Forecast ---
predictions = model.forecast(forecast_warmup, horizon=1000)
# shape: (batch, 1000, 3)
```

Or use a [premade model](guide/premade-models.md) in one line:

```python
from resdag.models import ott_esn

model = ott_esn(reservoir_size=500, feedback_size=3, output_size=3)
```

---

## Quick Links

| | |
|---|---|
| [Installation Guide](getting-started/installation.md) | pip, uv, optional extras |
| [5-Minute Quickstart](getting-started/quickstart.md) | Full working example |
| [Core Concepts](getting-started/concepts.md) | Reservoir computing theory |
| [ESN Layer Reference](guide/esn-layer.md) | All parameters explained |
| [NG-RC](guide/ngrc.md) | Next Generation RC |
| [Topologies](guide/topologies.md) | All 17 graph types |
| [HPO Guide](hpo/overview.md) | Hyperparameter optimization |
| [API Reference](api/index.md) | Full auto-generated API |
