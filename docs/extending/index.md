# Extending resdag

resdag is designed to be easily extended. Every major component uses a **registry pattern** — you register new implementations with a decorator, import them once, and they become available throughout the library by name.

<div class="rd-features">

<div class="rd-feature-card">
<span class="rd-feature-card__icon">🌐</span>
<p class="rd-feature-card__title"><a href="new-topology/">New Topology</a></p>
<p class="rd-feature-card__body">Add a custom graph topology for reservoir recurrent weights.</p>
</div>

<div class="rd-feature-card">
<span class="rd-feature-card__icon">🎯</span>
<p class="rd-feature-card__title"><a href="new-initializer/">New Initializer</a></p>
<p class="rd-feature-card__body">Add a custom input or feedback weight initialization strategy.</p>
</div>

<div class="rd-feature-card">
<span class="rd-feature-card__icon">🧲</span>
<p class="rd-feature-card__title"><a href="new-cell/">New Cell & Layer</a></p>
<p class="rd-feature-card__body">Implement a new reservoir cell type and wrap it as a layer.</p>
</div>

<div class="rd-feature-card">
<span class="rd-feature-card__icon">🏗️</span>
<p class="rd-feature-card__title"><a href="new-model/">New Premade Model</a></p>
<p class="rd-feature-card__body">Package a common architecture as a factory function.</p>
</div>

</div>

## Extension Pattern

All extensions follow the same pattern:

1. **Create** a new file in the appropriate module directory
2. **Implement** the required interface (base class or function signature)
3. **Register** using the provided decorator
4. **Import** in the module's `__init__.py` so registration runs at import time
5. **Use** by name string anywhere in the library
