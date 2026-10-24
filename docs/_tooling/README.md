# Documentation tooling

This directory is excluded from the published site. The existing theme and
figure generators are the canonical documentation tools; `scripts/build_docs_assets.py`
invokes the same pipeline.

| File | Purpose |
| --- | --- |
| `theme.json` | Colors and fonts shared by the site and figures |
| `apply_theme.py` | Render theme tokens into `docs/css/notebook.css` |
| `figures/_style.py` | Shared Matplotlib style and image output |
| `figures/forecast_data.py` | Validate fixed observations, predictions and display-case identities |
| `figures/forecast.py` | Full recorded Lorenz component traces and the 0.3-error valid prefix |
| `figures/phase.py` | Recorded Lorenz phase portraits through TSDynamics |
| `figures/hero_forecasts.py` | Recorded Lorenz/Rössler movies and posters through TSDynamics |
| `figures/hpo.py` | Standard Optuna plots reconstructed from recorded validation trials |
| `figures/topologies.py`, `figures/initializers.py` | Per-registry-entry connectivity and weight portraits |
| `figures/architectures.py`, `figures/readme.py` | DAG diagrams through `plot_model` |
| `figures/make_all.py` | Run the canonical figure generators |

## Regenerate figures

Use an environment with ResDAG's `docs`, `dynamics`, `hpo` and `viz` extras.
Graphviz renders SVG diagrams; FFmpeg writes MP4 recordings. From the repository
root:

```sh
uv run python docs/_tooling/apply_theme.py             # CSS tokens only
uv run python docs/_tooling/apply_theme.py --figures   # tokens and static figures
uv run python scripts/build_docs_assets.py            # the same static figures
uv run python scripts/build_docs_assets.py --animations
```

Individual generators can also run directly, for example:

```sh
uv run python docs/_tooling/figures/forecast.py
uv run python docs/_tooling/figures/hero_forecasts.py
```

A normal `mkdocs build --strict` consumes committed images and recordings. It
does not run these generators or require FFmpeg on the documentation builder.
Review rendered figures, captions and code together before committing assets.
After the strict build, check actual output links and raw HTML media paths too:

```sh
uv run mkdocs build --strict
uv run python tools/check_docs_site.py site --base-path /ResDAG/
```

The local check validates HTML anchors, image sources, video posters and source
files. It rejects duplicate anchors and unrendered reStructuredText roles or
directives in prose, while allowing literal examples and displayed source.
It does not contact external sites or run JavaScript. Theme overrides
are source templates, not standalone documentation pages, and are excluded from
the published site.

## Forecast provenance

Forecast plots and recordings read
`docs/assets/data/forecast-showcase-2026-10-24.json`. Each system uses final
trajectory 0, reservoir seed 7849201 and forecast start 0, chosen before final
results. The file stores original-unit observations and predictions, physical
timestamps, training-only error scales, the selected recipe and prediction
hashes. Its linked study records validation and final evaluation separately.

Rebuilding a figure never integrates an ODE, trains a model, searches for a
better seed or truncates a divergent path. HPO charts reconstruct plain Optuna
trials with their recorded distributions, values and complete/failed states.
They use validation scores only. Replacing a study requires a new recorded
experiment and an explicit display policy; do not edit arrays to improve a plot.
