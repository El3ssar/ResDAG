# Documentation tooling

This folder is excluded from the built site. It holds the single source of
truth for the documentation's look, and the scripts that generate every
figure.

| file | purpose |
| --- | --- |
| `theme.json` | All colors and fonts — the website CSS and every figure read from here |
| `apply_theme.py` | Renders `theme.json` into `docs/css/notebook.css` (`--figures` also regenerates all plots) |
| `figures/_style.py` | Shared matplotlib style built from `theme.json` |
| `figures/forecast.py` | The first-forecast component plot |
| `figures/phase.py` | The Lorenz phase portraits |
| `figures/topologies.py` | Per-topology connectivity portraits (one per registry entry) |
| `figures/initializers.py` | Per-initializer matrix portraits (one per registry entry) |
| `figures/architectures.py` | Per-architecture wiring diagrams via `plot_model` |
| `figures/hpo.py` | The Optuna study figures |
| `figures/hero_forecasts.py` | Trajectory pairs for the landing-page animation |
| `figures/make_all.py` | Runs everything above |

Changing the palette is a one-line edit in `theme.json` followed by:

```bash
uv run python docs/_tooling/apply_theme.py --figures
```
