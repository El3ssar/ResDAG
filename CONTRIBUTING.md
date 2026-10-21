# Contributing to ResDAG

ResDAG is a PyTorch-native reservoir computing library. Contributions are
welcome — components, fixes, docs, benchmarks. This guide covers the
mechanics; the design rationale lives in the
[documentation](https://el3ssar.github.io/ResDAG/).

> **Looking for something to work on?** The roadmap is tracked as a
> dependency-ordered set of issues, grouped under the `type:epic` tracking
> issues and organized by the `pillar:*` labels (correctness, speed, api,
> pipeline, hpo). Pick any issue labelled `status:ready`.

## Development setup

```bash
git clone https://github.com/El3ssar/ResDAG && cd ResDAG
uv sync --locked --extra dev # lint/format/test toolchain from the committed lock
uv run pytest --no-cov -q    # full suite, including golden forecast regressions
```

`pip install -e ".[dev]"` works too if you don't use uv. Add `hpo` (Optuna) or
`docs` (MkDocs + the figure pipeline) as needed — e.g. `uv sync --locked --extra dev
--extra docs` or `pip install -e ".[dev,docs]"`.

## Quality gate

Run before pushing:

```bash
uv run ruff check src/ tests/ tools/ci_environment.py
uv run black --check src/ tests/ tools/ci_environment.py
uv run mypy src/resdag/
uv run pytest --no-cov -q          # full suite
```

### Type checking

CI type-checks the **whole package** — `mypy src/resdag/`, not a single file —
and the tree is kept at **zero errors**. There is deliberately **no per-module
baseline and no `disable_error_code` escape hatch** in `pyproject.toml`: new and
changed code is gated immediately, so a typing error is a failing build, not a
warning. The only standing allowance is `ignore_missing_imports` (for untyped
third-party libraries such as `pytorch_symbolic`).

If your change surfaces a type error — even one that looks pre-existing — fix it
in your PR with a real annotation/narrowing rather than suppressing it. Reach
for `cast`/`assert` only when a runtime invariant genuinely guarantees the
narrowing (and say so in a short comment); never use them, or a blanket
`# type: ignore`, to paper over a possible-`None`/wrong-type bug. If you must
ignore, use a specific `# type: ignore[code]` with a one-line justification.

### Fast local iteration — run only the affected tests

The full suite takes a few minutes. While iterating, run only the tests that
your change can affect, computed from a static first-party import graph:

```bash
# What would run for your branch vs. main, and why:
uv run python tools/affected_tests.py --explain

# Run exactly that subset:
uv run pytest --no-cov -q $(uv run python tools/affected_tests.py --format args)
```

The selector errs toward running *more* tests, never fewer (broad-impact
changes such as `pyproject.toml`, `conftest.py`, or a package `__init__` fan
out to the whole suite). Still do one full `pytest` run before you push.

### Selective CI

Pull-request CI runs the **same** selector and tests only the affected subset,
so a typical single-module PR finishes in seconds instead of minutes. The full
suite is the safety net and still runs on every merge to `main` and on a nightly
schedule. To force a full run on a PR, add the **`ci-full`** label.

### Reproducible environments and dependency updates

Pull-request, nightly, release, benchmark, and documentation gates install the
versions in `uv.lock`, including development tools and the editable-build
backend. CI checks that the lock matches `pyproject.toml` before installing.
CPU jobs use the matching PyTorch CPU wheel and omit its CUDA-only dependency
tree. Python 3.11 through 3.14 all run against this baseline.

To reproduce that environment in a fresh virtual environment:

```bash
python -m pip install uv==0.13.3
python tools/ci_environment.py --mode locked --extras dev,hpo
```

Use `--extras docs` for the documentation environment; development tools are
always included. The installer prints the resolved packages and runs
`pip check` to catch missing or conflicting dependencies.

The separate **Latest dependency compatibility** workflow runs the full test
matrix and a strict documentation build every Wednesday and on manual dispatch.
It resolves current packages within the declared dependency ranges, retaining
the CPU PyTorch wheel. Its failures reveal upstream compatibility changes
without silently changing the normal PR baseline. Reproduce it with
`python tools/ci_environment.py --mode latest --extras dev,hpo`.
The existing **CI** and **Documentation** workflows also accept a
`dependency-mode` choice on manual dispatch, so either environment can be
verified on a branch before merge. Pull requests always use the locked baseline.

When changing dependencies, update `pyproject.toml`, run `uv lock`, and commit
both files. For an intentional upgrade, prefer `uv lock --upgrade-package NAME`
and review the resulting lock diff. Run the full suite and strict docs build
before proposing the new baseline.

### Test markers and GPU checks

The test suite mirrors `src/resdag/`. Markers:

- `gpu` — CUDA variants, auto-skip on machines without a GPU
- `benchmark` — performance assertions, deselected by default
  (`pytest -m benchmark` to run; they assert the GPU beats the CPU at scale)

The weekly GPU workflow requires a self-hosted runner carrying the `gpu` label.
After provisioning it, set the repository Actions variable
`RESDAG_GPU_RUNNER_ENABLED=true` to enable scheduled GPU runs. Manual dispatch
works independently of that variable. A skipped schedule is not CUDA
verification; only a completed run on a CUDA runner provides that evidence.

### Golden-forecast regression fixtures

`tests/test_models/test_golden_forecast.py` pins the behaviour of
`ESNModel.forecast` for each premade model against committed *golden*
trajectories under `tests/fixtures/golden_forecast/` — a fully-seeded
warmup → forecast on Lorenz-63, in CPU float64. It fails on any drift in the
forecast path. If you change that path — or the canonical `lorenz` /
`prepare_esn_data` generators it builds on — regenerate the fixtures and review
the diff before committing:

```bash
uv run python tools/regen_golden_forecasts.py --check   # verify only (drift check, non-zero exit on drift)
uv run python tools/regen_golden_forecasts.py --all      # regenerate every fixture
```

The model specs, data splits, metric, and tolerances live in
`tests/test_models/golden_forecast.py`, shared by the test and the tool so they
can never disagree.

## Commit messages and releases

> **Releases are live.** Every push to `main` whose commits since the last tag
> warrant a release publishes automatically (version bump, tag, GitHub release,
> PyPI via trusted publishing). **Never rename `release.yml`** — PyPI trusted
> publishing is bound to that filename. The mechanics below describe what runs.

Releases are fully automated from **conventional commits** — there are no
manual tags or version edits. On every push to `main`,
python-semantic-release parses the commits since the last release:

| prefix | effect |
| --- | --- |
| `feat: ...` | minor release (0.5.0 → 0.6.0) |
| `fix: ...`, `perf: ...` | patch release (0.5.0 → 0.5.1) |
| `feat!: ...` or `BREAKING CHANGE:` footer | major (including 0.x → 1.0; `major_on_zero=true`) |
| `docs:`, `chore:`, `ci:`, `refactor:`, `test:`, `style:`, `build:` | no release |

A release rewrites `__version__` in `src/resdag/__init__.py` and the
mirrored `version` field in `CITATION.cff` (both are listed in
`version_variables`, so they stay in lock-step automatically), tags
`vX.Y.Z`, publishes a GitHub release with generated notes, and uploads to
PyPI. Documentation deploys on the same push, and version strings in the
docs substitute automatically.

**Pull requests are squash-merged, and the PR title becomes the commit
that decides the bump** — so PR titles must be conventional
(`feat: add Rössler topology`). CI checks the title format on every PR.
Inside your branch, commit however you like; only the squash title counts.

### Recovering a failed package publication

Treat test results, the GitHub release, and PyPI publication as separate
outcomes. If PyPI reports that a filename belonged to a deleted upload, that
version cannot be reused: leave the existing GitHub tag and artifacts intact
and advance to a fresh patch version through a `fix(release): ...` commit.
Rerunning the unchanged release cannot resolve the collision. In particular,
the first 1.x PyPI publication must skip `1.0.0`, whose filenames are reserved
by a deleted historical upload.

## Adding components

Every component type is an open set with one extension point:

| component | how | docs page |
| --- | --- | --- |
| Topology (graph) | `@register_graph_topology("name", **defaults)` on a NetworkX generator | generated automatically from the registry |
| Topology (matrix) | `@register_matrix_topology("name", **defaults)` on an `fn(n) -> matrix` | generated automatically |
| Input/feedback initializer | `@register_input_feedback("name")` on a function or `InputFeedbackInitializer` subclass | generated automatically |
| Readout | subclass `ReadoutLayer`, implement `_fit_impl(states, targets)` | one file in `docs/build/readouts/` |
| Reservoir family | implement the `ReservoirCell` contract, wrap in a `BaseReservoirLayer` subclass | one file in `docs/build/layers/` |
| Premade architecture | factory function returning a model, in `src/resdag/models/` | one file in `docs/build/architectures/` |

Registered topologies and initializers document themselves: their docs
pages (figure, parameters, usage) are generated from the registry at build
time. For the other component types, drop a Markdown file in the listed
folder — navigation and index cards pick it up automatically.

New components need tests (place them in the mirror location under
`tests/`) and, for anything with tunable structure, a figure — see below.

## Documentation

```bash
uv run mkdocs serve            # live preview at 127.0.0.1:8000/ResDAG/
uv run mkdocs build --strict   # link-checking build, run before pushing
```

- **Colors and fonts** for the entire site *and* every figure live in
  `docs/_tooling/theme.json`. After editing:
  `uv run python docs/_tooling/apply_theme.py --figures`
- **Figures** are generated by the per-category scripts in
  `docs/_tooling/figures/` — never edit images in `docs/assets/figures/`
  by hand.
- The token block in `docs/css/notebook.css` between the
  `===== TOKENS =====` markers is generated; edit `theme.json` instead.
- The landing page is a single file: `docs/overrides/home.html`.

## Pull request checklist

- [ ] Conventional PR title (it becomes the release-deciding commit)
- [ ] `ruff check`, `black --check`, and `pytest` pass locally
- [ ] New code has tests; new components have docs (see table above)
- [ ] `mkdocs build --strict` passes if you touched docs
- [ ] Figures regenerated via the scripts if you changed visuals
