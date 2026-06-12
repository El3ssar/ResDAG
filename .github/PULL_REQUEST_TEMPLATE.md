<!--
PR title must be a conventional commit — it becomes the squash commit and
decides the release: feat -> minor, fix/perf -> patch, docs/chore/ci/
refactor/test -> no release. CI rejects non-conventional titles.
-->

## What

<!-- What changes, and why. Link the issue if one exists. -->

## Checklist

- [ ] Conventional PR title (`feat:`, `fix:`, `docs:`, ...)
- [ ] `uv run ruff check src/ tests/ && uv run black --check src/ tests/ && uv run pytest --no-cov -q` pass
- [ ] New code has tests in the mirror location under `tests/`
- [ ] New components have docs (registered topologies/initializers self-document; everything else: one file in the matching `docs/build/` folder)
- [ ] `uv run mkdocs build --strict` passes if docs were touched
