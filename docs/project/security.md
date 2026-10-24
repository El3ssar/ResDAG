---
description: Checkpoint trust boundaries and the dated dependency-audit disposition.
---

# Persistence and dependency audit

This record covers the 24 October 2026 hygiene sweep. It distinguishes repaired
library behavior from upstream dependency advisories; it is not a claim that
arbitrary model files or Python plugins can be safely executed.

## Model and data files

`Model.save` and `rd.load` use a portable DAG description and tensor state.
Installed custom classes are reconstructed through registered codecs. Loading
runs those installed constructors: register code you trust. Notebook bundles
are a separate, explicit `trusted=True` format. The
[save/load guide](../workflows/save-load.md) explains both forms.

Low-level `ESNModel`, parallel and coupled model weight checkpoints now use
`torch.load(weights_only=True)`, with no automatic unsafe fallback. Full Python
model files require `load_full(..., trusted=True)` before any file read. This
is an intentional API change. Unsupported objects, cyclic metadata and hidden
Python attributes are rejected before ordinary saves. Serialization writes a
temporary file beside the destination, flushes it and replaces the destination
only on success; a failed write preserves the previous model.

These checks have malicious-payload, tensor-attribute, nested-metadata,
fresh-process reload and failed-write regression tests. NumPy dataset loading
retains `allow_pickle=False`. A restricted loader still does not impose a
memory or CPU quota on hostile files; run externally supplied models and data
with appropriate resource isolation.

## Dependency disposition

The sweep ran `pip-audit` against the installed full development environment,
updated the affected non-numerical dependencies in `uv.lock`, and reran the
audit. The remaining report contains **two advisories for the locked Torch
2.10.0**, with no suppression. Torch, NumPy, SciPy and TSDynamics versions are
retained for reproducible scientific baselines. Ordinary pip installations can
resolve newer versions; a separate CI matrix tests current compatible packages.

Material for MkDocs now requires **9.7.7 or newer**. Its enabled search-suggestion
feature was affected by a DOM XSS flaw; the upstream patch fixes it. See the
[upstream changelog](https://squidfunk.github.io/mkdocs-material/changelog/#9.7.7)
and [maintainer advisory](https://github.com/squidfunk/mkdocs-material/security/advisories/GHSA-xvg9-69gf-fjrf).
Black, pytest and Markdown extension minima are also raised to patched releases.
The lock refresh includes certificate, HTTP, image, highlighting, template,
file-lock and filesystem dependencies. Full tests and rendered documentation
must pass with these actual versions before acceptance.

| Remaining advisory | Affected operation and decision |
| --- | --- |
| CVE-2026-4538 / PYSEC-2026-139 | The upstream report concerns Python unpickling through `torch.export.load` on `.pt2` archives; no fixed version is listed by the audit. ResDAG does not call this loader or offer `.pt2` loading. Its portable/raw checkpoints use the restricted Torch path described above. [Upstream report](https://github.com/pytorch/pytorch/pull/176791). |
| CVE-2025-3000 / PYSEC-2025-194 | The report concerns scripting a Python class with `torch.jit.script`; the advisory lists 2.13.0 as fixed. ResDAG does not call that compiler on supplied classes. Keep the 2.10 baseline for comparisons, and use a current patched Torch release for applications that script custom Python classes. [Upstream issue](https://github.com/pytorch/pytorch/issues/149623), [advisory](https://github.com/advisories/GHSA-rrmf-rvhw-rf47). |

The retained baseline is not a clean dependency scan. New upstream fixes need
review when the lock changes. PyTorch's
[security policy](https://github.com/pytorch/pytorch/blob/main/SECURITY.md)
explains its trust boundary for models, TorchScript and local Python execution.
Restricted checkpoint loading is available at the library's minimum Torch
2.10.0, which also contains the upstream restricted-unpickler repair described
in [GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).
