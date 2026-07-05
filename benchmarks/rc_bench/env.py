"""Shared reproducibility-metadata collection.

Every results artifact this harness writes (``results/latest.json`` from
:mod:`rc_bench.run` and ``results/internal_cells.json`` from
:mod:`rc_bench.internal_cells`) embeds the **same** ``env`` block, produced by
:func:`collect_env` here, so a JSON file always carries enough to reproduce it:
library versions (resdag / torch / numpy / scipy / reservoirpy), the Julia
version, the CUDA toolkit + GPU name, the CPU model, the git commit SHA, and a
timestamp.

Two deliberate choices, both required by the benchmark contract:

* **The timestamp is read from the shell, never ``datetime.now()``.** The
  harness process must not mint its own clock reading — :func:`shell_timestamp`
  shells out to ``date`` so the recorded time is the machine's, uniformly for
  every artifact.
* **No duplication.** :mod:`rc_bench.run` and :mod:`rc_bench.internal_cells`
  both call :func:`collect_env`; there is one source of truth for the ``env``
  schema.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parents[1]  # .../benchmarks
_REPO_ROOT = _BENCH_DIR.parent  # repo root (git lives here)


def shell_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp read from the shell (``date``), not from
    the Python process clock.

    The benchmark contract requires the harness *not* to call ``datetime.now()``
    itself — the recorded time must come from the machine's shell. Falls back to
    an empty string only if ``date`` is somehow unavailable.
    """
    try:
        out = subprocess.run(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - date is always present on POSIX
        return ""


def _module_version(mod_name: str) -> str | None:
    try:
        return __import__(mod_name).__version__
    except Exception:
        return None


def _cpu_model() -> str:
    """Best-effort human-readable CPU model.

    ``platform.processor()`` is empty on most Linux boxes, so fall back to
    ``/proc/cpuinfo`` (Linux) or the platform string.
    """
    proc = platform.processor()
    if proc:
        return proc
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            for line in cpuinfo.read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.machine()


def _julia_version() -> str | None:
    """``julia --version`` -> ``"1.12.5"``; ``None`` if Julia is absent."""
    if shutil.which("julia") is None:
        return None
    try:
        out = subprocess.run(
            ["julia", "--version"], capture_output=True, text=True, check=True
        )
        # "julia version 1.12.5"
        return out.stdout.strip().split()[-1]
    except Exception:
        return None


def _git_commit() -> str | None:
    """Short + long git SHA of the repo, or ``None`` outside a checkout."""
    if shutil.which("git") is None:
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes (so a SHA alone would
    not fully reproduce the run). ``None`` if git is unavailable."""
    if shutil.which("git") is None:
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def collect_env() -> dict:
    """Assemble the reproducibility ``env`` block embedded in every results JSON.

    Returns
    -------
    dict
        Keys: ``timestamp`` (shell-sourced UTC ISO-8601), ``versions``
        (python / resdag / torch / numpy / scipy / reservoirpy / julia), ``cuda``
        (toolkit version + availability + GPU name + total VRAM), ``cpu``
        (model + logical core count), ``platform``, and ``git`` (commit SHA +
        dirty flag).
    """
    import os

    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "resdag": _module_version("resdag"),
        "torch": _module_version("torch"),
        "numpy": _module_version("numpy"),
        "scipy": _module_version("scipy"),
        "reservoirpy": _module_version("reservoirpy"),
        "julia": _julia_version(),
    }

    cuda: dict[str, object] = {"available": False}
    try:
        import torch

        cuda["toolkit"] = torch.version.cuda or "n/a"
        cuda["available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            cuda["gpu"] = torch.cuda.get_device_name(0)
            cuda["vram_gb"] = round(props.total_memory / 1e9, 2)
    except Exception:
        pass

    return {
        "timestamp": shell_timestamp(),
        "versions": versions,
        "cuda": cuda,
        "cpu": {"model": _cpu_model(), "logical_cores": os.cpu_count()},
        "platform": platform.platform(),
        "git": {"commit": _git_commit(), "dirty": _git_dirty()},
    }
