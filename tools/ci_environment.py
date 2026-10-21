"""Install either the locked CPU CI baseline or the latest compatible dependencies.

The lock contains platform-specific CUDA dependencies from PyPI's Linux torch
wheel. CI uses the matching CPU wheel instead, so its accelerator-only dependency
subtrees are pruned while torch's pure-Python dependencies stay pinned. Install
without dependency resolution after export to prevent pip from adding them back.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def is_accelerator_package(name: str) -> bool:
    """Dependencies replaced by the CPU torch wheel, never needed in CPU CI."""
    name = name.lower().replace("_", "-")
    return name.startswith("nvidia-") or name in {"triton", "cuda-bindings", "cuda-pathfinder"}


def locked_torch_version(lock: dict) -> str:
    """Require one unambiguous torch version across the supported Python range."""
    versions = {package["version"] for package in lock["package"] if package["name"] == "torch"}
    if len(versions) != 1:
        raise ValueError(f"Expected one locked torch version, found {sorted(versions)}")
    return versions.pop()


def export_command(lock: dict, extras: list[str], output: Path) -> list[str]:
    """Export pure CPU dependencies at exactly the committed lock versions."""
    command = [
        "uv",
        "export",
        "--frozen",
        "--no-dev",
        "--no-emit-project",
        "--no-emit-package",
        "torch",
        "--no-hashes",
        "--no-header",
        "--no-annotate",
    ]
    for extra in extras:
        command.extend(["--extra", extra])
    for name in sorted({p["name"] for p in lock["package"] if is_accelerator_package(p["name"])}):
        command.extend(["--prune", name])
    return [*command, "--output-file", str(output)]


def run(command: list[str], root: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def install(mode: str, extras: list[str], root: Path = ROOT) -> None:
    """Build a checked CPU-only environment; never mutate the committed lock."""
    if mode not in {"locked", "latest"}:
        raise ValueError(f"Unknown dependency mode: {mode}")
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    extras = sorted({"dev", *extras})
    known = set(pyproject["project"]["optional-dependencies"])
    if set(extras) - known:
        raise ValueError(f"Unknown extras: {sorted(set(extras) - known)}")
    pip = [sys.executable, "-m", "pip"]
    with tempfile.TemporaryDirectory(prefix="resdag-ci-") as directory:
        output = Path(directory) / "requirements.txt"
        if mode == "locked":
            lock = tomllib.loads((root / "uv.lock").read_text())
            # --check rejects metadata/dependency drift; the dynamic local
            # project version is not pinned, so release bumps remain valid.
            run(["uv", "lock", "--check"], root)
            run(export_command(lock, extras, output), root)
            # Export is complete before installing anything. Guard against an
            # unexpected new accelerator package or torch escaping pruning.
            for line in output.read_text().splitlines():
                name = line.split("==", 1)[0].strip()
                if name == "torch" or is_accelerator_package(name):
                    raise ValueError(f"Non-CPU package escaped lock export: {line}")
            version = locked_torch_version(lock)
            run([*pip, "install", "--no-deps", "--index-url", CPU_INDEX, f"torch=={version}"], root)
            run([*pip, "install", "--no-deps", "--requirement", str(output)], root)
            # hatchling and its editable-build helper are in the dev lock.
            run([*pip, "install", "--no-deps", "--no-build-isolation", "--editable", "."], root)
        else:
            # Latest runtime compatibility is tested independently of baseline
            # formatting/type-tool versions. Keep the installed CPU wheel
            # constrained while pip resolves all other declared dependencies.
            run([*pip, "install", "--upgrade", "--index-url", CPU_INDEX, "torch"], root)
            version = subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "from importlib.metadata import version; print(version('torch'))",
                ],
                cwd=root,
                text=True,
            ).strip()
            output.write_text(f"torch=={version}\n")
            requirement = f".[{','.join(extras)}]"
            run(
                [
                    *pip,
                    "install",
                    "--upgrade",
                    "--constraint",
                    str(output),
                    "--editable",
                    requirement,
                ],
                root,
            )
        run([*pip, "check"], root)
        run([*pip, "freeze"], root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["locked", "latest"], default="locked")
    parser.add_argument(
        "--extras", default="dev,hpo", help="Comma-separated extras; dev is always included"
    )
    args = parser.parse_args()
    install(args.mode, [extra.strip() for extra in args.extras.split(",") if extra.strip()])


if __name__ == "__main__":
    main()
