"""CI environment contracts: locked versions, CPU-only wheels and explicit canary."""

import importlib.util
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("ci_environment", ROOT / "tools/ci_environment.py")
assert _spec is not None and _spec.loader is not None
CI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CI)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        '[project.optional-dependencies]\ndev = ["hatchling"]\nhpo = ["optuna"]\ndocs = ["mkdocs"]\n'
    )
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "torch"\nversion = "2.10.0"\n'
        '[[package]]\nname = "nvidia-cublas-cu12"\nversion = "12.1"\n'
        '[[package]]\nname = "filelock"\nversion = "3.20.1"\n'
    )
    return tmp_path


def _capture(monkeypatch, exported="filelock==3.20.1\n"):
    commands = []

    def run(command, root):
        commands.append(command)
        if command[:2] == ["uv", "export"]:
            Path(command[command.index("--output-file") + 1]).write_text(exported)

    monkeypatch.setattr(CI, "run", run)
    return commands


def test_export_prunes_accelerator_subtrees_but_preserves_torch_python_dependencies() -> None:
    lock = {
        "package": [
            {"name": name}
            for name in (
                "torch",
                "nvidia-cudnn-cu12",
                "cuda-bindings",
                "cuda-pathfinder",
                "triton",
                "filelock",
            )
        ]
    }
    command = CI.export_command(lock, ["dev", "hpo"], Path("requirements.txt"))
    pruned = {command[i + 1] for i, token in enumerate(command) if token == "--prune"}
    assert pruned == {"nvidia-cudnn-cu12", "cuda-bindings", "cuda-pathfinder", "triton"}
    assert "filelock" not in pruned and "torch" not in pruned
    assert command[command.index("--no-emit-package") + 1] == "torch"
    assert "--frozen" in command


def test_ambiguous_torch_lock_fails_closed() -> None:
    with pytest.raises(ValueError, match="one locked torch version"):
        CI.locked_torch_version(
            {
                "package": [
                    {"name": "torch", "version": "2.10.0"},
                    {"name": "torch", "version": "2.11.0"},
                ]
            }
        )


def test_locked_install_checks_lock_before_installing_exact_pins(project, monkeypatch) -> None:
    original = (project / "uv.lock").read_bytes()
    commands = _capture(monkeypatch)
    CI.install("locked", ["hpo"], project)
    assert commands[0] == ["uv", "lock", "--check"]
    assert commands[1][:2] == ["uv", "export"]
    torch = next(command for command in commands if "torch==2.10.0" in command)
    assert "--no-deps" in torch and CI.CPU_INDEX in torch
    editable = next(command for command in commands if "--editable" in command)
    assert "--no-build-isolation" in editable and "--no-deps" in editable
    assert not any("--upgrade" in command for command in commands)
    assert (project / "uv.lock").read_bytes() == original
    assert commands[-2][-1] == "check"
    assert commands[-1][-1] == "freeze"


def test_stale_lock_stops_before_any_package_install(project, monkeypatch) -> None:
    commands = []

    def run(command, root):
        commands.append(command)
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(CI, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        CI.install("locked", ["dev"], project)
    assert commands == [["uv", "lock", "--check"]]


def test_unexpected_accelerator_export_stops_before_install(project, monkeypatch) -> None:
    commands = _capture(monkeypatch, exported="nvidia-new-library==1.0\n")
    with pytest.raises(ValueError, match="Non-CPU package escaped"):
        CI.install("locked", ["dev"], project)
    assert len(commands) == 2


def test_latest_canary_constrains_the_installed_cpu_wheel(project, monkeypatch) -> None:
    commands = _capture(monkeypatch)
    monkeypatch.setattr(CI.subprocess, "check_output", lambda *a, **kw: "2.14.1+cpu\n")
    CI.install("latest", ["hpo"], project)
    install = next(command for command in commands if "--editable" in command)
    assert "--upgrade" in install and "--constraint" in install
    assert install[-1] == ".[dev,hpo]"
    assert not any(command[:2] == ["uv", "lock"] for command in commands)


def test_unknown_extra_fails_before_any_install(project, monkeypatch) -> None:
    commands = _capture(monkeypatch)
    with pytest.raises(ValueError, match="Unknown extras"):
        CI.install("locked", ["typo"], project)
    assert commands == []


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv exporter is not installed")
def test_real_lock_exports_without_accelerator_packages(tmp_path: Path) -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    output = tmp_path / "requirements.txt"
    command = CI.export_command(lock, ["dev", "hpo", "docs"], output)
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    requirements = output.read_text().splitlines()
    assert any(line.startswith("hatchling==") for line in requirements)
    assert any(line.startswith("editables==") for line in requirements)
    assert any(line.startswith("filelock==") for line in requirements)
    assert not any(CI.is_accelerator_package(line.split("==", 1)[0]) for line in requirements)
    assert not any(line.startswith("torch==") for line in requirements)
