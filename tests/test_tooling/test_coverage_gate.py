"""Exercise the configured coverage floor through a real isolated pytest run."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

COVERAGE_CONFIG = Path(__file__).resolve().parents[2] / "pyproject.toml"


@pytest.mark.parametrize(
    ("covered_lines", "expected_exit_code"),
    [(898, 1), (902, 0)],
    ids=["below-floor", "above-floor"],
)
def test_coverage_floor_controls_pytest_exit_status(
    tmp_path: Path, covered_lines: int, expected_exit_code: int
) -> None:
    # Two executed function definitions plus 998 assignment statements give
    # exactly 1000 executable lines. Only the first function's body is called.
    exercised = covered_lines - 2
    uncalled = 1000 - covered_lines
    subject = "def exercised():\n"
    subject += "".join(f"    value_{i} = {i}\n" for i in range(exercised))
    subject += "\ndef uncalled():\n"
    subject += "".join(f"    value_{i} = {i}\n" for i in range(uncalled))
    (tmp_path / "measured_subject.py").write_text(subject)
    (tmp_path / "test_probe.py").write_text(
        "from measured_subject import exercised\n"
        "def test_probe():\n"
        "    assert exercised() is None\n"
    )
    pytest_config = tmp_path / "pytest.ini"
    pytest_config.write_text("[pytest]\n")
    report_path = tmp_path / "result.json"

    excluded = {
        "COVERAGE_FILE",
        "COVERAGE_PROCESS_START",
        "COVERAGE_RCFILE",
        "PYTEST_ADDOPTS",
        "PYTEST_CURRENT_TEST",
        "PYTEST_PLUGINS",
        "PYTHONPATH",
    }
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("COV_CORE_") and name not in excluded
    }
    env.update(
        COVERAGE_FILE=str(tmp_path / "measurements.sqlite"),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "pytest_cov.plugin",
            "-c",
            str(pytest_config),
            "--confcutdir",
            str(tmp_path),
            "--color=no",
            "-q",
            "--cov=measured_subject",
            f"--cov-config={COVERAGE_CONFIG}",
            "--cov-report=term",
            f"--cov-report=json:{report_path}",
            str(tmp_path / "test_probe.py"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    output = result.stdout + result.stderr
    assert "1 passed" in output, output
    totals = json.loads(report_path.read_text())["totals"]
    assert totals["num_statements"] == 1000, output
    assert totals["covered_lines"] == covered_lines, output
    assert totals["percent_covered"] == pytest.approx(covered_lines / 10), output
    assert result.returncode == expected_exit_code, output
