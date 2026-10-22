"""Run every named-trajectory example against the actual optional package."""

import re
from pathlib import Path

import pytest


def test_named_trajectory_guide_executes(tmp_path, monkeypatch):
    pytest.importorskip("tsdynamics")
    path = Path(__file__).parents[1] / "docs" / "start" / "named-trajectories.md"
    blocks = re.findall(r"^```python\n(.*?)^```", path.read_text(), re.MULTILINE | re.DOTALL)
    assert blocks, "The named-trajectory guide must contain executable examples"
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__named_trajectory_guide__"}
    for block in blocks:
        exec(compile(block, str(path), "exec"), namespace)
