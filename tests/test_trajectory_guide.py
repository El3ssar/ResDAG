"""Every Python example in the trajectory guide runs against the optional extra."""

import re
from pathlib import Path

import pytest
import torch


def test_trajectory_guide_runs_end_to_end():
    ts = pytest.importorskip("tsdynamics")
    path = Path(__file__).parents[1] / "docs" / "start" / "trajectory-workflow.md"
    source = path.read_text()
    namespace = {"__name__": "__main__"}
    examples = list(re.finditer(r"```python\n(.*?)\n```", source, re.S))
    assert examples
    for match in examples:
        padding = "\n" * source[: match.start(1)].count("\n")
        exec(compile(padding + match.group(1), str(path), "exec"), namespace)
    assert isinstance(namespace["prediction"], ts.Trajectory)
    assert isinstance(namespace["tensor_future"], torch.Tensor)
    assert namespace["restored_future"].variables == ("position",)
