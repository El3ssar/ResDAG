"""Execute every Python block in the new user guides in reading order."""

import re
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "guide",
    [
        "build/initialization/recurrent.md",
        "build/initialization/custom.md",
        "build/layers/oscillator.md",
        "build/layers/legendre-memory.md",
        "build/layers/pytorch-recurrent.md",
        "workflows/calibration.md",
    ],
)
def test_documented_workflow_executes(guide):
    path = Path(__file__).parents[1] / "docs" / guide
    source = path.read_text()
    blocks = re.findall(r"^```python\n(.*?)^```", source, flags=re.MULTILINE | re.DOTALL)
    assert blocks, f"No executable examples in {guide}"
    namespace = {"__name__": "__documented_example__"}
    for block in blocks:
        exec(compile(block, str(path), "exec"), namespace)
