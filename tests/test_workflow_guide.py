"""The common-model guide executes its complete training and PyTorch examples."""

import re
from pathlib import Path

import torch

import resdag as rd


def test_common_workflow_guide_runs_end_to_end():
    path = Path(__file__).parents[1] / "docs" / "start" / "model-workflow.md"
    namespace = {"__name__": "__main__"}
    source = path.read_text()
    for match in re.finditer(r"```python\n(.*?)\n```", source, re.S):
        padding = "\n" * source[: match.start(1)].count("\n")
        exec(compile(padding + match.group(1), str(path), "exec"), namespace)
    assert namespace["prediction"].shape == namespace["truth"].shape == (40, 2)
    assert namespace["preset_prediction"].shape == (40, 2)
    assert torch.isfinite(namespace["batch"].grad).all()
    assert isinstance(namespace["model"], torch.nn.Module)
    assert rd.core.Model is rd.Model
    assert "Model" in rd.__all__ and "Model" in rd.core.__all__
