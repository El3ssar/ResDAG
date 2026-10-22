"""Architecture guides must execute and forecast finite, held-out samples."""

import re
from pathlib import Path

import pytest
import torch

from resdag import ESNTrainer


@pytest.mark.parametrize("guide", ["classic-esn", "ott-esn", "power-augmented"])
def test_architecture_guide_forecasts_held_out_data(guide, monkeypatch):
    targets = []
    original = ESNTrainer.fit

    def capture(self, *args, **kwargs):
        targets.append(kwargs["targets"]["output"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ESNTrainer, "fit", capture)
    path = Path(__file__).parents[1] / "docs" / "build" / "architectures" / f"{guide}.md"
    namespace = {"__name__": "__main__"}
    for block in re.findall(r"```python\n(.*?)\n```", path.read_text(), re.S):
        exec(compile(block, str(path), "exec"), namespace)
    prediction, truth = namespace["preds"], namespace["truth"]
    assert prediction.shape == truth.shape == (1, 200, 3)
    assert torch.isfinite(prediction).all()
    assert len(targets) == 1
    target = targets[0]
    # The guide slices one trajectory: the final supervised target must end
    # before the first held-out observation, not forecast back inside training.
    assert target.untyped_storage().data_ptr() == truth.untyped_storage().data_ptr()
    last_target = target[:, -1]
    assert last_target.data_ptr() + last_target.numel() * target.element_size() <= truth.data_ptr()
