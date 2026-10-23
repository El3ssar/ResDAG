"""Execute the published memory workflow and solver examples as written."""

import re
from pathlib import Path

import pytest


@pytest.mark.parametrize("page", ["workflows/sparse-reservoirs.md", "reference/lsmr-readout.md"])
def test_memory_workflow_examples(page, tmp_path, monkeypatch):
    from resdag.init.topology import registry

    root = Path(__file__).resolve().parents[2]
    source = (root / "docs" / page).read_text()
    blocks = re.findall(r"```python\n(.*?)```", source, re.S)
    assert blocks
    monkeypatch.setattr(registry, "_TOPOLOGY_REGISTRY", dict(registry._TOPOLOGY_REGISTRY))
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__main__"}
    for block in blocks:
        exec(compile(block, str(root / "docs" / page), "exec"), namespace)
