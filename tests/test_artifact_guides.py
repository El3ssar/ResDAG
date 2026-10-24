"""Execute complete persistence and internal-target guide snippets."""

import re
from pathlib import Path

import torch


def test_save_load_guide_executes_in_temporary_directory(
    tmp_path, monkeypatch, graphviz_rendering, isolated_component_registries
):
    source_path = Path(__file__).parents[1] / "docs" / "workflows" / "save-load.md"
    source = source_path.read_text()
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__main__"}
    for match in re.finditer(r"```python\n(.*?)\n```", source, re.S):
        padding = "\n" * source[: match.start(1)].count("\n")
        exec(compile(padding + match.group(1), str(source_path), "exec"), namespace)
    for name in (
        "run.pt",
        "structure.dot",
        "structure.svg",
        "structure.png",
        "structure.pdf",
        "wide.svg",
        "custom.pt",
        "notebook.pt",
        "annotated.pt",
    ):
        assert (tmp_path / name).is_file()
    assert namespace["metadata"] == {"experiment": "sine", "seed": 7}
    assert isinstance(namespace["notebook_model"], torch.nn.Module)


def test_internal_readout_training_recipe_executes():
    source_path = Path(__file__).parents[1] / "docs" / "workflows" / "train.md"
    source = source_path.read_text().split(
        "### Internal readouts without extra prediction outputs", 1
    )[1]
    source = source.split("**What `alpha` does.**", 1)[0]
    namespace = {"__name__": "__main__"}
    for match in re.finditer(r"```python\n(.*?)\n```", source, re.S):
        exec(compile(match.group(1), str(source_path), "exec"), namespace)
    assert namespace["forecast"].shape == (20, 1)
    assert bool(torch.isfinite(namespace["forecast"]).all())
