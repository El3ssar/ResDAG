"""Run the progressive tuning guide as users read it, at its published budgets."""

import importlib.util
import random
import re
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("optuna")
pytest.importorskip("joblib")


def test_progressive_tuning_guide_examples_execute_in_reading_order():
    guide = Path(__file__).resolve().parents[2] / "docs/workflows/tune.md"
    blocks = re.findall(r"```python\n(.*?)\n```", guide.read_text(), flags=re.S)
    assert blocks, "The guide must retain runnable Python examples"
    namespace = {"__name__": "__main__"}
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            for number, code in enumerate(blocks, 1):
                if "import tsdynamics" in code and importlib.util.find_spec("tsdynamics") is None:
                    continue
                exec(compile(code, f"{guide}:python-block-{number}", "exec"), namespace)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
