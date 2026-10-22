"""The public component recipes execute and agree with independent references."""

import re
from pathlib import Path

import torch


def test_custom_component_guide_executes_public_recipes():
    path = Path(__file__).parents[1] / "docs" / "advanced" / "custom-components.md"
    namespace = {"__name__": "__main__"}
    source = path.read_text()
    for match in re.finditer(r"```python\n(.*?)\n```", source, re.S):
        padding = "\n" * source[: match.start(1)].count("\n")
        exec(compile(padding + match.group(1), str(path), "exec"), namespace)
    custom = namespace["custom"]
    assert custom.is_fitted
    assert callable(namespace["LeastSquaresReadout"].solve)
    memory = namespace["memory"]
    data = namespace["training"][:7].unsqueeze(0)
    memory.reset_state()
    observed = memory(data)
    state = torch.zeros_like(data[:, 0])
    manual = []
    for frame in data.unbind(1):
        state = 0.7 * state + 0.3 * frame
        manual.append(state)
    torch.testing.assert_close(observed, torch.stack(manual, dim=1), rtol=1e-14, atol=1e-14)
    source = namespace["training"][-4:].clone().requires_grad_()
    custom.reset_state()
    custom(source).square().sum().backward()
    assert torch.isfinite(source.grad).all() and source.grad.abs().sum() > 0
