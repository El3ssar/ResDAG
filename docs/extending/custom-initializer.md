# Add an input/feedback initializer

Subclass `InputFeedbackInitializer` and register.

```python
import torch
from resdag.init.input_feedback import InputFeedbackInitializer
from resdag.init.input_feedback.registry import register_input_feedback


@register_input_feedback("alternating", scale=1.0)
class AlternatingInitializer(InputFeedbackInitializer):
    def initialize(self, weight: torch.Tensor, **kwargs) -> None:
        rows, cols = weight.shape
        pattern = torch.tensor([1.0, -1.0])
        w = pattern.repeat(rows * cols)[: rows * cols].view(rows, cols)
        weight.copy_(w * self.scale)
```

```python
ESNLayer(300, feedback_size=4, feedback_initializer="alternating")
```

Study [`chebyshev.py`](https://github.com/El3ssar/resdag/blob/main/src/resdag/init/input_feedback/chebyshev.py) for a full example with hyperparameters on the decorator.

## See also

- [Input & feedback initializers](../learn/input-feedback-initializers.md)
