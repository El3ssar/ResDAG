import torch

from .base import InputFeedbackInitializer
from .registry import register_input_feedback


@register_input_feedback("zeros")
class ZeroInitializer(InputFeedbackInitializer):
    """Initializer that sets all weights to zero."""

    def __init__(self) -> None:
        """Initialize the ZeroInitializer."""
        super().__init__()

    def initialize(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        """Initialize weight tensor with zero values."""
        with torch.no_grad():
            weight.zero_()
        return weight

    def __repr__(self) -> str:
        return "ZeroInitializer"
