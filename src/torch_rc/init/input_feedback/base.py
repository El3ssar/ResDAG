"""Base classes for input/feedback weight initializers."""

from abc import ABC, abstractmethod

import torch


class InputFeedbackInitializer(ABC):
    """Base class for input/feedback weight initialization."""

    @abstractmethod
    def initialize(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        """Initialize a weight tensor."""
        pass

    def __call__(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        """Callable interface for initialization."""
        return self.initialize(weight, **kwargs)


def _resolve_shape(weight: torch.Tensor) -> tuple:
    """Resolve tensor shape, ensuring it's 2D."""
    if weight.ndim != 2:
        raise ValueError(f"Weight must be 2D, got shape {weight.shape}")
    return weight.shape[0], weight.shape[1]
