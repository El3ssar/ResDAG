"""Base ReadoutLayer implementation for torch_rc.

This module provides ReadoutLayer, a per-timestep linear layer with support
for classical ESN training (ridge regression fitting).
"""

from typing import Optional

import torch
import torch.nn as nn


class ReadoutLayer(nn.Linear):
    """Per-timestep linear layer with custom fitting for ESN training.

    This layer extends nn.Linear with:
    - Per-timestep application to sequence tensors (B, T, F)
    - Named identification for multi-readout architectures
    - Custom fit() method for classical ESN training (Phase 4)

    The layer applies the same linear transformation independently to each
    timestep in a sequence:
        Input: (B, T, F_in) -> Reshape to (B*T, F_in)
        Apply: linear(x) = x @ W.T + b
        Output: (B*T, F_out) -> Reshape to (B, T, F_out)

    This matches classical ESN semantics where readouts are fitted across
    the entire sequence at once using ridge regression.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to use bias (default: True)
        name: Optional name for this readout (used in multi-readout training)

    Example:
        # Single readout
        readout = ReadoutLayer(in_features=100, out_features=10)
        x = torch.randn(2, 20, 100)  # (batch, seq_len, features)
        y = readout(x)  # (2, 20, 10)

        # Named readout (for multi-readout architectures)
        readout1 = ReadoutLayer(100, 10, name="output1")
        readout2 = ReadoutLayer(100, 5, name="output2")
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: Optional[str] = None,
        trainable: bool = False,
    ) -> None:
        """Initialize the ReadoutLayer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to use bias
            name: Optional name for multi-readout identification
            trainable: Whether weights should be trainable (default: False)
        """
        super().__init__(in_features, out_features, bias)

        # Store name for trainer identification
        self._name = name
        self.trainable = trainable

        # Flag to track if this readout has been fitted (Phase 4)
        self._is_fitted = False

        # Freeze weights if not trainable
        if not self.trainable:
            self._freeze_weights()

    def _freeze_weights(self) -> None:
        """Freeze all weights by setting requires_grad=False."""
        for param in self.parameters():
            param.requires_grad_(False)

    @property
    def name(self) -> Optional[str]:
        """Get the readout name.

        Returns:
            Name of this readout, or None if unnamed
        """
        return self._name

    @property
    def is_fitted(self) -> bool:
        """Check if this readout has been fitted.

        Returns:
            True if fit() has been called, False otherwise
        """
        return self._is_fitted

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with per-timestep application.

        Handles both 2D (batch, features) and 3D (batch, seq_len, features) inputs.
        For 3D inputs, applies the linear transformation independently to each timestep.

        Args:
            input: Input tensor of shape (B, F) or (B, T, F)
                   where B = batch size, T = sequence length, F = features

        Returns:
            Output tensor of shape (B, F_out) or (B, T, F_out)

        Raises:
            ValueError: If input has invalid number of dimensions
        """
        # Use no_grad context when not trainable for better performance
        if not self.trainable:
            with torch.no_grad():
                return self._forward_impl(input)
        else:
            return self._forward_impl(input)

    def _forward_impl(self, input: torch.Tensor) -> torch.Tensor:
        """Internal forward implementation."""
        if input.dim() == 2:
            # Standard 2D input: (B, F) -> (B, F_out)
            return super().forward(input)

        elif input.dim() == 3:
            # 3D sequence input: (B, T, F) -> (B, T, F_out)
            batch_size, seq_len, features = input.shape

            # Reshape to (B*T, F) for per-timestep processing
            input_reshaped = input.reshape(batch_size * seq_len, features)

            # Apply linear transformation
            output_reshaped = super().forward(input_reshaped)  # (B*T, F_out)

            # Reshape back to (B, T, F_out)
            output = output_reshaped.reshape(batch_size, seq_len, self.out_features)

            return output

        else:
            raise ValueError(
                f"ReadoutLayer expects 2D (B, F) or 3D (B, T, F) input, "
                f"got {input.dim()}D tensor with shape {input.shape}"
            )

    def fit(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
        ridge: float = 1e-6,
    ) -> None:
        """Fit readout weights using ridge regression.

        This method is a stub for Phase 1. Full implementation will be added
        in Phase 4 when the conjugate gradient solver is implemented.

        Solves: states @ W = targets
        Using ridge regression: (states.T @ states + ridge*I)^-1 @ states.T @ targets

        Args:
            states: Input states of shape (B, T, F_in) or (B*T, F_in)
            targets: Target outputs of shape (B, T, F_out) or (B*T, F_out)
            ridge: Ridge regularization parameter (default: 1e-6)

        Raises:
            NotImplementedError: This is a Phase 4 feature
        """
        raise NotImplementedError(
            "ReadoutLayer.fit() will be implemented in Phase 4 "
            "when the conjugate gradient solver is added. "
            "For now, use standard PyTorch training (loss.backward() + optimizer.step())"
        )

    def __repr__(self) -> str:
        """String representation.

        Returns:
            String showing layer configuration
        """
        name_str = f", name='{self._name}'" if self._name is not None else ""
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
            f"{name_str}"
            f")"
        )
