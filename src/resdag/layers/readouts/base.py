"""
Base Readout Layer
==================

This module provides :class:`ReadoutLayer`, a per-timestep linear layer
with support for classical ESN training via ridge regression.

See Also
--------
resdag.layers.readouts.CGReadoutLayer : Conjugate gradient readout implementation.
resdag.training.ESNTrainer : Trainer for fitting readout layers.
"""

import torch
import torch.nn as nn


class ReadoutLayer(nn.Linear):
    """
    Per-timestep linear layer with custom fitting for ESN training.

    This layer extends :class:`torch.nn.Linear` with:

    - Per-timestep application to sequence tensors ``(B, T, F)``
    - Named identification for multi-readout architectures
    - Custom ``fit()`` interface for classical ESN training

    The layer applies the same linear transformation independently to each
    timestep in a sequence:

    .. code-block:: text

        Input:  (B, T, F_in)  -> Reshape to (B*T, F_in)
        Apply:  linear(x) = x @ W.T + b
        Output: (B*T, F_out) -> Reshape to (B, T, F_out)

    This matches classical ESN semantics where readouts are fitted across
    the entire sequence at once using ridge regression.

    Parameters
    ----------
    in_features : int
        Size of input features.
    out_features : int
        Size of output features.
    bias : bool, default=True
        Whether to include a bias term.
    name : str, optional
        Name for this readout layer. Used for identification in
        multi-readout architectures and by :class:`ESNTrainer`.
    trainable : bool, default=False
        If True, weights are trainable via backpropagation.
        If False, weights are frozen (standard ESN behavior).

    Attributes
    ----------
    weight : torch.nn.Parameter
        Weight matrix of shape ``(out_features, in_features)``.
    bias : torch.nn.Parameter or None
        Bias vector of shape ``(out_features,)``, or None if ``bias=False``.
    name : str or None
        Name of this readout layer.
    is_fitted : bool
        True if ``fit()`` has been called successfully.

    Examples
    --------
    Basic usage:

    >>> readout = ReadoutLayer(in_features=100, out_features=10)
    >>> x = torch.randn(2, 20, 100)  # (batch, seq_len, features)
    >>> y = readout(x)
    >>> print(y.shape)
    torch.Size([2, 20, 10])

    Named readout for multi-output architectures:

    >>> readout1 = ReadoutLayer(100, 10, name="position")
    >>> readout2 = ReadoutLayer(100, 3, name="velocity")

    See Also
    --------
    CGReadoutLayer : Readout with Conjugate Gradient solver.
    resdag.training.ESNTrainer : Trainer for fitting readouts.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: str | None = None,
        trainable: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias)

        self._name = name
        self.trainable = trainable
        self._is_fitted = False

        if not self.trainable:
            self._freeze_weights()

    def _freeze_weights(self) -> None:
        """Freeze all weights by setting requires_grad=False."""
        for param in self.parameters():
            param.requires_grad_(False)

    @property
    def name(self) -> str | None:
        """
        str or None : Name of this readout layer.
        """
        return self._name

    @property
    def is_fitted(self) -> bool:
        """
        bool : True if ``fit()`` has been called successfully.
        """
        return self._is_fitted

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation to input.

        Handles both 2D ``(batch, features)`` and 3D ``(batch, seq_len, features)``
        inputs. For 3D inputs, applies the linear transformation independently
        to each timestep.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape ``(B, F)`` or ``(B, T, F)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, F_out)`` or ``(B, T, F_out)``.

        Raises
        ------
        ValueError
            If input has neither 2 nor 3 dimensions.

        Examples
        --------
        >>> readout = ReadoutLayer(100, 10)
        >>> x_2d = torch.randn(4, 100)
        >>> y_2d = readout(x_2d)  # (4, 10)
        >>> x_3d = torch.randn(4, 50, 100)
        >>> y_3d = readout(x_3d)  # (4, 50, 10)
        """
        if input.dim() == 2:
            return super().forward(input)

        elif input.dim() == 3:
            batch_size, seq_len, features = input.shape
            input_reshaped = input.reshape(batch_size * seq_len, features)
            output_reshaped = super().forward(input_reshaped)
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
    ) -> None:
        """
        Fit readout weights using ridge regression.

        This is an abstract method that should be overridden by subclasses.
        See :class:`CGReadoutLayer` for a concrete implementation.

        Parameters
        ----------
        states : torch.Tensor
            Input states of shape ``(B, T, F_in)`` or ``(N, F_in)``.
        targets : torch.Tensor
            Target outputs of shape ``(B, T, F_out)`` or ``(N, F_out)``.

        Raises
        ------
        NotImplementedError
            This base class does not implement fitting.

        See Also
        --------
        CGReadoutLayer.fit : Concrete implementation using Conjugate Gradient.
        """
        raise NotImplementedError(
            "ReadoutLayer.fit() is not implemented in the base class. "
            "Use CGReadoutLayer for ridge regression fitting."
        )

    def __repr__(self) -> str:
        """Return string representation."""
        name_str = f", name='{self._name}'" if self._name is not None else ""
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
            f"{name_str}"
            f")"
        )
