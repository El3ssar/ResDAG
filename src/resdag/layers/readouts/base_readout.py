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

    @torch.no_grad()
    def fit(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Fit readout weights to the given (states, targets) pair.

        This base implementation handles the bookkeeping that every readout
        needs: shape normalisation from ``(B, T, F)`` → ``(B*T, F)`` plus
        sample-count and out-features validation.  The actual algebraic
        solve is delegated to :meth:`_fit_impl`, which subclasses override.

        Parameters
        ----------
        states : torch.Tensor
            Input states of shape ``(B, T, F_in)`` or ``(N, F_in)``.
        targets : torch.Tensor
            Target outputs of shape ``(B, T, F_out)`` or ``(N, F_out)``.

        Raises
        ------
        NotImplementedError
            The base ``ReadoutLayer`` does not implement an algebraic solver;
            use a subclass such as :class:`CGReadoutLayer`.
        ValueError
            If ``states`` and ``targets`` disagree on the sample dimension
            after flattening, or if the target's feature dimension does not
            match ``self.out_features``.

        Notes
        -----
        After ``fit()`` returns successfully, :attr:`is_fitted` is ``True``.

        See Also
        --------
        CGReadoutLayer : Concrete implementation using Conjugate Gradient.
        """
        if states.dim() == 3:
            b, t, f = states.shape
            states = states.reshape(b * t, f)
        if targets.dim() == 3:
            b, t, f = targets.shape
            targets = targets.reshape(b * t, f)

        readout_id = f"'{self._name}'" if self._name is not None else type(self).__name__
        if states.shape[0] != targets.shape[0]:
            raise ValueError(
                f"{type(self).__name__}.fit({readout_id}): sample count mismatch. "
                f"States have {states.shape[0]} samples after flattening, "
                f"targets have {targets.shape[0]}."
            )
        if targets.shape[1] != self.out_features:
            raise ValueError(
                f"{type(self).__name__}.fit({readout_id}): target feature dimension "
                f"({targets.shape[1]}) does not match readout out_features "
                f"({self.out_features})."
            )

        coefs, intercept = self._fit_impl(states, targets)

        self.weight.copy_(coefs.T.to(self.weight.dtype))
        if self.bias is not None and intercept is not None:
            self.bias.copy_(intercept.to(self.bias.dtype))

        self._is_fitted = True

    def _fit_impl(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Solve the readout fit on *already-flattened* inputs.

        Subclasses override this hook instead of :meth:`fit` so they don't
        need to re-implement shape normalisation, validation, or parameter
        copy-back.

        Parameters
        ----------
        states : torch.Tensor
            Flattened input states of shape ``(N, in_features)``.
        targets : torch.Tensor
            Flattened targets of shape ``(N, out_features)``.

        Returns
        -------
        coefs : torch.Tensor
            Coefficient matrix of shape ``(in_features, out_features)``.
            The base ``fit`` transposes this into the ``(out, in)`` layout
            expected by ``nn.Linear``.
        intercept : torch.Tensor or None
            Bias vector of shape ``(out_features,)``, or ``None`` to leave
            the layer's bias untouched.

        Raises
        ------
        NotImplementedError
            The base class does not implement an algebraic solve.
        """
        raise NotImplementedError(
            "ReadoutLayer._fit_impl() is not implemented in the base class. "
            "Use CGReadoutLayer for ridge regression fitting, or subclass "
            "ReadoutLayer and override _fit_impl()."
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
