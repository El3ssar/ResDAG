"""
Base Reservoir Classes
======================

This module provides abstract base class for the reservoir cell:

- :class:`ReservoirCell` — abstract single-timestep reservoir update (owns parameters).

See Also
--------
resdag.layers.esn : Concrete ESN implementation (ESNCell, ESNLayer).
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ReservoirCell(nn.Module, ABC):
    """
    Abstract base for a single-timestep reservoir state update.

    Owns all trainable (or frozen) parameters.  Sequence iteration is
    handled by the enclosing :class:`BaseReservoirLayer`, not by the cell
    itself.

    Parameters
    ----------
    (none — concrete subclasses define their own signatures)

    Notes
    -----
    Subclasses must implement :attr:`state_size` and :meth:`forward`.

    ``inputs[0]`` passed to :meth:`forward` is always the feedback slice.
    Additional elements are driving inputs in the order they were passed to
    the layer's ``forward``.

    See Also
    --------
    resdag.layers.esn.ESNCell : Concrete ESN cell implementation.
    resdag.layers.base.BaseReservoirLayer : Layer that drives the cell.
    """

    @property
    @abstractmethod
    def state_size(self) -> int:
        """Dimensionality of the hidden state vector."""
        ...

    @abstractmethod
    def forward(
        self,
        inputs: list[torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the next hidden state from current inputs and state.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            Per-timestep input slices, one per input stream, each of shape
            ``(batch, feature_dim)``.  ``inputs[0]`` is always the feedback
            slice; additional elements are driving inputs.
        state : torch.Tensor
            Current hidden state of shape ``(batch, state_size)``.

        Returns
        -------
        torch.Tensor
            Next hidden state of shape ``(batch, state_size)``.
        """
        ...
