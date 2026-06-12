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

    Notes
    -----
    Subclasses must implement :attr:`state_size`, :attr:`output_size`,
    :meth:`init_state`, and :meth:`forward`.

    ``inputs[0]`` passed to :meth:`forward` is always the feedback slice.
    Additional elements are driving inputs in the order they were passed to
    the layer's ``forward``.

    For cells where the output and the state are the same tensor (e.g.
    :class:`ESNCell`), ``output_size == state_size``.  For cells where they
    differ (e.g. :class:`NGCell` whose output is a feature vector but whose
    state is a delay buffer), the two properties return different values.

    See Also
    --------
    resdag.layers.esn.ESNCell : Concrete ESN cell implementation.
    resdag.layers.cells.ngrc_cell.NGCell : NG-RC cell implementation.
    resdag.layers.base.BaseReservoirLayer : Layer that drives the cell.
    """

    @property
    @abstractmethod
    def state_size(self) -> int:
        """Size of the state tensor (second dimension)."""
        ...

    @property
    @abstractmethod
    def output_size(self) -> int:
        """Dimensionality of the per-step output vector."""
        ...

    @abstractmethod
    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Return a zero initial state tensor.

        Parameters
        ----------
        batch_size : int
            Number of samples in the batch.
        device : torch.device
            Target device.
        dtype : torch.dtype
            Target dtype.

        Returns
        -------
        torch.Tensor
            Zero-filled initial state.
        """
        ...

    @abstractmethod
    def forward(
        self,
        inputs: list[torch.Tensor],
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the per-step output and next state from current inputs and state.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            Per-timestep input slices, one per input stream, each of shape
            ``(batch, feature_dim)``.  ``inputs[0]`` is always the feedback
            slice; additional elements are driving inputs.
        state : torch.Tensor
            Current state tensor.

        Returns
        -------
        output : torch.Tensor
            Per-step output of shape ``(batch, output_size)``.
        new_state : torch.Tensor
            Updated state tensor.
        """
        ...

    def project_inputs(self, inputs: list[torch.Tensor]) -> torch.Tensor | None:
        """
        Optional sequence-level fast path: precompute input contributions.

        Cells whose pre-activation splits into an input-dependent part and a
        state-dependent part (e.g. the leaky ESN) can compute the
        input-dependent part for the *whole* sequence in one batched matmul,
        leaving only the recurrent term inside the time loop.  This cuts the
        per-step kernel count roughly in half — the difference between the
        GPU being slower and faster than the CPU for typical reservoir sizes.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            Full input sequences, one per stream, each of shape
            ``(batch, timesteps, feature_dim)``.  ``inputs[0]`` is the
            feedback sequence.

        Returns
        -------
        torch.Tensor or None
            Precomputed projection of shape ``(batch, timesteps, state_size)``
            to be consumed step-by-step via :meth:`step`, or ``None`` if the
            cell does not support the fast path (the layer then falls back to
            calling :meth:`forward` per timestep).
        """
        return None

    def step(
        self,
        projected_t: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step update consuming a slice of :meth:`project_inputs`.

        Only called by the layer when :meth:`project_inputs` returned a
        tensor.  Subclasses implementing the fast path must override both
        methods together.

        Parameters
        ----------
        projected_t : torch.Tensor
            Per-timestep slice of the precomputed projection, shape
            ``(batch, state_size)``.
        state : torch.Tensor
            Current state tensor.

        Returns
        -------
        output : torch.Tensor
            Per-step output.
        new_state : torch.Tensor
            Updated state tensor.
        """
        raise NotImplementedError(
            f"{type(self).__name__} returned a projection from project_inputs() "
            f"but does not implement step()."
        )

    def validate_state(self, state: torch.Tensor) -> None:
        """
        Validate that ``state`` matches the layout this cell expects.

        The base implementation enforces the 2-D ``(batch, state_size)``
        contract used by classical RNN-style cells (e.g. :class:`ESNCell`).
        Cells with a different state layout — for example
        :class:`~resdag.layers.cells.ngrc_cell.NGCell` whose state is a 3-D
        delay buffer — override this method to check their own shape.

        Parameters
        ----------
        state : torch.Tensor
            Candidate state tensor to validate.

        Raises
        ------
        ValueError
            If ``state`` does not match the expected layout.  The error
            message names the cell class and the offending shape so callers
            can act on it without digging into the layer/cell internals.
        """
        if state.dim() != 2 or state.shape[-1] != self.state_size:
            raise ValueError(
                f"{type(self).__name__}.validate_state: expected a 2-D state of shape "
                f"(batch, {self.state_size}); got tensor of shape {tuple(state.shape)}."
            )
