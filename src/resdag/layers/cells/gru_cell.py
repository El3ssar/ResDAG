"""
GRU Reservoir Cell
==================

This module provides a Gated Recurrent Unit (GRU) reservoir cell with frozen
random weights:

- :class:`GRUCell` — single-timestep GRU update with non-trainable weights.

Reference: Cho et al., "Learning Phrase Representations using RNN
Encoder-Decoder for Statistical Machine Translation" (2014).

See Also
--------
resdag.layers.cells.base_cell : Abstract base class (ReservoirCell).
resdag.layers.reservoirs.gru : GRULayer that sequences this cell.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_cell import ReservoirCell


class GRUCell(ReservoirCell):
    """
    Single-timestep GRU reservoir update with frozen random weights.

    Implements the standard Gated Recurrent Unit equations with all weight
    matrices registered as non-trainable buffers.  The gating mechanism
    provides per-dimension, input-dependent memory dynamics that differ
    qualitatively from a scalar leaky integrator ESN.

    The update equations are:

    .. math::

        z(t) = \\sigma(W_z x(t) + U_z h(t-1) + b_z)

        r(t) = \\sigma(W_r x(t) + U_r h(t-1) + b_r)

        \\tilde{h}(t) = \\tanh(W_h x(t) + U_h (r(t) \\odot h(t-1)) + b_h)

        h(t) = (1 - z(t)) \\odot h(t-1) + z(t) \\odot \\tilde{h}(t)

    where :math:`\\sigma` is the sigmoid function and :math:`\\odot` is the
    element-wise product.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector.  When used via :class:`GRULayer`,
        this is the total concatenated size of feedback plus any driving inputs.
    hidden_dim : int
        Dimensionality of the hidden state.
    w_in_scale : float, default=1.0
        Scale for input weight matrices.  Each entry of ``W_z``, ``W_r``, and
        ``W_h`` is drawn from ``Uniform(-w_in_scale, w_in_scale)``.
    spectral_radius : float, default=0.9
        Target spectral radius for each of the three recurrent weight matrices
        ``U_z``, ``U_r``, and ``U_h``.  Each is scaled independently.
    sparsity : float, default=0.0
        Fraction of recurrent weight entries to zero out.  Applied
        independently to each of ``U_z``, ``U_r``, ``U_h``.
    bias : bool, default=True
        Whether to include bias vectors.  If ``True``, biases are initialised
        from ``Uniform(-1, 1)`` shifted by ``gate_bias_init``.
    gate_bias_init : float, default=0.0
        Constant added to each bias entry after uniform initialisation.
        Positive values push gates toward open (update/reset); negative values
        push toward closed (memory retention).

    Attributes
    ----------
    W_z, W_r, W_h : torch.Tensor
        Input weight matrices of shape ``(hidden_dim, input_dim)``; buffers.
    U_z, U_r, U_h : torch.Tensor
        Recurrent weight matrices of shape ``(hidden_dim, hidden_dim)``; buffers.
    b_z, b_r, b_h : torch.Tensor or None
        Bias vectors of shape ``(hidden_dim,)``; buffers, or ``None`` when
        ``bias=False``.

    See Also
    --------
    resdag.layers.reservoirs.gru.GRULayer : Layer that sequences this cell.
    resdag.layers.cells.base_cell.ReservoirCell : Abstract cell interface.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        w_in_scale: float = 1.0,
        spectral_radius: float = 0.9,
        sparsity: float = 0.0,
        bias: bool = True,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._has_bias = bias

        # --- Input weight matrices (frozen uniform) ---
        W_z = torch.empty(hidden_dim, input_dim).uniform_(-w_in_scale, w_in_scale)
        W_r = torch.empty(hidden_dim, input_dim).uniform_(-w_in_scale, w_in_scale)
        W_h = torch.empty(hidden_dim, input_dim).uniform_(-w_in_scale, w_in_scale)
        self.register_buffer("W_z", W_z)
        self.register_buffer("W_r", W_r)
        self.register_buffer("W_h", W_h)

        # --- Recurrent weight matrices (independently scaled, frozen) ---
        self.register_buffer("U_z", self._init_recurrent(hidden_dim, spectral_radius, sparsity))
        self.register_buffer("U_r", self._init_recurrent(hidden_dim, spectral_radius, sparsity))
        self.register_buffer("U_h", self._init_recurrent(hidden_dim, spectral_radius, sparsity))

        # --- Bias vectors ---
        if bias:
            b_z = torch.empty(hidden_dim).uniform_(-1.0, 1.0).add_(gate_bias_init)
            b_r = torch.empty(hidden_dim).uniform_(-1.0, 1.0).add_(gate_bias_init)
            b_h = torch.empty(hidden_dim).uniform_(-1.0, 1.0).add_(gate_bias_init)
            self.register_buffer("b_z", b_z)
            self.register_buffer("b_r", b_r)
            self.register_buffer("b_h", b_h)
        else:
            self.register_buffer("b_z", None)
            self.register_buffer("b_r", None)
            self.register_buffer("b_h", None)

    # ------------------------------------------------------------------
    # ReservoirCell interface
    # ------------------------------------------------------------------

    @property
    def hidden_dim(self) -> int:
        """Dimensionality of the hidden state."""
        return self._hidden_dim

    @property
    def state_size(self) -> int:
        """Dimensionality of the hidden state vector (alias for hidden_dim)."""
        return self._hidden_dim

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Return a zero initial hidden state.

        Parameters
        ----------
        batch_size : int
            Number of sequences in the batch.
        device : torch.device
            Target device.
        dtype : torch.dtype
            Target dtype.

        Returns
        -------
        torch.Tensor
            Zero tensor of shape ``(batch_size, hidden_dim)``.
        """
        return torch.zeros(batch_size, self._hidden_dim, device=device, dtype=dtype)

    def forward(
        self,
        inputs: list[torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the next GRU hidden state for a single timestep.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            Per-timestep input slices.  ``inputs[0]`` is always the feedback
            slice of shape ``(batch, feedback_dim)``.  If a driving input is
            present, ``inputs[1]`` has shape ``(batch, driving_dim)``.  All
            tensors are concatenated along the feature axis to form ``x``.
        state : torch.Tensor
            Current hidden state of shape ``(batch, hidden_dim)``.

        Returns
        -------
        torch.Tensor
            Next hidden state of shape ``(batch, hidden_dim)``.
        """
        x = torch.cat(inputs, dim=-1) if len(inputs) > 1 else inputs[0]
        h = state

        z = torch.sigmoid(F.linear(x, self.W_z, self.b_z) + F.linear(h, self.U_z))
        r = torch.sigmoid(F.linear(x, self.W_r, self.b_r) + F.linear(h, self.U_r))
        h_tilde = torch.tanh(F.linear(x, self.W_h, self.b_h) + F.linear(r * h, self.U_h))
        new_h = (1.0 - z) * h + z * h_tilde

        return new_h

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_recurrent(hidden_dim: int, spectral_radius: float, sparsity: float) -> torch.Tensor:
        """
        Create a single recurrent weight matrix scaled to ``spectral_radius``.

        Parameters
        ----------
        hidden_dim : int
            Square matrix side length.
        spectral_radius : float
            Target spectral radius.
        sparsity : float
            Fraction of entries set to zero before scaling.

        Returns
        -------
        torch.Tensor
            Float tensor of shape ``(hidden_dim, hidden_dim)`` in the default dtype.
        """
        # Work in float64 for numerical stability of the eigendecomposition.
        U = torch.randn(hidden_dim, hidden_dim, dtype=torch.float64)

        if sparsity > 0.0:
            mask = torch.rand(hidden_dim, hidden_dim) < sparsity
            U[mask] = 0.0

        eigvals = torch.linalg.eigvals(U)
        actual_sr = torch.max(torch.abs(eigvals)).item()

        if actual_sr > 0.0:
            U = U * (spectral_radius / actual_sr)

        return U.to(torch.get_default_dtype())

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"GRUCell("
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self._hidden_dim}, "
            f"bias={self._has_bias}"
            f")"
        )
