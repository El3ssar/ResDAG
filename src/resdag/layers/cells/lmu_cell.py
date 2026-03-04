"""
LMU Cell
========

This module provides the Legendre Memory Unit (LMU) implementation of the
reservoir cell:

- :class:`LMUCell` — single-timestep LMU update; owns all parameters.

The LMU is derived from the paper:
    Voelker, Kajić, Eliasmith, "Legendre Memory Units: Continuous-Time
    Representation in Recurrent Neural Networks" (NeurIPS 2019).

The core idea is to maintain a linear memory that optimally represents the
continuous-time history of its input over a sliding window using shifted
Legendre polynomials as a basis.

See Also
--------
resdag.layers.cells.base_cell : Abstract base class (ReservoirCell).
resdag.layers.reservoirs.lmu : LMULayer wrapping this cell in a sequence loop.
"""

import warnings
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_cell import ReservoirCell


def _compute_legendre_matrices(order: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Legendre memory matrices A and B analytically.

    These are derived from the Padé approximant to the pure time-delay
    operator via shifted Legendre polynomials.

    Parameters
    ----------
    order : int
        Order of the approximation (memory state dimension).

    Returns
    -------
    A : np.ndarray of shape (order, order)
        Recurrent matrix of the continuous-time LTI system.
    B : np.ndarray of shape (order, 1)
        Input matrix of the continuous-time LTI system.
    """
    A = np.zeros((order, order), dtype=np.float64)
    B = np.zeros((order, 1), dtype=np.float64)
    for i in range(order):
        B[i, 0] = (2 * i + 1) * ((-1) ** i)
        for j in range(order):
            if i > j:
                A[i, j] = (2 * i + 1) * ((-1) ** (i - j))
            elif i == j:
                A[i, j] = -(2 * i + 1)
            else:  # i < j
                A[i, j] = (2 * i + 1)
    return A, B


def _discretize(
    A_np: np.ndarray,
    B_np: np.ndarray,
    theta: float,
    dt: float,
    method: Literal["zoh", "euler"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Discretize the continuous-time LTI system via Zero-Order Hold or Euler.

    Parameters
    ----------
    A_np : np.ndarray, shape (order, order)
        Continuous-time recurrent matrix.
    B_np : np.ndarray, shape (order, 1)
        Continuous-time input matrix.
    theta : float
        Window length (time constant).
    dt : float
        Simulation time step.
    method : {'zoh', 'euler'}
        Discretization method.

    Returns
    -------
    A_bar : torch.Tensor of shape (order, order), dtype float64
        Discretized recurrent matrix.
    B_bar : torch.Tensor of shape (order, 1), dtype float64
        Discretized input matrix.

    Raises
    ------
    ValueError
        If ``method`` is not 'zoh' or 'euler'.
    """
    order = A_np.shape[0]
    A_t = torch.tensor(A_np, dtype=torch.float64)
    B_t = torch.tensor(B_np, dtype=torch.float64)
    eye = torch.eye(order, dtype=torch.float64)

    if method == "zoh":
        A_bar = torch.linalg.matrix_exp(A_t * (dt / theta))
        # B_bar = inv(A) @ (A_bar - I) @ B  via linalg.solve for numerical stability
        rhs = (A_bar - eye) @ B_t  # shape (order, 1)
        B_bar = torch.linalg.solve(A_t, rhs)  # shape (order, 1)
    elif method == "euler":
        A_bar = eye + A_t * (dt / theta)
        B_bar = B_t * (dt / theta)
    else:
        raise ValueError(f"Unknown discretization method '{method}'. Use 'zoh' or 'euler'.")

    return A_bar, B_bar


class LMUCell(ReservoirCell):
    """
    Single-timestep Legendre Memory Unit update.

    Maintains a linear memory state that optimally represents the
    continuous-time history of the input signal over a sliding window of
    length ``theta``, using shifted Legendre polynomials as a basis.

    All dynamics are fixed (non-trainable): the discretized LTI matrices
    ``A_bar`` and ``B_bar`` are computed analytically.

    For a ``d_input``-dimensional input ``x(t)``, the cell maintains
    ``d_input`` independent memory vectors of size ``order``, each running
    its own LTI system with the same ``A_bar`` and ``B_bar``. The total
    memory state has shape ``(batch, input_dim * order)``.

    Parameters
    ----------
    input_dim : int
        Total input dimension (feedback_size + optional input_size).
    order : int, default=8
        Order of the Legendre approximation (memory state dimension per
        input channel).
    theta : float, default=1.0
        Window length (time constant) of the sliding memory window.
    dt : float, default=1.0
        Simulation time step for discretization.
    discretization : {'zoh', 'euler'}, default='zoh'
        Discretization method. 'zoh' (Zero-Order Hold) is more faithful
        to the continuous-time dynamics; 'euler' is simpler.
    nonlinear_hidden : bool, default=False
        If ``False`` (default), the cell output is the flat memory state
        ``m(t)`` only — a purely linear reservoir.  If ``True``, a
        random nonlinear hidden state ``h(t)`` is appended, combining LMU
        memory with ESN-style nonlinear dynamics.
    hidden_dim : int, default=64
        Dimension of the nonlinear hidden state. Only used when
        ``nonlinear_hidden=True``.
    w_x_scale : float, default=1.0
        Scaling factor for the random input-to-hidden weight matrix
        ``W_x``. Only used when ``nonlinear_hidden=True``.

    Attributes
    ----------
    A_bar : torch.Tensor, shape (order, order)
        Discretized recurrent matrix (fixed buffer).
    B_bar : torch.Tensor, shape (order, 1)
        Discretized input matrix (fixed buffer).
    W_h : torch.Tensor, shape (hidden_dim, hidden_dim)
        Hidden-to-hidden weights (fixed buffer). Present only when
        ``nonlinear_hidden=True``.
    W_m : torch.Tensor, shape (hidden_dim, input_dim * order)
        Memory-to-hidden weights (fixed buffer). Present only when
        ``nonlinear_hidden=True``.
    W_x : torch.Tensor, shape (hidden_dim, input_dim)
        Input-to-hidden weights (fixed buffer). Present only when
        ``nonlinear_hidden=True``.
    b : torch.Tensor, shape (hidden_dim,)
        Hidden bias (fixed buffer). Present only when
        ``nonlinear_hidden=True``.

    See Also
    --------
    resdag.layers.reservoirs.lmu.LMULayer : Layer that sequences this cell.
    resdag.layers.cells.base_cell.ReservoirCell : Abstract cell interface.
    """

    def __init__(
        self,
        input_dim: int,
        order: int = 8,
        theta: float = 1.0,
        dt: float = 1.0,
        discretization: Literal["zoh", "euler"] = "zoh",
        nonlinear_hidden: bool = False,
        hidden_dim: int = 64,
        w_x_scale: float = 1.0,
    ) -> None:
        super().__init__()

        if order > 256:
            warnings.warn(
                f"LMU order={order} > 256 may cause numerical instability in "
                "the matrix exponential during ZOH discretization.",
                UserWarning,
                stacklevel=2,
            )

        self.input_dim = input_dim
        self._order = order
        self._theta = theta
        self._dt = dt
        self._discretization = discretization
        self._nonlinear_hidden = nonlinear_hidden
        self._hidden_dim = hidden_dim if nonlinear_hidden else 0
        self._memory_dim = input_dim * order

        # Compute discretized LTI matrices in float64, then convert
        A_np, B_np = _compute_legendre_matrices(order)
        A_bar_f64, B_bar_f64 = _discretize(A_np, B_np, theta, dt, discretization)

        self.register_buffer("A_bar", A_bar_f64.to(torch.float32))  # (order, order)
        self.register_buffer("B_bar", B_bar_f64.to(torch.float32))  # (order, 1)

        if nonlinear_hidden:
            self._init_nonlinear(input_dim, order, hidden_dim, w_x_scale)

    def _init_nonlinear(
        self,
        input_dim: int,
        order: int,
        hidden_dim: int,
        w_x_scale: float,
    ) -> None:
        """Initialize frozen random weights for the nonlinear hidden state."""
        memory_dim = input_dim * order

        # W_h: (hidden_dim, hidden_dim) — scaled to spectral radius < 1
        W_h = torch.empty(hidden_dim, hidden_dim)
        nn.init.uniform_(W_h, -1.0, 1.0)
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvals(W_h)
            sr = torch.max(torch.abs(eigenvalues)).real.item()
            if sr > 0:
                W_h = W_h * (0.9 / sr)
        self.register_buffer("W_h", W_h)

        # W_m: (hidden_dim, memory_dim)
        W_m = torch.empty(hidden_dim, memory_dim)
        nn.init.uniform_(W_m, -1.0, 1.0)
        self.register_buffer("W_m", W_m)

        # W_x: (hidden_dim, input_dim), scaled by w_x_scale
        W_x = torch.empty(hidden_dim, input_dim)
        nn.init.uniform_(W_x, -1.0, 1.0)
        self.register_buffer("W_x", W_x * w_x_scale)

        # b: (hidden_dim,), uniform in [-1, 1]
        b = torch.empty(hidden_dim)
        nn.init.uniform_(b, -1.0, 1.0)
        self.register_buffer("b", b)

    # ------------------------------------------------------------------
    # ReservoirCell interface
    # ------------------------------------------------------------------

    @property
    def state_size(self) -> int:
        """Dimensionality of the full state vector."""
        return self._memory_dim + self._hidden_dim

    @property
    def memory_dim(self) -> int:
        """Dimensionality of the linear memory portion of the state."""
        return self._memory_dim

    @property
    def output_dim(self) -> int:
        """Output dimensionality (equals ``state_size``)."""
        return self.state_size

    @property
    def order(self) -> int:
        """Order of the Legendre approximation."""
        return self._order

    @property
    def theta(self) -> float:
        """Window length (time constant)."""
        return self._theta

    def forward(
        self,
        inputs: list[torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the next LMU state for a single timestep.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            Per-timestep input slices.  ``inputs[0]`` is the feedback slice
            of shape ``(batch, feedback_size)``.  If a driving input is
            present, ``inputs[1]`` has shape ``(batch, driving_size)``.
            All elements are concatenated along the feature axis to form
            the full input ``x`` of shape ``(batch, input_dim)``.
        state : torch.Tensor
            Current state of shape ``(batch, state_size)``.

        Returns
        -------
        torch.Tensor
            Next state of shape ``(batch, state_size)``.
        """
        # Concatenate all input streams into x: (batch, input_dim)
        x = torch.cat(inputs, dim=-1) if len(inputs) > 1 else inputs[0]
        batch = x.shape[0]

        # Extract flat memory: (batch, input_dim * order)
        m_flat = state[:, : self._memory_dim]
        # Reshape to (batch, input_dim, order) for vectorised LTI update
        m = m_flat.reshape(batch, self.input_dim, self._order)

        # Linear memory update (no activation — linearity is the key property):
        #   m_new[k] = A_bar @ m[k] + B_bar * x[k]   for each input dim k
        #
        # Vectorised:
        #   m @ A_bar.T  : (batch, input_dim, order) @ (order, order)
        #   x.unsqueeze(-1) * B_bar.T : (batch, input_dim, 1) * (1, order)
        A_bar = self.A_bar.to(x.dtype)
        B_bar = self.B_bar.to(x.dtype)
        m_new = m @ A_bar.T + x.unsqueeze(-1) * B_bar.T  # (batch, input_dim, order)
        m_new_flat = m_new.reshape(batch, self._memory_dim)

        if not self._nonlinear_hidden:
            return m_new_flat

        # Nonlinear hidden state: h_new = tanh(W_h @ h + W_m @ m + W_x @ x + b)
        h = state[:, self._memory_dim :]  # (batch, hidden_dim)
        h_new = torch.tanh(
            F.linear(h, self.W_h.to(x.dtype))
            + F.linear(m_new_flat, self.W_m.to(x.dtype))
            + F.linear(x, self.W_x.to(x.dtype))
            + self.b.to(x.dtype)
        )
        return torch.cat([m_new_flat, h_new], dim=-1)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"LMUCell("
            f"input_dim={self.input_dim}, "
            f"order={self._order}, "
            f"theta={self._theta}, "
            f"dt={self._dt}, "
            f"discretization='{self._discretization}', "
            f"nonlinear_hidden={self._nonlinear_hidden}"
            f")"
        )
