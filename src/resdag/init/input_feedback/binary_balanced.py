"""Binary balanced initializer using Hadamard structure."""

from math import gcd

import numpy as np
import torch

from .base import InputFeedbackInitializer, _resolve_shape
from .registry import register_input_feedback


@register_input_feedback(
    "binary_balanced", input_scaling=None, balance_global=True, step=None, seed=None
)
class BinaryBalancedInitializer(InputFeedbackInitializer):
    """Deterministic binary balanced initializer using Walsh-Hadamard structure.

    Generates a dense input matrix with entries in {-1, +1}, column-wise balance
    (sum close to zero), and low inter-column correlation via truncated
    Walsh-Hadamard structure.

    The matrix is built **without randomness** (deterministic) and is suitable for
    ESN/RC setups where each column (reservoir unit) should receive a balanced mix
    of positive/negative signs from the input channels, and columns should be nearly
    orthogonal.

    Parameters
    ----------
    input_scaling : float, optional
        Scaling factor applied to the {-1, +1} matrix.
    balance_global : bool, default=True
        When rows is odd, enforce near-equal global count of +1 vs -1 column-sums
        by flipping full columns.
    step : int, optional
        Preferred column selection step. If None or not coprime with L, the
        initializer will choose the smallest odd step that is coprime with L.
    seed : int, optional
        Unused (kept for API compatibility). The initializer is fully deterministic.

    Notes
    -----
    - Column-wise balance: sum over rows per column is exactly 0 if rows even, else ±1.
    - Low correlation: columns originate from orthogonal Hadamard columns.
    - Deterministic: no RNG is used; the output is reproducible given the shape.

    Examples
    --------
    >>> from resdag.init.input_feedback import BinaryBalancedInitializer
    >>>
    >>> init = BinaryBalancedInitializer(input_scaling=0.5, balance_global=True)
    >>> weight = torch.empty(100, 10)  # (reservoir_size, input_dim)
    >>> init.initialize(weight)
    >>>
    >>> # Each column will have sum close to 0 (balanced)
    >>> column_sums = weight.sum(dim=0)
    >>> print(column_sums)  # Close to zero
    """

    def __init__(
        self,
        input_scaling: float | None = None,
        balance_global: bool = True,
        step: int | None = None,
        seed: int | None = None,  # Unused, for API consistency
    ) -> None:
        """Initialize the BinaryBalancedInitializer."""
        self.input_scaling = input_scaling
        self.balance_global = balance_global
        self.step = step
        self.seed = seed  # Kept but unused

    @staticmethod
    def _next_pow2(x: int) -> int:
        """Return the next power of two >= x."""
        return 1 << (x - 1).bit_length()

    @staticmethod
    def _hadamard(L: int) -> np.ndarray:
        """Construct Sylvester Hadamard matrix H_L ∈ {+1, -1}^{L×L}."""
        H = np.array([[1]], dtype=np.int8)
        while H.shape[0] < L:
            H = np.block([[H, H], [H, -H]]).astype(np.int8)
        return H

    @staticmethod
    def _choose_step(L: int, preferred: int | None = None) -> int:
        """Choose a step s coprime with L for column selection."""
        if preferred is not None and gcd(preferred, L) == 1:
            return int(preferred)
        s = 1
        while gcd(s, L) != 1:
            s += 2
        return s

    @staticmethod
    def _balance_columns_zero_sum(Vw: np.ndarray) -> None:
        """Make each column sum to 0 by minimal sign flips (in-place)."""
        n_work, m = Vw.shape
        for j in range(m):
            s_j = int(Vw[:, j].sum())
            if s_j == 0:
                continue
            # Flip signs bottom-to-top
            for i in range(n_work - 1, -1, -1):
                if s_j == 0:
                    break
                if s_j > 0 and Vw[i, j] == 1:
                    Vw[i, j] = -1
                    s_j -= 2
                elif s_j < 0 and Vw[i, j] == -1:
                    Vw[i, j] = 1
                    s_j += 2

    @staticmethod
    def _delete_least_bias_row(Vw: np.ndarray) -> np.ndarray:
        """Delete one row deterministically to minimize overall bias."""
        row_sums = Vw.sum(axis=1)
        r_del = int(np.argmin(np.abs(row_sums)))
        return np.delete(Vw, r_del, axis=0)

    @staticmethod
    def _balance_global_column_counts(V: np.ndarray) -> None:
        """Balance global counts of +1 and -1 column-sums (in-place)."""
        m = V.shape[1]
        col_sums = V.sum(axis=0)
        total = int(col_sums.sum())
        target = 0 if (m % 2 == 0) else 1

        if total > target:
            flips = (total - target) // 2
            cnt = 0
            for j in range(m):
                if col_sums[j] == 1 and cnt < flips:
                    V[:, j] *= -1
                    col_sums[j] = -1
                    cnt += 1
        elif total < -target:
            flips = (-target - total) // 2
            cnt = 0
            for j in range(m):
                if col_sums[j] == -1 and cnt < flips:
                    V[:, j] *= -1
                    col_sums[j] = 1
                    cnt += 1

    def initialize(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        """Initialize weight tensor with binary balanced structure.

        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor of shape (out_features, in_features)

        Returns
        -------
        torch.Tensor
            Initialized weight tensor
        """
        out_features, in_features = _resolve_shape(weight)
        device = weight.device
        dtype = weight.dtype

        rows, cols = out_features, in_features

        if rows == 0 or cols == 0:
            weight.zero_()
            return weight

        # Work with even row count
        n_work = rows if (rows % 2 == 0) else rows + 1

        # Build Hadamard matrix
        L = self._next_pow2(max(n_work, cols + 1, 2))
        H = self._hadamard(L)

        s = self._choose_step(L, self.step)
        idxs = [1 + (j * s) % (L - 1) for j in range(cols)]  # Exclude DC column

        Vw = H[:n_work, idxs].copy()

        # Balance each column to sum zero
        self._balance_columns_zero_sum(Vw)

        # If rows is even, done; else delete one row and optionally balance
        if n_work == rows:
            V = Vw
        else:
            V = self._delete_least_bias_row(Vw)
            if self.balance_global:
                self._balance_global_column_counts(V)

        values = V.astype(np.float32)

        if self.input_scaling is not None:
            values *= float(self.input_scaling)

        # Convert to tensor and copy to weight
        weight_data = torch.from_numpy(values).to(device=device, dtype=dtype)

        with torch.no_grad():
            weight.copy_(weight_data)

        return weight

    def __repr__(self) -> str:
        return (
            f"BinaryBalancedInitializer(input_scaling={self.input_scaling}, "
            f"balance_global={self.balance_global}, step={self.step})"
        )
