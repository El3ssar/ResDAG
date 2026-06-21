"""
Input/Feedback Initializer Base Class
=====================================

This module provides the abstract base class for all input and feedback
weight initializers used in reservoir layers, along with the **shared
scaling/connectivity contract** every initializer honors.

The scaling contract
--------------------
``input_scaling`` is the single, uniform magnitude knob shared by every
initializer:

- ``input_scaling=None`` means **no scaling** — the initializer returns its
  natural-range matrix untouched (the identity transform).
- ``input_scaling=s`` (a finite float) applies ``W <- s * W`` as the
  *documented final transform*, so the relevant magnitude statistic of the
  matrix scales **linearly** with ``s``.

The "relevant magnitude statistic" is ``max|W|`` for the elementwise
initializers (random, chessboard, chebyshev, ...) and the **per-channel L2
norm** for the structured ring initializers (``opposite_anchors``,
``ring_window``), whose value *is* the scaling target. Either way,
``input_scaling=0.5`` halves it and ``input_scaling=2.0`` doubles it.

``connectivity`` is an optional density knob: the fraction of nonzero entries
to keep per column (input channel). ``connectivity=None`` (the default) leaves
the produced sparsity pattern untouched; a float in ``(0, 1]`` randomly zeros a
fraction of each column's entries. Structured initializers that *define* their
own connectivity pattern (only the core ring receives input, a fixed window per
channel, ...) document whether they honor this knob.

Subclasses cooperate with the contract by:

1. calling ``super().__init__(input_scaling=..., connectivity=..., seed=...)``,
2. building their natural-range matrix, and
3. returning ``self._apply_scaling(self._apply_connectivity(values, ...))`` —
   or applying only the pieces that make sense for their structure.

Classes
-------
InputFeedbackInitializer
    Abstract base class for weight initialization (owns the scaling contract).

See Also
--------
resdag.init.input_feedback.registry : Registry of available initializers.
resdag.layers.ESNLayer : Uses these initializers for weight matrices.
"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar, cast

import numpy as np
import torch

from resdag.utils.general import SeedLike, coerce_seed_to_int, create_torch_generator

# ``_apply_scaling`` / ``_apply_connectivity`` are dtype-preserving on the two
# array kinds initializers build with: ``numpy.ndarray`` and ``torch.Tensor``.
ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)


class InputFeedbackInitializer(ABC):
    """
    Abstract base class for input/feedback weight initialization.

    All input/feedback weight initializers inherit from this class, implement
    :meth:`initialize`, and honor the **shared scaling contract** owned here.
    These initializers create weight matrices for:

    - Input connections: shape ``(reservoir_size, input_size)``
    - Feedback connections: shape ``(reservoir_size, feedback_size)``

    The scaling contract
    --------------------
    ``input_scaling`` is the single uniform magnitude knob:

    - ``None`` — no scaling; the natural-range matrix is returned as-is.
    - ``s`` (finite float) — multiply the whole matrix by ``s`` as the
      documented final transform, so the matrix's magnitude statistic
      (``max|W|`` for elementwise initializers, per-channel L2 norm for the
      structured ring initializers) scales linearly with ``s``.

    ``connectivity`` is an optional density knob in ``(0, 1]``: the fraction of
    nonzero entries kept per column. ``None`` leaves the produced sparsity
    untouched.

    Subclasses must implement :meth:`initialize`, which builds the weight
    tensor and returns it (modified in-place). They opt into the contract via
    :meth:`_apply_scaling` (and optionally :meth:`_apply_connectivity`).

    Parameters
    ----------
    input_scaling : float, optional
        Uniform multiplicative scaling applied as the final transform. ``None``
        (the default) applies no scaling. A float ``s`` scales the matrix's
        magnitude statistic linearly (``input_scaling=0.5`` halves it).
    connectivity : float, optional
        Fraction of nonzero entries to keep per column, in ``(0, 1]``. ``None``
        (the default) leaves the produced sparsity pattern untouched.
    seed : int, torch.Generator, or None, optional
        Reproducibility seed for any subclass randomness (the uniform/binary
        draws of the random initializers) and the connectivity mask. Accepts a
        plain ``int``, a :class:`torch.Generator` (whose ``initial_seed()`` is
        used so a generator and the equivalent int agree), or ``None`` (defer to
        the global RNG). The torch-native random initializers draw on the target
        tensor's device via :meth:`_torch_generator_for`, so the same ``seed``
        is reproducible per device — including on CUDA.

    Examples
    --------
    Creating a custom initializer that honors the contract:

    >>> class MyInitializer(InputFeedbackInitializer):
    ...     def initialize(self, weight, **kwargs):
    ...         values = torch.empty_like(weight).uniform_(-1, 1)
    ...         values = self._apply_scaling(values)
    ...         with torch.no_grad():
    ...             weight.copy_(values)
    ...         return weight
    >>>
    >>> initializer = MyInitializer(input_scaling=0.5)
    >>> weight = torch.empty(100, 10)
    >>> initializer(weight)

    Using with ESNLayer:

    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     reservoir_size=100,
    ...     feedback_size=10,
    ...     feedback_initializer=MyInitializer(input_scaling=0.5),
    ... )

    See Also
    --------
    resdag.init.input_feedback.registry : Get initializers by name.
    resdag.layers.ESNLayer : Uses these initializers.
    """

    def __init__(
        self,
        input_scaling: float | None = None,
        connectivity: float | None = None,
        seed: SeedLike = None,
    ) -> None:
        """Store and validate the shared scaling/connectivity parameters."""
        self.input_scaling = _validate_input_scaling(input_scaling)
        self.connectivity = _validate_connectivity(connectivity)
        self.seed = seed

    @abstractmethod
    def initialize(self, weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Initialize a weight tensor.

        Parameters
        ----------
        weight : torch.Tensor
            2D tensor of shape ``(reservoir_size, input_size)`` to initialize.
            Modified in-place.
        **kwargs
            Additional keyword arguments for specific initializers.

        Returns
        -------
        torch.Tensor
            The initialized weight tensor (same as input, modified in-place).
        """
        pass

    def _apply_scaling(self, values: ArrayT) -> ArrayT:
        """Apply the uniform ``input_scaling`` transform to ``values``.

        This is the documented final transform of the scaling contract:
        ``input_scaling=None`` returns ``values`` unchanged, while a finite
        float ``s`` returns ``s * values``. The array dtype is preserved, so
        callers may apply this to a float64 intermediate before narrowing.

        Parameters
        ----------
        values : numpy.ndarray or torch.Tensor
            The natural-range weight matrix to scale.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            ``values`` unchanged when ``input_scaling is None``; otherwise
            ``input_scaling * values`` (a new array; ``values`` is not mutated).
        """
        if self.input_scaling is None:
            return values
        if isinstance(values, np.ndarray):
            # Cast the scalar to the array dtype so a float32 intermediate stays
            # float32 (a bare Python float would upcast the product to float64).
            scaled = values * np.asarray(self.input_scaling, dtype=values.dtype)
            return cast(ArrayT, scaled)
        return values * self.input_scaling

    def _apply_connectivity(
        self,
        values: ArrayT,
        rng: np.random.Generator | None = None,
    ) -> ArrayT:
        """Sparsify ``values`` to the configured per-column ``connectivity``.

        For each column (input channel) a random fraction ``1 - connectivity``
        of the entries is zeroed, keeping at least one nonzero per column. When
        ``connectivity is None`` the array is returned untouched.

        Parameters
        ----------
        values : numpy.ndarray or torch.Tensor
            The dense (or natural-sparsity) weight matrix to mask.
        rng : numpy.random.Generator, optional
            Generator for the mask. Defaults to ``numpy.random.default_rng``
            seeded with ``self.seed`` so masking is reproducible.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            ``values`` unchanged when ``connectivity is None``; otherwise a
            masked copy with the requested per-column density.
        """
        if self.connectivity is None:
            return values
        if rng is None:
            # Reduce a torch.Generator/int/None seed to the int/None NumPy
            # default_rng accepts, so a generator seed still masks reproducibly.
            rng = np.random.default_rng(coerce_seed_to_int(self.seed))

        n_rows = int(values.shape[0])
        n_cols = int(values.shape[1])
        keep = max(1, int(round(self.connectivity * n_rows)))

        mask = np.zeros((n_rows, n_cols), dtype=bool)
        for col in range(n_cols):
            idx = rng.choice(n_rows, size=keep, replace=False)
            mask[idx, col] = True

        if isinstance(values, np.ndarray):
            return np.where(mask, values, np.zeros_like(values))
        torch_mask = torch.from_numpy(mask).to(device=values.device)
        return torch.where(torch_mask, values, torch.zeros_like(values))

    def _torch_generator_for(self, device: torch.device) -> torch.Generator:
        """Build a device-native :class:`torch.Generator` seeded from ``self.seed``.

        This is the single torch-RNG entry point the torch-native random
        initializers use, so every draw happens **on the target tensor's
        device** (no CPU build + copy) and is reproducible per device under the
        same ``seed``.

        The seed is first reduced to a plain ``int`` via
        :func:`~resdag.utils.general.coerce_seed_to_int` so that an ``int`` seed
        ``k`` and a ``torch.Generator().manual_seed(k)`` agree; the resulting
        generator is then created freshly on ``device``. When ``self.seed is
        None`` the generator is seeded from torch's global RNG, so global
        ``torch.manual_seed`` still propagates.

        Parameters
        ----------
        device : torch.device
            Device the draws must happen on (the target weight's device).

        Returns
        -------
        torch.Generator
            A generator on ``device``, deterministic in ``self.seed``.

        Notes
        -----
        torch's CPU and CUDA RNG streams differ, so the *same* seed yields a
        different (but per-device reproducible) matrix on each backend. Two
        builds on the **same** device with the same seed are byte-identical.
        """
        seed_int = coerce_seed_to_int(self.seed)
        return create_torch_generator(seed_int, device=device)

    def __call__(self, weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Callable interface for initialization.

        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor to initialize.
        **kwargs
            Additional arguments passed to :meth:`initialize`.

        Returns
        -------
        torch.Tensor
            The initialized weight tensor.
        """
        return self.initialize(weight, **kwargs)


def _validate_input_scaling(input_scaling: float | None) -> float | None:
    """Validate the shared ``input_scaling`` parameter.

    Parameters
    ----------
    input_scaling : float or None
        ``None`` (no scaling) or a finite float.

    Returns
    -------
    float or None
        ``None`` unchanged, otherwise ``float(input_scaling)``.

    Raises
    ------
    ValueError
        If ``input_scaling`` is not finite.
    """
    if input_scaling is None:
        return None
    value = float(input_scaling)
    if not np.isfinite(value):
        raise ValueError(f"input_scaling must be a finite float or None; got {input_scaling!r}.")
    return value


def _validate_connectivity(connectivity: float | None) -> float | None:
    """Validate the shared ``connectivity`` parameter.

    Parameters
    ----------
    connectivity : float or None
        ``None`` (keep the produced sparsity) or a float in ``(0, 1]``.

    Returns
    -------
    float or None
        ``None`` unchanged, otherwise ``float(connectivity)``.

    Raises
    ------
    ValueError
        If ``connectivity`` is not in ``(0, 1]``.
    """
    if connectivity is None:
        return None
    value = float(connectivity)
    if not (0.0 < value <= 1.0):
        raise ValueError(f"connectivity must be in (0, 1] or None; got {connectivity!r}.")
    return value


def _resolve_shape(weight: torch.Tensor) -> tuple[int, int]:
    """Resolve tensor shape, ensuring it's 2D."""
    if weight.ndim != 2:
        raise ValueError(f"Weight must be 2D, got shape {weight.shape}")
    return weight.shape[0], weight.shape[1]


def _numpy_compute_dtype(weight_dtype: torch.dtype) -> type[np.floating]:
    """Pick the NumPy dtype to build initializer intermediates in.

    Structured initializers compute their values in NumPy before copying the
    result into the target ``weight``. The intermediate dtype must match the
    target's precision so that no precision is silently truncated: building a
    ``float32`` intermediate for a ``float64`` weight rounds every value to
    single precision before the widening ``.to()`` ever runs.

    The mapping mirrors the target precision exactly where NumPy has a matching
    type, and falls back to ``float64`` for the half-precision torch dtypes
    (``float16``/``bfloat16``), which have no convenient NumPy compute type — the
    final ``.to(dtype=weight.dtype)`` narrows the double-precision result down to
    the target with a single rounding.

    Parameters
    ----------
    weight_dtype : torch.dtype
        Dtype of the target weight tensor.

    Returns
    -------
    type of numpy.floating
        ``numpy.float32`` for ``torch.float32`` targets, otherwise
        ``numpy.float64``. Computing a ``float32`` target in ``float32`` keeps
        its result bit-for-bit identical to the historical behavior, while every
        other target gains genuine double-precision intermediates.

    Examples
    --------
    >>> import torch
    >>> _numpy_compute_dtype(torch.float64) is np.float64
    True
    >>> _numpy_compute_dtype(torch.float32) is np.float32
    True
    """
    if weight_dtype == torch.float32:
        return np.float32
    return np.float64


def _resolve_scaling_alias(
    input_scaling: float | None,
    gain: float | None,
    *,
    default: float | None,
) -> float | None:
    """Resolve the deprecated ``gain`` alias to ``input_scaling``.

    Some structured initializers historically exposed their magnitude knob as
    ``gain``. ``gain`` is now a deprecated alias for ``input_scaling`` with
    identical meaning (the per-channel L2 norm target). This helper centralizes
    the alias handling: it emits a :class:`DeprecationWarning` when ``gain`` is
    supplied, rejects supplying *both* names, and falls back to ``default`` when
    neither is given.

    Parameters
    ----------
    input_scaling : float or None
        The new-name value as passed by the caller (``None`` if unset).
    gain : float or None
        The deprecated-name value as passed by the caller (``None`` if unset).
    default : float or None
        Value to use when neither ``input_scaling`` nor ``gain`` is supplied.

    Returns
    -------
    float or None
        The resolved scaling value.

    Raises
    ------
    ValueError
        If both ``input_scaling`` and ``gain`` are supplied.
    """
    if gain is not None:
        if input_scaling is not None:
            raise ValueError(
                "Pass only one of 'input_scaling' or the deprecated 'gain' (they are aliases)."
            )
        import warnings

        warnings.warn(
            "'gain' is a deprecated alias for 'input_scaling' and will be removed in a "
            "future release; pass 'input_scaling' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return gain
    if input_scaling is not None:
        return input_scaling
    return default
