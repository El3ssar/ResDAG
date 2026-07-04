"""
Topology Initializer Base Classes
=================================

This module provides base classes for topology-based weight initialization
in reservoir computing networks. Topologies define the structure of the
recurrent weight matrix — via graph connectivity (:class:`GraphTopology`)
or via any function that builds a matrix directly
(:class:`MatrixTopology`).

Classes
-------
TopologyInitializer
    Abstract base class for topology initializers.
GraphTopology
    Concrete implementation using NetworkX graphs.
MatrixTopology
    Concrete implementation wrapping any matrix-building callable.

See Also
--------
resdag.init.graphs : Graph generation functions.
resdag.init.matrices : Direct matrix-construction functions.
resdag.layers.ESNLayer : Uses topologies for weight initialization.
"""

import inspect
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, cast, overload

import networkx as nx
import numpy as np
import torch

from resdag.utils.general import create_torch_generator

# Below this size the dense ``eigvals`` path is already cheap and exact, so it
# is used directly rather than any iterative (approximate) method.  The dense
# ``O(n^3)`` eigendecomposition only becomes worth avoiding well past ~1000.
_DENSE_EIGVALS_MAX_N = 1000

# A matrix at or below this non-zero fraction takes the sparse CSR ``eigs``
# path; denser matrices go through the (exact or convergence-verified) dense
# path. Every path is exact or verified, so the estimate does NOT depend on
# which side of this threshold a matrix falls — the threshold only trades a CSR
# Arnoldi iteration against a dense matmul-based one.
_SPARSE_DENSITY_THRESHOLD = 0.1

# Number of eigenvalues ARPACK converges in the sparse ``eigs`` path.  With
# ``k=1``, ``which='LM'`` ARPACK can lock onto a sub-dominant Ritz value and
# miss the true largest by up to ~0.2% when the top eigenvalues are (near-)
# degenerate — a common case for reservoir adjacency spectra.  Converging a
# small cluster and taking ``max|lambda|`` over it reliably captures the
# dominant one (accurate to ~1e-4) at negligible extra cost.
_EIGS_K = 6

# Number of Arnoldi/Lanczos basis vectors scipy's ``eigs`` builds per restart.
# The default (``max(2*k+1, 20)``) is too small to converge the dominant
# eigenvalue of a circular-law reservoir matrix; a slightly larger subspace
# makes the iteration both faster and accurate to ~1e-8.
_EIGS_NCV = 48

# Power-iteration controls (scipy-less dense GPU-resident fallback).  Reservoir
# matrices from the circular law have a tight spectral gap, so a generous fixed
# budget with an early-exit tolerance trades a few cheap matmuls for accuracy.
_POWER_ITER_MAX_ITERS = 1000
_POWER_ITER_TOL = 1e-7
# Budget for the dense-path *probe*: a matrix with a well-separated real dominant
# eigenvalue converges in a handful of steps and wins over an O(n^3) eigvals,
# but the circular-law reservoir norm never converges — so the probe bails after
# this many cheap O(n^2) matmuls and hands off to the exact eigendecomposition,
# instead of burning the full 1000-iteration budget only to fall back anyway.
_POWER_ITER_DENSE_PROBE = 64


def estimate_spectral_radius(matrix: torch.Tensor) -> float:
    """Estimate the spectral radius (largest ``|eigenvalue|``) of a matrix.

    Every path is exact or convergence-verified, so — unlike the previous
    ``||A v||`` power iteration — the estimate is never silently biased by a
    complex-conjugate dominant eigenvalue pair (which real non-symmetric
    reservoir matrices almost always have), and it never depends on which side
    of the density threshold a matrix falls on:

    - **Small matrices** (``n <= 1000``) use the exact dense
      :func:`torch.linalg.eigvals` — already fast there and free of any
      approximation error.
    - **Large sparse matrices** (density ``<= 0.1``) use
      :func:`scipy.sparse.linalg.eigs` with ``k=1, which='LM'`` on a CSR copy,
      an Arnoldi iteration that never densifies the operator and returns the
      dominant *complex* eigenvalue directly.
    - **Large dense matrices** try a complex-pair-robust power iteration (the
      two-step geometric ratio ``sqrt(||A^2 v||/||v||)``); if it does not
      converge within tolerance — a tight spectral gap defeats it — the exact
      dense :func:`torch.linalg.eigvals` is used, so a biased estimate is never
      returned. scipy ``eigs`` is not used for the dense case: its dense matvec
      is a full ``O(n^2)`` matmul per Arnoldi step and, at reservoir sizes, is
      slower than the LAPACK dense eigendecomposition.

    Parameters
    ----------
    matrix : torch.Tensor
        Square matrix whose spectral radius is wanted.

    Returns
    -------
    float
        The estimated largest absolute eigenvalue ``max|lambda|``.

    See Also
    --------
    scale_to_spectral_radius : Uses this to rescale a matrix to a target radius.
    """
    n = matrix.shape[0]

    if n <= _DENSE_EIGVALS_MAX_N:
        return _dense_eigvals_radius(matrix)

    if _is_sparse(matrix):
        radius = _sparse_eigs_radius(matrix)
        if radius is not None:
            return radius

    # Dense (or scipy unavailable): a short complex-pair-robust power-iteration
    # probe, with the exact dense eigendecomposition as the guaranteed-correct
    # fallback. The probe converges cheaply only when the dominant eigenvalue is
    # well separated; the circular-law reservoir norm does not converge, so the
    # small budget bails quickly to eigvals instead of wasting the full budget.
    estimate, converged = _power_iteration_radius(
        matrix, max_iters=_POWER_ITER_DENSE_PROBE, return_converged=True
    )
    if converged:
        return estimate
    return _dense_eigvals_radius(matrix)


def _dense_eigvals_radius(matrix: torch.Tensor) -> float:
    """Exact spectral radius via the dense (non-symmetric) eigendecomposition."""
    eigenvalues = torch.linalg.eigvals(matrix)
    return float(torch.max(torch.abs(eigenvalues)).item())


def _is_sparse(matrix: torch.Tensor) -> bool:
    """Return ``True`` when the matrix is sparse enough for the CSR ``eigs`` path.

    This chooses only the *representation* fed to :func:`scipy.sparse.linalg.eigs`
    (a CSR copy for sparse matrices, a dense-matvec ``LinearOperator`` otherwise);
    both feed the same Arnoldi iteration, so the resulting spectral-radius
    estimate is independent of which side of the threshold a matrix falls on.
    """
    n = matrix.shape[0]
    nnz = int(torch.count_nonzero(matrix).item())
    return nnz <= _SPARSE_DENSITY_THRESHOLD * n * n


def _sparse_eigs_radius(matrix: torch.Tensor) -> float | None:
    """Spectral radius of a sparse matrix via ``scipy.sparse.linalg.eigs``.

    Builds a CSR copy on the CPU and runs an Arnoldi iteration for the largest-
    magnitude eigenvalues (``which='LM'``), returning ``max|lambda|`` over the
    converged cluster. ``eigs`` returns *complex* eigenvalues directly, so
    complex-conjugate dominant pairs (the norm of real non-symmetric reservoir
    matrices) are handled exactly, unlike ``||A v||`` power iteration. A small
    cluster (``k > 1``) is requested rather than ``k=1`` because ARPACK can lock
    onto a sub-dominant Ritz value and miss the true largest when the top
    eigenvalues are near-degenerate. Returns ``None`` when scipy is unavailable
    or the iteration fails, so the caller can fall back.

    Parameters
    ----------
    matrix : torch.Tensor
        Square (sparse) matrix.

    Returns
    -------
    float or None
        ``max|lambda|`` if scipy succeeds, otherwise ``None``.
    """
    try:
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigs
    except ImportError:
        return None

    n = matrix.shape[0]
    # ``eigs`` requires ``k < n - 1`` and a Krylov subspace ``ncv`` with
    # ``k < ncv <= n``; for tiny matrices this can fail, so signal a fall back.
    if n <= 2:
        return None

    dense = matrix.detach().to(device="cpu", dtype=torch.float64).numpy()
    sparse = sp.csr_matrix(dense)
    k = min(_EIGS_K, n - 2)
    ncv = min(n, max(2 * k + 1, _EIGS_NCV))

    try:
        eigenvalues = eigs(
            sparse,
            k=k,
            which="LM",
            ncv=ncv,
            maxiter=n * 10,
            tol=1e-9,
            return_eigenvectors=False,
        )
    except Exception:
        return None

    return float(np.max(np.abs(eigenvalues)))


@overload
def _power_iteration_radius(
    matrix: torch.Tensor,
    max_iters: int = ...,
    tol: float = ...,
    return_converged: Literal[False] = ...,
) -> float: ...


@overload
def _power_iteration_radius(
    matrix: torch.Tensor,
    max_iters: int = ...,
    tol: float = ...,
    *,
    return_converged: Literal[True],
) -> tuple[float, bool]: ...


def _power_iteration_radius(
    matrix: torch.Tensor,
    max_iters: int = _POWER_ITER_MAX_ITERS,
    tol: float = _POWER_ITER_TOL,
    return_converged: bool = False,
) -> float | tuple[float, bool]:
    """Estimate ``max|lambda|`` of a dense matrix by power iteration.

    Iterates ``v <- A v / ||A v||`` but tracks the dominant eigenvalue
    magnitude via the **two-step geometric ratio** ``sqrt(||A^2 v|| / ||v||)``
    rather than the single-step ``||A v||``. For a real non-symmetric operator
    the dominant eigenvalue is usually a complex-conjugate pair ``r e^{±i θ}``:
    the single-step norm ratio then *oscillates* and never settles, but the
    two-step ratio converges to ``r^2`` regardless of ``θ``, so its square root
    recovers ``r``. This removes the systematic bias (measured up to ~5% of the
    realized radius) of the old ``||A v||`` estimator. All work stays on
    ``matrix.device`` and in ``matrix.dtype`` (promoted to at least
    ``float32``), so the estimate is GPU-resident.

    Parameters
    ----------
    matrix : torch.Tensor
        Square matrix.
    max_iters : int, optional
        Maximum number of iterations (fixed budget).
    tol : float, optional
        Relative-change early-exit tolerance on the eigenvalue estimate.
    return_converged : bool, optional
        If ``True``, return ``(estimate, converged)`` where ``converged`` is
        ``True`` only when the early-exit tolerance was met before the budget
        ran out. A non-converged estimate lets the caller fall back to the exact
        dense eigendecomposition instead of trusting a biased value. Defaults to
        ``False`` (returns the estimate alone, for backward compatibility).

    Returns
    -------
    float or tuple[float, bool]
        Estimated largest absolute eigenvalue, or ``(estimate, converged)`` when
        ``return_converged`` is ``True``.
    """
    n = matrix.shape[0]
    device = matrix.device
    # Power iteration needs floating point; promote integer/half matrices to a
    # stable working precision without forcing a host round-trip.
    work_dtype = matrix.dtype if matrix.dtype in (torch.float32, torch.float64) else torch.float32
    a = matrix.detach().to(dtype=work_dtype)

    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    v = torch.randn(n, generator=generator, device=device, dtype=work_dtype)
    v_norm = torch.linalg.vector_norm(v)
    if v_norm == 0:
        return (0.0, True) if return_converged else 0.0
    v = v / v_norm

    prev = torch.zeros((), device=device, dtype=work_dtype)
    estimate = prev
    converged = False
    for _ in range(max_iters):
        av = a @ v
        av_norm = torch.linalg.vector_norm(av)
        if av_norm <= 0:
            return (0.0, True) if return_converged else 0.0
        a2v = a @ av
        a2v_norm = torch.linalg.vector_norm(a2v)
        # ||A^2 v|| / ||v|| -> r^2 for a dominant complex pair of modulus r
        # (||v|| == 1 here); the sqrt recovers r without the single-step
        # oscillation of ||A v||.
        estimate = torch.sqrt(a2v_norm)
        # Renormalize on the *second* iterate so the two-step ratio stays a
        # clean geometric mean and the vector cannot overflow/underflow.
        v = a2v / a2v_norm
        if torch.abs(estimate - prev) <= tol * torch.clamp(estimate, min=1e-30):
            converged = True
            break
        prev = estimate

    result = float(estimate.item())
    return (result, converged) if return_converged else result


def scale_to_spectral_radius(matrix: torch.Tensor, target_radius: float) -> torch.Tensor:
    """Rescale a square matrix so its spectral radius equals ``target_radius``.

    The current spectral radius is obtained from
    :func:`estimate_spectral_radius`, which picks the exact dense ``eigvals``
    (small N), scipy ``eigs`` (large N, sparse or dense), or a power-iteration
    fallback automatically.  Returns the matrix unchanged when its current
    spectral radius is (numerically) zero — e.g. the ``zero`` topology or a
    nilpotent matrix.

    This is the single shared rescale implementation used by both
    :class:`GraphTopology`/:class:`MatrixTopology` and
    :class:`resdag.layers.cells.esn_cell.ESNCell`.

    Parameters
    ----------
    matrix : torch.Tensor
        Square matrix to rescale.
    target_radius : float
        Desired largest absolute eigenvalue.

    Returns
    -------
    torch.Tensor
        The rescaled matrix (new tensor).
    """
    current_radius = estimate_spectral_radius(matrix)

    if current_radius > 1e-8:
        matrix = matrix * (target_radius / current_radius)
    elif abs(target_radius) > 1e-8:
        # A nilpotent (all-eigenvalues-zero, e.g. the pure ``delay_line``) or
        # all-zero matrix cannot be rescaled to a nonzero radius — scaling a zero
        # spectrum leaves it zero. Warn rather than silently ignore the request,
        # so ``ESNLayer(topology="delay_line", spectral_radius=0.9)`` does not look
        # like it took effect. Control the scale through the topology's own weight
        # parameter instead.
        warnings.warn(
            f"spectral_radius={target_radius:g} was requested but the recurrent "
            f"matrix has a zero spectral radius (nilpotent or all-zero, e.g. the "
            f"'delay_line' / 'zeros' topologies); it cannot be rescaled to a nonzero "
            f"radius and is left unchanged. Set the scale via the topology's weight "
            f"parameter instead.",
            RuntimeWarning,
            stacklevel=2,
        )

    return matrix


def _warn_prescaled_override(topology_name: str, spectral_radius: float) -> None:
    """Warn that a layer ``spectral_radius`` is ignored by a pre-scaled topology.

    Pre-scaled topologies (``prescaled=True``) bake their own spectral structure
    into the recurrent matrix — graded per-clique radii, unit singular values, or
    an analytically fixed radius. The outer
    :func:`scale_to_spectral_radius` rescale would silently overwrite that
    structure, so it is skipped and this warning makes the (otherwise silent)
    collision of the two meanings of ``spectral_radius`` explicit and
    deterministic.

    Parameters
    ----------
    topology_name : str
        Name of the topology builder, used in the warning message.
    spectral_radius : float
        The ignored layer-level target spectral radius.
    """
    warnings.warn(
        f"Topology '{topology_name}' is pre-scaled: it bakes its own spectral "
        f"structure into the recurrent matrix, so the layer-level "
        f"spectral_radius={spectral_radius} is ignored (no outer rescale is "
        f"applied). Control the radius through the topology's own parameters, or "
        f"leave the layer spectral_radius as None to silence this warning.",
        UserWarning,
        stacklevel=3,
    )


class TopologyInitializer(ABC):
    """
    Abstract base class for topology-based weight initialization.

    Topology initializers convert graph structures into PyTorch weight tensors
    for reservoir layers. They extract the required size from the tensor shape,
    generate a graph, convert it to an adjacency matrix, and optionally apply
    spectral radius scaling.

    Subclasses must implement the :meth:`initialize` method.

    Attributes
    ----------
    prescaled : bool
        Whether the topology bakes its own spectral structure into the
        recurrent matrix (graded radii, unit singular values, an analytically
        fixed spectral radius, ...). When ``True``, :meth:`initialize` skips the
        outer :func:`scale_to_spectral_radius` rescale so that structure is
        preserved, and warns if a layer-level ``spectral_radius`` is also passed
        (the two collide). Defaults to ``False``.

    See Also
    --------
    GraphTopology : Concrete implementation using NetworkX graphs.
    resdag.layers.ESNLayer : Uses topology initializers.
    """

    #: Topologies that own their spectral scaling set this to ``True`` so the
    #: base classes skip the outer rescale.  Overridden per-instance in
    #: ``__init__`` of the concrete subclasses.
    prescaled: bool = False

    @abstractmethod
    def initialize(
        self,
        weight: torch.Tensor,
        spectral_radius: float | None = None,
    ) -> torch.Tensor:
        """
        Initialize a weight tensor using graph topology.

        Parameters
        ----------
        weight : torch.Tensor
            The weight tensor to initialize, shape ``(n, n)`` for recurrent
            weights. This tensor is modified in-place.
        spectral_radius : float, optional
            Target spectral radius for scaling. If None, no scaling is applied.

        Returns
        -------
        torch.Tensor
            The initialized weight tensor (same as input, modified in-place).
        """
        pass


class GraphTopology(TopologyInitializer):
    """
    Topology initializer based on NetworkX graph functions.

    This class wraps a graph generation function and converts it into a weight
    initializer. The graph function must accept ``n`` (number of nodes) as its
    first argument.

    Parameters
    ----------
    graph_func : callable
        A function with signature ``(n: int, **kwargs) -> nx.Graph | nx.DiGraph``.
        Must return a NetworkX graph with weighted edges.
    graph_kwargs : dict, optional
        Keyword arguments to pass to the graph function.
    prescaled : bool, default=False
        If ``True``, the graph builder already bakes its own spectral structure
        into the adjacency matrix (e.g. ``spectral_cascade``'s graded per-clique
        radii). The outer :func:`scale_to_spectral_radius` rescale is then
        skipped in :meth:`initialize`, and a warning is emitted if a
        ``spectral_radius`` is nonetheless requested, since the two meanings of
        ``spectral_radius`` collide.

    Attributes
    ----------
    graph_func : callable
        The graph generation function.
    graph_kwargs : dict
        Keyword arguments for the graph function.
    prescaled : bool
        Whether the outer spectral-radius rescale is suppressed (see above).

    Examples
    --------
    Using a registered graph function:

    >>> from resdag.init.graphs import erdos_renyi_graph
    >>> import torch
    >>>
    >>> topology = GraphTopology(erdos_renyi_graph, {"p": 0.1, "directed": True})
    >>> weight = torch.empty(100, 100)
    >>> topology.initialize(weight, spectral_radius=0.9)

    With the registry helper:

    >>> from resdag.init.topology import get_topology
    >>> topology = get_topology("erdos_renyi", p=0.15)
    >>> topology.initialize(weight, spectral_radius=0.95)

    See Also
    --------
    resdag.init.graphs : Available graph generation functions.
    get_topology : Get pre-configured topology by name.
    """

    def __init__(
        self,
        graph_func: Callable,
        graph_kwargs: dict[str, Any] | None = None,
        prescaled: bool = False,
    ):
        self.graph_func = graph_func
        self.graph_kwargs = graph_kwargs or {}
        self.prescaled = prescaled

    def initialize(
        self,
        weight: torch.Tensor,
        spectral_radius: float | None = None,
    ) -> torch.Tensor:
        """
        Initialize weight tensor from graph topology.

        Generates a graph using the stored function, converts it to an
        adjacency matrix, and optionally scales to the target spectral radius.

        When this topology is :attr:`prescaled`, the outer spectral-radius
        rescale is skipped (the builder's own spectral structure is kept) and a
        warning is emitted if ``spectral_radius`` is not ``None``.

        Parameters
        ----------
        weight : torch.Tensor
            Square tensor to initialize, shape ``(n, n)``.
        spectral_radius : float, optional
            Target spectral radius for the weight matrix. Ignored (with a
            warning) when the topology is :attr:`prescaled`.

        Returns
        -------
        torch.Tensor
            Initialized weight tensor (modified in-place).

        Raises
        ------
        ValueError
            If weight is not 2D or not square.
        """
        if weight.ndim != 2:
            raise ValueError(f"Weight must be 2D, got shape {weight.shape}")

        if weight.shape[0] != weight.shape[1]:
            raise ValueError(f"Weight must be square, got shape {weight.shape}")

        n = weight.shape[0]
        device = weight.device
        dtype = weight.dtype

        # Dense fast path: generators decorated with ``register_dense_adjacency``
        # advertise a vectorized NumPy builder, so the dense adjacency is built
        # directly and the (pure-overhead) NetworkX round-trip is skipped.
        adj_matrix = self._build_adjacency(n)

        # Convert to torch tensor
        weight_data = torch.from_numpy(adj_matrix).to(device=device, dtype=dtype)

        # Apply spectral radius scaling if requested.  Pre-scaled topologies bake
        # their own spectral structure into the adjacency matrix, so the outer
        # rescale is suppressed (and the collision is surfaced as a warning).
        if spectral_radius is not None:
            if self.prescaled:
                _warn_prescaled_override(
                    getattr(self.graph_func, "__name__", repr(self.graph_func)),
                    spectral_radius,
                )
            else:
                weight_data = self._scale_spectral_radius(weight_data, spectral_radius)

        # Copy into weight tensor
        with torch.no_grad():
            weight.copy_(weight_data)

        return weight

    def _build_adjacency(self, n: int) -> np.ndarray:
        """Build the ``(n, n)`` adjacency matrix for the wrapped graph function.

        Prefers the vectorized dense fast path when the graph function was
        decorated with
        :func:`~resdag.init.graphs._dense.register_dense_adjacency`: the dense
        weighted adjacency is built directly in NumPy and the NetworkX object is
        never materialised. Falls back to building the graph and converting it
        with :func:`networkx.to_numpy_array` for genuinely graph-shaped
        generators.

        Parameters
        ----------
        n : int
            Number of nodes / matrix dimension.

        Returns
        -------
        numpy.ndarray
            An ``(n, n)`` ``float32`` weighted adjacency matrix.

        Raises
        ------
        ValueError
            If the produced adjacency does not have shape ``(n, n)``.
        """
        from resdag.init.graphs._dense import dense_adjacency_builder

        adjacency_builder = dense_adjacency_builder(self.graph_func)
        if adjacency_builder is not None:
            adjacency = np.asarray(adjacency_builder(n, **self.graph_kwargs), dtype=np.float32)
            if adjacency.shape != (n, n):
                raise ValueError(
                    f"Dense adjacency builder produced shape {adjacency.shape}, "
                    f"expected ({n}, {n})"
                )
            return adjacency

        G = self.graph_func(n, **self.graph_kwargs)
        return self._graph_to_adjacency(G, n)

    def _graph_to_adjacency(
        self,
        G: nx.Graph | nx.DiGraph,
        n: int,
    ) -> np.ndarray:
        """Convert NetworkX graph to adjacency matrix."""
        adj_matrix = nx.to_numpy_array(
            G,
            nodelist=sorted(G.nodes()),
            weight="weight",
            dtype=np.float32,
        )

        if adj_matrix.shape != (n, n):
            raise ValueError(
                f"Graph produced adjacency matrix of shape {adj_matrix.shape}, expected ({n}, {n})"
            )

        return cast(np.ndarray, adj_matrix)

    def _scale_spectral_radius(
        self,
        weight: torch.Tensor,
        target_radius: float,
    ) -> torch.Tensor:
        """Scale weight matrix to target spectral radius."""
        return scale_to_spectral_radius(weight, target_radius)

    def __repr__(self) -> str:
        """Return string representation."""
        prescaled = ", prescaled=True" if self.prescaled else ""
        return (
            f"{self.__class__.__name__}(graph_func={self.graph_func.__name__}, "
            f"kwargs={self.graph_kwargs}{prescaled})"
        )


class MatrixTopology(TopologyInitializer):
    """
    Topology initializer wrapping any matrix-building callable.

    This is the general-purpose escape hatch of the topology system: any
    function with logic for constructing a recurrent weight matrix becomes a
    topology, with full access to spectral-radius scaling, the registry, and
    the ``ESNLayer(topology=...)`` shorthand. No graph required.

    Two calling conventions are supported, tried in order:

    1. **Build style** — ``fn(n, **kwargs)`` returning an ``(n, n)``
       array-like: a ``torch.Tensor``, a ``numpy.ndarray``, or even a
       ``networkx`` graph (converted to its adjacency matrix).
    2. **In-place style** — ``fn(tensor, **kwargs)`` mutating a tensor
       in place, e.g. any ``torch.nn.init.*_`` function.

    Parameters
    ----------
    matrix_func : callable
        The matrix-building function (build style or in-place style).
    matrix_kwargs : dict, optional
        Keyword arguments bound to the function.
    prescaled : bool, default=False
        If ``True``, the builder already fixes the matrix's spectral structure
        (e.g. ``orthogonal``'s unit singular values, or
        ``fast_spectral_initialization``'s analytically targeted radius). The
        outer :func:`scale_to_spectral_radius` rescale is then skipped in
        :meth:`initialize`, and a warning is emitted if a ``spectral_radius`` is
        nonetheless requested.
    seed : int, optional
        Reproducibility seed for in-place builders that take a ``generator=``
        keyword (e.g. :func:`torch.nn.init.orthogonal_`). At :meth:`initialize`
        time a :class:`torch.Generator` seeded from ``seed`` on the weight's
        device is passed to the builder, so two topologies built with the same
        seed produce bit-identical recurrent matrices. Builders that take a
        ``seed`` argument directly receive it through ``matrix_kwargs`` instead
        and ignore this. Defaults to ``None`` (draws from torch's global RNG).

    Examples
    --------
    A plain function as a topology:

    >>> import torch
    >>> def block_diagonal(n, blocks=4):
    ...     w = torch.zeros(n, n)
    ...     size = n // blocks
    ...     for b in range(blocks):
    ...         s = b * size
    ...         w[s : s + size, s : s + size] = torch.randn(size, size)
    ...     return w
    >>> topology = MatrixTopology(block_diagonal, {"blocks": 5})
    >>> weight = torch.empty(100, 100)
    >>> topology.initialize(weight, spectral_radius=0.9)

    Bare callables passed to ``ESNLayer`` are wrapped automatically:

    >>> from resdag.layers import ESNLayer
    >>> layer = ESNLayer(100, feedback_size=3, topology=block_diagonal)
    >>> layer = ESNLayer(100, feedback_size=3, topology=(block_diagonal, {"blocks": 2}))

    ``torch.nn.init`` functions work directly (in-place style):

    >>> layer = ESNLayer(100, feedback_size=3, topology=torch.nn.init.orthogonal_)

    See Also
    --------
    GraphTopology : Graph-based topologies.
    resdag.init.topology.register_matrix_topology : Register by name.
    """

    def __init__(
        self,
        matrix_func: Callable,
        matrix_kwargs: dict[str, Any] | None = None,
        prescaled: bool = False,
        seed: int | None = None,
    ):
        self.matrix_func = matrix_func
        self.matrix_kwargs = matrix_kwargs or {}
        self.prescaled = prescaled
        self.seed = seed

    def initialize(
        self,
        weight: torch.Tensor,
        spectral_radius: float | None = None,
    ) -> torch.Tensor:
        """
        Initialize a square weight tensor from the wrapped callable.

        When this topology is :attr:`prescaled`, the outer spectral-radius
        rescale is skipped (the builder's own spectral structure is kept) and a
        warning is emitted if ``spectral_radius`` is not ``None``.

        Parameters
        ----------
        weight : torch.Tensor
            Square tensor to initialize, shape ``(n, n)``. Modified in-place.
        spectral_radius : float, optional
            Target spectral radius. If None, no scaling is applied. Ignored
            (with a warning) when the topology is :attr:`prescaled`.

        Returns
        -------
        torch.Tensor
            Initialized weight tensor (modified in-place).

        Raises
        ------
        ValueError
            If ``weight`` is not 2-D square, if the callable matches neither
            calling convention, or if the built matrix has the wrong shape.
        """
        if weight.ndim != 2:
            raise ValueError(f"Weight must be 2D, got shape {weight.shape}")
        if weight.shape[0] != weight.shape[1]:
            raise ValueError(f"Weight must be square, got shape {weight.shape}")

        n = weight.shape[0]

        result = _call_matrix_builder(
            self.matrix_func, weight, (n,), self.matrix_kwargs, seed=self.seed
        )
        matrix = _coerce_to_matrix(result, (n, n), weight.device, weight.dtype)

        # Pre-scaled builders fix their own spectral structure (unit singular
        # values, analytic radius, ...), so the outer rescale is suppressed and
        # the collision is surfaced as a warning rather than silently applied.
        if spectral_radius is not None:
            if self.prescaled:
                _warn_prescaled_override(
                    getattr(self.matrix_func, "__name__", repr(self.matrix_func)),
                    spectral_radius,
                )
            else:
                matrix = scale_to_spectral_radius(matrix, spectral_radius)

        with torch.no_grad():
            weight.copy_(matrix)

        return weight

    def __repr__(self) -> str:
        """Return string representation."""
        name = getattr(self.matrix_func, "__name__", repr(self.matrix_func))
        prescaled = ", prescaled=True" if self.prescaled else ""
        return (
            f"{self.__class__.__name__}(matrix_func={name}, kwargs={self.matrix_kwargs}{prescaled})"
        )


def _accepts_generator(fn: Callable) -> bool:
    """Return ``True`` when ``fn``'s signature names a ``generator`` parameter.

    In-place initializers like :func:`torch.nn.init.orthogonal_` take a
    ``generator=`` keyword rather than a ``seed=`` one, so seeding them
    reproducibly means passing a :class:`torch.Generator`, not an int. When
    introspection fails (e.g. a C builtin without a signature) this returns
    ``False`` and the callable draws from torch's global RNG as before.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "generator" in params


def _call_matrix_builder(
    fn: Callable,
    weight: torch.Tensor,
    build_args: tuple[Any, ...],
    kwargs: dict[str, Any],
    seed: int | None = None,
) -> Any:
    """Call ``fn`` in build style, falling back to in-place style.

    Build style passes the dimensions (``build_args``) and expects a matrix
    back. In-place style passes a scratch tensor of the right shape and uses
    whatever the function leaves in it (``torch.nn.init.*_`` convention).
    Functions that take dimensions raise ``TypeError``/``AttributeError``
    when handed a tensor and vice versa, which is what the fallback keys on.

    Parameters
    ----------
    fn : callable
        The matrix builder (build style or in-place style).
    weight : torch.Tensor
        Destination tensor; its shape/device/dtype seed the scratch tensor.
    build_args : tuple
        Positional dimension arguments for the build-style call.
    kwargs : dict
        Keyword arguments bound to the call.
    seed : int, optional
        Reproducibility seed. When ``fn`` takes a ``generator=`` keyword (e.g.
        :func:`torch.nn.init.orthogonal_`), ``kwargs`` does not already bind one,
        and a ``seed`` is given, a :class:`torch.Generator` seeded from ``seed``
        on the scratch tensor's device is injected so two calls with the same
        seed produce bit-identical weights instead of drawing from torch's
        global RNG. Ignored for builders that take ``seed`` directly (which
        already thread it through ``kwargs``) or accept neither.
    """
    try:
        return fn(*build_args, **kwargs)
    except (TypeError, AttributeError) as build_err:
        scratch = torch.empty_like(weight, memory_format=torch.contiguous_format)
        inplace_kwargs = kwargs
        if seed is not None and "generator" not in kwargs and _accepts_generator(fn):
            generator = create_torch_generator(seed, device=scratch.device)
            inplace_kwargs = {**kwargs, "generator": generator}
        try:
            with torch.no_grad():
                out = fn(scratch, **inplace_kwargs)
        except Exception as inplace_err:
            fn_name = getattr(fn, "__name__", repr(fn))
            raise ValueError(
                f"Callable '{fn_name}' matches neither initializer convention. "
                f"Build style fn({', '.join(map(str, build_args))}, **kwargs) raised: "
                f"{build_err!r}. In-place style fn(tensor, **kwargs) raised: "
                f"{inplace_err!r}."
            ) from inplace_err
        return out if isinstance(out, torch.Tensor) else scratch


def _coerce_to_matrix(
    result: Any,
    expected_shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a builder's return value into a validated torch matrix.

    Accepts ``torch.Tensor``, ``numpy.ndarray`` (or anything ``np.asarray``
    understands, e.g. nested lists), and ``networkx`` graphs.
    """
    if isinstance(result, (nx.Graph, nx.DiGraph)):
        adjacency = nx.to_numpy_array(
            result,
            nodelist=sorted(result.nodes()),
            weight="weight",
            dtype=np.float32,
        )
        matrix = torch.from_numpy(adjacency)
    elif isinstance(result, torch.Tensor):
        matrix = result
    else:
        try:
            matrix = torch.from_numpy(np.asarray(result, dtype=np.float32))
        except Exception as err:
            raise ValueError(
                f"Matrix builder returned {type(result).__name__}, which cannot "
                f"be converted to a torch.Tensor. Return a torch.Tensor, a "
                f"numpy.ndarray, or a networkx graph."
            ) from err

    if tuple(matrix.shape) != expected_shape:
        raise ValueError(
            f"Matrix builder produced shape {tuple(matrix.shape)}, expected {expected_shape}."
        )

    return matrix.to(device=device, dtype=dtype)
