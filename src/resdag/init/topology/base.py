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
import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, cast

import networkx as nx
import numpy as np
import torch

from resdag.utils.general import coerce_seed_to_int, create_torch_generator

# Below this size the dense ``eigvals`` path is already cheap and exact, so it
# is used directly rather than any iterative (approximate) method.  The dense
# ``O(n^3)`` eigendecomposition only becomes worth avoiding well past ~1000.
_DENSE_EIGVALS_MAX_N = 1000

# Large matrices at or below this density use a CPU CSR Arnoldi solve.
# Denser inputs use the dense eigensolver. Both are numerical algorithms;
# iterative convergence is not a proof that every dominant mode was found.
_SPARSE_DENSITY_THRESHOLD = 0.1

# Request a small cluster rather than one Ritz value to reduce sensitivity to
# near-degenerate leading eigenvalues. The Krylov basis is larger than SciPy's
# default so circular-law spectra have room to converge.
_EIGS_K = 6
_EIGS_NCV = 48


def estimate_spectral_radius(matrix: torch.Tensor) -> float:
    """Estimate the spectral radius (largest ``|eigenvalue|``) of a matrix.

    Dense inputs use :func:`torch.linalg.eigvals`, or SciPy's sparse Arnoldi
    solver (``which="LM"``) for large matrices with density at most 0.1, with
    a dense fallback if Arnoldi is unavailable or unsuccessful. CSR inputs use
    analytic zero, triangular and weighted-cycle cases when applicable, then
    Arnoldi; failure raises without allocating a dense fallback. Only CSR sizes
    1--2 use a bounded dense solve. Arnoldi is an iterative numerical
    approximation; requesting several dominant eigenvalues reduces sensitivity
    to clustered spectra.

    A stationary power-iteration estimate does not establish that the dominant
    eigenspace was found, so it is never used for automatic spectral scaling.
    Dense eigendecomposition can be expensive at large sizes. Use a builder with
    known spectral structure and ``prescaled=True`` to avoid this measurement.
    MPS inputs are measured on CPU because MPS lacks complex eigensolves.

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
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("matrix must be a non-empty square matrix.")
    values = matrix.values() if matrix.layout == torch.sparse_csr else matrix
    if matrix.layout == torch.sparse_csr and matrix.is_complex():
        # Some Torch complex operations inspect the view's sparse parent for
        # strides. An independent O(nnz) value tensor keeps validation strided.
        values = values.clone()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("matrix must contain only finite values.")
    # MPS has no complex eigvals or complex power-iteration support. Radius
    # measurement is an initialization/diagnostic operation, so use the CPU
    # implementation while keeping recurrent weights and CUDA work in place.
    if matrix.device.type == "mps":
        return estimate_spectral_radius(matrix.detach().cpu())
    n = matrix.shape[0]

    if matrix.layout == torch.sparse_csr:
        if n <= 2:
            return _dense_eigvals_radius(matrix.to_dense())
        # Stored CSR slots can remain after in-place zeroing or a zero target.
        if matrix._nnz() == 0 or not bool((values != 0).any()):
            return 0.0
        rows = torch.repeat_interleave(
            torch.arange(n, device=matrix.device), matrix.crow_indices().diff()
        )
        columns = matrix.col_indices()
        if bool((columns >= rows).all()) or bool((columns <= rows).all()):
            diagonal = values[columns == rows]
            return float(diagonal.abs().max().item()) if diagonal.numel() else 0.0
        radius = _functional_csr_radius(matrix)
        if radius is not None:
            return radius
        radius = _sparse_eigs_radius(matrix)
        if radius is None:
            raise RuntimeError(
                "Sparse spectral-radius estimation did not converge; no dense fallback was "
                "allocated. Use a prescaled initializer or omit automatic spectral scaling."
            )
        return radius

    if n <= _DENSE_EIGVALS_MAX_N:
        return _dense_eigvals_radius(matrix)

    if _is_sparse(matrix):
        radius = _sparse_eigs_radius(matrix)
        if radius is not None:
            return radius

    # A stationary power estimate can belong to a subdominant eigenspace.
    # Dense scaling therefore uses the eigensolver, never that heuristic.
    return _dense_eigvals_radius(matrix)


def _functional_csr_radius(matrix: torch.Tensor) -> float | None:
    """Measure CSR functional graphs analytically, including trees and zero edges.

    With at most one stored entry per row, each nonzero strongly connected
    component is either a directed weighted cycle or a singleton without a
    self-loop. A length-k cycle satisfies ``lambda**k = product(weights)``;
    incoming trees and terminal rows contribute only zero eigenvalues. Work
    and temporary storage are linear, with no eigensolve or dense square.
    """
    if not bool((matrix.crow_indices().diff() <= 1).all()):
        return None
    try:
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components
    except ImportError:
        return None

    # Promote before taking complex magnitudes, and keep SciPy's connectivity
    # arrays independent: eliminate_zeros must not mutate the caller's indices.
    dtype = torch.complex128 if matrix.is_complex() else torch.float64
    tensor_values = matrix.values().detach()
    if matrix.is_complex():
        tensor_values = tensor_values.clone()
    values = tensor_values.to(device="cpu", dtype=dtype).numpy()
    columns = matrix.col_indices().detach().cpu().numpy()
    pointers = matrix.crow_indices().detach().cpu().numpy()
    graph = sp.csr_matrix(
        (np.where(values != 0, 1.0, 0.0), columns, pointers),
        shape=tuple(matrix.shape),
        copy=True,
    )
    graph.eliminate_zeros()
    _, labels = connected_components(graph, directed=True, connection="strong")
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(pointers))
    cycles: dict[int, list[float | complex]] = {}
    for row, column, value in zip(rows, columns, values.tolist()):
        if value != 0 and labels[row] == labels[column]:
            cycles.setdefault(int(labels[row]), []).append(value)

    radius = 0.0
    for weights in cycles.values():
        # Preserve a constant magnitude exactly, including signed permutations
        # and the ordinary simple-cycle builder, rather than exp(log(weight)).
        magnitudes = []
        for weight in weights:
            try:
                magnitudes.append(abs(weight))
            except OverflowError:
                magnitudes.append(math.inf)
        if math.isfinite(magnitudes[0]) and all(value == magnitudes[0] for value in magnitudes):
            candidate = magnitudes[0]
        else:
            logarithms = []
            for weight in weights:
                if isinstance(weight, complex):
                    scale = max(abs(weight.real), abs(weight.imag))
                    ratio = min(abs(weight.real), abs(weight.imag)) / scale
                    logarithms.append(math.log(scale) + 0.5 * math.log1p(ratio * ratio))
                else:
                    logarithms.append(math.log(abs(weight)))
            try:
                candidate = math.exp(math.fsum(logarithms) / len(weights))
            except OverflowError:
                candidate = math.inf
        radius = max(radius, candidate)
    return float(radius)


def _dense_eigvals_radius(matrix: torch.Tensor) -> float:
    """Exact spectral radius via the dense (non-symmetric) eigendecomposition."""
    if not matrix.is_floating_point() and not matrix.is_complex():
        matrix = matrix.to(torch.float64)
    elif matrix.dtype in (torch.float16, torch.bfloat16):
        matrix = matrix.float()
    eigenvalues = torch.linalg.eigvals(matrix)
    return float(torch.max(torch.abs(eigenvalues)).item())


def _is_sparse(matrix: torch.Tensor) -> bool:
    """Return ``True`` when the matrix is sparse enough for the CSR ``eigs`` path.

    The density threshold selects an iterative CPU CSR solve for large matrices;
    denser matrices use a dense eigendecomposition.
    """
    n = matrix.shape[0]
    nnz = int(torch.count_nonzero(matrix).item())
    return nnz <= _SPARSE_DENSITY_THRESHOLD * n * n


def _sparse_eigs_radius(matrix: torch.Tensor) -> float | None:
    """Spectral radius of a sparse matrix via ``scipy.sparse.linalg.eigs``.

    Builds a CSR copy on the CPU and runs an Arnoldi iteration for the largest-
    magnitude eigenvalues (``which='LM'``), returning ``max|lambda|`` over the
    converged cluster. Complex eigenvalues are retained, including conjugate
    pairs. This is an iterative estimate, subject to the solver tolerance. A small
    cluster (``k > 1``) is requested rather than ``k=1`` because ARPACK can lock
    onto a sub-dominant Ritz value and miss the true largest when the top
    eigenvalues are near-degenerate. Returns ``None`` when SciPy is unavailable
    or the iteration fails. Dense callers can fall back to dense eigvals; CSR
    callers raise without a dense allocation.

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

    # A real 3-by-3 Arnoldi solve can request only one Ritz value. Some
    # ARPACK versions return a spurious zero when its dominant mode is a
    # complex conjugate pair. Complex arithmetic represents that mode with
    # one Ritz value and keeps the solve sparse.
    work_dtype = torch.complex128 if matrix.is_complex() or n == 3 else torch.float64
    work = matrix.detach().to(device="cpu", dtype=work_dtype)
    if work.layout == torch.sparse_csr:
        sparse = sp.csr_matrix(
            (work.values().numpy(), work.col_indices().numpy(), work.crow_indices().numpy()),
            shape=tuple(work.shape),
        )
    else:
        sparse = sp.csr_matrix(work.numpy())
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
            v0=np.random.default_rng(0).standard_normal(n),
        )
    except Exception:
        return None

    return float(np.max(np.abs(eigenvalues)))


def scale_to_spectral_radius(matrix: torch.Tensor, target_radius: float) -> torch.Tensor:
    """Rescale a square matrix so its spectral radius equals ``target_radius``.

    The current spectral radius is obtained from
    :func:`estimate_spectral_radius`, which uses analytic structured-CSR cases,
    dense ``eigvals`` or sparse Arnoldi. General CSR failure raises without a
    dense fallback; dense inputs retain their dense fallback. Returns the
    matrix unchanged when its current measured spectral radius is zero — e.g.
    the ``zero`` topology or a nilpotent matrix.

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
    if not np.isfinite(target_radius) or target_radius < 0:
        raise ValueError("target_radius must be finite and non-negative.")
    current_radius = estimate_spectral_radius(matrix)

    if current_radius > 0.0:
        # Normalize before multiplying: target/current may overflow in the
        # matrix dtype even when the normalized matrix is representable.
        if matrix.layout == torch.sparse_csr:
            matrix = matrix.clone()
            matrix.values().div_(current_radius).mul_(target_radius)
        else:
            matrix = (matrix / current_radius) * target_radius
    elif target_radius > 0.0:
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

    Initializers build recurrent weights from graph structures or direct
    matrix builders. They take the required size, dtype and device from a
    destination tensor and optionally apply spectral radius scaling.

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
    rescale : bool or None
        Explicit outer scaling policy. None keeps the registration's
        ``prescaled`` policy; False skips spectral estimation/scaling without
        a warning; True applies a requested layer radius even when prescaled.

    See Also
    --------
    GraphTopology : Concrete implementation using NetworkX graphs.
    resdag.layers.ESNLayer : Uses topology initializers.
    """

    #: Topologies that own their spectral scaling set this to ``True`` so the
    #: base classes skip the outer rescale.  Overridden per-instance in
    #: ``__init__`` of the concrete subclasses.
    prescaled: bool = False
    # None preserves the historical registration policy. A configured policy
    # can explicitly skip estimation, or opt into scaling a prescaled builder.
    rescale: bool | None = None

    def build_sparse(
        self,
        n: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        spectral_radius: float | None = None,
    ) -> torch.Tensor:
        """Build CSR weights; override to avoid a dense initialization scratch.

        Existing in-place custom initializers remain usable. Their default
        implementation needs a temporary dense matrix, released after conversion.
        GraphTopology builds directly from edges, including bounded ER draws.
        """
        dense = torch.empty(n, n, dtype=dtype, device=device)
        self.initialize(dense, spectral_radius=spectral_radius)
        return dense.to_sparse_csr()

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
    rescale : bool or None, default=None
        Explicit outer scaling policy. False skips scaling silently; True
        applies a requested radius even for prescaled graphs. None retains
        the legacy prescaled policy and warning.

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
        rescale: bool | None = None,
    ):
        self.graph_func = graph_func
        self.graph_kwargs = graph_kwargs or {}
        self.prescaled = prescaled
        if rescale is not None and not isinstance(rescale, bool):
            raise TypeError("rescale must be True, False, or None")
        self.rescale = rescale

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
        warning is emitted if ``spectral_radius`` is not ``None``. An explicit
        :attr:`rescale` policy overrides this: False skips silently, True scales.

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
        if spectral_radius is not None and self.rescale is not False:
            if self.prescaled and self.rescale is None:
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

    def build_sparse(
        self,
        n: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        spectral_radius: float | None = None,
    ) -> torch.Tensor:
        """Build from graph edges without a dense Torch adjacency allocation."""
        from resdag.init.graphs._sparse import sparse_adjacency

        adjacency = sparse_adjacency(self.graph_func, n, self.graph_kwargs)
        result = torch.sparse_csr_tensor(
            torch.as_tensor(adjacency.indptr, dtype=torch.int64, device=device),
            torch.as_tensor(adjacency.indices, dtype=torch.int64, device=device),
            torch.as_tensor(adjacency.data, dtype=dtype, device=device),
            size=(n, n),
        )
        if spectral_radius is not None and self.rescale is not False:
            if self.prescaled and self.rescale is None:
                _warn_prescaled_override(
                    getattr(self.graph_func, "__name__", repr(self.graph_func)), spectral_radius
                )
            else:
                result = scale_to_spectral_radius(result, spectral_radius)
        return result

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
            An ``(n, n)`` weighted adjacency matrix. Custom builder precision is preserved.

        Raises
        ------
        ValueError
            If the produced adjacency does not have shape ``(n, n)``.
        """
        from resdag.init.graphs._dense import dense_adjacency_builder

        adjacency_builder = dense_adjacency_builder(self.graph_func)
        if adjacency_builder is not None:
            adjacency = np.asarray(adjacency_builder(n, **self.graph_kwargs))
            if adjacency.shape != (n, n):
                raise ValueError(
                    f"Dense adjacency builder produced shape {adjacency.shape}, expected ({n}, {n})"
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
            dtype=np.float64,
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
        rescale = f", rescale={self.rescale}" if self.rescale is not None else ""
        return (
            f"{self.__class__.__name__}(graph_func={self.graph_func.__name__}, "
            f"kwargs={self.graph_kwargs}{prescaled}{rescale})"
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
        (e.g. ``orthogonal``'s unit singular values or an explicitly prescribed
        eigenvalue spectrum). The
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
    rescale : bool or None, default=None
        Explicit outer scaling policy. False skips spectral estimation and
        scaling silently; True applies a requested layer radius even for a
        prescaled builder. None retains the legacy policy and warning.

    Notes
    -----
    Build-style callbacks with explicit ``dtype`` or ``device`` parameters
    receive the destination's values unless ``matrix_kwargs`` already binds
    them. This lets built-in builders compute directly in float64 or on CUDA
    instead of casting a prebuilt float32 CPU matrix. Generic ``**kwargs``
    callbacks receive no additional placement keywords. MPS spectral-radius
    measurements use CPU because MPS does not implement complex eigvals.

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
        rescale: bool | None = None,
    ):
        self.matrix_func = matrix_func
        self.matrix_kwargs = dict(matrix_kwargs or {})
        self.prescaled = prescaled
        self.seed = _pop_generator_seed(matrix_func, self.matrix_kwargs, seed)
        if rescale is not None and not isinstance(rescale, bool):
            raise TypeError("rescale must be True, False, or None")
        self.rescale = rescale

    def build_sparse(
        self,
        n: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        spectral_radius: float | None = None,
    ) -> torch.Tensor:
        """Accept registered builders returning CSR/SciPy sparse data.

        Build-style callables need no dense destination. In-place callables
        allocate their scratch tensor lazily through the shared convention.
        """
        from ._structured import known_matrix_csr

        matrix = known_matrix_csr(
            self.matrix_func, n, self.matrix_kwargs, dtype=dtype, device=device
        )
        if matrix is None:
            reference = torch.empty(0, device=device, dtype=dtype)
            result = _call_matrix_builder(
                self.matrix_func,
                reference,
                (n,),
                self.matrix_kwargs,
                seed=self.seed,
                scratch_shape=(n, n),
            )
            matrix = _coerce_to_matrix(result, (n, n), device, dtype, compact=True)
        if spectral_radius is not None and self.rescale is not False:
            if self.prescaled and self.rescale is None:
                _warn_prescaled_override(
                    getattr(self.matrix_func, "__name__", repr(self.matrix_func)), spectral_radius
                )
            else:
                matrix = scale_to_spectral_radius(matrix, spectral_radius)
        return matrix

    def initialize(
        self,
        weight: torch.Tensor,
        spectral_radius: float | None = None,
    ) -> torch.Tensor:
        """
        Initialize a square weight tensor from the wrapped callable.

        When this topology is :attr:`prescaled`, the outer spectral-radius
        rescale is skipped (the builder's own spectral structure is kept) and a
        warning is emitted if ``spectral_radius`` is not ``None``. An explicit
        :attr:`rescale` policy overrides this: False skips silently, True scales.

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
        if spectral_radius is not None and self.rescale is not False:
            if self.prescaled and self.rescale is None:
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
        rescale = f", rescale={self.rescale}" if self.rescale is not None else ""
        return (
            f"{self.__class__.__name__}(matrix_func={name}, "
            f"kwargs={self.matrix_kwargs}{prescaled}{rescale})"
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
        params = dict(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return False
    return "generator" in params


def _pop_generator_seed(
    fn: Callable, kwargs: dict[str, Any], seed: int | None = None
) -> int | None:
    """Adapt a bound seed only when the callback explicitly takes ``generator``.

    Callbacks that accept ``seed`` or arbitrary keywords keep their original
    calling convention. An explicitly bound generator always controls its own
    state and takes precedence when the callback is invoked.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return seed
    if (
        "seed" in kwargs
        and "seed" not in params
        and "generator" in params
        and not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    ):
        return coerce_seed_to_int(kwargs.pop("seed"))
    return seed


def _call_matrix_builder(
    fn: Callable,
    weight: torch.Tensor,
    build_args: tuple[Any, ...],
    kwargs: dict[str, Any],
    seed: int | None = None,
    scratch_shape: tuple[int, int] | None = None,
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
    # Only named placement parameters opt in. Do not pass them to arbitrary
    # **kwargs callbacks, which may forward unknown keywords elsewhere.
    build_kwargs = kwargs.copy()
    try:
        params = dict(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        params = {}
    for key, value in (("device", weight.device), ("dtype", weight.dtype)):
        parameter = params.get(key)
        if (
            parameter is not None
            and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
            and key not in build_kwargs
        ):
            build_kwargs[key] = value
    if seed is not None and "generator" not in build_kwargs and _accepts_generator(fn):
        build_kwargs["generator"] = create_torch_generator(seed, device=weight.device)
    try:
        return fn(*build_args, **build_kwargs)
    except (TypeError, AttributeError) as build_err:
        scratch = (
            torch.empty_like(weight, memory_format=torch.contiguous_format)
            if scratch_shape is None
            else torch.empty(scratch_shape, device=weight.device, dtype=weight.dtype)
        )
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
    *,
    compact: bool = False,
) -> torch.Tensor:
    """Convert a builder's return value into a validated torch matrix.

    Accepts ``torch.Tensor``, ``numpy.ndarray`` (or anything ``np.asarray``
    understands, e.g. nested lists), and ``networkx`` graphs.
    """
    import scipy.sparse as sp

    if isinstance(result, (nx.Graph, nx.DiGraph)) and compact:
        result = nx.to_scipy_sparse_array(
            result, nodelist=sorted(result.nodes()), weight="weight", dtype=np.float64
        )
    if sp.issparse(result):
        sparse = result.tocsr()
        matrix = torch.sparse_csr_tensor(
            torch.as_tensor(sparse.indptr, dtype=torch.int64),
            torch.as_tensor(sparse.indices, dtype=torch.int64),
            torch.as_tensor(sparse.data),
            size=sparse.shape,
        )
    elif isinstance(result, (nx.Graph, nx.DiGraph)):
        adjacency = nx.to_numpy_array(
            result,
            nodelist=sorted(result.nodes()),
            weight="weight",
            dtype=np.float64,
        )
        matrix = torch.from_numpy(adjacency)
    elif isinstance(result, torch.Tensor):
        matrix = result
    else:
        try:
            # Preserve a callback's numerical precision until the final cast
            # to the destination. In particular float64 must not round via32.
            array = np.asarray(result)
            if not array.dtype.isnative:
                array = array.astype(array.dtype.newbyteorder("="))
            matrix = torch.from_numpy(np.ascontiguousarray(array))
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

    if compact:
        if matrix.layout != torch.sparse_csr:
            matrix = matrix.to_sparse_csr()
        if not bool(torch.isfinite(matrix.values()).all()):
            raise ValueError("Matrix builder must produce finite weights")
    elif matrix.layout != torch.strided:
        matrix = matrix.to_dense()
    return matrix.to(device=device, dtype=dtype)
