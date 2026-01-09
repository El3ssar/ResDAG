"""Base topology interface for weight initialization."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union

import networkx as nx
import numpy as np
import torch


class TopologyInitializer(ABC):
    """Base class for topology-based weight initialization.

    Topology initializers convert graph structures into PyTorch weight tensors
    for reservoir layers. They extract the required size from the tensor shape,
    generate a graph, convert it to an adjacency matrix, and optionally apply
    spectral radius scaling.
    """

    @abstractmethod
    def initialize(
        self,
        weight: torch.Tensor,
        spectral_radius: Optional[float] = None,
    ) -> torch.Tensor:
        """Initialize a weight tensor using graph topology.

        Parameters
        ----------
        weight : torch.Tensor
            The weight tensor to initialize (shape: (n, n) for recurrent weights).
            This tensor is modified in-place.
        spectral_radius : float, optional
            Target spectral radius for scaling. If None, no scaling is applied.

        Returns
        -------
        torch.Tensor
            The initialized weight tensor (same as input, modified in-place).
        """
        pass


class GraphTopology(TopologyInitializer):
    """Topology initializer based on NetworkX graph functions.

    This class wraps a graph generation function and converts it into a weight
    initializer. The graph function must accept `n` as its first argument.

    Parameters
    ----------
    graph_func : callable
        A function with signature (n: int, *args, **kwargs) -> nx.Graph/DiGraph
    graph_kwargs : dict, optional
        Keyword arguments to pass to the graph function

    Examples
    --------
    >>> from torch_rc.graphs import erdos_renyi_graph
    >>> topology = GraphTopology(erdos_renyi_graph, {"p": 0.1, "directed": True})
    >>> weight = torch.empty(100, 100)
    >>> topology.initialize(weight, spectral_radius=0.9)
    """

    def __init__(
        self,
        graph_func: Callable,
        graph_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.graph_func = graph_func
        self.graph_kwargs = graph_kwargs or {}

    def initialize(
        self,
        weight: torch.Tensor,
        spectral_radius: Optional[float] = None,
    ) -> torch.Tensor:
        """Initialize weight tensor from graph topology.

        Parameters
        ----------
        weight : torch.Tensor
            Square tensor to initialize (n, n)
        spectral_radius : float, optional
            Target spectral radius for scaling

        Returns
        -------
        torch.Tensor
            Initialized weight tensor
        """
        if weight.ndim != 2:
            raise ValueError(f"Weight must be 2D, got shape {weight.shape}")

        if weight.shape[0] != weight.shape[1]:
            raise ValueError(f"Weight must be square, got shape {weight.shape}")

        n = weight.shape[0]
        device = weight.device
        dtype = weight.dtype

        # Generate graph
        G = self.graph_func(n, **self.graph_kwargs)

        # Convert to adjacency matrix
        adj_matrix = self._graph_to_adjacency(G, n)

        # Convert to torch tensor
        weight_data = torch.from_numpy(adj_matrix).to(device=device, dtype=dtype)

        # Apply spectral radius scaling if requested
        if spectral_radius is not None:
            weight_data = self._scale_spectral_radius(weight_data, spectral_radius)

        # Copy into weight tensor (using no_grad to avoid in-place operation on leaf variable)
        with torch.no_grad():
            weight.copy_(weight_data)

        return weight

    def _graph_to_adjacency(
        self,
        G: nx.Graph | nx.DiGraph,
        n: int,
    ) -> np.ndarray:
        """Convert NetworkX graph to adjacency matrix.

        Parameters
        ----------
        G : nx.Graph or nx.DiGraph
            NetworkX graph with weighted edges
        n : int
            Number of nodes

        Returns
        -------
        np.ndarray
            Adjacency matrix (n, n) with edge weights
        """
        # Use nx.to_numpy_array to handle weighted graphs properly
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

        return adj_matrix

    def _scale_spectral_radius(
        self,
        weight: torch.Tensor,
        target_radius: float,
    ) -> torch.Tensor:
        """Scale weight matrix to have target spectral radius.

        Parameters
        ----------
        weight : torch.Tensor
            Weight matrix to scale
        target_radius : float
            Target spectral radius

        Returns
        -------
        torch.Tensor
            Scaled weight matrix
        """
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(weight)
        current_radius = torch.max(torch.abs(eigenvalues)).item()

        if current_radius > 1e-8:
            scale_factor = target_radius / current_radius
            weight = weight * scale_factor

        return weight

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(graph_func={self.graph_func.__name__}, kwargs={self.graph_kwargs})"
