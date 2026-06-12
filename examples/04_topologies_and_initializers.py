"""04 — Topologies and initializers: shaping the reservoir's weight matrices.

The recurrent matrix of an ``ESNLayer`` is built by a *topology*; the
input/feedback matrices are built by *initializers*. Both accept the same
five spec formats::

    None                      library default (dense random)
    "name"                    registry lookup with registered defaults
    ("name", {...})           registry lookup with overrides
    fn  /  (fn, {...})        any bare callable (NEW)
    configured object         GraphTopology / MatrixTopology / Initializer

What it shows
-------------
1. Browsing the registries (show_topologies, show_input_initializers)
2. Graph topologies by name and (name, params) tuple
3. NEW: bare callable topologies — plain functions and torch.nn.init.*
4. NEW: register_matrix_topology + the built-in "orthogonal" matrix topology
5. Input/feedback initializers with the same spec formats

Expected runtime: ~5 s on CPU.
"""

import torch

from resdag.init.input_feedback import get_input_feedback, show_input_initializers
from resdag.init.topology import get_topology, register_matrix_topology, show_topologies
from resdag.layers import ESNLayer


def spectral_radius(matrix: torch.Tensor) -> float:
    """Largest absolute eigenvalue of a square matrix."""
    return torch.max(torch.abs(torch.linalg.eigvals(matrix))).item()


def main() -> None:
    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # 1. Browse the registries
    # ------------------------------------------------------------------
    print("=" * 70)
    print("1. What is available")
    print("=" * 70)

    names = show_topologies()  # prints the list, also returns it
    print(f"(returned as a list of {len(names)} names)")

    print("Parameter details for one topology:")
    show_topologies("watts_strogatz")

    # ------------------------------------------------------------------
    # 2. Graph topologies by name / tuple
    # ------------------------------------------------------------------
    print("=" * 70)
    print("2. Graph topologies in ESNLayer")
    print("=" * 70)

    # String spec: registered defaults
    layer = ESNLayer(200, feedback_size=3, topology="erdos_renyi", spectral_radius=0.9)
    print(f'topology="erdos_renyi"          rho(W) = {spectral_radius(layer.weight_hh):.4f}')

    # Tuple spec: override registered defaults
    layer = ESNLayer(
        200,
        feedback_size=3,
        topology=("watts_strogatz", {"k": 6, "p": 0.3, "seed": 42}),
        spectral_radius=0.9,
    )
    print(f'topology=("watts_strogatz",...) rho(W) = {spectral_radius(layer.weight_hh):.4f}')

    # Object spec: configure once, reuse everywhere
    topo = get_topology("barabasi_albert", m=3, seed=42)
    layer = ESNLayer(200, feedback_size=3, topology=topo, spectral_radius=0.9)
    density = (layer.weight_hh != 0).float().mean().item()
    print(f"topology=get_topology(...)      rho(W) = {spectral_radius(layer.weight_hh):.4f}")
    print(f"Spectral radius is rescaled to the requested value; density here {density:.3f}")

    # ------------------------------------------------------------------
    # 3. NEW: bare callables as topologies
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. Bare callables as topologies (no registry needed)")
    print("=" * 70)
    print("Build style: fn(n, **kwargs) -> (n, n) tensor / ndarray / nx graph")

    def block_diagonal(n: int, blocks: int = 4) -> torch.Tensor:
        """Independent sub-reservoirs along the diagonal."""
        w = torch.zeros(n, n)
        size = n // blocks
        for b in range(blocks):
            s = b * size
            w[s : s + size, s : s + size] = torch.randn(size, size)
        return w

    layer = ESNLayer(200, feedback_size=3, topology=block_diagonal, spectral_radius=0.9)
    print(f"topology=block_diagonal         rho(W) = {spectral_radius(layer.weight_hh):.4f}")

    # (fn, params) binds keyword arguments to the callable
    layer = ESNLayer(
        200, feedback_size=3, topology=(block_diagonal, {"blocks": 2}), spectral_radius=0.9
    )
    off_block = layer.weight_hh[:100, 100:]
    print(
        f"topology=(block_diagonal, ...)  off-block weights all zero: {bool((off_block == 0).all())}"
    )

    print("\nIn-place style: fn(tensor, **kwargs), i.e. any torch.nn.init.*_")
    layer = ESNLayer(200, feedback_size=3, topology=torch.nn.init.orthogonal_, spectral_radius=1.0)
    print(f"topology=torch.nn.init.orthogonal_  rho(W) = {spectral_radius(layer.weight_hh):.4f}")
    layer = ESNLayer(
        200,
        feedback_size=3,
        topology=(torch.nn.init.sparse_, {"sparsity": 0.9}),
        spectral_radius=0.9,
    )
    print(f"topology=(torch.nn.init.sparse_, ...) rho(W) = {spectral_radius(layer.weight_hh):.4f}")

    # ------------------------------------------------------------------
    # 4. NEW: registering matrix topologies
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. register_matrix_topology: name any matrix builder")
    print("=" * 70)
    print("register_graph_topology is for NetworkX builders; this is the")
    print("complement for functions that construct the matrix directly.")

    @register_matrix_topology("two_blocks", blocks=2)
    def two_blocks(n: int, blocks: int = 2, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        return block_diagonal(n, blocks=blocks)

    layer = ESNLayer(200, feedback_size=3, topology="two_blocks", spectral_radius=0.9)
    print(f"registered + used by name:      rho(W) = {spectral_radius(layer.weight_hh):.4f}")

    # The library ships one matrix topology out of the box: "orthogonal"
    layer = ESNLayer(
        200, feedback_size=3, topology=("orthogonal", {"seed": 42}), spectral_radius=1.0
    )
    wtw = layer.weight_hh.T @ layer.weight_hh
    ortho_err = (wtw - torch.eye(200)).abs().max().item()
    print(f'built-in "orthogonal":          max |W^T W - I| = {ortho_err:.2e}')

    # ------------------------------------------------------------------
    # 5. Input / feedback initializers
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. Input/feedback initializers (same spec formats)")
    print("=" * 70)

    show_input_initializers()

    # String spec
    layer = ESNLayer(200, feedback_size=3, feedback_initializer="pseudo_diagonal")
    nonzero_per_row = (layer.weight_feedback != 0).sum(dim=1).float().mean().item()
    print(
        f'"pseudo_diagonal": shape {tuple(layer.weight_feedback.shape)}, '
        f"~{nonzero_per_row:.1f} nonzero/row (each neuron sees one input dim)"
    )

    # Tuple spec with overrides
    layer = ESNLayer(
        200, feedback_size=3, feedback_initializer=("random", {"input_scaling": 0.1, "seed": 7})
    )
    print(
        f'("random", {{"input_scaling": 0.1}}): max |W_fb| = '
        f"{layer.weight_feedback.abs().max().item():.4f}"
    )

    # Object spec
    init = get_input_feedback("chebyshev", input_scaling=0.5)
    layer = ESNLayer(200, feedback_size=3, feedback_initializer=init)
    print(
        f"get_input_feedback('chebyshev'): mean |W_fb| = "
        f"{layer.weight_feedback.abs().mean().item():.4f}"
    )

    # Bare callable spec works here too (in-place torch.nn.init style)
    layer = ESNLayer(200, feedback_size=3, feedback_initializer=torch.nn.init.xavier_uniform_)
    print(
        f"torch.nn.init.xavier_uniform_:   std(W_fb) = " f"{layer.weight_feedback.std().item():.4f}"
    )

    # Driving-input matrix gets its own initializer via input_initializer=
    layer = ESNLayer(
        200,
        feedback_size=3,
        input_size=5,
        feedback_initializer="chebyshev",
        input_initializer=("random", {"input_scaling": 0.5}),
    )
    print(f"separate input_initializer:      W_in shape {tuple(layer.weight_input.shape)}")

    print("\nDone. These specs work identically in every premade factory (topology=...,")
    print("feedback_initializer=...). Next: 05_training_paths.py")


if __name__ == "__main__":
    main()
