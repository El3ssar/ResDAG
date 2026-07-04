"""Antisymmetric (skew-symmetric) matrix initializer.

Pins down the ``antisymmetric`` matrix topology (``A - A^T``):

- exact antisymmetry (``M == -M^T``) and purely-imaginary eigenvalues,
- registry resolution by name and end-to-end use through ``ESNLayer``,
- spectral-radius scaling (an antisymmetric matrix has a nonzero radius = the
  largest imaginary-eigenvalue magnitude, so the outer rescale applies cleanly
  and preserves antisymmetry),
- reproducibility under a seed, dtype/shape contract.

Imports are from the concrete module file: the public ``__init__`` exports are
wired separately.
"""

import pytest
import torch

import resdag.init.matrices  # noqa: F401 - triggers topology registration
from resdag.init.matrices.antisymmetric import antisymmetric_matrix
from resdag.init.topology import MatrixTopology, estimate_spectral_radius, get_topology
from resdag.init.utils import resolve_topology
from resdag.layers import ESNLayer


class TestAntisymmetry:
    """The core algebraic property: M == -M^T with imaginary spectrum."""

    @pytest.mark.parametrize("n", [4, 16, 64])
    def test_is_antisymmetric(self, n: int) -> None:
        """M == -M^T to float precision."""
        m = antisymmetric_matrix(n, seed=0)
        assert m.shape == (n, n)
        assert torch.allclose(m, -m.t(), atol=1e-5)

    def test_diagonal_is_zero(self) -> None:
        """A - A^T has an exactly zero diagonal."""
        m = antisymmetric_matrix(32, seed=1)
        assert torch.allclose(torch.diagonal(m), torch.zeros(32), atol=1e-6)

    def test_eigenvalues_purely_imaginary(self) -> None:
        """A real antisymmetric matrix has purely imaginary eigenvalues."""
        m = antisymmetric_matrix(48, seed=2)
        eigs = torch.linalg.eigvals(m)
        assert eigs.real.abs().max().item() < 1e-4

    def test_scaling_preserves_antisymmetry(self) -> None:
        """The ``scaling`` gain scales magnitude but keeps M == -M^T."""
        base = antisymmetric_matrix(20, scaling=1.0, seed=3)
        scaled = antisymmetric_matrix(20, scaling=2.5, seed=3)
        assert torch.allclose(scaled, 2.5 * base, atol=1e-5)
        assert torch.allclose(scaled, -scaled.t(), atol=1e-5)


class TestRegistration:
    """Registry resolution and the layer path."""

    def test_get_topology_resolves(self) -> None:
        """``antisymmetric`` resolves to a MatrixTopology."""
        topo = get_topology("antisymmetric")
        assert isinstance(topo, MatrixTopology)

    def test_resolve_topology_by_name(self) -> None:
        """A string spec resolves to a topology initializer."""
        topo = resolve_topology("antisymmetric")
        assert topo is not None

    def test_not_prescaled(self) -> None:
        """The topology is NOT prescaled — the outer spectral rescale applies."""
        topo = get_topology("antisymmetric")
        assert topo.prescaled is False

    def test_esn_layer_builds_antisymmetric(self) -> None:
        """ESNLayer(topology='antisymmetric') builds a skew recurrent matrix."""
        reservoir = ESNLayer(64, feedback_size=3, topology="antisymmetric", seed=1)
        w = reservoir.weight_hh.detach()
        assert torch.allclose(w, -w.t(), atol=1e-5)

    def test_spectral_radius_scaling(self) -> None:
        """A layer spectral_radius rescales to the target (imag-eigenvalue radius)."""
        target = 0.8
        reservoir = ESNLayer(
            128, feedback_size=3, topology="antisymmetric", spectral_radius=target, seed=1
        )
        w = reservoir.weight_hh.detach()
        # Still antisymmetric after scaling.
        assert torch.allclose(w, -w.t(), atol=1e-5)
        # Radius (largest |imaginary eigenvalue|) is on target.
        assert estimate_spectral_radius(w) == pytest.approx(target, abs=1e-3)


class TestOutputContract:
    """dtype/shape/reproducibility, matching orthogonal.py/fsi.py."""

    def test_dtype_and_shape(self) -> None:
        """Returns an (n, n) float32 tensor."""
        m = antisymmetric_matrix(24, seed=0)
        assert m.dtype == torch.float32
        assert m.shape == (24, 24)

    def test_seed_reproducible(self) -> None:
        """Same seed -> identical matrix; different seed -> different."""
        a = antisymmetric_matrix(32, seed=7)
        b = antisymmetric_matrix(32, seed=7)
        c = antisymmetric_matrix(32, seed=8)
        assert torch.equal(a, b)
        assert not torch.equal(a, c)

    def test_positive_size_required(self) -> None:
        """Non-positive n raises."""
        with pytest.raises(ValueError, match="positive"):
            antisymmetric_matrix(0)
