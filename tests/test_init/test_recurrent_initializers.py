"""Mathematical contracts, placement, and compatibility of recurrent builders."""

from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch

from resdag.init import (
    get_recurrent_initializer,
    register_recurrent_initializer,
    resolve_recurrent_initializer,
    select_recurrent_spec,
    show_recurrent_initializers,
)
from resdag.init.matrices import (
    antisymmetric_matrix,
    band,
    block_diagonal,
    cycle_jumps,
    delay_line,
    delay_line_backward,
    eigen_spectrum,
    fast_spectral_initialization,
    householder_orthogonal,
    orthogonal_matrix,
    self_loop_cycle,
    signed_permutation,
    simple_cycle,
    singular_spectrum,
    toeplitz,
)
from resdag.init.topology import (
    GraphTopology,
    MatrixTopology,
    get_topology,
    register_matrix_topology,
)
from resdag.init.topology.registry import _TOPOLOGY_REGISTRY
from resdag.init.utils import resolve_topology

NEW_NAMES = ["singular_spectrum", "eigen_spectrum", "signed_permutation", "householder_orthogonal"]
BUILDERS = [
    orthogonal_matrix,
    antisymmetric_matrix,
    fast_spectral_initialization,
    delay_line,
    delay_line_backward,
    simple_cycle,
    cycle_jumps,
    self_loop_cycle,
    toeplitz,
    band,
    block_diagonal,
    singular_spectrum,
    eigen_spectrum,
    signed_permutation,
    householder_orthogonal,
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("builder", BUILDERS)
def test_direct_builder_dtype_and_placement(builder: Callable, dtype: torch.dtype) -> None:
    matrix = builder(7, dtype=dtype, device="cpu")
    assert matrix.shape == (7, 7)
    assert matrix.dtype == dtype
    assert matrix.device == torch.device("cpu")
    assert torch.isfinite(matrix).all()


@pytest.mark.parametrize("builder", BUILDERS)
def test_direct_builder_rejects_nonfloating_dtype(builder: Callable) -> None:
    with pytest.raises(ValueError, match="floating-point"):
        builder(7, dtype=torch.int64)


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize("size", [0, -1, True, 2.5])
def test_direct_builder_rejects_invalid_size(builder: Callable, size: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        builder(size)


@pytest.mark.parametrize(
    "name", NEW_NAMES + ["orthogonal", "antisymmetric", "fast_spectral_initialization"]
)
def test_explicit_seed_is_local_and_reproducible(name: str) -> None:
    before = torch.random.get_rng_state().clone()
    first = torch.empty(9, 9, dtype=torch.float64)
    second = torch.empty_like(first)
    resolve_recurrent_initializer(name, seed=28).initialize(first)
    resolve_recurrent_initializer(name, seed=28).initialize(second)
    assert torch.equal(first, second)
    assert torch.equal(before, torch.random.get_rng_state())


def test_prescribed_singular_values_include_zeros_and_preserve_float64() -> None:
    values = [0.0, 0.123456789123456, 0.9, 1.23456789123456, 0.2]
    matrix = singular_spectrum(5, values, seed=2, dtype=torch.float64)
    torch.testing.assert_close(
        torch.linalg.svdvals(matrix),
        torch.tensor(sorted(values, reverse=True), dtype=torch.float64),
        rtol=1e-12,
        atol=1e-14,
    )
    assert torch.linalg.matrix_rank(matrix).item() == 4


def test_singular_spectrum_is_not_an_eigenvalue_specification() -> None:
    values = torch.tensor([1.0, 0.7, 0.1], dtype=torch.float64)
    matrix = singular_spectrum(3, values, seed=30, dtype=torch.float64)
    # Independent left/right factors give the specified linear gains without
    # constraining eigenvalues to the same magnitudes.
    assert not torch.allclose(
        torch.linalg.eigvals(matrix).abs().sort().values, values.sort().values
    )


@pytest.mark.parametrize(
    "values", [[1.0, -0.1], [1.0, float("nan")], [1.0], [[1.0, 0.2]], [1.0, 1j]]
)
def test_invalid_singular_spectrum(values: list) -> None:
    with pytest.raises(ValueError, match="singular_values"):
        singular_spectrum(2, values)


def test_prescribed_eigenvalues_are_real_matrix_and_normal() -> None:
    values = [0.123456789123456, 0.3 + 0.8j, -0.45, 0.3 - 0.8j, -0.2 - 0.1j, -0.2 + 0.1j]
    matrix = eigen_spectrum(6, values, seed=13, dtype=torch.float64)
    assert not matrix.is_complex()
    expected = torch.tensor(values, dtype=torch.complex128)
    actual = torch.linalg.eigvals(matrix)
    # Compare unordered sets with one-to-one matching (not a fragile complex sort).
    for value in expected:
        distances = (actual - value).abs()
        closest = distances.argmin()
        assert distances[closest] < 2e-14
        actual = torch.cat((actual[:closest], actual[closest + 1 :]))
    torch.testing.assert_close(matrix @ matrix.T, matrix.T @ matrix, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(
        torch.linalg.svdvals(matrix),
        expected.abs().sort(descending=True).values,
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "values", [[1j, 0.0], [float("inf"), 0.0], [1.0], [[1.0, 0.2]], [0.2 + 0.3j, 0.2 - 0.4j]]
)
def test_invalid_eigen_spectrum(values: list) -> None:
    with pytest.raises(ValueError, match="eigenvalues|conjugate"):
        eigen_spectrum(2, values)


def test_signed_permutation_has_one_signed_entry_per_row_and_column() -> None:
    matrix = signed_permutation(37, gain=0.73, seed=21, dtype=torch.float64)
    assert torch.all((matrix != 0).sum(dim=0) == 1)
    assert torch.all((matrix != 0).sum(dim=1) == 1)
    assert torch.all(matrix[matrix != 0].abs() == 0.73)
    assert (matrix > 0).any() and (matrix < 0).any()
    torch.testing.assert_close(matrix.T @ matrix, 0.73**2 * torch.eye(37, dtype=torch.float64))
    torch.testing.assert_close(
        torch.linalg.eigvals(matrix).abs(),
        torch.full((37,), 0.73, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize("count", [0, 1, 2, 7, 13])
def test_householder_product_is_orthogonal_with_expected_determinant(count: int) -> None:
    matrix = householder_orthogonal(7, reflectors=count, seed=31, dtype=torch.float64)
    torch.testing.assert_close(
        matrix.T @ matrix, torch.eye(7, dtype=torch.float64), rtol=1e-12, atol=1e-14
    )
    assert torch.linalg.det(matrix).item() == pytest.approx((-1.0) ** count, abs=1e-13)
    # k reflections act as identity on the orthogonal complement of their span.
    if count < 7:
        assert torch.linalg.matrix_rank(matrix - torch.eye(7)).item() <= count


def test_single_householder_reflection_is_symmetric_involution() -> None:
    matrix = householder_orthogonal(11, reflectors=1, seed=9, dtype=torch.float64)
    torch.testing.assert_close(matrix, matrix.T, atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(
        matrix @ matrix, torch.eye(11, dtype=torch.float64), atol=1e-14, rtol=1e-12
    )


@pytest.mark.parametrize("count", [-1, 2.5, True])
def test_invalid_reflector_count(count: int) -> None:
    with pytest.raises(ValueError, match="reflectors"):
        householder_orthogonal(7, reflectors=count)


@pytest.mark.parametrize("builder", [householder_orthogonal, signed_permutation])
@pytest.mark.parametrize("gain", [0.0, -0.5, 1.5])
def test_orthogonal_structures_gain(builder: Callable, gain: float) -> None:
    matrix = builder(7, gain=gain, seed=2, dtype=torch.float64)
    torch.testing.assert_close(matrix.T @ matrix, gain**2 * torch.eye(7, dtype=torch.float64))


def test_builder_receives_destination_dtype_without_roundtrip_through_float32() -> None:
    params = {"singular_values": [1.0, 0.123456789123456, 0.2], "seed": 9}
    weight = torch.empty(3, 3, dtype=torch.float64)
    get_recurrent_initializer("singular_spectrum", **params).initialize(weight)
    expected = singular_spectrum(3, **params, dtype=torch.float64)
    assert torch.equal(weight, expected)
    assert not torch.equal(weight, expected.to(torch.float32).to(torch.float64))


def test_explicit_builder_placement_overrides_destination_defaults() -> None:
    received = []

    def builder(n: int, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        received.append((dtype, torch.device(device)))
        return torch.eye(n, dtype=dtype, device=device)

    weight = torch.empty(3, 3, dtype=torch.float64)
    MatrixTopology(builder, {"dtype": torch.float32, "device": "cpu"}).initialize(weight)
    assert received == [(torch.float32, torch.device("cpu"))]
    assert weight.dtype == torch.float64


def test_arbitrary_kwargs_callbacks_do_not_receive_new_placement_keywords() -> None:
    received = []

    def builder(n: int, **kwargs: object) -> torch.Tensor:
        received.append(kwargs)
        return torch.eye(n)

    MatrixTopology(builder, {"custom": "value"}).initialize(torch.empty(3, 3))
    assert received == [{"custom": "value"}]


def test_select_recurrent_spec_retains_legacy_alias_and_rejects_conflicts() -> None:
    assert select_recurrent_spec() is None
    assert select_recurrent_spec(topology="simple_cycle") == "simple_cycle"
    assert select_recurrent_spec(recurrent_initializer="signed_permutation") == "signed_permutation"
    with pytest.raises(ValueError, match="only one"):
        select_recurrent_spec(topology="simple_cycle", recurrent_initializer="signed_permutation")


def test_shared_registry_supports_new_and_legacy_custom_registration() -> None:
    names = ["test_recurrent_custom", "test_legacy_custom"]

    def builder(n: int, rescale: float = 2.0) -> torch.Tensor:
        # An existing custom builder may already own a keyword called rescale.
        return torch.eye(n) * rescale

    try:
        register_recurrent_initializer(names[0], rescale=3.0)(builder)
        register_matrix_topology(names[1], rescale=4.0)(builder)
        for name in names:
            for resolver in (resolve_topology, resolve_recurrent_initializer):
                initializer = resolver((name, {"rescale": 5.0}))
                weight = torch.empty(3, 3)
                initializer.initialize(weight)
                assert torch.equal(weight, 5 * torch.eye(3))
        with pytest.raises(ValueError, match="already registered"):
            register_recurrent_initializer(names[1])(builder)
    finally:
        for name in names:
            _TOPOLOGY_REGISTRY.pop(name, None)


@pytest.mark.parametrize(
    "spec",
    [
        lambda n: torch.eye(n),
        (lambda n, gain: gain * torch.eye(n), {"gain": 1.0}),
        torch.nn.init.eye_,
    ],
)
def test_recurrent_alias_accepts_callable_and_tuple_forms(spec: object) -> None:
    weight = torch.empty(3, 3)
    resolve_recurrent_initializer(spec, seed=4).initialize(weight)
    assert torch.equal(weight, torch.eye(3))


def test_recurrent_alias_inplace_initializer_seed() -> None:
    first = torch.empty(7, 7)
    second = torch.empty_like(first)
    resolve_recurrent_initializer(torch.nn.init.orthogonal_, seed=8).initialize(first)
    resolve_recurrent_initializer(torch.nn.init.orthogonal_, seed=8).initialize(second)
    assert torch.equal(first, second)


def test_graph_registration_is_discoverable_through_recurrent_api() -> None:
    assert isinstance(get_recurrent_initializer("erdos_renyi"), GraphTopology)
    names = show_recurrent_initializers()
    assert "erdos_renyi" in names
    assert all(name in names for name in NEW_NAMES)


def test_fsi_opt_out_avoids_all_outer_spectral_estimation() -> None:
    initializer = get_recurrent_initializer(
        "fast_spectral_initialization", spectral_radius=0.73, seed=19, rescale=False
    )
    weight = torch.empty(7, 7, dtype=torch.float64)
    with patch("resdag.init.topology.base.scale_to_spectral_radius", side_effect=AssertionError):
        initializer.initialize(weight, spectral_radius=0.1)
    expected = fast_spectral_initialization(7, 0.73, seed=19, dtype=torch.float64)
    assert torch.equal(weight, expected)


def test_fsi_legacy_policy_still_rescales_to_requested_radius() -> None:
    weight = torch.empty(7, 7, dtype=torch.float64)
    get_topology("fast_spectral_initialization", seed=19).initialize(weight, spectral_radius=0.17)
    assert torch.linalg.eigvals(weight).abs().max().item() == pytest.approx(0.17, abs=1e-13)


def test_explicit_rescale_overrides_prescaled_policy() -> None:
    weight = torch.empty(7, 7, dtype=torch.float64)
    get_recurrent_initializer("signed_permutation", gain=0.8, seed=12, rescale=True).initialize(
        weight, spectral_radius=0.2
    )
    torch.testing.assert_close(
        torch.linalg.svdvals(weight), torch.full((7,), 0.2, dtype=torch.float64)
    )


@pytest.mark.parametrize("name", NEW_NAMES)
def test_new_spectra_are_prescaled_by_default(name: str) -> None:
    assert get_recurrent_initializer(name).prescaled
    first = torch.empty(7, 7, dtype=torch.float64)
    second = torch.empty_like(first)
    get_recurrent_initializer(name, seed=4).initialize(first)
    with pytest.warns(UserWarning, match="pre-scaled"):
        get_recurrent_initializer(name, seed=4).initialize(second, spectral_radius=0.1)
    assert torch.equal(first, second)


@pytest.mark.parametrize("name", ["simple_cycle", "signed_permutation", "erdos_renyi"])
def test_explicit_skip_scaling_is_silent(name: str) -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        get_recurrent_initializer(name, rescale=False).initialize(
            torch.empty(7, 7), spectral_radius=0.1
        )


def test_invalid_scaling_policy_is_clear() -> None:
    with pytest.raises(TypeError, match="rescale"):
        get_recurrent_initializer("simple_cycle", rescale="never")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("builder", BUILDERS)
def test_builder_native_mps_placement(builder: Callable) -> None:
    # This verifies native allocation/precision, not accelerator performance.
    matrix = builder(7, dtype=torch.float32, device="mps")
    assert matrix.device.type == "mps"
    assert matrix.dtype == torch.float32
    assert bool(torch.isfinite(matrix).all())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("builder", BUILDERS)
def test_builder_native_cuda_placement(builder: Callable) -> None:
    matrix = builder(7, dtype=torch.float64, device="cuda")
    assert matrix.device.type == "cuda"
    assert matrix.dtype == torch.float64
    assert bool(torch.isfinite(matrix).all())
