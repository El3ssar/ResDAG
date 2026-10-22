"""Native-precision, placement and failed-fit contracts for Standardize."""

import copy

import pytest
import torch

from resdag import Standardize


def _training(dtype=torch.float32, device="cpu"):
    return torch.tensor(
        [[[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]], [[7.0, 2.0], [9.0, 2.0], [11.0, 2.0]]],
        dtype=dtype,
        device=device,
    )


def test_float64_fit_retains_variation_below_float32_precision():
    training = torch.tensor([1.0000000001, 1.0000000002, 1.0000000003], dtype=torch.float64)[
        :, None
    ]
    layer = Standardize(1, eps=0.0).fit(training)
    assert layer.mean.dtype == layer.std.dtype == torch.float64
    assert layer.std.item() > 0
    torch.testing.assert_close(layer.mean, training.mean(0), rtol=0, atol=0)
    torch.testing.assert_close(layer.std, training.std(0, unbiased=False), rtol=0, atol=0)
    normalized = layer(training)
    torch.testing.assert_close(normalized.std(0, unbiased=False), torch.ones(1).double())
    torch.testing.assert_close(layer.inverse(normalized), training, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_fit_adopts_native_dtype_and_constant_channels_stay_finite(dtype):
    training = _training(dtype).requires_grad_()
    layer = Standardize(2).fit(training)
    assert layer.mean.dtype == layer.std.dtype == dtype
    assert set(dict(layer.named_buffers())) == {"mean", "std"}
    flat = training.reshape(-1, 2)
    torch.testing.assert_close(layer.mean, flat.mean(0), rtol=0, atol=0)
    torch.testing.assert_close(layer.std, flat.std(0, unbiased=False), rtol=0, atol=0)
    normalized = layer(training)
    assert normalized.dtype == dtype and bool(torch.isfinite(normalized).all())
    torch.testing.assert_close(normalized[..., 1], torch.zeros_like(normalized[..., 1]))
    torch.testing.assert_close(layer.inverse(normalized), training)
    normalized.sum().backward()
    assert training.grad is not None and bool(torch.isfinite(training.grad).all())
    assert bool((training.grad != 0).all())


def test_float32_fitting_preserves_existing_population_std_and_eps_arithmetic_exactly():
    generator = torch.Generator().manual_seed(9)
    training = torch.randn(2, 7, 3, generator=generator).transpose(0, 1)
    flat = training.reshape(-1, 3)
    expected_mean, expected_std = flat.mean(0), flat.std(0, unbiased=False)
    layer = Standardize(3).fit(training)
    torch.testing.assert_close(layer.mean, expected_mean, rtol=0, atol=0)
    torch.testing.assert_close(layer.std, expected_std, rtol=0, atol=0)
    torch.testing.assert_close(
        layer(training), (training - expected_mean) / (expected_std + layer.eps), rtol=0, atol=0
    )


@pytest.mark.parametrize("shape", [(2,), (3, 2), (2, 3, 4, 2)])
def test_fit_accepts_any_nonempty_leading_dimensions_and_single_observation(shape):
    training = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float64).reshape(shape)
    layer = Standardize(2).fit(training)
    assert layer.mean.shape == layer.std.shape == (2,)
    assert bool(torch.isfinite(layer(training)).all())
    torch.testing.assert_close(layer.inverse(layer(training)), training)


@pytest.mark.parametrize("created_inside", [False, True])
def test_inference_mode_fit_leaves_detached_buffers_usable_for_upstream_gradients(created_inside):
    layer = None if created_inside else Standardize(2)
    with torch.inference_mode():
        if created_inside:
            layer = Standardize(2)
        training = _training(torch.float64)
        assert layer.fit(training) is layer
    assert not layer.mean.is_inference() and not layer.std.is_inference()
    assert not layer.mean.requires_grad and not layer.std.requires_grad
    source = _training(torch.float64).requires_grad_()
    layer(source).sum().backward()
    torch.testing.assert_close(source.grad, (1 / (layer.std + layer.eps)).expand_as(source))
    assert bool(torch.isfinite(source.grad).all())


def test_fitted_statistics_detach_training_and_refitting_replaces_dtype_without_aliasing():
    training = _training(torch.float64).requires_grad_()
    layer = Standardize(2).fit(training)
    assert not layer.mean.requires_grad and not layer.std.requires_grad
    snapshot = copy.deepcopy(layer.state_dict())
    with torch.no_grad():
        training.add_(100)
    for key, value in snapshot.items():
        torch.testing.assert_close(layer.state_dict()[key], value, rtol=0, atol=0)
    replacement = _training(torch.float32) * 3
    layer.fit(replacement)
    assert layer.mean.dtype == layer.std.dtype == torch.float32
    torch.testing.assert_close(layer.mean, replacement.reshape(-1, 2).mean(0), rtol=0, atol=0)
    assert training.grad is None


@pytest.mark.parametrize(
    "invalid,error",
    [
        (None, TypeError),
        ([[1.0, 2.0]], TypeError),
        (torch.ones(2, 2, dtype=torch.int64), TypeError),
        (torch.ones(2, 2, dtype=torch.complex64), TypeError),
        (torch.tensor(1.0), ValueError),
        (torch.empty(2, 0, 2), ValueError),
        (torch.empty(2, 0), ValueError),
        (torch.ones(2, 3), ValueError),
        (torch.tensor([[float("nan"), 0.0]]), ValueError),
        (torch.tensor([[0.0, float("inf")]]), ValueError),
        (torch.ones(2, 2, device="meta"), ValueError),
        (torch.full((2, 2), torch.finfo(torch.float32).max), ValueError),
    ],
)
def test_failed_fit_preserves_both_previous_native_statistics_atomically(invalid, error):
    layer = Standardize(2).fit(_training(torch.float64))
    mean, std = layer.mean, layer.std
    snapshot = copy.deepcopy(layer.state_dict())
    with pytest.raises(error):
        layer.fit(invalid)
    assert layer.mean is mean and layer.std is std
    for key, value in snapshot.items():
        torch.testing.assert_close(layer.state_dict()[key], value, rtol=0, atol=0)


def test_native_fit_statistics_survive_checkpoint_and_ordinary_dtype_conversion():
    source = Standardize(2).fit(_training(torch.float64))
    restored = Standardize(2).double()
    restored.load_state_dict(source.state_dict())
    torch.testing.assert_close(restored(_training(torch.float64)), source(_training(torch.float64)))
    restored.float()
    assert restored.mean.dtype == restored.std.dtype == torch.float32
    torch.testing.assert_close(restored(_training()), source(_training(torch.float64)).float())


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_native_statistics_and_constant_channel_scale_remain_scriptable(dtype):
    training = _training(dtype)
    layer = Standardize(2).fit(training)
    scripted = torch.jit.script(layer)
    torch.testing.assert_close(scripted(training), layer(training), rtol=0, atol=0)
    assert bool(torch.isfinite(scripted(training)).all())
    torch.testing.assert_close(scripted.inverse(scripted(training)), training)


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_native_accelerator_fit_moves_statistics_and_preserves_roundtrip_and_gradients(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    layer = Standardize(2).fit(_training(device=device))
    assert layer.mean.device.type == layer.std.device.type == device
    source = _training(device=device).requires_grad_()
    normalized = layer(source)
    assert normalized.device.type == device and bool(torch.isfinite(normalized).all())
    torch.testing.assert_close(layer.inverse(normalized), source)
    normalized.sum().backward()
    torch.testing.assert_close(source.grad, (1 / (layer.std + layer.eps)).expand_as(source))
    assert bool(torch.isfinite(source.grad).all())
    expected = layer.mean.detach().cpu().clone(), layer.std.detach().cpu().clone()
    layer.cpu()
    torch.testing.assert_close(layer.mean, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(layer.std, expected[1], rtol=0, atol=0)


@pytest.mark.parametrize("eps", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_constructor_rejects_negative_or_nonfinite_epsilon(eps):
    with pytest.raises(ValueError, match="eps must be finite and nonnegative"):
        Standardize(1, eps=eps)


@pytest.mark.parametrize("eps", [1j, "1e-8", None])
def test_constructor_rejects_nonreal_epsilon_with_useful_error(eps):
    with pytest.raises(TypeError, match="eps must be a finite, nonnegative real number"):
        Standardize(1, eps=eps)


def test_zero_epsilon_preserves_exact_unpadded_inverse_and_scripted_roundtrip():
    layer = Standardize(2, mean=torch.tensor([1.0, -2.0]), std=torch.tensor([2.0, 4.0]), eps=0)
    source = torch.tensor([[3.0, 6.0], [5.0, -10.0]])
    expected = torch.tensor([[1.0, 2.0], [2.0, -2.0]])
    scripted = torch.jit.script(layer)
    assert layer.eps == 0.0
    torch.testing.assert_close(scripted(source), expected, rtol=0, atol=0)
    torch.testing.assert_close(scripted.inverse(expected), source, rtol=0, atol=0)


def test_float64_constructor_preserves_small_offsets_and_detaches_without_aliasing():
    mean = torch.tensor([1.0000000001], dtype=torch.float64, requires_grad=True)
    std = torch.tensor([0.0000000001], dtype=torch.float64, requires_grad=True)
    layer = Standardize(1, mean=mean, std=std, eps=0.0)
    source = torch.tensor([[1.0000000003]], dtype=torch.float64, requires_grad=True)
    expected = (source.detach() - mean.detach()) / std.detach()
    assert layer.mean.dtype == layer.std.dtype == torch.float64
    assert not layer.mean.requires_grad and not layer.std.requires_grad
    torch.testing.assert_close(layer(source), expected, rtol=0, atol=0)
    layer(source).sum().backward()
    torch.testing.assert_close(source.grad, torch.tensor([[1e10]], dtype=torch.float64))
    assert mean.grad is None and std.grad is None
    mean_snapshot, std_snapshot = layer.mean.clone(), layer.std.clone()
    with torch.no_grad():
        mean.add_(10)
        std.add_(10)
    torch.testing.assert_close(layer.mean, mean_snapshot, rtol=0, atol=0)
    torch.testing.assert_close(layer.std, std_snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("stat_name", ["mean", "std"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_missing_constructor_stat_uses_supplied_float_precision(stat_name, dtype):
    supplied = torch.tensor([2.0, 4.0], dtype=dtype)
    layer = Standardize(2, **{stat_name: supplied})
    assert layer.mean.dtype == layer.std.dtype == dtype
    torch.testing.assert_close(getattr(layer, stat_name), supplied, rtol=0, atol=0)
    missing = layer.std if stat_name == "mean" else layer.mean
    expected = torch.ones_like(supplied) if stat_name == "mean" else torch.zeros_like(supplied)
    torch.testing.assert_close(missing, expected, rtol=0, atol=0)


def test_two_floating_constructor_stats_use_common_promoted_precision():
    mean = torch.tensor([1.0, -2.0], dtype=torch.float16)
    std = torch.tensor([2.0000000001, 4.0000000001], dtype=torch.float64)
    layer = Standardize(2, mean=mean, std=std, eps=0.0)
    assert layer.mean.dtype == layer.std.dtype == torch.float64
    torch.testing.assert_close(layer.mean, mean.double(), rtol=0, atol=0)
    torch.testing.assert_close(layer.std, std, rtol=0, atol=0)


@pytest.mark.parametrize("supplied", [[2, 4], [2.0, 4.0], torch.tensor([2, 4])])
def test_list_and_integer_constructor_statistics_keep_default_float_coercion(supplied):
    layer = Standardize(2, mean=supplied)
    assert layer.mean.dtype == layer.std.dtype == torch.get_default_dtype()
    torch.testing.assert_close(layer.mean, torch.tensor([2.0, 4.0]), rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs,error,message",
    [
        ({"mean": torch.tensor([float("nan")])}, ValueError, "mean must contain finite"),
        ({"mean": torch.tensor([float("inf")])}, ValueError, "mean must contain finite"),
        ({"std": torch.tensor([float("nan")])}, ValueError, "std must contain finite"),
        ({"std": torch.tensor([float("inf")])}, ValueError, "std must contain finite"),
        ({"std": torch.tensor([-1.0])}, ValueError, "std must be nonnegative"),
        ({"std": [-1e-100]}, ValueError, "std must be nonnegative"),
        ({"mean": torch.tensor([1j])}, TypeError, "mean must contain real"),
        ({"std": [1j]}, TypeError, "std must contain real"),
        ({"mean": torch.ones(1, device="meta")}, ValueError, "concrete values"),
    ],
)
def test_invalid_constructor_statistics_fail_with_actionable_errors(kwargs, error, message):
    with pytest.raises(error, match=message):
        Standardize(1, **kwargs)


def test_tensor_statistics_on_different_devices_reject_before_any_transfer():
    with pytest.raises(ValueError, match="same device; move them explicitly"):
        Standardize(1, mean=torch.zeros(1), std=torch.ones(1, device="meta"))


@pytest.mark.parametrize("supplied", [False, True])
def test_inference_mode_constructor_buffers_allow_later_autograd(supplied):
    with torch.inference_mode():
        mean = torch.tensor([1.0, 2.0], dtype=torch.float64) if supplied else None
        std = torch.tensor([2.0, 4.0], dtype=torch.float64) if supplied else None
        layer = Standardize(2, mean=mean, std=std, eps=0.0)
    assert not layer.mean.is_inference() and not layer.std.is_inference()
    source = torch.tensor([[3.0, 6.0]], dtype=layer.mean.dtype, requires_grad=True)
    normalized = layer(source)
    torch.testing.assert_close(layer.inverse(normalized), source)
    normalized.sum().backward()
    expected = torch.tensor([[0.5, 0.25]], dtype=torch.float64) if supplied else torch.ones(1, 2)
    torch.testing.assert_close(source.grad, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("stat_name", ["mean", "std"])
def test_native_mps_constructor_preserves_device_and_precision_and_autograd(dtype, stat_name):
    with torch.inference_mode():
        stat = torch.tensor([1.0, 2.0], dtype=dtype, device="mps")
        layer = Standardize(2, **{stat_name: stat}, eps=0.0)
    assert layer.mean.device.type == layer.std.device.type == "mps"
    assert layer.mean.dtype == layer.std.dtype == dtype
    assert not layer.mean.is_inference() and not layer.std.is_inference()
    source = torch.tensor([[3.0, 6.0]], dtype=dtype, device="mps", requires_grad=True)
    expected = [[2.0, 4.0]] if stat_name == "mean" else [[3.0, 3.0]]
    torch.testing.assert_close(layer(source), torch.tensor(expected, dtype=dtype, device="mps"))
    torch.testing.assert_close(layer.inverse(layer(source)), source)
    layer(source).sum().backward()
    expected_gradient = [[1.0, 1.0]] if stat_name == "mean" else [[1.0, 0.5]]
    torch.testing.assert_close(
        source.grad, torch.tensor(expected_gradient, dtype=dtype, device="mps"), rtol=0, atol=0
    )
    mixed = Standardize(2, mean=stat, std=[2.0, 4.0])
    assert mixed.mean.device.type == mixed.std.device.type == "mps"
    assert mixed.mean.dtype == mixed.std.dtype == torch.get_default_dtype()
    with pytest.raises(ValueError, match="same device; move them explicitly"):
        Standardize(2, mean=stat, std=torch.ones(2))
