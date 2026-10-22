"""Independent numerical and PyTorch contracts for fitted preprocessing."""

import copy
import io

import numpy as np
import pytest
import torch
from torch import nn

from resdag.core._preprocessing import FittedNormalization

_METHODS = ("minmax", "standard", "noncentered", "meanpreserving")


def _training(dtype=torch.float64):
    return torch.tensor(
        [
            [[-5, 4, 0, 7], [-1, 5, 0, 7], [2, 9, 0, 7]],
            [[7, 2, 0, 7], [11, 8, 0, 7], [8, 13, 0, 7]],
        ],
        dtype=dtype,
    )


def _reference(training, data, method):
    values = training.detach().cpu().double().numpy().reshape(-1, training.shape[-1])
    x = data.detach().cpu().double().numpy()
    if method == "minmax":
        offset = values.min(0)
        scale = values.max(0) - offset
        scale[scale == 0] = 1
        return 2 * (x - offset) / scale - 1, 2 / scale
    if method == "standard":
        offset = values.mean(0)
        scale = values.std(0, ddof=1 if len(values) > 1 else 0)
        scale[scale == 0] = 1
        return (x - offset) / scale, 1 / scale
    if method == "noncentered":
        scale = np.max(np.abs(values), 0)
        scale[scale == 0] = 1
        return x / scale, 1 / scale
    offset = values.mean(0)
    scale = np.max(np.abs(values - offset), 0)
    scale[scale == 0] = 1
    return (x - offset) / scale + offset, 1 / scale


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("two_dimensional", [False, True])
def test_fitted_values_match_independent_numpy_reference_and_roundtrip(method, two_dimensional):
    training = _training()
    if two_dimensional:
        training = training.reshape(-1, training.shape[-1])
    data = training + torch.tensor([2.5, -3.0, 4.0, -0.25])
    normalizer = FittedNormalization(method)
    assert normalizer.fit(training) is normalizer
    actual = normalizer(data)
    expected, _ = _reference(training, data, method)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(normalizer.inverse(actual), data, rtol=1e-14, atol=1e-14)
    assert actual.shape == data.shape
    assert normalizer.n_features == 4
    assert list(normalizer.parameters()) == []


@pytest.mark.parametrize("method", _METHODS)
def test_future_values_do_not_refit_or_mutate_training_statistics(method):
    training = _training()
    normalizer = FittedNormalization(method).fit(training)
    original = copy.deepcopy(normalizer.state_dict())
    future = training + 1e9
    first = normalizer(future)
    normalizer.inverse(first)
    normalizer(training * -1000)
    for name, value in original.items():
        torch.testing.assert_close(normalizer.state_dict()[name], value, rtol=0, atol=0)
    expected, _ = _reference(training, future, method)
    np.testing.assert_allclose(first.numpy(), expected, rtol=1e-14, atol=1e-7)


@pytest.mark.parametrize("method", _METHODS)
def test_training_statistics_are_detached_but_input_gradients_are_preserved(method):
    training = _training().requires_grad_()
    x = (training.detach() + 0.5).requires_grad_()
    normalizer = FittedNormalization(method).fit(training)
    result = normalizer(x)
    result.sum().backward()
    _, slope = _reference(training, x, method)
    expected = torch.from_numpy(slope).expand_as(x)
    torch.testing.assert_close(x.grad, expected, rtol=1e-14, atol=1e-14)
    assert training.grad is None
    assert all(not value.requires_grad and value.grad_fn is None for value in normalizer.buffers())
    assert torch.autograd.gradcheck(normalizer, (x.detach().requires_grad_(),))
    assert torch.autograd.gradcheck(normalizer.inverse, (result.detach().requires_grad_(),))


@pytest.mark.parametrize("method", _METHODS)
def test_fit_inside_inference_mode_does_not_poison_later_autograd(method):
    with torch.inference_mode():
        normalizer = FittedNormalization(method).fit(_training())
    x = _training().requires_grad_()
    normalizer.inverse(normalizer(x)).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=1e-14)


@pytest.mark.parametrize("method", _METHODS)
def test_single_observation_and_constant_channels_are_finite(method):
    training = torch.tensor([[0.0, 7.0, -2.0]], dtype=torch.float64)
    normalizer = FittedNormalization(method).fit(training)
    actual = normalizer(training)
    expected, _ = _reference(training, training, method)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(normalizer.inverse(actual), training, rtol=0, atol=0)


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_transform_preserves_floating_input_dtype(method, dtype):
    training = _training(dtype)
    normalizer = FittedNormalization(method).fit(training)
    result = normalizer(training)
    assert result.dtype == dtype
    assert result.device == training.device
    tolerance = {
        torch.float16: 0.02,
        torch.bfloat16: 0.1,
        torch.float32: 1e-5,
        torch.float64: 1e-14,
    }[dtype]
    torch.testing.assert_close(normalizer.inverse(result), training, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("method", _METHODS)
def test_state_dict_restores_rule_feature_width_and_predictions_in_fresh_module(method):
    original = FittedNormalization(method).fit(_training())
    stream = io.BytesIO()
    torch.save(original.state_dict(), stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=True)
    # The recipient does not need the fitted width or normalization rule.
    restored = FittedNormalization().double()
    restored.load_state_dict(state)
    assert restored.method == method
    assert restored.n_features == 4
    x = _training() * 2
    torch.testing.assert_close(restored(x), original(x), rtol=0, atol=0)
    torch.testing.assert_close(restored.inverse(original(x)), x, rtol=1e-14, atol=1e-14)


def test_prefixed_checkpoint_and_requested_destination_dtype_work_normally():
    original = nn.Sequential(FittedNormalization("minmax").fit(_training()))
    restored = nn.Sequential(FittedNormalization().float())
    restored.load_state_dict(original.state_dict())
    assert restored[0].method == "minmax"
    assert all(
        value.dtype == torch.float32 for value in restored[0].buffers() if value.is_floating_point()
    )
    data = _training(torch.float32)
    torch.testing.assert_close(restored(data), original(data), rtol=0, atol=0)
    restored.double()
    assert all(
        value.dtype == torch.float64 for value in restored[0].buffers() if value.is_floating_point()
    )


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("source_inference", [False, True])
def test_checkpoint_loading_inside_inference_mode_preserves_later_autograd(
    method, assign, source_inference
):
    original = FittedNormalization(method).fit(_training(torch.float32))
    if source_inference:
        with torch.inference_mode():
            state = {key: value.clone() for key, value in original.state_dict().items()}
    else:
        state = original.state_dict()
    with torch.inference_mode():
        restored = FittedNormalization()
        restored.load_state_dict(state, assign=assign)
    assert all(not value.is_inference() and not value.requires_grad for value in restored.buffers())
    x = _training(torch.float32).requires_grad_()
    restored.inverse(restored(x)).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    torch.testing.assert_close(restored(x), original(x), rtol=0, atol=0)


@pytest.mark.parametrize("magnitude", [1e-50, 1e50])
def test_checkpoint_cast_underflow_or_overflow_is_rejected_without_mutation(magnitude):
    training = torch.tensor([[magnitude], [2 * magnitude]], dtype=torch.float64)
    source = FittedNormalization("noncentered").fit(training)
    restored = FittedNormalization("minmax").fit(_training(torch.float32))
    original = copy.deepcopy(restored.state_dict())
    with pytest.raises(RuntimeError, match="destination dtype.*double"):
        restored.load_state_dict(source.state_dict())
    assert restored.method == "minmax"
    assert restored.n_features == 4
    for key, value in original.items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    # Either standard explicit-dtype loading or assignment retains the values.
    restored.double().load_state_dict(source.state_dict())
    assigned = FittedNormalization()
    assigned.load_state_dict(source.state_dict(), assign=True)
    assert all(
        value.dtype == torch.float64 for value in assigned.buffers() if value.is_floating_point()
    )
    expected = torch.tensor([[0.5], [1.0]], dtype=torch.float64)
    torch.testing.assert_close(restored(training), expected, rtol=0, atol=0)
    torch.testing.assert_close(assigned(training), expected, rtol=0, atol=0)


def test_assigned_checkpoint_statistics_remain_detached_buffers():
    source = FittedNormalization("standard").fit(_training())
    state = {
        key: nn.Parameter(value.clone(), requires_grad=value.is_floating_point())
        for key, value in source.state_dict().items()
    }
    restored = FittedNormalization()
    restored.load_state_dict(state, assign=True)
    assert list(restored.parameters()) == []
    assert len(list(restored.buffers())) == 3
    assert all(not value.requires_grad for value in restored.buffers())
    x = _training().requires_grad_()
    restored(x).sum().backward()
    assert all(value.grad is None for value in state.values())
    torch.testing.assert_close(restored(x), source(x), rtol=0, atol=0)


@pytest.mark.parametrize("method", ["minmax", "standard", "meanpreserving"])
def test_finite_training_with_overflowing_statistics_is_rejected_atomically(method):
    normalizer = FittedNormalization(method).fit(_training(torch.float32))
    original = copy.deepcopy(normalizer.state_dict())
    extreme = torch.tensor(
        [[3e38 if method == "meanpreserving" else -3e38], [3e38]], dtype=torch.float32
    )
    with pytest.raises(ValueError, match="statistics overflowed.*higher"):
        normalizer.fit(extreme)
    for key, value in original.items():
        torch.testing.assert_close(normalizer.state_dict()[key], value, rtol=0, atol=0)


def test_unfitted_checkpoint_clears_an_existing_fit():
    original = FittedNormalization("minmax")
    restored = FittedNormalization().fit(_training())
    restored.load_state_dict(original.state_dict())
    assert restored.method == "minmax"
    assert not restored.is_fitted
    assert restored.n_features is None
    with pytest.raises(RuntimeError, match="not fitted"):
        restored(_training())


def test_refitting_replaces_width_and_does_not_alias_the_training_data():
    training = _training()
    normalizer = FittedNormalization().fit(training)
    expected = normalizer(training).clone()
    training.zero_()
    torch.testing.assert_close(normalizer(_training()), expected, rtol=0, atol=0)
    replacement = torch.tensor([[1.0, 4.0], [3.0, 8.0]])
    normalizer.fit(replacement)
    assert normalizer.n_features == 2
    assert normalizer(replacement).dtype == replacement.dtype
    expected_replacement = torch.full_like(replacement, 2**-0.5)
    expected_replacement[0].neg_()
    torch.testing.assert_close(normalizer(replacement), expected_replacement)


@pytest.mark.parametrize("method", _METHODS)
def test_invalid_training_fit_leaves_previous_statistics_unchanged(method):
    normalizer = FittedNormalization(method).fit(_training())
    before = copy.deepcopy(normalizer.state_dict())
    for invalid in [torch.full((1, 2, 4), float("nan")), torch.full((1, 2, 4), float("inf"))]:
        with pytest.raises(ValueError, match="finite"):
            normalizer.fit(invalid)
        for key, value in before.items():
            torch.testing.assert_close(normalizer.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["forward", "inverse"])
def test_unfitted_transform_has_an_actionable_error(operation):
    with pytest.raises(RuntimeError, match=r"call fit\(training_data\)"):
        getattr(FittedNormalization(), operation)(_training())


@pytest.mark.parametrize("data", [None, [[1.0, 2.0]], np.ones((2, 4))])
def test_non_tensor_inputs_are_rejected(data):
    normalizer = FittedNormalization().fit(_training())
    for operation in [normalizer.fit, normalizer.forward, normalizer.inverse]:
        with pytest.raises(TypeError, match="torch.Tensor"):
            operation(data)


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.complex64])
def test_non_real_floating_inputs_are_rejected(dtype):
    normalizer = FittedNormalization().fit(_training())
    invalid = torch.ones(2, 4, dtype=dtype)
    for operation in [normalizer.fit, normalizer.forward, normalizer.inverse]:
        with pytest.raises(TypeError, match="real floating"):
            operation(invalid)


@pytest.mark.parametrize("shape", [(4,), (1, 2, 3, 4), (0, 3, 4), (1, 0, 4), (1, 2, 0)])
def test_invalid_dimensions_and_empty_axes_are_rejected(shape):
    normalizer = FittedNormalization().fit(_training())
    invalid = torch.empty(shape)
    for operation in [normalizer.fit, normalizer.forward, normalizer.inverse]:
        with pytest.raises(ValueError, match="non-empty"):
            operation(invalid)


def test_transform_rejects_a_different_feature_width():
    normalizer = FittedNormalization().fit(_training())
    with pytest.raises(ValueError, match="fitted with 4 features, got 3"):
        normalizer(torch.ones(1, 2, 3))


@pytest.mark.parametrize("change", ["method", "shape", "width", "scale", "nonfinite", "partial"])
def test_corrupt_checkpoint_rejected_without_replacing_existing_statistics(change):
    normalizer = FittedNormalization("standard").fit(_training())
    original = copy.deepcopy(normalizer.state_dict())
    state = copy.deepcopy(original)
    if change == "method":
        state["_method_code"] = torch.tensor(100)
    elif change == "shape":
        state["_stat0"] = torch.zeros(4)
    elif change == "width":
        state["_stat1"] = torch.ones(1, 1, 3)
    elif change == "scale":
        state["_stat1"].zero_()
    elif change == "nonfinite":
        state["_stat0"].fill_(float("nan"))
    else:
        del state["_stat1"]
    with pytest.raises(RuntimeError, match="normalization checkpoint"):
        normalizer.load_state_dict(state)
    for key, value in original.items():
        torch.testing.assert_close(normalizer.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_native_mps_buffers_values_and_upstream_gradients(method):
    training = _training(torch.float32)
    normalizer = FittedNormalization(method).fit(training).to("mps")
    assert all(value.device.type == "mps" for value in normalizer.buffers())
    x = (training + 0.5).to("mps").requires_grad_()
    result = normalizer(x)
    expected, slope = _reference(training, x, method)
    np.testing.assert_allclose(result.detach().cpu().numpy(), expected, rtol=1e-5, atol=1e-5)
    result.sum().backward()
    gradient = torch.tensor(slope, dtype=torch.float32).expand_as(training)
    torch.testing.assert_close(x.grad.cpu(), gradient, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(normalizer.inverse(result).detach().cpu(), x.detach().cpu())
    normalizer.cpu().double()
    assert all(value.device.type == "cpu" for value in normalizer.buffers())
    assert all(
        value.dtype == torch.float64 for value in normalizer.buffers() if value.is_floating_point()
    )
