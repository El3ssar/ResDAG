"""Single tensors and explicit streams express the same fitting operation."""

import copy

import pytest
import torch

import resdag as rd


def test_single_tensor_fit_matches_named_target_fit():
    torch.manual_seed(2)
    model = rd.classic_esn(12, 2, 2, readout="svd", readout_name="prediction")
    explicit = copy.deepcopy(model)
    x = torch.randn(2, 35, 2)
    rd.ESNTrainer(model).fit(x[:, :5], x[:, 5:-1], x[:, 6:])
    rd.ESNTrainer(explicit).fit((x[:, :5],), (x[:, 5:-1],), {"prediction": x[:, 6:]})
    actual, _ = model.forward_stateless(x)
    expected, _ = explicit.forward_stateless(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_single_tensor_stream_matches_explicit_stream():
    torch.manual_seed(2)
    model = rd.classic_esn(12, 2, 2, readout="incremental")
    explicit = copy.deepcopy(model)
    x = torch.randn(2, 35, 2)
    chunks = [(x[:, 5:20], x[:, 6:21]), (x[:, 20:34], x[:, 21:35])]
    rd.ESNTrainer(model).fit_stream(x[:, :5], iter(chunks))
    rd.ESNTrainer(explicit).fit_stream(
        (x[:, :5],), (((inputs,), {"output": targets}) for inputs, targets in chunks)
    )
    actual, _ = model.forward_stateless(x)
    expected, _ = explicit.forward_stateless(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_multiple_readouts_require_named_targets():
    inp = rd.reservoir_input(2)
    h = rd.ESNLayer(8, 2)(inp)
    model = rd.ESNModel(
        inp, [rd.SVDReadoutLayer(8, 2, name="a")(h), rd.SVDReadoutLayer(8, 2, name="b")(h)]
    )
    x = torch.randn(1, 12, 2)
    with pytest.raises(ValueError, match="dictionary"):
        rd.ESNTrainer(model).fit(x[:, :2], x[:, 2:-1], x[:, 3:])


def test_fit_reports_dtype_mismatch_before_mutating_state():
    model = rd.classic_esn(8, 2, 2)
    x = torch.randn(1, 12, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="dtype"):
        rd.ESNTrainer(model).fit(x[:, :2], x[:, 2:-1], x[:, 3:])
    assert model.get_state() == {}
