"""Every concrete ResDAG layer family has an explicit portable codec."""

import pytest
import torch

import resdag as rd
from resdag.artifacts import load, save_model
from resdag.layers import reservoirs

LAYER_SPECS = [
    (reservoirs.ESNLayer, dict(reservoir_size=4, feedback_size=2, seed=2)),
    (reservoirs.ES2NLayer, dict(reservoir_size=4, feedback_size=2, seed=2, beta=0.7)),
    (reservoirs.ResESNLayer, dict(reservoir_size=4, feedback_size=2, seed=2, alpha=0.3, beta=0.7)),
    (reservoirs.EuSNLayer, dict(reservoir_size=4, feedback_size=2, seed=2, epsilon=0.1)),
    (reservoirs.GatedReservoirLayer, dict(reservoir_size=4, feedback_size=2, seed=2)),
    (reservoirs.LIFReservoirLayer, dict(reservoir_size=4, feedback_size=2, seed=2)),
    (reservoirs.CoRNNLayer, dict(reservoir_size=4, feedback_size=2, seed=2)),
    (reservoirs.LegendreDelayLayer, dict(order=2, feedback_size=2, theta=8, seed=2)),
    (reservoirs.NGReservoirLayer, dict(input_dim=2, k=2, p=2)),
    (reservoirs.TorchRecurrentLayer, dict(cell=torch.nn.GRUCell(2, 4), feedback_size=2)),
]


@pytest.mark.parametrize("cls,kwargs", LAYER_SPECS, ids=[c.__name__ for c, _ in LAYER_SPECS])
def test_reservoir_wrappers_state_policies_and_next_chunk(tmp_path, cls, kwargs):
    layer = cls(**kwargs)
    layer.detach_state_between_calls = False
    layer(torch.randn(2, 3, 2))
    save_model(layer, tmp_path / "layer.pt")
    restored = load(tmp_path / "layer.pt")
    assert type(restored) is type(layer)
    assert not restored.detach_state_between_calls
    xx = torch.randn(2, 5, 2)
    torch.testing.assert_close(layer(xx), restored(xx), rtol=0, atol=0)


TRANSFORMS = [
    (rd.Concatenate, {}, True),
    (rd.ExtendedSquare, {}, False),
    (rd.FeaturePartitioner, {"partitions": 2, "overlap": 1}, False),
    (rd.NLAT2, {}, False),
    (rd.NLAT3, {}, False),
    (rd.Pad, {"padding": 0.4}, False),
    (rd.PartialSquare, {"eta": 0.5}, False),
    (rd.Power, {"exponent": 1.5, "sign_preserving": True}, False),
    (rd.SelectiveDropout, {"mask": [True, False, True, False]}, False),
    (rd.SelectiveExponentiation, {"index": 1, "exponent": 1.5, "sign_preserving": True}, False),
    (rd.Standardize, {"num_features": 4, "eps": 0.02}, False),
]


@pytest.mark.parametrize(
    "cls,kwargs,multiple", TRANSFORMS, ids=[c.__name__ for c, _, _ in TRANSFORMS]
)
def test_all_builtin_feature_transforms(tmp_path, cls, kwargs, multiple):
    layer = cls(**kwargs)
    xx = torch.randn(2, 5, 4)
    if isinstance(layer, rd.Standardize):
        layer.fit(xx)
    save_model(layer, tmp_path / "transform.pt")
    restored = load(tmp_path / "transform.pt")
    args = (xx, xx) if multiple else (xx,)
    expected, actual = layer(*args), restored(*args)
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_original_custom_initializer_is_not_reexecuted(tmp_path):
    calls = []

    def initializer(rows, cols, **kwargs):
        calls.append((rows, cols))
        return torch.ones(rows, cols)

    model = rd.ESNLayer(4, feedback_size=2, feedback_initializer=initializer, seed=2)
    count = len(calls)
    save_model(model, tmp_path / "realized.pt")
    load(tmp_path / "realized.pt")
    assert len(calls) == count
