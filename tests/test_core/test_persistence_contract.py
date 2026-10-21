"""Complete checkpoints preserve the same data across every model family."""

import copy

import pytest
import torch
from torch import nn

import resdag as rd


class WeightedMean(nn.Module):
    """An aggregation parameter must survive a weights-only checkpoint too."""

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([0.2, 0.8]))

    def forward(self, stacked):
        return (stacked * self.logits.softmax(0).view(-1, 1, 1, 1)).sum(0)


def _models():
    def classic():
        return rd.classic_esn(12, 2, 2, readout="svd")

    return [
        classic(),
        rd.models.parallel_esn(input_dim=2, core_width=1, halo=1, reservoir_size=12, readout="svd"),
        rd.CoupledEnsembleESNModel([classic(), classic()], aggregator=WeightedMean()),
    ]


@pytest.mark.parametrize("index", range(3), ids=["graph", "parallel", "ensemble"])
def test_weights_states_and_full_model_roundtrip(index, tmp_path):
    torch.manual_seed(3)
    model = _models()[index].double()
    x = torch.randn(2, 25, 2, dtype=torch.float64)
    if index == 0:
        rd.ESNTrainer(model).fit(x[:, :5], x[:, 5:-1], x[:, 6:])
    else:
        model.fit(x[:, :5], x[:, 5:-1], x[:, 6:])
    model(x)
    expected_state = model.get_state()
    expected, _ = model.forward_stateless(x)
    restored = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in restored.parameters():
            parameter.zero_()
    restored.reset_reservoirs()
    path = tmp_path / "nested" / "weights.pt"
    model.save(path, include_states=True, purpose="regression")
    type(model).load_from_file(path, model=restored, load_states=True, map_location="cpu")
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key
    for key, value in expected_state.items():
        assert torch.equal(value, restored.get_state()[key]), key
    actual, _ = restored.forward_stateless(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    full_path = tmp_path / "other" / "full.pt"
    model.save_full(full_path, purpose="full")
    full, metadata = type(model).load_full(full_path, return_metadata=True, map_location="cpu")
    assert metadata == {"purpose": "full"}
    full_output, _ = full.forward_stateless(x)
    torch.testing.assert_close(full_output, expected, rtol=0, atol=0)


def test_legacy_ensemble_checkpoint_is_still_accepted(tmp_path):
    model = rd.CoupledEnsembleESNModel([rd.classic_esn(8, 2, 2) for _ in range(2)])
    model(torch.randn(1, 4, 2))
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "state_dicts": [m.state_dict() for m in model.models],
            "reservoir_states": model.get_reservoir_states(),
        },
        path,
    )
    restored = copy.deepcopy(model)
    restored.reset_reservoirs()
    restored.load(path, load_states=True)
    for name, state in model.get_state().items():
        assert torch.equal(state, restored.get_state()[name])


def test_full_loader_checks_wrapped_model_type(tmp_path):
    path = tmp_path / "wrong.pt"
    torch.save({"resdag_full_model": nn.Linear(2, 2)}, path)
    with pytest.raises(ValueError, match="does not contain a full ESNModel"):
        rd.ESNModel.load_full(path)


@pytest.mark.parametrize("index", range(3), ids=["graph", "parallel", "ensemble"])
def test_loading_cold_state_clears_preexisting_stream_state(index, tmp_path):
    model = _models()[index]
    path = tmp_path / "cold.pt"
    model.save(path, include_states=True)
    model(torch.randn(1, 4, 2))
    assert model.get_state()
    model.load(path, load_states=True)
    assert model.get_state() == {}
