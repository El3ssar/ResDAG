"""Sparse workflow staging copies canonical weights, not CSR storage."""

import copy

import pytest
import torch

import resdag as rd
from resdag.models._recipes import builtin_recipe


def _series():
    time = torch.arange(100, dtype=torch.float64)
    return torch.stack((torch.sin(time / 5), torch.cos(time / 7)), dim=-1)


def test_sparse_cell_copy_rebuilds_from_current_weights_without_mutating_source_or_rng():
    cell = rd.ESNLayer(16, feedback_size=2, sparse=True, seed=13, dtype=torch.float64).cell
    original_cache = cell._weight_hh_csr
    cached_weights = original_cache.to_dense().clone()
    with torch.no_grad():
        cell.weight_hh.values().add_(0.01)
    random_state = torch.get_rng_state().clone()

    clone = copy.deepcopy(cell)

    assert torch.equal(torch.get_rng_state(), random_state)
    assert cell._weight_hh_csr is original_cache
    assert not torch.equal(original_cache.to_dense(), cached_weights)
    assert clone._weight_hh_csr.layout == torch.sparse_csr
    assert clone.weight_hh.values().data_ptr() != cell.weight_hh.values().data_ptr()
    assert clone._weight_hh_csr.values().data_ptr() != original_cache.values().data_ptr()
    torch.testing.assert_close(
        clone._weight_hh_csr.to_dense(), cell.weight_hh.to_dense(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        clone.weight_hh.to_dense(), cell.weight_hh.to_dense(), rtol=0, atol=0
    )


@pytest.mark.parametrize("family", ["classic", "ott", "esn"])
def test_sparse_workflow_fit_copy_and_portable_reload_preserve_forecasts(family, tmp_path):
    if family == "esn":
        model = rd.ESN(
            16,
            input_size=2,
            washout=3,
            readout="svd",
            sparse=True,
            seed=13,
            dtype=torch.float64,
        )
    else:
        factory = rd.classic_esn if family == "classic" else rd.ott_esn
        model = rd.Model(
            graph=factory(16, 2, 2, readout="svd", sparse=True, seed=13, dtype=torch.float64),
            washout=3,
        )
    data = _series()
    model.fit(data)
    expected = model.forecast(history=data[-20:], horizon=9)
    assert torch.isfinite(expected).all()
    assert builtin_recipe(model.graph) is not None
    clone = copy.deepcopy(model)
    model.save(tmp_path / "sparse.pt")
    restored = rd.load(tmp_path / "sparse.pt")
    for candidate in (clone, restored):
        assert builtin_recipe(candidate.graph) is not None
        cells = [
            module.cell for module in candidate.graph.modules() if isinstance(module, rd.ESNLayer)
        ]
        assert cells and all(cell._weight_hh_csr.layout == torch.sparse_csr for cell in cells)
        actual = candidate.forecast(history=data[-20:], horizon=9)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("family", ["classic", "ott", "esn"])
def test_sparse_builtin_can_tune_and_remain_ready_to_forecast(family):
    pytest.importorskip("optuna")
    pytest.importorskip("joblib")
    if family == "esn":
        model = rd.ESN(16, washout=3, readout="svd", sparse=True, dtype=torch.float64)
    else:
        factory = rd.classic_esn if family == "classic" else rd.ott_esn
        model = rd.Model(
            graph=factory(16, 2, 2, readout="svd", sparse=True, seed=13, dtype=torch.float64),
            washout=3,
        )
    model.tune(_series(), preset="quick", n_trials=1, horizon=5, seed=7, pruner="none")
    assert model.is_fitted
    assert model.forecast(horizon=5).shape == (5, 2)
    assert builtin_recipe(model.graph) is not None
    cells = [module.cell for module in model.graph.modules() if isinstance(module, rd.ESNLayer)]
    assert cells and all(cell._sparse_active for cell in cells)
