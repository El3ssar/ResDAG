"""Transactional workflow fitting retains diagnostics without checkpoint objects."""

import io

import pytest
import torch

import resdag as rd


def _model(kind):
    if kind == "esn":
        return rd.ESN(5, input_size=4, washout=3, alpha=0.1, seed=7, dtype=torch.float64)
    options = dict(reservoir_size=5, readout="auto", readout_alpha=0.1, seed=7, dtype=torch.float64)
    if kind == "parallel":
        graph = rd.parallel_esn(input_dim=4, core_width=2, halo=1, **options)
    elif kind == "coupled":
        graph = rd.coupled_ensemble_esn(
            2, model_factory=rd.classic_esn, feedback_size=4, output_size=4, **options
        )
    else:
        graph = rd.classic_esn(feedback_size=4, output_size=4, **options)
    return rd.Model(graph=graph, washout=3)


def _readouts(model):
    return {
        path: layer
        for path, layer in model.graph.named_modules()
        if isinstance(layer, rd.AutoReadoutLayer)
    }


def _data(length=60):
    return torch.randn(length, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(19))


@pytest.mark.parametrize("kind", ["model", "esn", "parallel", "coupled"])
def test_successful_fit_and_refit_commit_reports_by_path_without_replacing_layers(kind):
    model = _model(kind)
    original = _readouts(model)
    parameters = {path: layer.weight for path, layer in original.items()}
    assert original and all(layer.last_solve is None for layer in original.values())
    model.fit(_data())
    reports = {}
    for path, layer in _readouts(model).items():
        assert layer is original[path] and layer.weight is parameters[path]
        assert layer.is_fitted and layer.last_solve is not None
        assert layer.last_solve.n_samples > 0 and layer.last_solve.alpha == 0.1
        assert layer.last_solve.solve_device == "cpu"
        assert layer.last_solve.solve_dtype == torch.float64
        reports[path] = layer.last_solve
    assert torch.isfinite(model.forecast(horizon=3)).all()
    model.fit(_data(45))
    for path, layer in _readouts(model).items():
        assert layer is original[path] and layer.weight is parameters[path]
        assert layer.last_solve is not reports[path]
        assert layer.last_solve.n_samples < reports[path].n_samples


@pytest.mark.parametrize("kind", ["model", "esn", "parallel", "coupled"])
def test_failed_staged_refit_preserves_previous_reports_weights_and_fittedness(kind, monkeypatch):
    model = _model(kind).fit(_data())
    original = _readouts(model)
    reports = {path: layer.last_solve for path, layer in original.items()}
    weights = {path: layer.weight.clone() for path, layer in original.items()}
    solve = rd.AutoReadoutLayer.solve

    def fail_after_solving(self, states, targets):
        solve(self, states, targets)
        raise RuntimeError("deliberate staged fit failure")

    monkeypatch.setattr(rd.AutoReadoutLayer, "solve", fail_after_solving)
    with pytest.raises(RuntimeError, match="deliberate staged fit failure"):
        model.fit(_data(45))
    assert model.is_fitted
    for path, layer in _readouts(model).items():
        assert layer is original[path] and layer.last_solve is reports[path]
        assert layer.is_fitted
        torch.testing.assert_close(layer.weight, weights[path], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["model", "esn", "parallel", "coupled"])
def test_weights_only_checkpoint_restores_fit_but_not_past_reports(kind):
    model = _model(kind).fit(_data())
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    assert not any("last_solve" in key for key in checkpoint)
    restored = _model(kind)
    restored.load_state_dict(checkpoint)
    assert restored.is_fitted
    assert all(
        layer.is_fitted and layer.last_solve is None for layer in _readouts(restored).values()
    )
    torch.testing.assert_close(
        restored.forecast(horizon=3), model.forecast(horizon=3), rtol=0, atol=0
    )


def test_failed_first_fit_does_not_publish_report(monkeypatch):
    model = _model("model")
    solve = rd.AutoReadoutLayer.solve

    def fail_after_solving(self, states, targets):
        solve(self, states, targets)
        raise RuntimeError("deliberate staged fit failure")

    monkeypatch.setattr(rd.AutoReadoutLayer, "solve", fail_after_solving)
    with pytest.raises(RuntimeError, match="deliberate staged fit failure"):
        model.fit(_data())
    assert not model.is_fitted
    assert all(
        layer.last_solve is None and not layer.is_fitted for layer in _readouts(model).values()
    )
