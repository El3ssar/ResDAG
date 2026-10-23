import subprocess
import sys

import pytest
import torch
from torch import nn

import resdag as rd
from resdag.artifacts import load, save_model
from resdag.models.parallel_esn import ParallelESNModel


def _series(width=2):
    t = torch.arange(80.0)
    return torch.stack([torch.sin(t / (3 + i)) for i in range(width)], dim=-1)


def _parallel():
    raw = ParallelESNModel(
        rd.ESNLayer(7, feedback_size=3, seed=3),
        rd.AutoReadoutLayer(7, 1),
        input_dim=4,
        core_width=1,
        halo=1,
    )
    return rd.Model(graph=raw, input_names=["state"], output_names=["prediction"], washout=5)


def _coupled():
    members = [
        rd.classic_esn(7, feedback_size=2, output_size=2, seed=seed, readout="auto")
        for seed in (3, 9)
    ]
    next(m for m in members[1].modules() if isinstance(m, rd.ESNLayer)).cell.weight_hh = next(
        m for m in members[0].modules() if isinstance(m, rd.ESNLayer)
    ).cell.weight_hh
    aggregator = rd.OutliersFilteredMean(method="iqr", threshold=2)
    aggregator.supports_framewise = True
    raw = rd.CoupledEnsembleESNModel(members, aggregator=aggregator)
    return rd.Model(graph=raw, input_names=["state"], output_names=["prediction"], washout=5)


@pytest.mark.parametrize("factory,width", [(_parallel, 4), (_coupled, 2)])
@pytest.mark.parametrize("trusted", [False, True])
def test_parallel_coupled_geometry_parameters_and_forecast(tmp_path, factory, width, trusted):
    model = factory().fit(_series(width))
    model.eval()
    model(_series(width)[:4])
    save_model(model, tmp_path / "family.pt", trusted=trusted)
    restored = load(tmp_path / "family.pt", trusted=trusted)
    expected = model.forecast(horizon=9)
    torch.testing.assert_close(restored.forecast(horizon=9), expected, rtol=0, atol=0)
    if width == 2:
        reservoirs = [m.cell for m in restored.graph.modules() if isinstance(m, rd.ESNLayer)]
        assert reservoirs[0].weight_hh is reservoirs[1].weight_hh
    else:
        assert torch.equal(model.graph.patch_index, restored.graph.patch_index)


def test_named_branch_merge_multihead_delayed_feedback(tmp_path):
    x = rd.Input((3, 2))
    a = rd.ESNLayer(5, feedback_size=2, seed=5)(x)
    b = rd.ESNLayer(4, feedback_size=2, seed=7)(x)
    features = rd.Concatenate()(a, b)
    main = rd.AutoReadoutLayer(9, 2, name="main")(features)
    extra = rd.SVDReadoutLayer(9, 1, name="extra", alpha=0.01)(features)
    model = rd.Model(
        inputs={"state": x},
        outputs={"prediction": main, "energy": extra},
        washout=4,
        prediction_lead={"prediction": 2, "energy": 1},
        feedback=[rd.Feedback("prediction", "state", delay=2)],
    )
    series = _series()
    model.fit(
        series, targets={"prediction": series, "energy": series.square().sum(-1, keepdim=True)}
    )
    save_model(model, tmp_path / "named.pt")
    restored = load(tmp_path / "named.pt")
    assert restored.prediction_leads == model.prediction_leads
    assert restored.input_names == model.input_names
    expected = model.forecast(horizon=9)
    actual = restored.forecast(horizon=9)
    assert set(actual) == {"prediction", "energy"}
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_symbolic_multioutput_unpack_and_slices(tmp_path):
    x = rd.Input((4, 4))
    a, b = rd.FeaturePartitioner(2, 0)(x)
    model = rd.ESNModel(x, (a, b))
    save_model(model, tmp_path / "unpack.pt")
    restored = load(tmp_path / "unpack.pt")
    xx = torch.randn(2, 7, 4)
    expected = model(xx)
    actual = restored(xx)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    y = rd.Input((4, 4))
    sliced = rd.ESNModel(y, y[..., 1:3])
    save_model(sliced, tmp_path / "slice.pt")
    torch.testing.assert_close(load(tmp_path / "slice.pt")(xx), sliced(xx), rtol=0, atol=0)


def test_shared_nonrecurrent_invocations_and_parameter_gradients(tmp_path):
    x = rd.Input((3, 2))
    shared = nn.Linear(2, 2, bias=False)
    a = shared(x)
    b = shared(a)
    model = rd.ESNModel(x, b)
    save_model(model, tmp_path / "shared.pt")
    restored = load(tmp_path / "shared.pt")
    named = list(restored.named_parameters(remove_duplicate=False))
    assert named[0][1] is named[1][1]
    data = torch.randn(2, 5, 2, requires_grad=True)
    expected = model(data)
    actual = restored(data)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    a_grad = torch.autograd.grad(expected.sum(), shared.weight)[0]
    b_grad = torch.autograd.grad(actual.sum(), named[0][1])[0]
    torch.testing.assert_close(a_grad, b_grad, rtol=0, atol=0)


def test_custom_reversible_transform_preserves_fitted_stats(tmp_path):
    series = _series()
    model = rd.ESN(
        reservoir_size=6,
        washout=4,
        seed=5,
        input_transforms={"input": rd.Standardize(2, eps=0.01)},
        output_transforms={"output": rd.Standardize(2, eps=0.02)},
    ).fit(series)
    save_model(model, tmp_path / "transform.pt")
    restored = load(tmp_path / "transform.pt")
    assert restored.workflow.input_scalers[0].eps == 0.01
    torch.testing.assert_close(
        model.forecast(horizon=8), restored.forecast(horizon=8), rtol=0, atol=0
    )


def test_raw_lambda_initializer_does_not_need_original_builder(tmp_path):
    reservoir = rd.ESNLayer(
        5,
        feedback_size=2,
        feedback_initializer=lambda rows, cols, **kw: torch.ones(rows, cols),
        seed=4,
    )
    save_model(reservoir, tmp_path / "realized.pt")
    restored = load(tmp_path / "realized.pt")
    xx = torch.randn(2, 5, 2)
    torch.testing.assert_close(restored(xx), reservoir(xx), rtol=0, atol=0)


def test_actual_fresh_process_noisy_memory_and_rng_continuation(tmp_path):
    for trusted in (False, True):
        model = rd.ESNLayer(5, feedback_size=2, noise=0.1, seed=4).train()
        model(torch.ones(1, 3, 2))
        save_model(model, tmp_path / "stream.pt", trusted=trusted)
        chunk = torch.linspace(-1, 1, 12).reshape(1, 6, 2)
        expected = model(chunk)
        torch.save({"chunk": chunk, "expected": expected}, tmp_path / "oracle.pt")
        code = f"""import torch
from resdag.artifacts import load
m=load({str(tmp_path/'stream.pt')!r},trusted={trusted!r})
x=torch.load({str(tmp_path/'oracle.pt')!r},weights_only=True)
torch.testing.assert_close(m(x['chunk']),x['expected'],rtol=0,atol=0)
assert m.training
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
