"""All factory solvers retain fitting and checkpoint semantics behind Model."""

import copy
import warnings

import pytest
import torch

import resdag as rd
from resdag.training import ESNTrainer

SOLVERS = [
    ("cg", {"max_iter": 300, "tol": 1e-10}),
    ("svd", {}),
    ("pinv", {}),
    ("pinv", {"solver": "pinv"}),
    ("ridge", {}),
    ("ridge", {"solver": "solve"}),
    ("cholesky", {}),
    ("incremental", {}),
    ("rls", {}),
    ("fista", {"max_iter": 5000, "tol": 1e-4}),
]


def _graph(kind, solver, options):
    kwargs = dict(
        reservoir_size=5,
        readout=solver,
        readout_alpha=0.01,
        readout_kwargs=options,
        seed=23,
        dtype=torch.float64,
    )
    if kind == "parallel":
        return rd.parallel_esn(input_dim=4, core_width=2, halo=1, **kwargs)
    return rd.coupled_ensemble_esn(
        2, model_factory=rd.classic_esn, feedback_size=4, output_size=4, **kwargs
    )


@pytest.mark.parametrize("kind", ["parallel", "coupled"])
@pytest.mark.parametrize("solver,options", SOLVERS)
def test_solver_training_rollout_and_checkpoint_match_raw_family(kind, solver, options):
    data = torch.randn(1, 60, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(19))
    graph = _graph(kind, solver, options)
    raw = copy.deepcopy(graph)
    model = rd.Model(graph=graph, washout=3)
    mean, std = data[:, 3:-1].mean((0, 1), keepdim=True), data[:, 3:-1].std((0, 1), keepdim=True)
    scaled = (data - mean) / std
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model.fit(data)
        if kind == "parallel":
            raw.fit(scaled[:, :3], scaled[:, 3:-1], scaled[:, 4:])
        else:
            for member in raw.models:
                ESNTrainer(member).fit(
                    (scaled[:, :3],), (scaled[:, 3:-1],), {"output": scaled[:, 4:]}
                )
    for name, value in graph.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, raw.state_dict()[name], rtol=0, atol=0)
    raw.eval()
    expected = raw.forecast(scaled[:, -4:], horizon=3) * std + mean
    actual = model.forecast(horizon=3)
    # The cached tail is cloned, unlike the manual view into the full series;
    # layout-dependent BLAS reductions can differ by a few float64 ulps.
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    restored = rd.Model(graph=_graph(kind, solver, options), washout=0)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(restored.forecast(horizon=3), actual, rtol=0, atol=0)
