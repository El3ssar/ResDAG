"""Independent regressions for offset, streaming, precision and proximal fits."""

import copy
import io
import re
import warnings
from pathlib import Path

import pytest
import torch

import resdag as rd
from resdag.layers import (
    CGReadoutLayer,
    CholeskyReadoutLayer,
    FISTAReadoutLayer,
    IncrementalRidgeReadoutLayer,
    RidgeReadoutLayer,
    RLSReadout,
)


def _offset_problem() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(123)
    x = 1e8 + torch.randn(257, 3, dtype=torch.float64, generator=generator)
    y = (x - 1e8) @ torch.tensor([[1.0], [-2.0], [0.5]], dtype=torch.float64) + 3.0
    return x, y


def _svd_ridge(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    """Independent design-matrix oracle; no normal equations or shared helper."""
    xc = x - x.mean(0)
    yc = y - y.mean(0)
    u, s, vh = torch.linalg.svd(xc, full_matrices=False)
    return vh.T @ ((s / (s.square() + alpha))[:, None] * (u.T @ yc))


@pytest.mark.parametrize("cls", [RidgeReadoutLayer, CholeskyReadoutLayer, CGReadoutLayer])
def test_large_offset_batch_fit_matches_independent_svd(cls: type) -> None:
    x, y = _offset_problem()
    readout = cls(3, 1, alpha=1e-3).double()
    readout.fit(x, y)
    expected = _svd_ridge(x, y, 1e-3)
    torch.testing.assert_close(readout.weight, expected.T, rtol=1e-8, atol=1e-8)
    assert (readout(x) - y).square().mean() < 1e-8


@pytest.mark.parametrize("chunk_size", [1, 7, 64, 257])
def test_streaming_offset_covariance_matches_independent_svd(chunk_size: int) -> None:
    x, y = _offset_problem()
    readout = IncrementalRidgeReadoutLayer(3, 1, alpha=1e-3).double()
    for start in range(0, len(x), chunk_size):
        readout.partial_fit(x[start : start + chunk_size], y[start : start + chunk_size])
    readout.finalize()
    torch.testing.assert_close(readout.weight, _svd_ridge(x, y, 1e-3).T, rtol=1e-6, atol=1e-7)
    assert (readout(x) - y).square().mean() < 1e-8


@pytest.mark.parametrize("kind", ["incremental", "rls"])
def test_factory_preserves_internal_solver_precision(kind: str) -> None:
    model = rd.classic_esn(8, feedback_size=1, output_size=1, readout=kind)
    readout = next(
        module
        for module in model.modules()
        if isinstance(module, (IncrementalRidgeReadoutLayer, RLSReadout))
    )
    assert readout.weight.dtype == torch.get_default_dtype()
    internal = readout.XtX if isinstance(readout, IncrementalRidgeReadoutLayer) else readout.P
    assert internal.dtype == torch.float64


@pytest.mark.parametrize("cls", [IncrementalRidgeReadoutLayer, RLSReadout])
def test_parameter_cast_does_not_round_accumulated_solver_state(cls: type) -> None:
    readout = cls(3, 1).double()
    x = torch.tensor(
        [
            [0.123456789123, -0.29817298371, 0.38749328718],
            [0.918273645123, 0.11438973281, -0.23874923874],
        ],
        dtype=torch.float64,
    )
    readout.partial_fit(x, x[:, :1])
    name = "XtX" if isinstance(readout, IncrementalRidgeReadoutLayer) else "P"
    before = getattr(readout, name).clone()
    assert not torch.equal(before, before.float().double())
    readout.float()
    assert readout.weight.dtype == torch.float32
    assert getattr(readout, name).dtype == torch.float64
    assert torch.equal(getattr(readout, name), before)


@pytest.mark.parametrize("cls", [IncrementalRidgeReadoutLayer, RLSReadout])
def test_opted_out_solver_precision_follows_parameter_dtype(cls: type) -> None:
    readout = cls(3, 1, use_float64=False).double()
    name = "XtX" if isinstance(readout, IncrementalRidgeReadoutLayer) else "P"
    assert getattr(readout, name).dtype == torch.float64
    readout.float()
    assert getattr(readout, name).dtype == torch.float32


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
@pytest.mark.parametrize("cls", [IncrementalRidgeReadoutLayer, RLSReadout])
def test_unsupported_mps_precision_move_preserves_module(cls: type) -> None:
    readout = cls(3, 1)
    before = {name: tensor.clone() for name, tensor in readout.state_dict().items()}
    with pytest.raises(ValueError, match="MPS does not support float64 solver statistics"):
        readout.to("mps")
    for name, tensor in readout.state_dict().items():
        assert tensor.device.type == "cpu"
        assert torch.equal(tensor, before[name])


@pytest.mark.parametrize("bias", [False, True])
def test_rls_first_partial_fit_has_same_zero_prior_as_online_fit(bias: bool) -> None:
    x = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    y = torch.zeros(1, 1, dtype=torch.float64)
    stream = RLSReadout(2, 1, bias=bias, delta=1.0, dtype=torch.float64)
    online = RLSReadout(2, 1, bias=bias, delta=1.0, dtype=torch.float64)
    stream.partial_fit(x, y)
    online.fit_online(x, y)
    torch.testing.assert_close(stream.weight, online.weight, rtol=0, atol=0)
    assert torch.count_nonzero(stream.weight) == 0
    if bias:
        torch.testing.assert_close(stream.bias, online.bias, rtol=0, atol=0)


@pytest.mark.parametrize("bias", [False, True])
def test_legacy_incremental_statistics_load_and_continue(bias: bool) -> None:
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(50, 3, dtype=torch.float64, generator=generator) + 3.0
    y = x @ torch.tensor([[1.0], [-2.0], [0.5]], dtype=torch.float64)
    legacy = IncrementalRidgeReadoutLayer(3, 1, bias=bias, alpha=0.01).double().state_dict()
    for name in ("_mean_x", "_mean_y", "_statistics_version"):
        del legacy[name]
    legacy.update(
        XtX=x[:20].T @ x[:20],
        Xty=x[:20].T @ y[:20],
        sum_x=x[:20].sum(0),
        sum_y=y[:20].sum(0),
        _n=torch.tensor(20),
    )
    restored = IncrementalRidgeReadoutLayer(3, 1, bias=bias, alpha=0.01).double()
    with pytest.warns(RuntimeWarning, match="Upgraded legacy incremental"):
        restored.load_state_dict(legacy, strict=True)
    restored.partial_fit(x[20:], y[20:])
    restored.finalize()
    reference = RidgeReadoutLayer(3, 1, bias=bias, alpha=0.01).double()
    reference.fit(x, y)
    torch.testing.assert_close(restored.weight, reference.weight, rtol=1e-9, atol=1e-9)
    assert restored.n_seen == len(x)


def test_centered_incremental_checkpoint_continues_offset_stream() -> None:
    x, y = _offset_problem()
    source = IncrementalRidgeReadoutLayer(3, 1, alpha=1e-3).double()
    source.partial_fit(x[:100], y[:100])
    destination = copy.deepcopy(source)
    destination.reset_accumulators()
    destination.load_state_dict(source.state_dict())
    destination.partial_fit(x[100:], y[100:])
    destination.finalize()
    torch.testing.assert_close(destination.weight, _svd_ridge(x, y, 1e-3).T, rtol=1e-6, atol=1e-7)


def test_legacy_empty_incremental_checkpoint_loads_without_warning() -> None:
    readout = IncrementalRidgeReadoutLayer(3, 1)
    state = readout.state_dict()
    for name in ("_mean_x", "_mean_y", "_statistics_version"):
        del state[name]
    with warnings.catch_warnings(record=True) as seen:
        readout.load_state_dict(state)
    assert not seen
    assert readout.n_seen == 0


def test_legacy_full_pickle_upgrades_raw_statistics() -> None:
    generator = torch.Generator().manual_seed(12)
    x = torch.randn(30, 3, dtype=torch.float64, generator=generator) + 2
    y = x[:, :1] - x[:, 1:2]
    legacy = IncrementalRidgeReadoutLayer(3, 1, alpha=0.01).double()
    legacy.XtX.copy_(x.T @ x)
    legacy.Xty.copy_(x.T @ y)
    legacy.sum_x.copy_(x.sum(0))
    legacy.sum_y.copy_(y.sum(0))
    legacy._n.fill_(len(x))
    for name in ("_mean_x", "_mean_y", "_statistics_version"):
        del legacy._buffers[name]
    stream = io.BytesIO()
    torch.save(legacy, stream)
    stream.seek(0)
    with pytest.warns(RuntimeWarning, match="Upgraded legacy incremental"):
        restored = torch.load(stream, weights_only=False)
    restored.finalize()
    torch.testing.assert_close(restored.weight, _svd_ridge(x, y, 0.01).T, rtol=1e-9, atol=1e-9)


def test_legacy_statistics_migrate_under_a_model_prefix() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 5.0], [4.0, 1.0]], dtype=torch.float64)
    y = x[:, :1] + x[:, 1:2]
    model = torch.nn.Sequential(IncrementalRidgeReadoutLayer(2, 1, alpha=0.01).double())
    state = model.state_dict()
    for name in ("_mean_x", "_mean_y", "_statistics_version"):
        del state["0." + name]
    state.update(
        {
            "0.XtX": x.T @ x,
            "0.Xty": x.T @ y,
            "0.sum_x": x.sum(0),
            "0.sum_y": y.sum(0),
            "0._n": torch.tensor(len(x)),
        }
    )
    with pytest.warns(RuntimeWarning, match="Upgraded legacy incremental"):
        model.load_state_dict(state)
    model[0].finalize()
    torch.testing.assert_close(model[0].weight, _svd_ridge(x, y, 0.01).T, rtol=1e-9, atol=1e-9)


def test_unknown_streaming_statistics_version_is_rejected() -> None:
    readout = IncrementalRidgeReadoutLayer(3, 1)
    state = readout.state_dict()
    state["_statistics_version"] = torch.tensor(999)
    with pytest.raises(RuntimeError, match="Unsupported incremental ridge statistics version"):
        readout.load_state_dict(state)


def test_legacy_rls_pickle_keeps_weights_and_restores_working_precision() -> None:
    readout = RLSReadout(2, 1)
    x = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
    readout.partial_fit(x, x[:, :1])
    weights = readout.weight.detach().clone()
    readout.P = readout.P.float()  # Old model.float() checkpoints narrowed P.
    stream = io.BytesIO()
    torch.save(readout, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    assert restored.P.dtype == torch.float64
    assert torch.equal(restored.weight, weights)
    assert restored.n_seen == 2


@pytest.mark.parametrize("power_iters", [1, 30])
def test_fista_null_initial_power_vector_has_nonzero_fit(power_iters: int) -> None:
    x = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)
    y = x[:, :1]
    readout = FISTAReadoutLayer(
        2, 1, bias=False, alpha=0.01, power_iters=power_iters, tol=1e-10
    ).double()
    readout.fit(x, y)
    # The symmetric L1 solution has prediction gain 1-alpha/4 on these two rows.
    torch.testing.assert_close(readout(x), y * (1 - 0.01 / 4), rtol=1e-8, atol=1e-8)
    assert readout.converged_


def test_fista_backtracking_repairs_a_positive_underestimate() -> None:
    top = torch.tensor([1.0, -1.0 + 1e-5], dtype=torch.float64)
    top /= top.norm()
    orthogonal = torch.tensor([-top[1], top[0]])
    q = torch.stack((top, orthogonal), dim=1)
    x = torch.diag(torch.tensor([100.0, 1.0], dtype=torch.float64)) @ q.T
    y = x @ top[:, None]
    readout = FISTAReadoutLayer(
        2,
        1,
        bias=False,
        alpha=0.1,
        l1_ratio=0.0,
        power_iters=1,
        power_iter_margin=0.0,
        tol=1e-10,
    ).double()
    expected = torch.linalg.solve(x.T @ x + 0.05 * torch.eye(2), x.T @ y)
    readout.fit(x, y)
    torch.testing.assert_close(readout.weight, expected.T, rtol=1e-7, atol=1e-7)
    assert readout.converged_


@pytest.mark.parametrize("bias", [False, True])
def test_documented_custom_lstsq_solver_fits_intercept(bias: bool) -> None:
    path = Path(__file__).resolve().parents[2] / "docs/build/readouts/cg-readout.md"
    block = next(
        source
        for source in re.findall(r"```python\n(.*?)\n```", path.read_text(), re.S)
        if "class LstsqReadout" in source
    )
    namespace: dict = {}
    exec(compile(block, str(path), "exec"), namespace)
    readout = namespace["LstsqReadout"](2, 1, bias=bias).double()
    x = torch.tensor([[1.0, 2.0], [3.0, 1.0], [-2.0, 0.5], [0.0, 0.0]], dtype=torch.float64)
    y = x[:, :1] - 2 * x[:, 1:2] + (4 if bias else 0)
    readout.fit(x, y)
    torch.testing.assert_close(readout(x), y, rtol=1e-10, atol=1e-10)
