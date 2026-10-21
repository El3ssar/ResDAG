"""Independent math and pipeline contracts for oscillator/LDN/native recurrent cells."""

import copy
import io
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch
from scipy.linalg import expm
from torch import nn

from resdag.core import ESNModel, reservoir_input
from resdag.layers import RidgeReadoutLayer
from resdag.layers.cells import CoRNNCell, LegendreDelayCell, TorchRecurrentCell
from resdag.layers.reservoirs import CoRNNLayer, LegendreDelayLayer, TorchRecurrentLayer
from resdag.layers.reservoirs.base_reservoir import BaseReservoirLayer
from resdag.training import ESNTrainer


def make_layer(kind: str, *, driver: bool = True, trainable: bool = False) -> BaseReservoirLayer:
    options = dict(feedback_size=2, input_size=1 if driver else None, trainable=trainable)
    if kind == "cornn":
        return CoRNNLayer(5, seed=7, **options)
    if kind == "ldn":
        return LegendreDelayLayer(3, theta=7, seed=7, **options)
    native = (
        nn.GRUCell(3 if driver else 2, 5) if kind == "gru" else nn.LSTMCell(3 if driver else 2, 5)
    )
    return TorchRecurrentLayer(native, **options)


KINDS = ["cornn", "ldn", "gru", "lstm"]


def test_cornn_independent_scalar_equations_and_state_order() -> None:
    cell = CoRNNCell(2, 1, input_size=1, dt=0.2, gamma=0.7, epsilon=0.3, dtype=torch.float64)
    # Fixed asymmetric values distinguish h/v recurrence, damping and ordering.
    with torch.no_grad():
        cell.weight_hh.copy_(torch.tensor([[0.1, 0.2], [-0.4, 0.3]]))
        cell.weight_hv.copy_(torch.tensor([[-0.2, 0.7], [0.5, -0.1]]))
        cell.weight_feedback.copy_(torch.tensor([[0.4], [-0.5]]))
        assert cell.weight_input is not None and cell.bias_h is not None
        cell.weight_input.copy_(torch.tensor([[0.8], [-0.3]]))
        cell.bias_h.copy_(torch.tensor([0.03, -0.02]))
    x, u = 0.6, -0.2
    h, v = [0.2, -0.4], [0.3, 0.1]
    expected_h, expected_v = [], []
    for row in range(2):
        force = math.tanh(
            sum(
                float(cell.weight_hh[row, j]) * h[j] + float(cell.weight_hv[row, j]) * v[j]
                for j in range(2)
            )
            + float(cell.weight_feedback[row, 0]) * x
            + float(cell.weight_input[row, 0]) * u
            + float(cell.bias_h[row])
        )
        velocity = v[row] + 0.2 * (force - 0.7 * h[row] - 0.3 * v[row])
        expected_v.append(velocity)
        expected_h.append(h[row] + 0.2 * velocity)
    output, state = cell(
        [torch.tensor([[x]], dtype=torch.float64), torch.tensor([[u]], dtype=torch.float64)],
        torch.tensor([h + v], dtype=torch.float64),
    )
    torch.testing.assert_close(
        output, torch.tensor([expected_h], dtype=torch.float64), rtol=1e-14, atol=1e-14
    )
    torch.testing.assert_close(
        state, torch.tensor([expected_h + expected_v], dtype=torch.float64), rtol=1e-14, atol=1e-14
    )
    assert output.shape[-1] * 2 == state.shape[-1]


def test_cornn_initializer_roles_alias_and_spectral_scaling() -> None:
    cell = CoRNNCell(
        4,
        2,
        recurrent_initializer=nn.init.eye_,
        velocity_initializer=nn.init.zeros_,
        spectral_radius=0.3,
        seed=3,
        dtype=torch.float64,
    )
    torch.testing.assert_close(cell.weight_hh, torch.eye(4, dtype=torch.float64) * 0.3)
    assert torch.count_nonzero(cell.weight_hv) == 0
    alias = CoRNNCell(4, 2, topology=nn.init.eye_, spectral_radius=0.3, seed=3, dtype=torch.float64)
    torch.testing.assert_close(cell.weight_hh, alias.weight_hh)
    with pytest.raises(ValueError, match="both|mutually|only one|Specify"):
        CoRNNCell(4, 2, topology=nn.init.eye_, recurrent_initializer=nn.init.eye_)


@pytest.mark.parametrize("order", [1, 2, 5, 12])
def test_ldn_matches_independent_scipy_zoh_and_paper_coefficients(order: int) -> None:
    theta, dt = 4.7, 0.23
    cell = LegendreDelayCell(order, 2, input_size=1, theta=theta, dt=dt, dtype=torch.float64)
    # Construct equation (2) by scalar indices, then use a different matrix_exp backend.
    A = np.array(
        [
            [(2 * i + 1) * (-1 if i < j else (-1) ** (i - j + 1)) / theta for j in range(order)]
            for i in range(order)
        ]
    )
    B = np.array([(2 * i + 1) * (-1) ** i / theta for i in range(order)])
    augmented = np.zeros((order + 1, order + 1))
    augmented[:order, :order], augmented[:order, order] = A, B
    E = expm(dt * augmented)
    np.testing.assert_allclose(cell.continuous_A.numpy(), A, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(cell.continuous_B.numpy(), B, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(cell.A.numpy(), E[:order, :order], rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(cell.B.numpy(), E[:order, order], rtol=1e-11, atol=1e-14)
    state = torch.linspace(-0.2, 0.3, 2 * 3 * order, dtype=torch.float64).reshape(2, -1)
    fb = torch.tensor([[0.4, -0.1], [0.9, 0.2]], dtype=torch.float64)
    driver = torch.tensor([[0.3], [-0.7]], dtype=torch.float64)
    values = np.concatenate((fb.numpy(), driver.numpy()), axis=-1)
    expected = np.empty((2, 3, order))
    for batch in range(2):
        for channel in range(3):
            old = state.numpy()[batch, channel * order : (channel + 1) * order]
            expected[batch, channel] = (
                E[:order, :order] @ old + E[:order, order] * values[batch, channel]
            )
    output, new_state = cell([fb, driver], state)
    np.testing.assert_allclose(
        output.numpy().reshape(2, 3, order), expected, rtol=1e-11, atol=1e-14
    )
    assert output is new_state


def test_ldn_order_one_closed_form_and_constant_input_semigroup() -> None:
    cell = LegendreDelayCell(1, 1, theta=3.2, dt=0.4, dtype=torch.float64)
    decay = math.exp(-0.4 / 3.2)
    torch.testing.assert_close(
        cell.A, torch.tensor([[decay]], dtype=torch.float64), rtol=1e-14, atol=1e-14
    )
    torch.testing.assert_close(
        cell.B, torch.tensor([1 - decay], dtype=torch.float64), rtol=1e-14, atol=1e-14
    )
    for order in [1, 6]:
        small = LegendreDelayCell(order, 1, theta=9.1, dt=0.2, dtype=torch.float64)
        large = LegendreDelayCell(order, 1, theta=9.1, dt=1.0, dtype=torch.float64)
        state = torch.linspace(-0.1, 0.2, order, dtype=torch.float64).reshape(1, -1)
        x = torch.tensor([[0.9]], dtype=torch.float64)
        expected = large([x], state)[1]
        for _ in range(5):
            _, state = small([x], state)
        torch.testing.assert_close(state, expected, rtol=1e-11, atol=1e-13)


def test_ldn_encoders_train_but_dynamics_stay_fixed() -> None:
    cell = LegendreDelayCell(
        4,
        2,
        1,
        memory_channels=2,
        feedback_initializer=nn.init.ones_,
        input_initializer=nn.init.zeros_,
        trainable=True,
    )
    torch.testing.assert_close(cell.weight_feedback, torch.ones(2, 2))
    assert cell.weight_input is not None and torch.count_nonzero(cell.weight_input) == 0
    assert all(p.requires_grad for p in cell.parameters())
    assert all(not b.requires_grad for b in cell.buffers())
    output, _ = cell([torch.ones(1, 2), torch.ones(1, 1)], torch.zeros(1, 8))
    output.square().sum().backward()
    assert cell.weight_feedback.grad is not None
    assert all(b.grad is None for b in cell.buffers())


@pytest.mark.parametrize("native_class", [nn.GRUCell, nn.LSTMCell])
@pytest.mark.parametrize("bias", [True, False])
def test_native_adapter_matches_torch_values_parameter_and_input_gradients(
    native_class: type, bias: bool
) -> None:
    native = native_class(3, 4, bias=bias).double()
    adapted = TorchRecurrentCell(native, 2, input_size=1, trainable=True)
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    state = torch.randn(2, adapted.state_size, dtype=torch.float64, requires_grad=True)
    clone_x, clone_state = (
        x.detach().clone().requires_grad_(),
        state.detach().clone().requires_grad_(),
    )
    output, next_state = adapted([x[:, :2], x[:, 2:]], state)
    if native_class is nn.LSTMCell:
        hidden, carry = native(clone_x, tuple(clone_state.chunk(2, -1)))
        expected = torch.cat((hidden, carry), -1)
    else:
        hidden = native(clone_x, clone_state)
        expected = hidden
    torch.testing.assert_close(output, hidden, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(next_state, expected, rtol=1e-13, atol=1e-14)
    (output.square().sum() + next_state.square().sum()).backward()
    (hidden.square().sum() + expected.square().sum()).backward()
    torch.testing.assert_close(x.grad, clone_x.grad, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(state.grad, clone_state.grad, rtol=1e-12, atol=1e-13)
    for ours, reference in zip(adapted.recurrent_cell.parameters(), native.parameters()):
        assert ours is not reference
        torch.testing.assert_close(ours.grad, reference.grad, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("driver", [True, False])
def test_projected_sequence_chunking_empty_input_and_frozen_default(
    kind: str, driver: bool
) -> None:
    layer = make_layer(kind, driver=driver).double()
    streams = [torch.randn(2, 9, 2, dtype=torch.float64)]
    if driver:
        streams.append(torch.randn(2, 9, 1, dtype=torch.float64))
    state = torch.randn(2, layer.cell.state_size, dtype=torch.float64)
    original = state.clone()
    assert all(not p.requires_grad for p in layer.parameters())
    full, final = layer.forward_stateless(streams, state)
    projected = layer.cell.project_inputs(streams)
    assert projected is not None
    oracle, expected_state = [], state
    for t in range(9):
        output, new_state = layer.cell([stream[:, t] for stream in streams], expected_state)
        direct_output, direct_state = layer.cell.step(projected[:, t], expected_state)
        torch.testing.assert_close(output, direct_output, rtol=1e-12, atol=1e-13)
        torch.testing.assert_close(new_state, direct_state, rtol=1e-12, atol=1e-13)
        oracle.append(output)
        expected_state = new_state
    torch.testing.assert_close(full, torch.stack(oracle, 1), rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(final, expected_state, rtol=1e-12, atol=1e-13)
    left, middle = layer.forward_stateless([s[:, :4] for s in streams], state)
    right, chunk_final = layer.forward_stateless([s[:, 4:] for s in streams], middle)
    torch.testing.assert_close(full, torch.cat((left, right), 1), rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(final, chunk_final, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(state, original)
    empty, unchanged = layer.forward_stateless([s[:, :0] for s in streams], state)
    assert empty.shape == (2, 0, layer.cell.output_size) and unchanged is state
    assert layer.state is None


@pytest.mark.parametrize("kind", KINDS)
def test_gradcheck_frozen_input_and_initial_state(kind: str) -> None:
    layer = make_layer(kind).double()
    feedback = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)
    driver = torch.randn(1, 3, 1, dtype=torch.float64, requires_grad=True)
    state = torch.randn(1, layer.cell.state_size, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x, u, h: layer.forward_stateless([x, u], h),
        (feedback, driver, state),
        fast_mode=True,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_gradient_pipeline_and_live_freezing(kind: str) -> None:
    layer = make_layer(kind, trainable=True)
    # Ordinary upstream/downstream modules and BPTT over an explicit chunk boundary.
    encoder = nn.Linear(3, 3)
    head = nn.Linear(layer.cell.output_size, 1)
    raw = torch.randn(2, 7, 3)
    transformed = encoder(raw)
    state = layer.cell.init_state(2, raw.device, raw.dtype)
    left, middle = layer.forward_stateless([transformed[:, :3, :2], transformed[:, :3, 2:]], state)
    right, final = layer.forward_stateless([transformed[:, 3:, :2], transformed[:, 3:, 2:]], middle)
    loss = head(torch.cat((left, right), 1)).square().mean() + final.square().mean()
    loss.backward()
    assert encoder.weight.grad is not None and torch.count_nonzero(encoder.weight.grad)
    assert head.weight.grad is not None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    layer.trainable = False
    assert not layer.cell.trainable and all(not p.requires_grad for p in layer.parameters())
    layer.trainable = True
    assert layer.cell.trainable and all(p.requires_grad for p in layer.parameters())


@pytest.mark.parametrize("kind", KINDS)
def test_state_dict_pickle_warmed_state_and_dtype_migration(kind: str) -> None:
    layer = make_layer(kind)
    x, u = torch.randn(2, 4, 2), torch.randn(2, 4, 1)
    layer(x, u)
    buffer = io.BytesIO()
    torch.save(layer, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    assert restored.state is not None and layer.state is not None
    torch.testing.assert_close(restored.state, layer.state)
    expected = layer(x, u)
    torch.testing.assert_close(restored(x, u), expected)
    copied = copy.deepcopy(restored).double()
    assert copied.state is not None and copied.state.dtype == torch.float64
    copied(x.double(), u.double())
    assert all(p.dtype == torch.float64 for p in copied.parameters())
    state_dict = restored.state_dict()
    assert "state" not in state_dict
    fresh = make_layer(kind)
    fresh.load_state_dict(state_dict)
    fresh.reset_state()
    restored.reset_state()
    torch.testing.assert_close(fresh(x, u), restored(x, u))


@pytest.mark.parametrize("kind", KINDS)
def test_resdag_readout_fit_forecast_and_full_checkpoint(kind: str, tmp_path: Path) -> None:
    layer = make_layer(kind, driver=False)
    feedback = reservoir_input(2)
    features = layer(feedback)
    output = RidgeReadoutLayer(layer.cell.output_size, 2, alpha=1e-3, name="output")(features)
    model = ESNModel(feedback, output)
    time = torch.arange(100, dtype=torch.float32) * 0.08
    series = torch.stack((time.sin(), time.cos()), -1).unsqueeze(0)
    ESNTrainer(model).fit((series[:, :12],), (series[:, 12:79],), {"output": series[:, 13:80]})
    forecast = model.forecast(series[:, 65:80], horizon=6)
    assert forecast.shape == (1, 6, 2) and torch.isfinite(forecast).all()
    path = tmp_path / f"{kind}.pt"
    model.save_full(path)
    restored = ESNModel.load_full(path)
    torch.testing.assert_close(restored.forecast(series[:, 65:80], horizon=6), forecast)


@pytest.mark.parametrize("kind", KINDS)
def test_invalid_streams_and_states_fail_clearly(kind: str) -> None:
    layer = make_layer(kind)
    cell = layer.cell
    state = cell.init_state(2, torch.device("cpu"), torch.float32)
    with pytest.raises(ValueError, match="streams"):
        cell.project_inputs([torch.zeros(2, 2)])
    with pytest.raises(ValueError, match="Driver"):
        cell.project_inputs([torch.zeros(2, 2), torch.zeros(3, 1)])
    with pytest.raises(ValueError, match="Feedback"):
        cell.project_inputs([torch.zeros(2, 3), torch.zeros(2, 1)])
    with pytest.raises(ValueError, match="state"):
        cell([torch.zeros(2, 2), torch.zeros(2, 1)], state[:, :-1])
    with pytest.raises(ValueError, match="Projected"):
        cell.step(torch.zeros(3, cell.output_size), state)
    with pytest.raises(TypeError, match="trainable"):
        layer.trainable = 1


@pytest.mark.parametrize(
    "constructor,kwargs",
    [
        (CoRNNCell, {"reservoir_size": 0, "feedback_size": 1}),
        (CoRNNCell, {"reservoir_size": 2, "feedback_size": True}),
        (CoRNNCell, {"reservoir_size": 2, "feedback_size": 1, "input_size": 0}),
        (CoRNNCell, {"reservoir_size": 2, "feedback_size": 1, "dt": 0}),
        (CoRNNCell, {"reservoir_size": 2, "feedback_size": 1, "gamma": float("nan")}),
        (CoRNNCell, {"reservoir_size": 2, "feedback_size": 1, "epsilon": -1}),
        (CoRNNCell, {"reservoir_size": 2, "feedback_size": 1, "spectral_radius": float("inf")}),
        (LegendreDelayCell, {"order": 0, "feedback_size": 1}),
        (LegendreDelayCell, {"order": 2, "feedback_size": 1, "theta": 0}),
        (LegendreDelayCell, {"order": 2, "feedback_size": 1, "dt": float("inf")}),
        (LegendreDelayCell, {"order": 2, "feedback_size": 1, "memory_channels": False}),
    ],
)
def test_invalid_constructor_values(constructor: Callable, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        constructor(**kwargs)


def test_live_cornn_parameters_and_fixed_ldn_discretization() -> None:
    oscillator = CoRNNLayer(3, 1)
    for name in ["dt", "gamma", "epsilon"]:
        setattr(oscillator, name, 0.27)
        assert getattr(oscillator.cell, name) == 0.27 and name not in oscillator.__dict__
        with pytest.raises(ValueError):
            setattr(oscillator, name, -1)
    memory = LegendreDelayLayer(3, 1)
    for name in ["order", "memory_channels", "theta", "dt"]:
        with pytest.raises(AttributeError, match="fixed"):
            setattr(memory, name, 2)


def test_native_adapter_validates_type_input_width_and_preserves_original_flags() -> None:
    native = nn.LSTMCell(3, 4).double()
    adapter = TorchRecurrentCell(native, 2, 1)
    assert all(p.requires_grad for p in native.parameters())
    assert all(not p.requires_grad for p in adapter.parameters())
    assert all(p.dtype == torch.float64 for p in adapter.parameters())
    with pytest.raises(TypeError, match="GRUCell"):
        TorchRecurrentCell(nn.RNNCell(3, 4), 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="width"):
        TorchRecurrentCell(native, 1)


@pytest.mark.parametrize("cell_class", [CoRNNCell, LegendreDelayCell])
def test_seeded_native_precision_construction_is_repeatable_and_rng_isolated(
    cell_class: type,
) -> None:
    before = torch.random.get_rng_state().clone()
    first = cell_class(4, 2, 1, seed=99, dtype=torch.float64)
    assert torch.equal(before, torch.random.get_rng_state())
    second = cell_class(4, 2, 1, seed=99, dtype=torch.float64)
    assert all(p.dtype == torch.float64 for p in first.parameters())
    assert all(b.dtype == torch.float64 for b in first.buffers())
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("kind", KINDS)
def test_cuda_device_projection_and_state_correctness(kind: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cpu = make_layer(kind).double()
    gpu = copy.deepcopy(cpu).cuda()
    x, u = torch.randn(2, 5, 2, dtype=torch.float64), torch.randn(2, 5, 1, dtype=torch.float64)
    expected = cpu(x, u)
    actual = gpu(x.cuda(), u.cuda())
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-10, atol=1e-11)
    assert gpu.state is not None and gpu.state.is_cuda


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("kind", KINDS)
def test_mps_construction_projected_scan_and_gradients(kind: str) -> None:
    if kind == "cornn":
        gpu = CoRNNLayer(5, 2, input_size=1, seed=17, device="mps", trainable=True)
    elif kind == "ldn":
        gpu = LegendreDelayLayer(3, 2, input_size=1, theta=7, seed=17, device="mps", trainable=True)
    else:
        native = (
            nn.GRUCell(3, 5, device="mps") if kind == "gru" else nn.LSTMCell(3, 5, device="mps")
        )
        gpu = TorchRecurrentLayer(native, 2, 1, trainable=True)
    cpu = copy.deepcopy(gpu).cpu()
    x = torch.randn(2, 5, 2)
    u = torch.randn(2, 5, 1)
    expected = cpu(x, u)
    actual = gpu(x.to("mps"), u.to("mps"))
    assert all(p.device.type == "mps" for p in gpu.parameters())
    assert all(b.device.type == "mps" for b in gpu.buffers())
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)
    actual.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gpu.parameters())


@pytest.mark.parametrize("kind", KINDS)
def test_projected_step_traces_as_one_full_graph(kind: str) -> None:
    # CPU eager backend verifies graph capture, not CUDA kernel performance.
    layer = make_layer(kind)
    cell = layer.cell
    feedback, driver = torch.randn(2, 2), torch.randn(2, 1)
    projected = cell.project_inputs([feedback, driver])
    assert projected is not None
    state = torch.randn(2, cell.state_size)
    expected = cell.step(projected, state)
    compiled = torch.compile(cell.step, backend="eager", fullgraph=True)
    actual = compiled(projected, state)
    for ours, reference in zip(actual, expected):
        torch.testing.assert_close(ours, reference)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cell_class", [CoRNNCell, LegendreDelayCell])
def test_low_precision_construction_keeps_runtime_weights_and_buffers_in_dtype(
    cell_class: type, dtype: torch.dtype
) -> None:
    cell = cell_class(4, 2, 1, dtype=dtype, seed=7)
    assert all(p.dtype == dtype for p in cell.parameters())
    assert all(b.dtype == dtype for b in cell.buffers())
    feedback, driver = torch.ones(2, 2, dtype=dtype), torch.ones(2, 1, dtype=dtype)
    output, state = cell([feedback, driver], cell.init_state(2, torch.device("cpu"), dtype))
    assert output.dtype == dtype and state.dtype == dtype
    assert torch.isfinite(output).all() and torch.isfinite(state).all()


def test_ldn_rejects_unrepresentable_dynamics() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        LegendreDelayCell(2, 1, theta=1e-40, dtype=torch.float32)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("cell_class", [CoRNNCell, LegendreDelayCell])
def test_implicit_default_device_mps_and_explicit_float64_rejection(cell_class: type) -> None:
    with torch.device("mps"):
        cell = cell_class(4, 2, 1, seed=17)
        assert all(p.device.type == "mps" for p in cell.parameters())
        assert all(b.device.type == "mps" for b in cell.buffers())
        feedback, driver = torch.zeros(1, 2), torch.zeros(1, 1)
        output, _ = cell([feedback, driver], cell.init_state(1, feedback.device, feedback.dtype))
        assert output.device.type == "mps"
    with pytest.raises(TypeError, match="MPS.*float64"):
        cell_class(4, 2, device="mps", dtype=torch.float64)


@pytest.mark.parametrize("native_class", [nn.GRUCell, nn.LSTMCell])
@pytest.mark.parametrize("bias", [True, False])
def test_native_adapter_cpu_autocast_parity(native_class: type, bias: bool) -> None:
    torch.manual_seed(1)
    native = native_class(3, 4, bias=bias)
    adapted = TorchRecurrentCell(native, 2, 1)
    x = torch.randn(2, 3)
    state = torch.randn(2, adapted.state_size)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, new_state = adapted([x[:, :2], x[:, 2:]], state)
        if native_class is nn.LSTMCell:
            hidden, carry = native(x, tuple(state.chunk(2, -1)))
            expected_state = torch.cat((hidden, carry), -1)
        else:
            hidden = native(x, state)
            expected_state = hidden
    assert actual.dtype == hidden.dtype
    torch.testing.assert_close(actual, hidden, rtol=0, atol=0)
    torch.testing.assert_close(new_state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize(
    "theta,dt,dtype",
    [
        (1e-200, 1.0, torch.float64),
        (1e-30, 1.0, torch.float32),
        (1.0, 1e300, torch.float64),
        (1e-200, 1e200, torch.float64),
    ],
)
def test_ldn_rejects_extreme_finite_zoh_work_before_matrix_exp(
    theta: float, dt: float, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_exp(matrix: torch.Tensor) -> torch.Tensor:
        raise AssertionError("Unsafe matrix reached matrix_exp")

    monkeypatch.setattr(torch, "matrix_exp", forbidden_exp)
    with pytest.raises(ValueError, match="ZOH|non-finite"):
        LegendreDelayCell(2, 1, theta=theta, dt=dt, dtype=dtype)
