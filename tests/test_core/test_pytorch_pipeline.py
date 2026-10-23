"""Trained reservoirs inside trainable PyTorch pipelines, with independent oracles."""

import copy

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

import resdag as rd


@pytest.fixture
def fresh_compiler_cache():
    # Each model/call-shape acceptance starts from a fresh compiler cache.
    # Other full-suite compile tests also specialize the shared Module wrapper.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def _training():
    time = torch.arange(90, dtype=torch.float64) / 9
    return torch.stack((time.sin() + 3, 2 * time.cos() - 4), dim=-1)


def _model(kind):
    options = dict(seed=23, dtype=torch.float64, spectral_radius=0.5, leak_rate=0.4)
    if kind in {"vanilla", "sparse"}:
        model = rd.ESN(
            12,
            input_size=2,
            washout=5,
            readout="svd",
            alpha=0.01,
            sparse=kind == "sparse",
            **options,
        )
    elif kind == "ott":
        model = rd.Model(
            graph=rd.ott_esn(12, 2, 2, readout="svd", readout_alpha=0.01, **options),
            washout=5,
        )
    else:
        x = rd.reservoir_input(2, dtype=torch.float64)
        a = rd.ESNLayer(7, feedback_size=2, **options)(x)
        b = rd.ESNLayer(5, feedback_size=2, **{**options, "seed": 24})(x)
        merged = rd.Concatenate()(a, b)
        position = rd.SVDReadoutLayer(12, 2, alpha=0.01).double()(merged)
        velocity = rd.SVDReadoutLayer(12, 2, alpha=0.01).double()(merged)
        model = rd.Model(
            inputs={"signal": x},
            outputs={"position": position, "velocity": velocity},
            washout=5,
        )
    data = _training()
    targets = {"position": data, "velocity": 0.5 * data + 1} if kind == "custom" else None
    return model.fit(data, targets)


class _Pipeline(nn.Module):
    def __init__(self, reservoir):
        super().__init__()
        self.encoder = nn.Linear(2, 2, dtype=torch.float64)
        self.reservoir = reservoir
        workflow = reservoir.workflow if isinstance(reservoir, rd.ESN) else reservoir
        self.head = nn.Linear(2 * len(workflow.output_names), 1, dtype=torch.float64)
        # Explicit weights keep tests independent of the caller's global RNG.
        with torch.no_grad():
            self.encoder.weight.copy_(torch.tensor([[0.8, 0.1], [-0.2, 0.7]]))
            self.encoder.bias.copy_(torch.tensor([0.1, -0.1]))
            self.head.weight.fill_(0.2)
            self.head.bias.fill_(0.05)

    def predict(self, features):
        if isinstance(features, dict):
            features = torch.cat(tuple(features.values()), dim=-1)
        return self.head(features)

    def forward(self, data):
        return self.predict(self.reservoir(self.encoder(data)))

    def chunk(self, data, states=None):
        features, final = self.reservoir.forward_stateless(self.encoder(data), states=states)
        return self.predict(features), final


def _manual_vanilla(pipeline, inputs):
    """Ordinary tensor equations; do not call Model, the graph or the cell."""
    reservoir = pipeline.reservoir
    model = reservoir.workflow if isinstance(reservoir, rd.ESN) else reservoir
    cell = next(module.cell for module in model.modules() if isinstance(module, rd.ESNLayer))
    readout = next(module for module in model.modules() if isinstance(module, rd.SVDReadoutLayer))
    encoded = inputs @ pipeline.encoder.weight.T + pipeline.encoder.bias
    scaler = model.input_scalers[0]
    normalized = (encoded - scaler._stat0) / scaler._stat1
    state = normalized.new_zeros(inputs.shape[0], cell.reservoir_size)
    sequence = []
    for frame in normalized.unbind(1):
        preactivation = frame @ cell.weight_feedback.T + state @ cell.weight_hh.T + cell.bias_h
        activated = torch.tanh(preactivation)
        state = state + cell.leak_rate * (activated - state)
        sequence.append(state)
    features = torch.cat((normalized, torch.stack(sequence, 1)), dim=-1)
    prediction = features @ readout.weight.T + readout.bias
    output_scaler = model.output_scalers[0]
    prediction = prediction * output_scaler._stat1 + output_scaler._stat0
    return prediction @ pipeline.head.weight.T + pipeline.head.bias


def test_encoder_reservoir_head_matches_independent_equations_and_gradients():
    pipeline = _Pipeline(_model("vanilla")).eval()
    source = _training()[-8:].unsqueeze(0).requires_grad_()
    pipeline.reservoir.reset_state()
    actual = pipeline(source)
    expected = _manual_vanilla(pipeline, source)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    parameters = (source, pipeline.encoder.weight, pipeline.encoder.bias, pipeline.head.weight)
    observed = torch.autograd.grad(actual.square().mean(), parameters)
    reference = torch.autograd.grad(expected.square().mean(), parameters)
    for gradient, oracle in zip(observed, reference, strict=True):
        torch.testing.assert_close(gradient, oracle, rtol=1e-10, atol=1e-12)
        assert bool(torch.isfinite(gradient).all()) and bool(gradient.abs().sum() > 0)

    # A finite-difference oracle also avoids merely comparing two autograd graphs.
    epsilon = 1e-5
    with torch.no_grad():
        original = pipeline.encoder.weight[0, 0].item()
        values = []
        for offset in (epsilon, -epsilon):
            pipeline.encoder.weight[0, 0] = original + offset
            values.append(_manual_vanilla(pipeline, source).square().mean())
        pipeline.encoder.weight[0, 0] = original
    derivative = (values[0] - values[1]) / (2 * epsilon)
    torch.testing.assert_close(observed[1][0, 0], derivative, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("kind", ["vanilla", "ott", "custom", "sparse"])
def test_repeated_optimizer_steps_preserve_frozen_rc_and_detach_carried_state(kind):
    pipeline = _Pipeline(_model(kind)).train()
    reservoir = pipeline.reservoir
    reservoir.reset_state()
    fixed = {name: value.detach().clone() for name, value in reservoir.named_parameters()}
    parameter_ids = [id(parameter) for parameter in pipeline.parameters()]
    optimizer = torch.optim.SGD(pipeline.parameters(), lr=0.01)
    initial_encoder = pipeline.encoder.weight.detach().clone()
    initial_head = pipeline.head.weight.detach().clone()
    source = _training()[-12:].unsqueeze(0).repeat(3, 1, 1)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        prediction = pipeline(source)
        prediction.square().mean().backward()
        for module in (pipeline.encoder, pipeline.head):
            assert module.weight.grad is not None
            assert bool(torch.isfinite(module.weight.grad).all())
            assert bool(module.weight.grad.abs().sum() > 0)
        optimizer.step()
        assert all(parameter.grad is None for parameter in reservoir.parameters())
        for layer in reservoir.modules():
            if isinstance(layer, rd.BaseReservoirLayer):
                assert layer.state is not None and layer.state.grad_fn is None
    assert parameter_ids == [id(parameter) for parameter in pipeline.parameters()]
    for name, value in reservoir.named_parameters():
        torch.testing.assert_close(value, fixed[name], rtol=0, atol=0)
    assert not torch.equal(initial_encoder, pipeline.encoder.weight)
    assert not torch.equal(initial_head, pipeline.head.weight)


@pytest.mark.parametrize("kind", ["vanilla", "ott", "custom", "sparse"])
def test_explicit_state_pipeline_chunks_preserve_full_sequence_gradients(kind):
    pipeline = _Pipeline(_model(kind)).eval()
    source = _training()[-11:].unsqueeze(0).repeat(2, 1, 1).requires_grad_()
    whole, expected_state = pipeline.chunk(source)
    first, state = pipeline.chunk(source[:, :4])
    second, final = pipeline.chunk(source[:, 4:], state)
    observed = torch.cat((first, second), dim=1)
    torch.testing.assert_close(observed, whole, rtol=1e-11, atol=1e-11)
    parameters = (source, pipeline.encoder.weight, pipeline.head.weight)
    reference_gradients = torch.autograd.grad(whole.square().sum(), parameters)
    actual_gradients = torch.autograd.grad(observed.square().sum(), parameters)
    for actual, expected in zip(actual_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    assert final.keys() == expected_state.keys()
    for name in final:
        # Chunking changes projection matrix shapes and can change BLAS rounding.
        torch.testing.assert_close(final[name], expected_state[name], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["vanilla", "ott", "custom"])
def test_standard_pipeline_state_dict_preserves_parameters_and_chunk_predictions(kind):
    pipeline = _Pipeline(_model(kind)).eval()
    with torch.no_grad():
        pipeline.encoder.weight.add_(0.1)
        pipeline.head.bias.add_(0.2)
    source = _training()[-7:].unsqueeze(0)
    expected, _ = pipeline.chunk(source)
    destination = _Pipeline(_model(kind)).eval()
    parameter_ids = [id(parameter) for parameter in destination.parameters()]
    destination.load_state_dict(copy.deepcopy(pipeline.state_dict()))
    assert parameter_ids == [id(parameter) for parameter in destination.parameters()]
    actual, _ = destination.chunk(source)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["vanilla", "ott", "custom"])
def test_compiled_surrounding_pipeline_preserves_gradients_and_carried_state(
    kind, fresh_compiler_cache
):
    reference = _Pipeline(_model(kind)).eval()
    pipeline = copy.deepcopy(reference)
    reference.reservoir.reset_state()
    pipeline.reservoir.reset_state()
    # AOT eager checks full graph capture and compiled backward, not GPU speed.
    compiled = torch.compile(pipeline, backend="aot_eager", fullgraph=True)
    source = _training()[-7:].unsqueeze(0)
    for _ in range(2):
        expected_input = source.clone().requires_grad_()
        actual_input = source.clone().requires_grad_()
        expected = reference(expected_input)
        actual = compiled(actual_input)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
        expected_gradients = torch.autograd.grad(
            expected.square().sum(),
            (expected_input, reference.encoder.weight, reference.head.weight),
        )
        actual_gradients = torch.autograd.grad(
            actual.square().sum(),
            (actual_input, pipeline.encoder.weight, pipeline.head.weight),
        )
        for observed, oracle in zip(actual_gradients, expected_gradients, strict=True):
            torch.testing.assert_close(observed, oracle, rtol=1e-10, atol=1e-10)
        for layer in pipeline.modules():
            if isinstance(layer, rd.BaseReservoirLayer):
                assert layer.state is not None and layer.state.grad_fn is None


def test_compiled_pipeline_can_preserve_gradients_across_call_boundaries(fresh_compiler_cache):
    reference = _Pipeline(_model("vanilla")).eval()
    pipeline = copy.deepcopy(reference)
    for model in (reference, pipeline):
        model.reservoir.reset_state()
        for layer in model.modules():
            if isinstance(layer, rd.BaseReservoirLayer):
                layer.detach_state_between_calls = False
    compiled = torch.compile(pipeline, backend="aot_eager", fullgraph=True)
    source = _training()[-9:].unsqueeze(0)
    expected_input = source.clone().requires_grad_()
    actual_input = source.clone().requires_grad_()
    expected = reference(expected_input)
    actual = torch.cat((compiled(actual_input[:, :4]), compiled(actual_input[:, 4:])), dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    observed = torch.autograd.grad(actual.square().sum(), (actual_input, pipeline.encoder.weight))
    oracle = torch.autograd.grad(
        expected.square().sum(), (expected_input, reference.encoder.weight)
    )
    for gradient, target in zip(observed, oracle, strict=True):
        torch.testing.assert_close(gradient, target, rtol=1e-10, atol=1e-10)
    assert all(
        layer.state.grad_fn is not None
        for layer in pipeline.modules()
        if isinstance(layer, rd.BaseReservoirLayer)
    )


@pytest.mark.parametrize("kind", ["vanilla", "ott", "custom", "sparse"])
def test_mixed_precision_pipeline_keeps_encoder_and_head_gradients(kind):
    pipeline = _Pipeline(_model(kind)).float().train()
    source = _training()[-9:].float().unsqueeze(0).repeat(2, 1, 1).requires_grad_()
    pipeline.reservoir.reset_state()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = pipeline(source)
        loss = prediction.float().square().mean()
    loss.backward()
    assert prediction.dtype == torch.bfloat16
    for gradient in (source.grad, pipeline.encoder.weight.grad, pipeline.head.weight.grad):
        assert gradient is not None and bool(torch.isfinite(gradient).all())
        assert bool(gradient.abs().sum() > 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("topology", ["erdos_renyi", "simple_cycle"])
def test_sparse_cpu_autocast_recurrence_matches_stored_precision_oracle(dtype, topology):
    cell = rd.ESNCell(12, feedback_size=2, topology=topology, sparse=True, seed=7, dtype=dtype)
    source = torch.linspace(-0.4, 0.5, 24, dtype=dtype).reshape(2, 12).requires_grad_()
    expected = (cell.weight_hh.to_dense() @ source.T).T
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = cell._recurrent(source)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
    actual_gradient = torch.autograd.grad(actual.square().sum(), source)[0]
    reference_gradient = torch.autograd.grad(expected.square().sum(), source)[0]
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("vector_leak", [False, True])
def test_direct_sparse_cell_autocast_preserves_driver_and_state_gradients(bias, vector_leak):
    leak = torch.linspace(0.2, 0.8, 12) if vector_leak else 0.4
    cell = rd.ESNCell(
        12,
        feedback_size=2,
        input_size=1,
        sparse=True,
        bias=bias,
        leak_rate=leak,
        seed=7,
        dtype=torch.float32,
    )
    feedback = torch.tensor([[0.2, -0.1]], dtype=torch.bfloat16, requires_grad=True)
    driver = torch.tensor([[0.3]], dtype=torch.bfloat16, requires_grad=True)
    state = torch.linspace(-0.2, 0.3, 12, dtype=torch.bfloat16).reshape(1, 12)
    state.requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output, final = cell([feedback, driver], state)
    assert final is output and output.dtype == torch.float32
    output.square().sum().backward()
    for value in (feedback, driver, state):
        assert value.grad is not None and bool(torch.isfinite(value.grad).all())
        assert bool(value.grad.abs().sum() > 0)


class _DeviceOnly(TorchDispatchMode):
    """Reject scalar synchronization and cross-device tensor work in the hot path."""

    def __init__(self, device):
        super().__init__()
        self.device = device

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            raise AssertionError("The pipeline hot path synchronizes a tensor scalar")
        for value in tree_leaves((args, kwargs)):
            if isinstance(value, torch.Tensor):
                assert value.device.type == self.device
        result = func(*args, **(kwargs or {}))
        for value in tree_leaves(result):
            if isinstance(value, torch.Tensor):
                assert value.device.type == self.device
        return result


@pytest.mark.parametrize("kind", ["vanilla", "ott", "custom"])
@pytest.mark.parametrize(
    "target",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
            ],
        ),
    ],
)
def test_pipeline_forward_backward_stays_on_device_without_host_scalar_reads(kind, target):
    reference = _Pipeline(_model(kind)).float().eval()
    pipeline = copy.deepcopy(reference).to(target)
    source = _training()[-7:].float().unsqueeze(0).repeat(2, 1, 1)
    source = source.to(target).requires_grad_()
    pipeline.reservoir.reset_state()
    with _DeviceOnly(target):
        actual = pipeline(source)
        actual.square().mean().backward()
    cpu_source = source.detach().cpu().requires_grad_()
    reference.reservoir.reset_state()
    expected = reference(cpu_source)
    expected.square().mean().backward()
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(source.grad.cpu(), cpu_source.grad, rtol=5e-4, atol=2e-5)
    for part in ("encoder", "head"):
        observed = getattr(pipeline, part).weight.grad
        oracle = getattr(reference, part).weight.grad
        torch.testing.assert_close(observed.cpu(), oracle, rtol=5e-4, atol=2e-5)
