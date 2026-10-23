"""Sequence view preparation preserves the original loop and its gradients."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional

from resdag.layers.cells.base_cell import ReservoirCell
from resdag.layers.cells.esn_cell import ESNCell
from resdag.layers.cells.eusn_cell import EuSNCell
from resdag.layers.cells.gated_cell import GatedReservoirCell
from resdag.layers.cells.lif_cell import LIFCell
from resdag.layers.reservoirs.base_reservoir import BaseReservoirLayer


class _NonprojectedCell(ReservoirCell):
    """A custom fallback cell with different readable-output and state widths."""

    def __init__(self, trainable):
        super().__init__()
        self.weight_feedback = nn.Parameter(
            torch.arange(9, dtype=torch.float64).reshape(3, 3) / 30 - 0.1,
            requires_grad=trainable,
        )
        self.weight_driver = nn.Parameter(
            torch.arange(6, dtype=torch.float64).reshape(3, 2) / 20 - 0.1,
            requires_grad=trainable,
        )
        self.weight_state = nn.Parameter(
            torch.eye(3, dtype=torch.float64) * 0.2, requires_grad=trainable
        )

    @property
    def state_size(self):
        return 3

    @property
    def output_size(self):
        return 2

    def init_state(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.state_size, device=device, dtype=dtype)

    def forward(self, inputs, state):
        updated = torch.tanh(
            functional.linear(state, self.weight_state)
            + functional.linear(inputs[0], self.weight_feedback)
            + functional.linear(inputs[1], self.weight_driver)
        )
        return updated[:, :2], updated


_FAMILIES = ("esn", "gated", "lif", "continuous", "nonprojected")


def _make_layer(family, trainable):
    options = {
        "reservoir_size": 5,
        "feedback_size": 3,
        "input_size": 2,
        "trainable": trainable,
        "seed": 217,
        "dtype": torch.float64,
    }
    if family == "esn":
        cell = ESNCell(**options, leak_rate=0.4)
    elif family == "gated":
        cell = GatedReservoirCell(
            **options,
            activation="tanh",
            gate_feedback_initializer=("uniform", {"input_scaling": 0.2}),
        )
    elif family == "lif":
        cell = LIFCell(
            **options,
            synaptic_current=True,
            tau_mem=5.0,
            tau_syn=3.0,
            dt=0.5,
            v_threshold=0.15,
            bias_scaling=0.1,
        )
    elif family == "continuous":
        cell = EuSNCell(**options, epsilon=0.04, gamma=0.2, substeps=2, solver="rk4")
    else:
        assert family == "nonprojected"
        cell = _NonprojectedCell(trainable)
    return BaseReservoirLayer(cell).eval()


def _original_loop(cell, inputs, state):
    """Independent pre-optimization loop: retain three-axis timestep slicing."""
    projected = cell.project_inputs(inputs)
    outputs = []
    if projected is not None:
        for t in range(inputs[0].shape[1]):
            output, state = cell.step(projected[:, t, :], state)
            outputs.append(output)
    else:
        for t in range(inputs[0].shape[1]):
            output, state = cell([stream[:, t, :] for stream in inputs], state)
            outputs.append(output)
    if not outputs:
        return inputs[0].new_empty(inputs[0].shape[0], 0, cell.output_size), state
    return torch.stack(outputs, dim=1), state


def _strided_input(values):
    """Place values in an independent leaf view with gaps and transposed axes."""
    batch, time, features = values.shape
    storage = values.new_empty(batch, features, time * 2)
    view = storage.transpose(1, 2)[:, ::2, :]
    view.copy_(values)
    return view.detach().requires_grad_()


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("trainable", [False, True])
def test_prepared_sequence_views_match_original_values_states_and_gradients(
    family, batch, trainable
):
    layer = _make_layer(family, trainable)
    reference = copy.deepcopy(layer)
    generator = torch.Generator().manual_seed(81)
    values = [
        torch.randn(batch, 7, width, dtype=torch.float64, generator=generator) * 0.2
        for width in (3, 2)
    ]
    inputs = [_strided_input(value) for value in values]
    reference_inputs = [_strided_input(value) for value in values]
    assert all(not value.is_contiguous() for value in (*inputs, *reference_inputs))
    assert [value.stride() for value in inputs] == [value.stride() for value in reference_inputs]
    initial = layer.cell.init_state(batch, torch.device("cpu"), torch.float64)
    initial.add_(0.05).requires_grad_()
    reference_initial = initial.detach().clone().requires_grad_()
    saved_initial = initial.detach().clone()

    outputs, final = layer.forward_stateless(inputs, initial)
    expected, expected_final = _original_loop(reference.cell, reference_inputs, reference_initial)

    torch.testing.assert_close(outputs, expected, rtol=0, atol=0)
    torch.testing.assert_close(final, expected_final, rtol=0, atol=0)
    torch.testing.assert_close(initial, saved_initial, rtol=0, atol=0)
    assert layer.state is None and reference.state is None
    parameters = [(name, value) for name, value in layer.named_parameters() if value.requires_grad]
    reference_parameters = [
        (name, value) for name, value in reference.named_parameters() if value.requires_grad
    ]
    assert [name for name, _ in parameters] == [name for name, _ in reference_parameters]
    loss = outputs.square().sum() + 0.3 * final.sum()
    expected_loss = expected.square().sum() + 0.3 * expected_final.sum()
    gradients = torch.autograd.grad(loss, (*inputs, initial, *(p for _, p in parameters)))
    expected_gradients = torch.autograd.grad(
        expected_loss,
        (*reference_inputs, reference_initial, *(p for _, p in reference_parameters)),
    )
    for gradient, expected_gradient in zip(gradients, expected_gradients, strict=True):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, expected_gradient, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("family", _FAMILIES)
def test_empty_sequence_keeps_initial_state_and_its_gradient(family):
    layer = _make_layer(family, trainable=True)
    inputs = [torch.empty(3, 0, width, dtype=torch.float64) for width in (3, 2)]
    initial = layer.cell.init_state(3, torch.device("cpu"), torch.float64).requires_grad_()

    outputs, final = layer.forward_stateless(inputs, initial)
    expected, expected_final = _original_loop(layer.cell, inputs, initial)

    assert outputs.shape == expected.shape == (3, 0, layer.cell.output_size)
    assert outputs.dtype == expected.dtype == torch.float64
    assert final is expected_final is initial
    assert layer.state is None
    gradient = torch.autograd.grad(final.sum(), initial)[0]
    torch.testing.assert_close(gradient, torch.ones_like(initial), rtol=0, atol=0)
