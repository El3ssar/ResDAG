"""Direct contract tests for the shared ``_esn_builder`` helper.

These exercise the orthogonal building blocks of :func:`resdag.models._builder._esn_builder`
(augment node, input concatenation, optional readout) that the five premade
factories compose, including branches the public factories do not reach
(``concat_input=False`` and the invalid-``output_size`` guard).
"""

import pytest
import torch

from resdag.layers import Concatenate, Power
from resdag.layers.readouts import ReadoutLayer
from resdag.models._builder import _esn_builder


def test_readout_path_with_concat() -> None:
    """A readout with input concatenation produces (batch, time, output_size)."""
    model = _esn_builder(reservoir_size=40, feedback_size=2, output_size=3)

    x = torch.randn(4, 15, 2)
    out = model(x)

    assert out.shape == (4, 15, 3)
    assert any(isinstance(m, ReadoutLayer) for m in model.modules())
    assert any(isinstance(m, Concatenate) for m in model.modules())


def test_no_readout_returns_reservoir_states() -> None:
    """With ``output_size=None`` the model stops at the reservoir output."""
    model = _esn_builder(reservoir_size=40, feedback_size=2, output_size=None)

    x = torch.randn(4, 15, 2)
    out = model(x)

    assert out.shape == (4, 15, 40)
    assert not any(isinstance(m, ReadoutLayer) for m in model.modules())
    assert not any(isinstance(m, Concatenate) for m in model.modules())


def test_concat_input_false_sizes_readout_to_reservoir_only() -> None:
    """Disabling concat skips the Concatenate node and keeps the output shape."""
    model = _esn_builder(
        reservoir_size=40,
        feedback_size=2,
        output_size=3,
        concat_input=False,
    )

    x = torch.randn(4, 15, 2)
    out = model(x)

    assert out.shape == (4, 15, 3)
    assert not any(isinstance(m, Concatenate) for m in model.modules())


def test_augment_factory_is_applied() -> None:
    """The ``augment`` factory is instantiated and inserted into the graph."""
    model = _esn_builder(
        reservoir_size=40,
        feedback_size=2,
        output_size=3,
        augment=lambda: Power(exponent=2.0),
    )

    assert any(isinstance(m, Power) for m in model.modules())


def test_invalid_output_size_raises() -> None:
    """A non-positive ``output_size`` with a readout requested is rejected."""
    with pytest.raises(ValueError, match="output_size must be a positive integer"):
        _esn_builder(reservoir_size=40, feedback_size=2, output_size=0)
