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


@pytest.mark.parametrize(
    "readout", ["ridge", "cholesky", "svd", "pinv", "incremental", "rls", "fista"]
)
def test_builtin_readouts_fit_through_factory(readout: str) -> None:
    from resdag.training import ESNTrainer

    torch.manual_seed(5)
    model = _esn_builder(12, 2, 2, readout=readout, readout_alpha=0.1)
    series = torch.randn(1, 60, 2)
    ESNTrainer(model).fit((series[:, :10],), (series[:, 10:-1],), {"output": series[:, 11:]})
    result = model(series[:, -1:])
    assert result.shape == (1, 1, 2)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize(
    "factory_name",
    [
        "classic_esn",
        "ott_esn",
        "power_augmented",
        "deep_esn",
        "gated_esn",
        "eusn",
        "es2n",
        "resesn",
        "lsm",
        "hybrid_esn",
        "parallel_esn",
    ],
)
def test_readout_options_are_separate_from_reservoir_options(factory_name: str) -> None:
    import resdag.models as models

    factory = getattr(models, factory_name)
    kwargs: dict = {"reservoir_size": 12, "feedback_size": 2, "output_size": 2}
    if factory_name == "deep_esn":
        kwargs = {"reservoir_sizes": [12, 8], "feedback_size": 2, "output_size": 2}
    elif factory_name == "hybrid_esn":
        kwargs["knowledge_model"] = lambda x: x
    elif factory_name == "parallel_esn":
        kwargs = {"input_dim": 4, "core_width": 2, "halo": 1, "reservoir_size": 12}
    model = factory(**kwargs, readout="svd", readout_kwargs={"trainable": True, "rcond": 1e-8})
    readout = next(m for m in model.modules() if isinstance(m, ReadoutLayer))
    assert readout.trainable
    assert readout.rcond == 1e-8
    assert all(p.requires_grad for p in readout.parameters())


def test_readout_options_cannot_change_architecture_widths() -> None:
    with pytest.raises(ValueError, match="architecture dimensions"):
        _esn_builder(12, 2, 2, readout_kwargs={"in_features": 7})


def test_rls_delta_option_overrides_the_factory_penalty_default() -> None:
    from resdag.layers.readouts import RLSReadout

    model = _esn_builder(12, 2, 2, readout="rls", readout_kwargs={"delta": 0.2})
    readout = next(m for m in model.modules() if isinstance(m, RLSReadout))
    assert readout.delta == 0.2
    with pytest.raises(ValueError, match="not both"):
        _esn_builder(12, 2, 2, readout="rls", readout_kwargs={"delta": 0.2, "alpha": 0.1})
