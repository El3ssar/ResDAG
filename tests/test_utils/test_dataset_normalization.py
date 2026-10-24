"""Raw generator defaults keep future samples out of observed-prefix values."""

import inspect

import numpy as np
import pytest
import torch

import resdag as rd

GENERATORS = [
    (rd.lorenz, {}),
    (rd.rossler, {}),
    (rd.henon, {}),
    (rd.mackey_glass, {}),
    (rd.narma, {}),
    (rd.sine, {}),
    (rd.kuramoto_sivashinsky, {"n_grid": 16}),
]
CHANGED_DEFAULTS = (rd.lorenz, rd.rossler, rd.henon, rd.mackey_glass)


@pytest.mark.parametrize(
    "generator, options", GENERATORS, ids=lambda value: getattr(value, "__name__", "")
)
def test_every_generator_defaults_to_explicit_raw_values(generator, options):
    assert inspect.signature(generator).parameters["normalize"].default is None
    settings = {"seed": 7, "discard": 12, "dtype": torch.float64, **options}
    observed = generator(64, **settings)
    explicit_raw = generator(64, normalize=None, **settings)
    assert torch.equal(observed, explicit_raw)


@pytest.mark.parametrize("generator", CHANGED_DEFAULTS, ids=lambda value: value.__name__)
def test_raw_chaotic_prefix_does_not_depend_on_the_requested_future_length(generator):
    # Sine defines periods over the complete requested window, so extending its
    # length intentionally changes its sampling frequency rather than its scale.
    settings = {"seed": 7, "discard": 12, "dtype": torch.float64}
    observed = generator(64, **settings)
    extended = generator(96, **settings)
    assert torch.equal(observed, extended[:, :64])


@pytest.mark.parametrize("generator", CHANGED_DEFAULTS, ids=lambda value: value.__name__)
def test_explicit_standard_retains_the_original_population_normalization(generator):
    settings = {"seed": 7, "discard": 12, "dtype": torch.float64}
    raw = generator(96, normalize=None, **settings).numpy()[0]
    actual = generator(96, normalize="standard", **settings)
    scale = raw.std(axis=0, keepdims=True)
    scale = np.where(scale == 0.0, 1.0, scale)
    expected = torch.from_numpy((raw - raw.mean(axis=0, keepdims=True)) / scale).unsqueeze(0)
    assert torch.equal(actual, expected)
    # Explicit whole-window normalization still changes when future rows are added.
    extended = generator(160, normalize="standard", **settings)
    assert not torch.equal(actual, extended[:, :96])


def test_raw_generated_split_statistics_exclude_changed_held_out_observations():
    raw = rd.lorenz(97, seed=7, dtype=torch.float64)
    changed_future = raw.clone()
    changed_future[:, 65:] = changed_future[:, 65:] * 1000 + 500
    settings = {
        "warmup_steps": 8,
        "train_steps": 56,
        "val_steps": 32,
        "normalize": True,
        "norm_method": "standard",
        "return_stats": True,
    }
    original = rd.data.prepare_esn_data(raw, **settings)
    changed = rd.data.prepare_esn_data(changed_future, **settings)
    for expected, actual in zip(original[:4], changed[:4]):
        assert torch.equal(actual, expected)
    assert not torch.equal(original[4], changed[4])
    assert original[5].keys() == changed[5].keys() == {"mean", "std"}
    for name in original[5]:
        assert torch.equal(original[5][name], changed[5][name])
    expected_training = raw[:, 8:64]
    assert torch.equal(original[5]["mean"], expected_training.mean((0, 1), keepdim=True))
    assert torch.equal(original[5]["std"], expected_training.std((0, 1), keepdim=True))
