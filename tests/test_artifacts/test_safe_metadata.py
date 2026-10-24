"""Portable metadata validation includes serialized attributes before any write."""

from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from torch import nn

from resdag.artifacts import ArtifactError, load, save_model


def _marker(path):
    Path(path).write_text("executed")
    return None


class _PythonPayload:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return _marker, (str(self.path),)


def _carrier(kind):
    if kind == "tensor":
        return torch.tensor([1.0, 2.0])
    if kind == "parameter":
        return nn.Parameter(torch.tensor([1.0, 2.0]))
    return OrderedDict(answer=42)


@pytest.mark.parametrize("kind", ["tensor", "parameter", "ordered_dict"])
def test_python_attributes_are_rejected_without_replacing_existing_artifact(kind, tmp_path):
    marker = tmp_path / "executed"
    value = _carrier(kind)
    value.hidden = _PythonPayload(marker)
    destination = tmp_path / "keep.pt"
    destination.write_bytes(b"existing artifact")
    with pytest.raises(ArtifactError, match="unsupported Python type"):
        save_model(nn.Identity(), destination, metadata={"sample": value})
    assert destination.read_bytes() == b"existing artifact"
    assert not marker.exists()


@pytest.mark.parametrize("kind", ["tensor", "parameter", "ordered_dict", "list"])
def test_cycles_in_containers_or_serialized_attributes_fail_before_creating_paths(kind, tmp_path):
    if kind == "list":
        value = []
        value.append(value)
    else:
        value = _carrier(kind)
        value.recursive = value
    destination = tmp_path / "not-created" / "model.pt"
    with pytest.raises(ArtifactError, match="acyclic"):
        save_model(nn.Identity(), destination, metadata={"sample": value})
    assert not destination.parent.exists()


@pytest.mark.parametrize("kind", ["tensor", "parameter", "ordered_dict"])
def test_safe_serialized_attributes_survive_restricted_roundtrip(kind, tmp_path):
    value = _carrier(kind)
    value.description = {"epoch": 2, "coordinates": (1, 2.5), "flags": [True, None]}
    value._metadata = OrderedDict(root={"version": 1})
    destination = tmp_path / "safe.pt"
    save_model(nn.Identity(), destination, metadata={"sample": value})
    restored, metadata = load(destination, return_metadata=True)
    assert type(restored) is nn.Identity
    actual = metadata["sample"]
    assert type(actual) is type(value)
    assert actual.description == value.description
    assert actual._metadata == value._metadata
    if isinstance(value, torch.Tensor):
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
    else:
        assert actual == value


@pytest.mark.parametrize("unsupported", [float("nan"), float("inf"), 1 + 2j, {1, 2}])
def test_serialized_attributes_keep_the_existing_portable_value_policy(unsupported, tmp_path):
    sample = torch.ones(1)
    sample.metadata = {"unsupported": unsupported}
    destination = tmp_path / "keep.pt"
    destination.write_bytes(b"existing artifact")
    with pytest.raises(ArtifactError, match="unsupported value"):
        save_model(nn.Identity(), destination, metadata={"sample": sample})
    assert destination.read_bytes() == b"existing artifact"


def test_invalid_primitive_metadata_still_fails_before_any_file_creation(tmp_path):
    destination = tmp_path / "not-created" / "model.pt"
    with pytest.raises(ArtifactError, match="unsupported value"):
        save_model(nn.Identity(), destination, metadata={"loss": float("inf")})
    assert not destination.parent.exists()
