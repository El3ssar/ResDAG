"""Ordinary checkpoints never execute Python; full models require explicit trust."""

import copy
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

import resdag as rd


def _marker(path):
    Path(path).write_text("executed")
    return {}


class _PythonPayload:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return _marker, (str(self.path),)


def _model(family):
    if family == "graph":
        return rd.classic_esn(4, 1, 1, topology="simple_cycle", seed=7)
    if family == "parallel":
        return rd.parallel_esn(2, core_width=1, halo=1, reservoir_size=4, seed=7)
    return rd.CoupledEnsembleESNModel([rd.classic_esn(4, 1, 1, seed=7)])


@pytest.mark.parametrize("family", ["graph", "parallel", "ensemble"])
def test_ordinary_checkpoint_rejects_python_without_executing_or_mutating(family, tmp_path):
    model = _model(family)
    before = copy.deepcopy(model.state_dict())
    marker = tmp_path / "executed"
    path = tmp_path / "malformed.pt"
    torch.save({"state_dict": before, "metadata": _PythonPayload(marker)}, path)
    with pytest.raises(ValueError, match="Restricted checkpoint"):
        model.load(path)
    assert not marker.exists()
    for name, value in before.items():
        assert torch.equal(model.state_dict()[name], value)


@pytest.mark.parametrize("family", ["graph", "parallel", "ensemble"])
def test_full_model_requires_trust_before_any_file_read(family, tmp_path, monkeypatch):
    model = _model(family)

    def forbidden(*args, **kwargs):
        pytest.fail("An untrusted full-model file was opened")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="trusted=True"):
        type(model).load_full(tmp_path / "not-opened.pt")
    for value in (0, 1, "yes", None):
        with pytest.raises(TypeError, match="trusted must be a bool"):
            type(model).load_full(tmp_path / "not-opened.pt", trusted=value)


def test_unsupported_checkpoint_metadata_is_rejected_before_creating_a_file(tmp_path):
    path = tmp_path / "not-created" / "model.pt"
    with pytest.raises(TypeError, match="unsupported Python type"):
        _model("graph").save(path, custom=_PythonPayload(tmp_path / "executed"))
    assert not path.parent.exists()


def test_checkpoint_primitive_and_tensor_metadata_is_loadable(tmp_path):
    model = _model("graph")
    path = tmp_path / "safe.pt"
    sample = torch.ones(2)
    sample.description = "safe attributes are serialized too"
    model.save(
        path,
        note="training",
        epoch=2,
        loss=0.5,
        shape=(1, 2),
        sample=sample,
        choices={1, 2},
        frequency=1 + 2j,
    )
    payload = torch.load(path, weights_only=True)
    assert payload["metadata"]["epoch"] == 2
    model.load(path)
    assert payload["metadata"]["sample"].description == sample.description
    assert payload["metadata"]["frequency"] == 1 + 2j


@pytest.mark.parametrize("kind", ["tensor", "parameter", "ordered_dict"])
def test_serialized_python_attributes_cannot_bypass_validation(kind, tmp_path):
    value = {
        "tensor": torch.ones(1),
        "parameter": torch.nn.Parameter(torch.ones(1)),
        "ordered_dict": OrderedDict(a=1),
    }[kind]
    value.hidden = _PythonPayload(tmp_path / "executed")
    path = tmp_path / "not-created" / "model.pt"
    with pytest.raises(TypeError, match="unsupported Python type"):
        _model("graph").save(path, custom=value)
    assert not path.parent.exists()


def test_cyclic_metadata_fails_clearly_without_touching_existing_file(tmp_path):
    path = tmp_path / "keep.pt"
    path.write_bytes(b"keep")
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="acyclic"):
        _model("graph").save(path, custom=cyclic)
    assert path.read_bytes() == b"keep"


@pytest.mark.parametrize("method", ["save", "save_full"])
def test_failed_serialization_preserves_existing_checkpoint_and_removes_temporary(
    method, tmp_path, monkeypatch
):
    model = _model("graph")
    path = tmp_path / "keep.pt"
    path.write_bytes(b"keep")

    def broken(payload, destination):
        destination.write(b"partial")
        raise OSError("simulated serialization failure")

    monkeypatch.setattr(torch, "save", broken)
    with pytest.raises(OSError, match="simulated serialization failure"):
        getattr(model, method)(path)
    assert path.read_bytes() == b"keep"
    assert list(tmp_path.iterdir()) == [path]
