import pickle

import pytest
import torch
from torch import nn

import resdag as rd
from resdag.artifacts import ArtifactError, load, save_model


@pytest.mark.parametrize("trusted", [False, True])
def test_hooks_compiled_cache_and_nonleaf_state_are_excluded(tmp_path, trusted):
    model = rd.ESNLayer(5, feedback_size=2, trainable=True, seed=4)
    model.detach_state_between_calls = False
    model(torch.randn(1, 4, 2, requires_grad=True))
    assert model.state.grad_fn is not None
    events = []
    handle = model.register_forward_hook(lambda *args: events.append("called"))
    model._compiled_call_impl = None
    before = model.state
    modes = {name: child.training for name, child in model.named_modules()}
    save_model(model, tmp_path / "model.pt", trusted=trusted)
    assert events == []
    assert model.state is before
    assert model.state.grad_fn is not None
    restored = load(tmp_path / "model.pt", trusted=trusted)
    assert events == []
    assert restored.state.grad_fn is None
    assert not restored._forward_hooks
    assert {name: child.training for name, child in restored.named_modules()} == modes
    restored(torch.ones(1, 2, 2))
    assert events == []
    handle.remove()


@pytest.mark.parametrize("trusted", [False, True])
def test_loaded_parameters_remain_trainable_inside_inference_mode(tmp_path, trusted):
    model = nn.Linear(3, 2).double()
    save_model(model, tmp_path / "model.pt", trusted=trusted)
    with torch.inference_mode():
        restored = load(tmp_path / "model.pt", trusted=trusted)
    x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    restored(x).sum().backward()
    assert x.grad is not None
    assert restored.weight.grad is not None
    assert not restored.weight.is_inference()


def test_outer_pickle_never_executes_even_with_explicit_trust(tmp_path):
    marker = tmp_path / "executed"

    class Payload:
        def __reduce__(self):
            return (eval, (f"open({str(marker)!r},'w').write('bad')",))

    torch.save(
        {"format": "resdag.model", "version": 1, "trusted": True, "danger": Payload()},
        tmp_path / "outer.pt",
    )
    with pytest.raises(ArtifactError, match="Restricted loading"):
        load(tmp_path / "outer.pt", trusted=True)
    assert not marker.exists()


def test_embedded_python_waits_until_format_version_and_trust_validation(tmp_path):
    marker = tmp_path / "executed"

    class Payload:
        def __reduce__(self):
            return (eval, (f"open({str(marker)!r},'w').write('bad')",))

    for format_, version in [("foreign", 1), ("resdag.model", 999)]:
        torch.save(
            {
                "format": format_,
                "version": version,
                "trusted": True,
                "python": pickle.dumps(Payload()),
                "tensors": [],
            },
            tmp_path / "bad.pt",
        )
        with pytest.raises(ArtifactError):
            load(tmp_path / "bad.pt", trusted=True)
        assert not marker.exists()


def test_preserves_global_rng_on_load_and_save(tmp_path):
    model = rd.ESN(reservoir_size=8, input_size=2, seed=3)
    torch.manual_seed(92)
    rng = torch.get_rng_state().clone()
    for trusted in (False, True):
        save_model(model, tmp_path / "model.pt", trusted=trusted)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        load(tmp_path / "model.pt", trusted=trusted)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_scalar_functional_operators_have_explicit_codecs(tmp_path):
    x = rd.Input((3, 2))
    output = -(x * torch.tensor([2.0, 3.0]) + 1) / 2
    model = rd.ESNModel(x, output)
    save_model(model, tmp_path / "scalar.pt")
    restored = load(tmp_path / "scalar.pt")
    data = torch.randn(2, 6, 2)
    torch.testing.assert_close(restored(data), model(data), rtol=0, atol=0)


def test_duplicate_sequential_children_keep_invocation_count(tmp_path):
    layer = nn.Linear(2, 2)
    model = nn.Sequential(layer, layer)
    save_model(model, tmp_path / "sequence.pt")
    restored = load(tmp_path / "sequence.pt")
    assert len(restored) == 2
    assert restored[0] is restored[1]
    xx = torch.randn(2, 2)
    torch.testing.assert_close(model(xx), restored(xx), rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires native Metal")
@pytest.mark.parametrize("trusted", [False, True])
def test_native_mps_and_cpu_map_location_roundtrip(tmp_path, trusted):
    series = torch.sin(torch.arange(60.0) / 5).unsqueeze(-1)
    model = rd.ESN(reservoir_size=8, washout=4, seed=4).fit(series)
    expected = model.forecast(horizon=4)
    save_model(model, tmp_path / "mps.pt", trusted=trusted)
    native = load(tmp_path / "mps.pt", trusted=trusted, map_location="mps")
    actual = native.forecast(horizon=4)
    assert actual.device.type == "mps"
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
    save_model(native, tmp_path / "back.pt", trusted=trusted)
    cpu = load(tmp_path / "back.pt", trusted=trusted, map_location="cpu")
    assert cpu.forecast(horizon=4).device.type == "cpu"
    torch.testing.assert_close(cpu.forecast(horizon=4), expected, rtol=1e-4, atol=1e-5)


def test_feature_extractor_pytorch_pipeline_keeps_live_trainability(tmp_path):
    extractor = rd.ReservoirFeatureExtractor(
        reservoir_size=5, feedback_size=2, trainable=True, seed=3
    )
    pipeline = nn.Sequential(extractor, nn.Linear(5, 1))
    x = torch.randn(2, 4, 2)
    pipeline(x)
    save_model(pipeline, tmp_path / "pipeline.pt")
    restored = load(tmp_path / "pipeline.pt")
    assert restored[0].reservoirs[0].cell.trainable
    torch.testing.assert_close(pipeline(x), restored(x), rtol=0, atol=0)
    restored(x).sum().backward()
    assert restored[0].reservoirs[0].cell.weight_hh.grad is not None


@pytest.mark.parametrize("trusted", [False, True])
def test_transposed_parameter_views_keep_shared_storage(tmp_path, trusted):
    left = nn.Linear(2, 2, bias=False).double()
    right = nn.Linear(2, 2, bias=False).double()
    right.weight = nn.Parameter(left.weight.T)
    model = nn.Sequential(left, right)
    save_model(model, tmp_path / "views.pt", trusted=trusted)
    restored = load(tmp_path / "views.pt", trusted=trusted)
    assert restored[0].weight is not restored[1].weight
    assert (
        restored[0].weight.untyped_storage().data_ptr()
        == restored[1].weight.untyped_storage().data_ptr()
    )
    before = restored[1].weight.detach().clone()
    with torch.no_grad():
        restored[0].weight[0, 1].add_(10)
    assert restored[1].weight[1, 0] - before[1, 0] == 10


@pytest.mark.parametrize("trusted", [False, True])
def test_persistent_buffer_identity_and_requires_grad(tmp_path, trusted):
    from resdag.artifacts import register_component

    class Buffers(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("left", torch.ones(2, requires_grad=True))
            self.register_buffer("right", self.left)

        def forward(self, x):
            return x * self.left

    register_component(
        f"test.integrity.Buffers.{trusted}",
        Buffers,
        encode=lambda m: {},
        decode=lambda c: Buffers(),
    )
    model = Buffers()
    save_model(model, tmp_path / "buffers.pt", trusted=trusted)
    restored = load(tmp_path / "buffers.pt", trusted=trusted)
    assert restored.left is restored.right
    assert restored.left.requires_grad
    restored(torch.ones(2)).sum().backward()
    assert restored.left.grad is not None


@pytest.mark.parametrize("trusted", [False, True])
def test_customized_outer_forward_is_rejected_instead_of_dropped(tmp_path, trusted):
    from types import MethodType

    x = rd.Input((3, 2))
    model = rd.ESNModel(x, nn.Identity()(x))
    original = model.forward
    model.forward = MethodType(lambda self, value: original(value) + 7, model)
    with pytest.raises(ArtifactError, match="customized outer ESNModel.forward"):
        save_model(model, tmp_path / "custom-forward.pt", trusted=trusted)
