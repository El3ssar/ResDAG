"""Generator recipes retain their streams or visibly restart after remapping."""

import pytest
import torch
from torch import nn

from resdag.artifacts import load, register_component, save_model


class GeneratorLayer(nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x):
        return x + torch.rand(x.shape, device=x.device, generator=self.generator)


@pytest.fixture
def registered_generator_layer(isolated_component_registries):
    register_component(
        "test.generator_recipes.Layer",
        GeneratorLayer,
        encode=lambda m: {"generator": m.generator},
        decode=lambda c: GeneratorLayer(**c),
    )


@pytest.mark.parametrize("trusted", [False, True])
def test_cpu_generator_current_stream_is_preserved(tmp_path, trusted, registered_generator_layer):
    generator = torch.Generator(device="cpu").manual_seed(73)
    torch.rand(17, generator=generator)
    model = GeneratorLayer(generator)
    save_model(model, tmp_path / "generator.pt", trusted=trusted)
    restored = load(tmp_path / "generator.pt", trusted=trusted)
    assert restored.generator.device.type == "cpu"
    assert restored.generator.initial_seed() == generator.initial_seed()
    assert torch.equal(restored.generator.get_state(), generator.get_state())
    torch.testing.assert_close(model(torch.zeros(8)), restored(torch.zeros(8)), rtol=0, atol=0)


@pytest.mark.parametrize("trusted", [False, True])
def test_synthetic_cuda_to_cpu_metadata_restarts_seed_with_warning(
    tmp_path, trusted, registered_generator_layer
):
    generator = torch.Generator().manual_seed(73)
    torch.rand(17, generator=generator)
    save_model(GeneratorLayer(generator), tmp_path / "generator.pt", trusted=trusted)
    payload = torch.load(tmp_path / "generator.pt", weights_only=True)
    if trusted:
        descriptors = payload["generators"]
    else:
        descriptors = []

        def visit(value):
            if isinstance(value, dict):
                if value.get("tag") == "generator":
                    descriptors.append(value)
                for child in value.values():
                    visit(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    visit(child)

        visit(payload)
    assert descriptors
    for descriptor in descriptors:
        descriptor["device"] = "cuda:0"
        assert descriptor["reference"].device.type == "cpu"
    torch.save(payload, tmp_path / "remapped.pt")
    with pytest.warns(UserWarning, match="seeded stream restarts"):
        restored = load(tmp_path / "remapped.pt", trusted=trusted)
    fresh = torch.Generator().manual_seed(73)
    assert torch.equal(restored.generator.get_state(), fresh.get_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA hardware")
@pytest.mark.parametrize("trusted", [False, True])
def test_cuda_generator_same_backend_preserves_current_stream(
    tmp_path, trusted, registered_generator_layer
):
    generator = torch.Generator(device="cuda").manual_seed(73)
    torch.rand(17, device="cuda", generator=generator)
    model = GeneratorLayer(generator)
    save_model(model, tmp_path / "cuda.pt", trusted=trusted)
    restored = load(tmp_path / "cuda.pt", trusted=trusted, map_location="cuda")
    assert restored.generator.device.type == "cuda"
    assert torch.equal(restored.generator.get_state(), generator.get_state())
    x = torch.zeros(8, device="cuda")
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)


@pytest.mark.parametrize("trusted", [False, True])
def test_generator_object_identity_preserves_shared_stream(
    tmp_path, trusted, isolated_component_registries
):
    class Shared(nn.Module):
        def __init__(self, left, right):
            super().__init__()
            self.left, self.right = left, right

        def forward(self, x):
            return (
                x
                + torch.rand(x.shape, generator=self.left)
                + torch.rand(x.shape, generator=self.right)
            )

    register_component(
        f"test.generator_recipes.Shared.{trusted}",
        Shared,
        encode=lambda m: {"left": m.left, "right": m.right},
        decode=lambda c: Shared(**c),
    )
    generator = torch.Generator().manual_seed(73)
    model = Shared(generator, generator)
    save_model(model, tmp_path / "shared.pt", trusted=trusted)
    restored = load(tmp_path / "shared.pt", trusted=trusted)
    assert restored.left is restored.right
    x = torch.zeros(7)
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
