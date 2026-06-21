"""Contracts for :class:`resdag.core.ReservoirFeatureExtractor`.

Pins down the nn.Sequential-friendly adapter: forward/backward through a
sequential pipeline, the frozen-by-default gradient gate, the
freeze/unfreeze and state-reset (epoch-hook) API, the ``from_model``
parameter-sharing constructor, stacking, driving inputs, and the Path-B
reference recipe (frozen extractor + Adam-trained head) reducing the loss.
"""

import pytest
import torch
import torch.nn as nn

from resdag import ESNLayer, ESNModel, ReservoirFeatureExtractor, reservoir_input
from resdag.layers import NGReservoir


class TestForwardAndBackward:
    """Sequential composition, shapes, and gradient flow."""

    def test_sequential_forward_shape(self, seeded: None) -> None:
        """nn.Sequential(extractor, Linear) maps (B, T, F) -> (B, T, out)."""
        model = nn.Sequential(
            ReservoirFeatureExtractor(reservoir_size=64, feedback_size=3),
            nn.Linear(64, 2),
        )
        x = torch.randn(4, 30, 3)
        out = model(x)
        assert out.shape == (4, 30, 2)

    def test_gradients_reach_head_not_frozen_reservoir(self, seeded: None) -> None:
        """Backward reaches the head; a frozen reservoir gets no gradients."""
        extractor = ReservoirFeatureExtractor(reservoir_size=64, feedback_size=3)
        head = nn.Linear(64, 3)
        model = nn.Sequential(extractor, head)

        model(torch.randn(2, 20, 3)).pow(2).mean().backward()

        assert head.weight.grad is not None
        assert all(p.grad is None for p in extractor.parameters())

    def test_gradients_reach_reservoir_when_unfrozen(self, seeded: None) -> None:
        """trainable=True lets gradients flow into the reservoir weights."""
        extractor = ReservoirFeatureExtractor(64, feedback_size=3, trainable=True)
        model = nn.Sequential(extractor, nn.Linear(64, 3))

        model(torch.randn(2, 20, 3)).pow(2).mean().backward()

        grads = [p.grad for p in extractor.parameters() if p.requires_grad]
        assert grads and any(g is not None for g in grads)

    def test_output_size_property(self, seeded: None) -> None:
        """output_size reports the (last) reservoir width."""
        assert ReservoirFeatureExtractor(128, feedback_size=3).output_size == 128


class TestFreezeUnfreeze:
    """The freeze/unfreeze toggle and frozen-by-default contract."""

    def test_frozen_by_default(self, seeded: None) -> None:
        """A fresh extractor has no trainable reservoir parameters."""
        extractor = ReservoirFeatureExtractor(32, feedback_size=3)
        assert extractor.is_frozen
        assert all(not p.requires_grad for p in extractor.parameters())

    def test_trainable_passthrough(self, seeded: None) -> None:
        """trainable=True yields a non-frozen extractor."""
        extractor = ReservoirFeatureExtractor(32, feedback_size=3, trainable=True)
        assert not extractor.is_frozen
        assert all(p.requires_grad for p in extractor.parameters())

    def test_freeze_then_unfreeze_round_trip(self, seeded: None) -> None:
        """unfreeze() then freeze() toggles requires_grad and chains."""
        extractor = ReservoirFeatureExtractor(32, feedback_size=3)

        returned = extractor.unfreeze()
        assert returned is extractor  # chainable
        assert not extractor.is_frozen
        assert all(p.requires_grad for p in extractor.parameters())

        extractor.freeze()
        assert extractor.is_frozen
        assert all(not p.requires_grad for p in extractor.parameters())


class TestStateManagement:
    """reset_state / on_epoch_start hooks."""

    def test_reset_state_clears_states(self, seeded: None) -> None:
        """reset_state() returns reservoirs to the lazy (None) state."""
        extractor = ReservoirFeatureExtractor(32, feedback_size=3)
        extractor(torch.randn(1, 10, 3))
        assert all(r.state is not None for r in extractor.reservoirs)

        extractor.reset_state()
        assert all(r.state is None for r in extractor.reservoirs)

    def test_reset_state_with_batch_size(self, seeded: None) -> None:
        """reset_state(batch_size=B) materialises a zero state of width B."""
        extractor = ReservoirFeatureExtractor(32, feedback_size=3)
        extractor.reset_state(batch_size=5)
        for r in extractor.reservoirs:
            assert r.state is not None
            assert r.state.shape[0] == 5
            assert torch.count_nonzero(r.state) == 0

    def test_on_epoch_start_is_reset_alias(self, seeded: None) -> None:
        """The documented epoch hook re-zeroes the reservoir like reset_state."""
        extractor = ReservoirFeatureExtractor(32, feedback_size=3)
        extractor(torch.randn(1, 10, 3))

        extractor.on_epoch_start()
        assert all(r.state is None for r in extractor.reservoirs)

    def test_reset_state_makes_forward_reproducible(self, seeded: None) -> None:
        """Same input from a fresh state reproduces the same features."""
        extractor = ReservoirFeatureExtractor(32, feedback_size=3)
        x = torch.randn(1, 20, 3)

        extractor.reset_state()
        a = extractor(x)
        extractor.reset_state()
        b = extractor(x)
        assert torch.allclose(a, b, atol=1e-6)


class TestFromModel:
    """from_model reuses (shares) the source model's reservoir parameters."""

    def _make_esn(self) -> ESNModel:
        inp = reservoir_input(3)
        states = ESNLayer(48, feedback_size=3, spectral_radius=0.9)(inp)
        return ESNModel(inp, states)

    def test_shares_parameters_by_identity(self, seeded: None) -> None:
        """Wrapped reservoir parameters are the *same* objects (id identity)."""
        esn = self._make_esn()
        src = next(m for m in esn.modules() if hasattr(m, "weight_hh"))

        extractor = ReservoirFeatureExtractor.from_model(esn)

        assert extractor.reservoirs[0].weight_hh is src.weight_hh
        src_ids = {id(p) for p in src.parameters()}
        ext_ids = {id(p) for p in extractor.reservoirs[0].parameters()}
        assert ext_ids <= src_ids

    def test_shared_grads_propagate_to_source(self, seeded: None) -> None:
        """Unfrozen + shared: a backward writes grads on the source model too."""
        esn = self._make_esn()
        extractor = ReservoirFeatureExtractor.from_model(esn, trainable=True)
        src = next(m for m in esn.modules() if hasattr(m, "weight_hh"))

        head = nn.Linear(48, 3)
        out = head(extractor(torch.randn(1, 15, 3)))
        out.pow(2).mean().backward()

        assert src.weight_hh.grad is not None  # same tensor as extractor's

    def test_from_model_without_reservoir_raises(self) -> None:
        """A model with no reservoir layers cannot be wrapped."""
        with pytest.raises(ValueError, match="no reservoir layers"):
            ReservoirFeatureExtractor.from_model(nn.Linear(3, 3))


class TestStackingAndDrivingInputs:
    """Stacked reservoirs and optional driving input."""

    def test_stacked_chains_reservoirs(self, seeded: None) -> None:
        """A list of sizes chains reservoirs; output width is the last size."""
        extractor = ReservoirFeatureExtractor([16, 8], feedback_size=3)
        out = extractor(torch.randn(2, 12, 3))
        assert out.shape == (2, 12, 8)
        assert extractor.output_size == 8
        assert len(extractor.reservoirs) == 2

    def test_driving_input_forward(self, seeded: None) -> None:
        """input_size enables a second positional driving input."""
        extractor = ReservoirFeatureExtractor(32, feedback_size=3, input_size=2)
        out = extractor(torch.randn(1, 10, 3), torch.randn(1, 10, 2))
        assert out.shape == (1, 10, 32)

    def test_too_many_driving_inputs_raises(self, seeded: None) -> None:
        """At most one driving input is accepted."""
        extractor = ReservoirFeatureExtractor(32, feedback_size=3, input_size=2)
        with pytest.raises(ValueError, match="[Oo]nly one driving input"):
            extractor(torch.randn(1, 10, 3), torch.randn(1, 10, 2), torch.randn(1, 10, 2))

    def test_from_prebuilt_layers_shares_reference(self, seeded: None) -> None:
        """Wrapping a pre-built layer shares it by reference."""
        layer = ESNLayer(24, feedback_size=3)
        extractor = ReservoirFeatureExtractor(layers=layer)
        assert extractor.reservoirs[0] is layer

    def test_wraps_non_esn_reservoir_layer(self, seeded: None) -> None:
        """A non-ESN BaseReservoirLayer (NGReservoir) wraps and runs end to end."""
        layer = NGReservoir(input_dim=3, k=2, s=1, p=2)  # not an ESNLayer
        extractor = ReservoirFeatureExtractor(layers=layer)

        assert len(extractor.reservoirs) == 1  # single non-ESN layer accepted
        assert extractor.output_size == int(layer.cell.output_size)

        x = torch.randn(2, 12, 3)
        out = extractor(x)
        assert out.shape == (2, 12, extractor.output_size)


class TestConstructionErrors:
    """Argument validation."""

    def test_missing_sizes_raises(self) -> None:
        with pytest.raises(ValueError, match="reservoir_size"):
            ReservoirFeatureExtractor()

    def test_mixing_layers_and_sizes_raises(self) -> None:
        layer = ESNLayer(8, feedback_size=3)
        with pytest.raises(ValueError, match="not both"):
            ReservoirFeatureExtractor(reservoir_size=8, feedback_size=3, layers=layer)

    def test_empty_layers_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ReservoirFeatureExtractor(layers=[])


class TestReferenceRecipe:
    """The Path-B reference recipe: frozen extractor + Adam-trained head."""

    def test_frozen_extractor_adam_head_reduces_loss(self, seeded: None) -> None:
        """A frozen reservoir + Adam-trained head learns next-step prediction."""
        # A short, learnable sine task (batch, time, features).
        t = torch.linspace(0, 8 * torch.pi, 400)
        series = torch.stack([torch.sin(t), torch.cos(t), torch.sin(2 * t)], dim=-1)
        series = series.unsqueeze(0)
        warmup, train_in, train_tgt = series[:, :50], series[:, 50:-1], series[:, 51:]

        extractor = ReservoirFeatureExtractor(120, feedback_size=3, spectral_radius=0.9)
        head = nn.Sequential(nn.Linear(120, 32), nn.Tanh(), nn.Linear(32, 3))

        # Frozen features: compute once (the speed win of a frozen base).
        with torch.no_grad():
            extractor.on_epoch_start()
            feats = extractor(torch.cat([warmup, train_in], dim=1))[:, warmup.shape[1] :]

        optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
        criterion = nn.MSELoss()

        first_loss = criterion(head(feats), train_tgt).item()
        for _ in range(200):
            loss = criterion(head(feats), train_tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        last_loss = loss.item()

        # The head learns: the loss drops substantially, and the reservoir
        # stayed frozen throughout.
        assert last_loss < first_loss * 0.1
        assert extractor.is_frozen
