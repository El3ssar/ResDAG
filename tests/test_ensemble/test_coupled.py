"""Tests for CoupledEnsembleESNModel and its aggregators."""

import tempfile
from pathlib import Path

import pytest
import torch

import resdag as rd
from resdag.ensemble import CoupledEnsembleESNModel
from resdag.ensemble.aggregators import OutliersFilteredMean


def _toy_data(seq_len: int = 60, feature_size: int = 2):
    """Smooth synthetic 2-D time series."""
    torch.manual_seed(0)
    raw = torch.randn(1, seq_len + 1, feature_size)
    kernel = torch.tensor([0.25, 0.5, 0.25])
    out = torch.empty_like(raw)
    for d in range(feature_size):
        out[:, :, d] = torch.nn.functional.conv1d(
            raw[:, :, d].unsqueeze(1), kernel.view(1, 1, 3), padding=1
        ).squeeze(1)
    return out[:, :seq_len, :]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_factory_builds_n_sub_models(self):
        ens = rd.coupled_ensemble_esn(
            n_models=4, reservoir_size=20, feedback_size=2, output_size=2
        )
        assert isinstance(ens, CoupledEnsembleESNModel)
        assert ens.n_models == 4
        assert len(ens.models) == 4

    def test_empty_models_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            CoupledEnsembleESNModel(models=[], aggregator="mean")

    def test_unknown_aggregator_string_raises(self):
        models = [
            rd.classic_esn(reservoir_size=10, feedback_size=2, output_size=2)
            for _ in range(2)
        ]
        with pytest.raises(ValueError, match="aggregator"):
            CoupledEnsembleESNModel(models=models, aggregator="argmin")

    def test_seed_reproducibility(self):
        """Two ensembles built with the same seed have identical weights."""
        e1 = rd.coupled_ensemble_esn(
            n_models=3, reservoir_size=20, feedback_size=2, output_size=2, seed=11
        )
        e2 = rd.coupled_ensemble_esn(
            n_models=3, reservoir_size=20, feedback_size=2, output_size=2, seed=11
        )
        for m1, m2 in zip(e1.models, e2.models):
            cell1 = next(
                m for m in m1.modules()
                if hasattr(m, "weight_hh") and hasattr(m, "reservoir_size")
            )
            cell2 = next(
                m for m in m2.modules()
                if hasattr(m, "weight_hh") and hasattr(m, "reservoir_size")
            )
            assert torch.equal(cell1.weight_hh.data, cell2.weight_hh.data)


# ---------------------------------------------------------------------------
# Fit / forecast
# ---------------------------------------------------------------------------


class TestFitForecast:
    def test_fit_and_forecast_mean(self):
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = rd.coupled_ensemble_esn(
            n_models=3, reservoir_size=30, feedback_size=2, output_size=2,
            seed=0,
        )
        ens.fit((warmup,), (train,), {"output": train.clone()})
        out = ens.forecast(f_warm, horizon=15)
        assert out.shape == (1, 15, 2)

    def test_forecast_return_individuals(self):
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = rd.coupled_ensemble_esn(
            n_models=4, reservoir_size=20, feedback_size=2, output_size=2, seed=0
        )
        ens.fit((warmup,), (train,), {"output": train.clone()})
        agg, indiv = ens.forecast(f_warm, horizon=10, return_individuals=True)
        assert agg.shape == (1, 10, 2)
        assert len(indiv) == 4
        for buf in indiv:
            assert buf.shape == (1, 10, 2)

    def test_forecast_return_warmup(self):
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = rd.coupled_ensemble_esn(
            n_models=3, reservoir_size=20, feedback_size=2, output_size=2, seed=0
        )
        ens.fit((warmup,), (train,), {"output": train.clone()})
        full = ens.forecast(f_warm, horizon=20, return_warmup=True)
        # Warmup phase has 20 steps, autoregressive horizon adds 20 → 40.
        assert full.shape[1] == 20 + 20

    def test_fit_parallel_n_workers(self):
        x = _toy_data()
        warmup, train = x[:, :10], x[:, 10:40]

        # Same seed → fits should yield bit-identical results sequential vs
        # threaded (CG solve is deterministic for fixed inputs/seeds).
        ens_seq = rd.coupled_ensemble_esn(
            n_models=2, reservoir_size=20, feedback_size=2, output_size=2, seed=99
        )
        ens_par = rd.coupled_ensemble_esn(
            n_models=2, reservoir_size=20, feedback_size=2, output_size=2, seed=99
        )
        ens_seq.fit((warmup,), (train,), {"output": train.clone()})
        ens_par.fit((warmup,), (train,), {"output": train.clone()}, n_workers=2)

        for m1, m2 in zip(ens_seq.models, ens_par.models):
            for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
                assert n1 == n2
                assert torch.allclose(p1, p2, atol=1e-6, rtol=1e-5), (
                    f"{n1}: sequential vs parallel diverged"
                )

    def test_fit_invalid_n_workers(self):
        ens = rd.coupled_ensemble_esn(
            n_models=2, reservoir_size=10, feedback_size=2, output_size=2
        )
        x = _toy_data()
        with pytest.raises(ValueError, match="n_workers"):
            ens.fit((x[:, :10],), (x[:, 10:30],),
                    {"output": x[:, 10:30].clone()}, n_workers=0)


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------


class TestAggregators:
    def test_custom_module_aggregator(self):
        """A user-supplied nn.Module aggregator is invoked at each step."""
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        agg = OutliersFilteredMean(method="z_score", threshold=2.0)
        ens = rd.coupled_ensemble_esn(
            n_models=4, reservoir_size=20, feedback_size=2, output_size=2,
            aggregate=agg, seed=0,
        )
        ens.fit((warmup,), (train,), {"output": train.clone()})
        out = ens.forecast(f_warm, horizon=10)
        assert out.shape == (1, 10, 2)

    def test_outliers_filtered_mean_all_outliers_fallback(self):
        """Regression for the bug fixed in commit f0bd4a7: at a position
        where *every* sample is classed as an outlier, the layer must
        return the plain mean (not zero)."""
        layer = OutliersFilteredMean(method="z_score", threshold=0.0)
        # threshold=0 means *nothing* is within tolerance — every sample
        # at every (batch, timestep) is an outlier.
        samples = torch.tensor(
            [
                [[[1.0, 2.0]]],   # sample 0
                [[[3.0, 4.0]]],   # sample 1
                [[[5.0, 6.0]]],   # sample 2
            ]
        )  # shape (samples=3, batch=1, T=1, F=2)
        result = layer(samples)
        expected = samples.mean(dim=0)
        assert torch.allclose(result, expected), (
            f"all-outlier fallback returned {result}, expected plain mean {expected}"
        )


# ---------------------------------------------------------------------------
# State handling + persistence
# ---------------------------------------------------------------------------


class TestStateAndPersistence:
    def test_get_set_reservoir_states_roundtrip(self):
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = rd.coupled_ensemble_esn(
            n_models=2, reservoir_size=20, feedback_size=2, output_size=2, seed=0
        )
        ens.fit((warmup,), (train,), {"output": train.clone()})
        ens.warmup(f_warm)
        saved = ens.get_reservoir_states()
        # Mutate then restore
        ens.reset_reservoirs()
        ens.set_reservoir_states(saved)
        restored = ens.get_reservoir_states()
        for s_old, s_new in zip(saved, restored):
            for key in s_old:
                assert torch.equal(s_old[key], s_new[key])

    def test_set_reservoir_states_wrong_length_raises(self):
        ens = rd.coupled_ensemble_esn(
            n_models=2, reservoir_size=20, feedback_size=2, output_size=2
        )
        with pytest.raises(ValueError, match="state dict"):
            ens.set_reservoir_states([{}])  # only 1 dict for 2 sub-models

    def test_save_load_roundtrip(self):
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = rd.coupled_ensemble_esn(
            n_models=2, reservoir_size=20, feedback_size=2, output_size=2, seed=0
        )
        ens.fit((warmup,), (train,), {"output": train.clone()})
        # Run a warmup to populate states
        ens.warmup(f_warm)
        saved_states = ens.get_reservoir_states()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ensemble.pt"
            ens.save(str(path), include_states=True, epoch=1)

            # Construct a fresh ensemble with the same architecture and load
            ens2 = rd.coupled_ensemble_esn(
                n_models=2, reservoir_size=20, feedback_size=2, output_size=2, seed=999
            )
            ens2.load(str(path), load_states=True)

            # Weights match
            for m1, m2 in zip(ens.models, ens2.models):
                for (_, p1), (_, p2) in zip(
                    m1.named_parameters(), m2.named_parameters()
                ):
                    assert torch.equal(p1, p2)

            # States restored
            for s1, s2 in zip(saved_states, ens2.get_reservoir_states()):
                for key in s1:
                    assert torch.equal(s1[key], s2[key])

    def test_load_size_mismatch_raises(self):
        ens_small = rd.coupled_ensemble_esn(
            n_models=2, reservoir_size=15, feedback_size=2, output_size=2
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ens.pt"
            ens_small.save(str(path))
            ens_big = rd.coupled_ensemble_esn(
                n_models=3, reservoir_size=15, feedback_size=2, output_size=2
            )
            with pytest.raises(ValueError, match="sub-model"):
                ens_big.load(str(path))
