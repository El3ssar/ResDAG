"""Tests for the dataset dt / lambda_max metadata helpers.

These live in :mod:`resdag.utils.data.datasets` but are metric-facing, so their
tests sit with the metrics suite.  They also guard the single-source contract
between ``DATASET_DT`` (in datasets) and the private ``_DATASET_DT`` copy (in
:mod:`resdag.metrics.forecast`) that keeps the two modules import-cycle-free.
"""

from __future__ import annotations

from resdag.metrics import CANONICAL_LAMBDA_MAX
from resdag.metrics.forecast import _DATASET_DT
from resdag.utils.data import DATASET_DT, dataset_lambda_max


class TestDatasetDt:
    def test_matches_generator_defaults(self) -> None:
        # These mirror the dt= defaults in the generator signatures.
        assert DATASET_DT["lorenz"] == 0.02
        assert DATASET_DT["rossler"] == 0.05
        assert DATASET_DT["mackey_glass"] == 1.0
        assert DATASET_DT["henon"] == 1.0
        assert DATASET_DT["narma"] == 1.0
        assert DATASET_DT["sine"] == 1.0

    def test_single_source_copy_in_sync(self) -> None:
        # The metrics module keeps a private byte-identical copy to avoid a
        # circular import; assert the two never drift.
        assert DATASET_DT == _DATASET_DT


class TestDatasetLambdaMax:
    def test_chaotic_values(self) -> None:
        assert dataset_lambda_max("lorenz") == CANONICAL_LAMBDA_MAX["lorenz"]
        assert dataset_lambda_max("rossler") == CANONICAL_LAMBDA_MAX["rossler"]
        assert dataset_lambda_max("mackey_glass", tau=17) == 0.0086
        assert dataset_lambda_max("mackey_glass", tau=30) == 0.0071

    def test_case_insensitive(self) -> None:
        assert dataset_lambda_max("Lorenz") == CANONICAL_LAMBDA_MAX["lorenz"]

    def test_non_chaotic_returns_none(self) -> None:
        assert dataset_lambda_max("sine") is None
        assert dataset_lambda_max("narma") is None

    def test_unknown_tau_returns_none(self) -> None:
        assert dataset_lambda_max("mackey_glass", tau=99) is None

    def test_unknown_name_returns_none(self) -> None:
        assert dataset_lambda_max("not_a_dataset") is None
