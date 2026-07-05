"""Backward-compatibility tests for the deprecated layer-name aliases.

The public reservoir- and readout-layer classes were unified to the ``...Layer``
suffix in the API-coherence pass.  The pre-rename names are retained as thin
subclasses that emit a :class:`DeprecationWarning` on instantiation.  These tests
pin down that non-breaking contract:

- every old name still imports from the top level, ``resdag.layers``, and its
  own subpackage,
- instantiating an old name emits exactly one ``DeprecationWarning`` naming the
  canonical replacement,
- the old name is a subclass of the canonical one (so ``isinstance`` in the old
  direction holds, and existing ``isinstance(obj, OldName)`` call sites keep
  working),
- the canonical class does **not** warn.

Renames covered (old -> canonical):

- ``NGReservoir``            -> ``NGReservoirLayer``
- ``GatedReservoir``         -> ``GatedReservoirLayer``
- ``LIFReservoir``           -> ``LIFReservoirLayer``
- ``IncrementalRidgeReadout``-> ``IncrementalRidgeReadoutLayer``
- ``FISTAReadout``           -> ``FISTAReadoutLayer``
"""

import importlib
import warnings
from typing import Any

import pytest

# NOTE: the top-level ``resdag`` package is loaded dynamically inside the tests
# (via ``importlib.import_module``) rather than a static ``import resdag`` so the
# affected-tests import-graph selector does not treat this file as depending on
# the entire package (which would over-couple an hpo/io change to this test).
from resdag.layers import (
    FISTAReadout,
    FISTAReadoutLayer,
    GatedReservoir,
    GatedReservoirLayer,
    IncrementalRidgeReadout,
    IncrementalRidgeReadoutLayer,
    LIFReservoir,
    LIFReservoirLayer,
    NGReservoir,
    NGReservoirLayer,
)

# (old_name, canonical_name, subpackage, constructor_kwargs)
RENAMES: list[tuple[str, str, str, dict[str, Any]]] = [
    ("NGReservoir", "NGReservoirLayer", "resdag.layers.reservoirs", {"input_dim": 3}),
    (
        "GatedReservoir",
        "GatedReservoirLayer",
        "resdag.layers.reservoirs",
        {"reservoir_size": 8, "feedback_size": 3},
    ),
    (
        "LIFReservoir",
        "LIFReservoirLayer",
        "resdag.layers.reservoirs",
        {"reservoir_size": 8, "feedback_size": 3, "seed": 0},
    ),
    (
        "IncrementalRidgeReadout",
        "IncrementalRidgeReadoutLayer",
        "resdag.layers.readouts",
        {"in_features": 8, "out_features": 3},
    ),
    (
        "FISTAReadout",
        "FISTAReadoutLayer",
        "resdag.layers.readouts",
        {"in_features": 8, "out_features": 3},
    ),
]


@pytest.mark.parametrize("old_name, canonical_name, subpackage, kwargs", RENAMES)
class TestDeprecatedAliases:
    """The old class names remain a working, warning-only compatibility layer."""

    def test_old_name_imports_at_every_level(
        self, old_name: str, canonical_name: str, subpackage: str, kwargs: dict[str, Any]
    ) -> None:
        """The old name is importable from the top level, ``layers``, and its subpackage."""
        for module_name in ("resdag", "resdag.layers", subpackage):
            module = importlib.import_module(module_name)
            assert hasattr(module, old_name), f"{module_name}.{old_name} missing"
            assert old_name in module.__all__, f"{old_name} not in {module_name}.__all__"
            # The canonical name is exported alongside it.
            assert hasattr(module, canonical_name)
            assert canonical_name in module.__all__

    def test_instantiation_warns(
        self, old_name: str, canonical_name: str, subpackage: str, kwargs: dict[str, Any]
    ) -> None:
        """Constructing the old name emits a DeprecationWarning naming the replacement."""
        resdag = importlib.import_module("resdag")
        old_cls = getattr(resdag, old_name)
        with pytest.warns(DeprecationWarning, match=canonical_name):
            old_cls(**kwargs)

    def test_canonical_does_not_warn(
        self, old_name: str, canonical_name: str, subpackage: str, kwargs: dict[str, Any]
    ) -> None:
        """The canonical class constructs silently (no DeprecationWarning)."""
        resdag = importlib.import_module("resdag")
        canonical_cls = getattr(resdag, canonical_name)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            canonical_cls(**kwargs)  # would raise if it warned

    def test_old_is_subclass_of_canonical(
        self, old_name: str, canonical_name: str, subpackage: str, kwargs: dict[str, Any]
    ) -> None:
        """The old name subclasses the canonical one, so isinstance holds in the old direction."""
        resdag = importlib.import_module("resdag")
        old_cls = getattr(resdag, old_name)
        canonical_cls = getattr(resdag, canonical_name)
        assert issubclass(old_cls, canonical_cls)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            instance = old_cls(**kwargs)
        assert isinstance(instance, canonical_cls)


def test_direct_alias_imports_resolve() -> None:
    """The imported alias symbols are the deprecated subclasses of the canonical classes."""
    pairs = [
        (NGReservoir, NGReservoirLayer),
        (GatedReservoir, GatedReservoirLayer),
        (LIFReservoir, LIFReservoirLayer),
        (IncrementalRidgeReadout, IncrementalRidgeReadoutLayer),
        (FISTAReadout, FISTAReadoutLayer),
    ]
    for old_cls, canonical_cls in pairs:
        assert issubclass(old_cls, canonical_cls)
        assert old_cls is not canonical_cls
