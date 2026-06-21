"""Tests for HPO study-naming robustness.

:func:`resdag.hpo.utils.make_study_name` runs on the default path whenever
``study_name is None`` (see :func:`resdag.hpo.run.run_hpo`). It must therefore
tolerate every callable shape a user might hand to ``run_hpo`` as a
``model_creator``:

- A :class:`functools.partial` — the natural way to pin fixed kwargs onto a
  premade factory. A bare ``partial`` makes :func:`inspect.getsourcefile` raise
  ``TypeError``, which previously aborted the study before any trial ran.
- Lambdas — two logically-distinct lambdas previously both produced
  ``'<string>:<lambda>'``, silently sharing persisted storage and resuming into
  each other.
- Callable instances — objects with ``__call__`` but no usable ``__name__``.

These tests pin the fix: a stable, non-crashing, collision-free name for each.
"""

import functools

import pytest

optuna = pytest.importorskip("optuna")

from resdag.hpo.utils import make_study_name  # noqa: E402


# ── Module-level callables (top-level so they stay introspectable/picklable) ──
def plain_creator(units: int = 10):
    """A plain top-level factory used as the baseline case."""
    return units


def other_creator(units: int = 20):
    """A second distinct top-level factory."""
    return units


class CallableCreator:
    """A callable class instance with no usable ``__name__``."""

    def __init__(self, units: int) -> None:
        self.units = units

    def __call__(self) -> int:
        return self.units


def test_plain_function_name() -> None:
    """A plain function yields ``filename:function_name``."""
    name = make_study_name(plain_creator)
    assert name.endswith(":plain_creator")
    assert ":" in name


def test_partial_does_not_crash() -> None:
    """A ``functools.partial`` produces a name instead of raising ``TypeError``."""
    creator = functools.partial(plain_creator, units=42)
    # Should not raise (regression: inspect.getsourcefile(partial) -> TypeError).
    name = make_study_name(creator)
    assert isinstance(name, str)
    assert name


def test_partial_recovers_underlying_name() -> None:
    """A ``partial`` is unwrapped to recover the wrapped factory's name + source."""
    creator = functools.partial(plain_creator, units=42)
    assert make_study_name(creator) == make_study_name(plain_creator)


def test_nested_partial_recovers_underlying_name() -> None:
    """Nested partials are unwrapped all the way to the underlying factory."""
    creator = functools.partial(functools.partial(plain_creator, units=1), units=2)
    assert make_study_name(creator) == make_study_name(plain_creator)


def test_partials_of_distinct_factories_differ() -> None:
    """Partials wrapping different factories produce different names."""
    a = functools.partial(plain_creator, units=1)
    b = functools.partial(other_creator, units=1)
    assert make_study_name(a) != make_study_name(b)


def test_lambda_does_not_crash() -> None:
    """A lambda creator produces a usable, non-crashing name."""
    name = make_study_name(lambda: 1)
    assert isinstance(name, str)
    assert name


def test_distinct_lambdas_do_not_collide() -> None:
    """Two distinct lambda creators must not share a study name."""
    creator_a = lambda: 1  # noqa: E731
    creator_b = lambda: 2  # noqa: E731
    assert make_study_name(creator_a) != make_study_name(creator_b)


def test_same_lambda_is_stable() -> None:
    """The same lambda object yields the same name across calls (stable hash)."""
    creator = lambda: 1  # noqa: E731
    assert make_study_name(creator) == make_study_name(creator)


def test_callable_instance_does_not_crash() -> None:
    """A callable class instance (no ``__name__``) produces a usable name."""
    name = make_study_name(CallableCreator(7))
    assert isinstance(name, str)
    assert name
    # Falls back to the type name, not a bare ``model_creator`` placeholder.
    assert "CallableCreator" in name


def test_distinct_callable_instances_do_not_collide() -> None:
    """Two distinct callable instances must not share a study name."""
    a = CallableCreator(1)
    b = CallableCreator(2)
    assert make_study_name(a) != make_study_name(b)


def test_callable_instance_is_stable() -> None:
    """The same callable instance yields the same name across calls."""
    instance = CallableCreator(3)
    assert make_study_name(instance) == make_study_name(instance)
