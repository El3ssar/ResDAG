"""Shared task identities and portable resolved settings for Optuna studies."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import marshal
import os
import sys
import types
from collections.abc import Mapping
from functools import partial
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

if TYPE_CHECKING:
    import optuna

PROTOCOL_KEY = "resdag_tuning_protocol_v1"
RESOLVED_SETTINGS_KEY = "resolved_params_literal"
REUSED_SETTINGS_KEY = "resdag_reused_settings_literal"


def canonical_config(value: Any, seen: set[int] | None = None) -> Any:
    """Describe effective configuration/data without serializing executable code.

    Tensor/array content is hashed, independent of storage addresses. Paths
    describe their spelling, not the contents of external files: callers must
    bind external data by an explicit version/content identity or validate the
    materialized data before admitting an old score.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if inspect.ismodule(value):
        return ("module", value.__name__, getattr(value, "__version__", None))
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return ("tensor", str(value.dtype), tuple(value.shape), hashlib.sha256(raw).hexdigest())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return ("object_array", value.shape, canonical_config(value.tolist(), seen))
        return (
            "array",
            value.dtype.str,
            value.shape,
            hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        )
    if isinstance(value, os.PathLike):
        return ("path", os.fspath(value))
    stack = set() if seen is None else seen
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    if id(value) in stack:
        return ("recursive", kind)
    stack.add(id(value))
    try:
        if isinstance(value, Mapping):
            return tuple(
                (canonical_config(key, stack), canonical_config(item, stack))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            )
        if isinstance(value, (tuple, list)):
            return tuple(canonical_config(item, stack) for item in value)
        if isinstance(value, partial):
            return (
                "partial",
                canonical_config(value.func, stack),
                canonical_config(value.args, stack),
                canonical_config(value.keywords, stack),
            )
        if isinstance(value, type) or hasattr(value, "__qualname__"):
            return (
                "callable",
                getattr(value, "__module__", type(value).__module__),
                value.__qualname__,
            )
        return (kind, canonical_config(vars(value), stack) if hasattr(value, "__dict__") else None)
    finally:
        stack.remove(id(value))


def dependency_identity(value: Any, seen: set[int]) -> Any:
    """Include callable defaults, closures and referenced global dependencies."""
    if inspect.ismodule(value):
        return ("module", value.__name__, getattr(value, "__version__", None))
    if inspect.isclass(value):
        return class_identity(value, seen)
    if inspect.isfunction(value) or inspect.ismethod(value) or isinstance(value, partial):
        return callable_identity(value, seen)
    if isinstance(value, (Mapping, tuple, list)):
        if id(value) in seen:
            return ("recursive", type(value).__qualname__)
        seen = seen | {id(value)}
        if isinstance(value, Mapping):
            return tuple(
                (str(key), dependency_identity(item, seen))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            )
        return tuple(dependency_identity(item, seen) for item in value)
    return canonical_config(value)


def _method_identity(function: Any) -> Any:
    """Snapshot an active method without walking an entire framework graph."""
    if function is None:
        return None
    closure = getattr(function, "__closure__", None)
    return (
        getattr(function, "__module__", ""),
        getattr(function, "__qualname__", type(function).__qualname__),
        _implementation(function),
        canonical_config(getattr(function, "__defaults__", None)),
        canonical_config(getattr(function, "__kwdefaults__", None)),
        canonical_config(tuple(cell.cell_contents for cell in closure) if closure else None),
    )


def class_identity(value: type, seen: set[int]) -> Any:
    """Include active method bodies, attributes and inherited implementations.

    Direct method snapshots keep framework classes practical to inspect and
    catch replacements after definition. Dynamic dispatch, methods' external
    resources/global helper implementations and native binaries are not a
    semantic equivalence proof; explicit task versions remain appropriate for
    such dependencies. Package versions identify installed native dependencies.
    """
    identity = (value.__module__, value.__qualname__)
    if id(value) in seen:
        return ("recursive_class", *identity)
    seen = seen | {id(value)}
    members = []
    for name, member in sorted(vars(value).items()):
        if name in ("__dict__", "__weakref__", "__doc__", "__module__", "__qualname__"):
            continue
        if isinstance(member, (staticmethod, classmethod)):
            content = (type(member).__name__, _method_identity(member.__func__))
        elif isinstance(member, property):
            content = (
                "property",
                *(_method_identity(item) for item in (member.fget, member.fset, member.fdel)),
            )
        elif inspect.isfunction(member) or inspect.ismethod(member):
            content = _method_identity(member)
        elif inspect.isclass(member):
            content = class_identity(member, seen)
        else:
            content = canonical_config(member)
        members.append((name, content))
    package = sys.modules.get(value.__module__.split(".")[0])
    return (
        "class",
        *identity,
        getattr(package, "__version__", None),
        tuple(class_identity(base, seen) for base in value.__bases__),
        hashlib.sha256(repr(canonical_config(members)).encode()).hexdigest(),
    )


def _implementation(function: Any) -> str:
    try:
        return inspect.getsource(function)
    except (OSError, TypeError):
        code = getattr(function, "__code__", None)
        return (
            marshal.dumps(_portable_code(code)).hex() if code is not None else repr(type(function))
        )


def _portable_code(code: types.CodeType) -> types.CodeType:
    """Remove machine paths from code objects unavailable through source."""
    return code.replace(
        co_filename="",
        co_firstlineno=0,
        co_consts=tuple(
            _portable_code(item) if isinstance(item, types.CodeType) else item
            for item in code.co_consts
        ),
    )


def callable_identity(function: Any, seen: set[int] | None = None) -> Any:
    """Fingerprint the implementation and captured configuration of a callback."""
    if function is None:
        return None
    seen = set() if seen is None else seen
    if inspect.isclass(function):
        return class_identity(function, seen)
    if id(function) in seen:
        return (
            "recursive_callable",
            getattr(function, "__qualname__", type(function).__qualname__),
        )
    seen = seen | {id(function)}
    if isinstance(function, partial):
        return (
            "partial",
            callable_identity(function.func, seen),
            dependency_identity(function.args, seen),
            dependency_identity(function.keywords, seen),
        )
    source = _implementation(function)
    closure = getattr(function, "__closure__", None)
    dependencies: Mapping[str, Any] = {}
    if inspect.isfunction(function) or inspect.ismethod(function):
        dependencies = inspect.getclosurevars(function).globals
    return (
        getattr(function, "__module__", ""),
        getattr(function, "__qualname__", type(function).__qualname__),
        source,
        (
            class_identity(type(function), seen)
            if callable(function)
            and not (inspect.isfunction(function) or inspect.ismethod(function))
            else None
        ),
        dependency_identity(getattr(function, "__defaults__", None), seen),
        dependency_identity(getattr(function, "__kwdefaults__", None), seen),
        dependency_identity(
            tuple(cell.cell_contents for cell in closure) if closure else None, seen
        ),
        canonical_config(getattr(function, "__dict__", None)),
        canonical_config(getattr(function, "__self__", None)),
        dependency_identity(dependencies, seen),
    )


def fingerprint(payload: Any) -> str:
    """Return a SHA256 identity for a canonical data/configuration payload."""
    return hashlib.sha256(repr(canonical_config(payload)).encode()).hexdigest()


def resolved_settings(value: Mapping[str, Any]) -> dict[str, Any]:
    """Retain fixed settings alongside suggestions as finite portable literals."""
    if not isinstance(value, Mapping) or not all(isinstance(name, str) for name in value):
        raise TypeError("search_space must return a mapping with string keys")
    try:
        json.dumps(dict(value), allow_nan=False)
        return cast(dict[str, Any], ast.literal_eval(repr(dict(value))))
    except (TypeError, ValueError, SyntaxError) as error:
        raise TypeError(
            "search_space values must be finite JSON data; capture custom modules, "
            "initializers and callables in the model factory instead"
        ) from error


def trial_settings(trial: Any) -> dict[str, Any]:
    """Recover complete recorded settings, falling back to external suggestions."""
    literal = trial.user_attrs.get(RESOLVED_SETTINGS_KEY)
    if literal is None:
        return resolved_settings(trial.params)
    try:
        value = ast.literal_eval(literal)
    except (ValueError, SyntaxError, TypeError) as error:
        raise ValueError("Source trial contains invalid resolved settings") from error
    return resolved_settings(value)


def resolve_trial_settings(trial: Any, sampled: Mapping[str, Any]) -> dict[str, Any]:
    """Honor a queued complete configuration without dropping its fixed values."""
    current = resolved_settings(sampled)
    literal = trial.user_attrs.get(REUSED_SETTINGS_KEY)
    if literal is None:
        return current
    try:
        return {**current, **resolved_settings(ast.literal_eval(literal))}
    except (ValueError, SyntaxError, TypeError) as error:
        raise ValueError(
            "Queued reused settings are not valid portable resolved settings"
        ) from error


def bind_protocol(study: optuna.Study, protocol_hash: str) -> None:
    """Bind a fresh/queued study or reject changed and unproven evaluated tasks."""
    import optuna

    previous = study.user_attrs.get(PROTOCOL_KEY)
    evaluated = any(trial.state != optuna.trial.TrialState.WAITING for trial in study.trials)
    if previous != protocol_hash and (previous is not None or evaluated):
        raise ValueError(
            "Study protocol differs (data, validation, settings or objective); "
            "reuse its settings in a fresh study and re-evaluate them"
        )
    study.set_user_attr(PROTOCOL_KEY, protocol_hash)
