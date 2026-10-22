"""A standard PyTorch ESN preset using the common :class:`resdag.Model` workflow."""

from __future__ import annotations

import copy
import hashlib
import operator
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from functools import partial
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import nn

from resdag._contracts import module_reference
from resdag.core._execution import ReservoirStates
from resdag.core._feedback import Feedback, feedback_specs, prediction_leads
from resdag.core._workflow_transforms import (
    commit_transform_state,
    transform_schema,
    validate_custom_transform,
    validate_transform_ownership,
)
from resdag.core.model import ESNModel
from resdag.core.workflow import Model, Preprocessing, Streams, _washout
from resdag.data._trajectory import is_trajectory, validate_resampling_options
from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.models._builder import ReadoutSpec
from resdag.models.classic_esn import classic_esn

if TYPE_CHECKING:
    from tsdynamics.data import Trajectory


def _positive_width(value: int, name: str) -> int:
    try:
        width = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be a positive integer") from None
    if isinstance(value, bool) or width < 1:
        raise ValueError(f"{name} must be a positive integer")
    return width


def _configuration(value: Any, seen: set[int] | None = None) -> Any:
    """A portable configuration fingerprint, never a generic object serializer."""
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
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return ("tensor", str(value.dtype), tuple(value.shape), hashlib.sha256(raw).hexdigest())
    stack = set() if seen is None else seen
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    if id(value) in stack:
        return ("recursive", kind)
    stack.add(id(value))
    try:
        if isinstance(value, Mapping):
            return tuple(
                (_configuration(key, stack), _configuration(item, stack))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            )
        if isinstance(value, (tuple, list)):
            return tuple(_configuration(item, stack) for item in value)
        if isinstance(value, partial):
            return (
                "partial",
                _configuration(value.func, stack),
                _configuration(value.args, stack),
                _configuration(value.keywords, stack),
            )
        if isinstance(value, type) or hasattr(value, "__qualname__"):
            return ("callable", value.__module__, value.__qualname__)
        return (kind, _configuration(vars(value), stack) if hasattr(value, "__dict__") else None)
    finally:
        stack.remove(id(value))


class ESN(nn.Module):
    """A classic ESN preset with the same original-unit workflow as :class:`Model`.

    ``ESN(input_size=3)`` builds eagerly, making all parameters available before
    attaching an optimizer. Without ``input_size``, :meth:`fit` infers the width
    and attaches a successfully fitted workflow. Call :meth:`build` explicitly
    when constructing a larger PyTorch pipeline before its first fit.

    ``fit`` accepts floating Torch tensors or a TSDynamics trajectory. Ordinary
    ``forward`` returns Torch tensors, carries recurrent state, preserves layout
    and gradients, and uses fitted training-only scaling. ``forecast`` has the
    same keyword-only horizon and return conventions as :class:`Model`.

    The single registered :attr:`workflow` child is a :class:`Model`; its raw
    symbolic graph is exposed by :attr:`graph`. Custom DAGs, drivers and multiple
    outputs use ``Model`` directly. ``readout``/``readout_kwargs`` expose solver
    choices; additional reservoir settings and custom initializers pass through
    to :func:`resdag.models.classic_esn`.

    An untouched lazy preset adopts the first floating tensor's device/dtype.
    Explicit constructor placement or a pre-build ``to``/``double`` conversion
    fixes placement; later fitting follows ordinary Torch placement validation.
    ``build`` uses the current placement unless explicitly overridden. Checkpoints
    record width and preset configuration, allowing a matching unbuilt preset to
    load; arbitrary custom Python configurations must still be reconstructed.
    """

    workflow: Model | None
    _reference: torch.Tensor

    def __init__(
        self,
        reservoir_size: int = 500,
        *,
        input_size: int | None = None,
        spectral_radius: float = 0.9,
        leak_rate: float = 1.0,
        washout: int = 100,
        alpha: float = 1e-6,
        preprocessing: Preprocessing = "standard",
        input_transforms: Mapping[str, nn.Module] | None = None,
        output_transforms: Mapping[str, nn.Module] | None = None,
        prediction_lead: int | Mapping[str, int] = 1,
        feedback: Sequence[Feedback] | None = None,
        readout: ReadoutSpec = "auto",
        readout_kwargs: dict[str, Any] | None = None,
        topology: TopologySpec = None,
        feedback_initializer: InitializerSpec = None,
        activation: str = "tanh",
        bias: bool = True,
        readout_bias: bool = True,
        trainable: bool = False,
        seed: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        resample_dt: float | None = None,
        resample_method: Literal["linear", "pchip", "cubic"] = "linear",
        **reservoir_kwargs: Any,
    ) -> None:
        super().__init__()
        reservoir_size = _positive_width(reservoir_size, "reservoir_size")
        washout = _washout(washout)
        lead = prediction_leads(prediction_lead, ("output",))[0]
        connections = feedback_specs(feedback)
        resample_dt, validated_method = validate_resampling_options(resample_dt, resample_method)
        if preprocessing not in (
            None,
            "off",
            "standard",
            "minmax",
            "noncentered",
            "meanpreserving",
        ):
            raise ValueError("preprocessing must be a known normalization method, 'off', or None")
        reference = torch.empty(0, device=device, dtype=dtype)
        if not reference.is_floating_point():
            raise ValueError("dtype must be a real floating Torch dtype")
        reserved = {"feedback_size", "output_size", "readout_alpha"}.intersection(reservoir_kwargs)
        if reserved:
            raise ValueError(
                f"Preset dimensions/readout settings cannot be overridden: {sorted(reserved)}"
            )
        self.register_buffer("_reference", reference, persistent=False)
        self.add_module("workflow", None)
        self._input_size: int | None = None
        self._device_pinned, self._dtype_pinned = device is not None, dtype is not None
        overrides = []
        for mapping, name in ((input_transforms, "input"), (output_transforms, "output")):
            if mapping is not None and (not isinstance(mapping, Mapping) or set(mapping) - {name}):
                raise ValueError(
                    f"ESN transform override keys must be declared port names: [{name!r}]"
                )
            overrides.append(dict(mapping or {}))
        originals = tuple(module for mapping in overrides for module in mapping.values())
        if any(not isinstance(module, nn.Module) for module in originals):
            raise TypeError("Transform overrides must be torch.nn.Module instances")
        validate_transform_ownership(nn.Identity(), originals)
        # Originals remain ordinary registered modules while the lazy graph is
        # absent. A staged first fit only transfers them after it succeeds.
        self._pending_input_transforms: nn.ModuleDict | None = (
            nn.ModuleDict(overrides[0]) if overrides[0] else None
        )
        self._pending_output_transforms: nn.ModuleDict | None = (
            nn.ModuleDict(overrides[1]) if overrides[1] else None
        )
        self._workflow_options: dict[str, Any] = {
            "washout": washout,
            "preprocessing": preprocessing,
            "prediction_lead": lead,
            "feedback": connections,
            "resample_dt": resample_dt,
            "resample_method": validated_method,
            "input_transforms": overrides[0] or None,
            "output_transforms": overrides[1] or None,
        }
        self._factory_options = {
            "reservoir_size": reservoir_size,
            "spectral_radius": spectral_radius,
            "leak_rate": leak_rate,
            "readout_alpha": alpha,
            "readout": readout,
            "readout_kwargs": dict(readout_kwargs or {}),
            "topology": topology,
            "feedback_initializer": feedback_initializer,
            "activation": activation,
            "bias": bias,
            "readout_bias": readout_bias,
            "trainable": trainable,
            "seed": seed,
            **reservoir_kwargs,
        }
        self._architecture = (
            _configuration(self._factory_options),
            tuple(
                tuple((name, transform_schema(module, 0)) for name, module in mapping.items())
                for mapping in overrides
            ),
        )
        if input_size is not None:
            self.build(input_size)

    def _apply(self, fn: Any, recurse: bool = True) -> ESN:
        result = super()._apply(fn, recurse=recurse)
        self._device_pinned = self._dtype_pinned = True
        return cast(ESN, result)

    @property
    def input_size(self) -> int | None:
        """Observed feature width, available after eager construction/build/fit."""
        return self._input_size

    @property
    def is_built(self) -> bool:
        return self.workflow is not None

    @property
    def is_fitted(self) -> bool:
        return self.workflow is not None and self.workflow.is_fitted

    @property
    def graph(self) -> ESNModel:
        """The original raw symbolic graph; call build() or fit() before access."""
        return cast(ESNModel, self._require_workflow().graph)

    @property
    def washout(self) -> int:
        return (
            self.workflow.washout
            if self.workflow is not None
            else cast(int, self._workflow_options["washout"])
        )

    @property
    def preprocessing(self) -> str | None:
        if self.workflow is not None:
            return self.workflow.preprocessing
        value = self._workflow_options["preprocessing"]
        return None if value in (None, "off") else cast(str, value)

    @property
    def prediction_leads(self) -> dict[str, int]:
        if self.workflow is not None:
            return self.workflow.prediction_leads
        return {"output": self._workflow_options["prediction_lead"]}

    @property
    def feedback(self) -> tuple[Feedback, ...] | None:
        if self.workflow is not None:
            return self.workflow.feedback
        return cast(tuple[Feedback, ...] | None, self._workflow_options["feedback"])

    def _require_workflow(self) -> Model:
        if self.workflow is None:
            raise RuntimeError("ESN is not built; call build(input_size) or fit(training) first")
        return self.workflow

    def _new_workflow(self, width: int, device: torch.device, dtype: torch.dtype) -> Model:
        originals = tuple(
            module
            for key in ("input_transforms", "output_transforms")
            for module in (self._workflow_options[key] or {}).values()
        )
        validate_transform_ownership(nn.Identity(), originals)
        for module in originals:
            validate_custom_transform(module, width)
        seed = self._factory_options["seed"]
        with ExitStack() as isolation:
            if seed is not None:
                cuda_devices = (
                    list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
                )
                if device.type == "cuda" and torch.cuda.is_available():
                    index = (
                        device.index if device.index is not None else torch.cuda.current_device()
                    )
                    if index not in cuda_devices:
                        cuda_devices.append(index)
                isolation.enter_context(torch.random.fork_rng(devices=cuda_devices))
                isolate_mps = torch.backends.mps.is_available()
                if isolate_mps:
                    isolation.enter_context(torch.random.fork_rng(devices=[0], device_type="mps"))
                torch.default_generator.manual_seed(seed)
                for index in cuda_devices:
                    torch.cuda.default_generators[index].manual_seed(seed)
                if isolate_mps:
                    torch.mps.manual_seed(seed)
            graph = classic_esn(
                feedback_size=width,
                output_size=width,
                device=device,
                dtype=dtype,
                **self._factory_options,
            )
            options = dict(self._workflow_options)
            incoming, outgoing = copy.deepcopy(
                (options["input_transforms"], options["output_transforms"])
            )
            options.update(input_transforms=incoming, output_transforms=outgoing)
            workflow = Model(graph=graph, **options)
            workflow.train(self.training)
            return workflow

    def _attach(self, workflow: Model, width: int) -> None:
        reference = module_reference(workflow.graph)
        assert reference is not None
        with torch.inference_mode(False):
            for pending, scalers in (
                (self._pending_input_transforms, workflow.input_scalers),
                (self._pending_output_transforms, workflow.output_scalers),
            ):
                if pending:
                    original = next(iter(pending.values()))
                    original.to(device=reference.device, dtype=reference.dtype)
                    commit_transform_state(original, scalers[0])
                    scalers[0] = original
            self._pending_input_transforms = self._pending_output_transforms = None
            self.workflow = workflow
            self._input_size = width
            self._reference = torch.empty(0, device=reference.device, dtype=reference.dtype)

    def build(
        self,
        input_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ESN:
        """Materialize parameters before attaching optimizers; same-width calls are idempotent."""
        width = _positive_width(input_size, "input_size")
        placement = self._reference.device if device is None else torch.device(device)
        precision = self._reference.dtype if dtype is None else dtype
        if self.workflow is not None:
            reference = module_reference(self.graph)
            assert reference is not None
            if width != self.input_size:
                raise ValueError(
                    "ESN input_size is fixed after build; construct a new preset for another width"
                )
            if (device is not None and placement != reference.device) or (
                dtype is not None and precision != reference.dtype
            ):
                raise ValueError("ESN is already built; use .to() to change its device/dtype")
            return self
        if not torch.empty(0, dtype=precision).is_floating_point():
            raise ValueError("dtype must be a real floating Torch dtype")
        with torch.inference_mode(False):
            candidate = self._new_workflow(width, placement, precision)
        self._attach(candidate, width)
        self._device_pinned = self._dtype_pinned = True
        return self

    def fit(
        self,
        training: Streams | Trajectory | Mapping[str, Trajectory],
        targets: Streams | Trajectory | Mapping[str, Trajectory] | None = None,
    ) -> ESN:
        """Fit with Model's train-only scaling, explicit targets and transactional refitting."""
        if self.workflow is not None:
            self.workflow.fit(training, targets)
            return self
        data: Any = training
        if isinstance(training, Mapping):
            if set(training) != {"input"}:
                raise ValueError("ESN training mapping must contain exactly the key 'input'")
            data = training["input"]
        if is_trajectory(data):
            data = torch.as_tensor(cast(Any, data).y)
        if not isinstance(data, torch.Tensor):
            raise TypeError(
                "ESN training requires a floating torch.Tensor or a TSDynamics trajectory"
            )
        if data.ndim not in (2, 3):
            raise ValueError("ESN training requires 2-D or 3-D time-series tensors")
        if not data.is_floating_point():
            raise TypeError("ESN training requires a real floating tensor")
        width = _positive_width(data.shape[-1], "input_size")
        placement = self._reference.device if self._device_pinned else data.device
        precision = self._reference.dtype if self._dtype_pinned else data.dtype
        with torch.inference_mode(False):
            candidate = self._new_workflow(width, placement, precision)
        candidate.fit(training, targets)
        self._attach(candidate, width)
        self._device_pinned = self._dtype_pinned = True
        return self

    def tune(
        self,
        training: Streams | Trajectory | Mapping[str, Trajectory],
        targets: Streams | Trajectory | Mapping[str, Trajectory] | None = None,
        *,
        preset: Literal["quick", "balanced", "thorough"] = "balanced",
        n_trials: int | None = None,
        horizon: int | None = None,
        n_jobs: int = 1,
        **options: Any,
    ) -> ESN:
        """Tune preset settings, refit the winner and return this same ESN.

        Requires ``resdag[hpo]``. Presets automatically rebuild reservoir
        sizes, initializations and readouts together. ``search_space(trial)``
        overrides the default Optuna search. See the tuning guide for explicit
        validation, seeds, objectives, study storage and :attr:`tuning_report`.
        Failed tuning preserves the current model. Create optimizers afterward.
        """
        from resdag.hpo import _require_optuna

        _require_optuna()
        from resdag.hpo.tuning import tune_model

        tune_model(
            self,
            training,
            targets,
            preset=preset,
            n_trials=n_trials,
            horizon=horizon,
            n_jobs=n_jobs,
            **options,
        )
        return self

    def forward(self, inputs: Streams) -> torch.Tensor:
        """Predict original units with carried state and ordinary Torch autograd/hooks."""
        return cast(torch.Tensor, self._require_workflow()(inputs))

    def forward_stateless(
        self,
        inputs: Streams,
        *,
        states: Mapping[str, torch.Tensor] | None = None,
        compile: bool = False,
    ) -> tuple[torch.Tensor, ReservoirStates]:
        output, final = self._require_workflow().forward_stateless(
            inputs, states=states, compile=compile
        )
        return cast(torch.Tensor, output), final

    def forecast(
        self,
        history: Streams | Trajectory | Mapping[str, Trajectory] | None = None,
        *,
        horizon: int,
        future_inputs: Streams | Trajectory | Mapping[str, Trajectory] | None = None,
        return_warmup: bool = False,
        compile: bool = False,
        execution: Literal["fast", "history"] = "fast",
        return_type: Literal["auto", "tensor", "trajectory"] = "auto",
    ) -> torch.Tensor | Trajectory:
        return cast(
            "torch.Tensor | Trajectory",
            self._require_workflow().forecast(
                history,
                horizon=horizon,
                future_inputs=future_inputs,
                return_warmup=return_warmup,
                compile=compile,
                return_type=return_type,
                execution=execution,
            ),
        )

    def reset_state(self) -> None:
        self._require_workflow().reset_state()

    def get_state(self) -> ReservoirStates:
        return self._require_workflow().get_state()

    def set_state(self, states: Mapping[str, torch.Tensor], strict: bool = True) -> None:
        self._require_workflow().set_state(states, strict=strict)

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": 2,
            "input_size": self.input_size,
            "architecture": copy.deepcopy(self._architecture),
        }

    def _validate_metadata(self, state: Any) -> int | None:
        if (
            not isinstance(state, dict)
            or type(state.get("version")) is not int
            or state["version"] != 2
        ):
            raise RuntimeError(
                "ESN checkpoint requires current metadata version 2; "
                "reconstruct and refit models saved with older metadata"
            )
        if "input_size" not in state:
            raise RuntimeError("Invalid ESN checkpoint: missing input_size")
        if state.get("architecture") != self._architecture:
            raise RuntimeError(
                "ESN checkpoint architecture differs; reconstruct the same preset settings"
            )
        width = state.get("input_size")
        if width is not None:
            try:
                width = _positive_width(width, "input_size")
            except ValueError as error:
                raise RuntimeError("Invalid ESN checkpoint input_size") from error
        if self.is_built and width != self.input_size:
            raise RuntimeError("ESN checkpoint input_size differs from this built preset")
        return cast(int | None, width)

    def set_extra_state(self, state: Any) -> None:
        self._validate_metadata(state)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        width = self._validate_metadata(state_dict.get(prefix + "_extra_state"))
        if width is not None and not self.is_built:
            self.build(width)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def extra_repr(self) -> str:
        return f"reservoir_size={self._factory_options['reservoir_size']}, input_size={self.input_size}, fitted={self.is_fitted}"


__all__ = ["ESN"]
