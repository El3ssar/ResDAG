"""Named functional models with a fitted, standard PyTorch workflow.

This first workflow increment supports existing :class:`ESNModel` graphs.
Algebraic fitting requires one distinct terminal readout per declared output.
Other graph families retain their existing model/trainer interfaces.
"""

from __future__ import annotations

import copy
import operator
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import torch
from pytorch_symbolic.symbolic_data import SymbolicData
from torch import nn

from resdag._contracts import module_reference, validate_sequence_inputs
from resdag.core._execution import ModelOutput, ReservoirStates
from resdag.core._forecast import positive_step_count
from resdag.core._modes import evaluating
from resdag.core._preprocessing import FittedNormalization
from resdag.core.model import ESNModel
from resdag.data.prepare import NormMethod
from resdag.layers.readouts import ReadoutLayer
from resdag.layers.reservoirs import BaseReservoirLayer

Streams = torch.Tensor | Mapping[str, torch.Tensor]
WorkflowOutput = torch.Tensor | dict[str, torch.Tensor]
Preprocessing = NormMethod | Literal["off"] | None


def _port_names(names: Sequence[str], count: int, label: str) -> tuple[str, ...]:
    if isinstance(names, str):
        raise TypeError(f"{label} must be a sequence of names, not a string")
    result = tuple(names)
    if len(result) != count or any(not isinstance(name, str) or not name for name in result):
        raise ValueError(f"{label} must contain {count} non-empty string names")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must contain distinct names")
    return result


def _washout(value: int) -> int:
    try:
        count = operator.index(value)
    except TypeError:
        raise ValueError("washout must be a non-negative integer") from None
    if isinstance(value, bool) or count < 0:
        raise ValueError("washout must be a non-negative integer")
    return count


class Model(nn.Module):
    """Wrap a named functional DAG in a common training/prediction workflow.

    Supply ``inputs={name: symbolic_input}`` and ``outputs={name: symbolic_output}``,
    or ``graph=existing_esn_model`` with optional ordered ``input_names`` and
    ``output_names``. The graph is registered once as :attr:`graph`; it remains
    available for raw, already-normalized tensor execution.

    ``fit(series)`` learns next-step prediction for one input/output of equal
    width. ``fit(inputs, targets={output_name: tensor})`` instead fits explicit,
    time-aligned targets; all streams include the same washout prefix. Only
    terminal readouts are fitted in this first increment. Internal fitted
    readouts require the lower-level trainer's explicit target mapping.

    ``forward`` accepts floating Torch tensors, preserves 2-D/3-D layout and
    gradients, applies fitted scaling and returns original units. Calls carry
    recurrent state by default; call :meth:`reset_state` between unrelated
    sequences. One head returns a Tensor; several return a named dictionary.

    Preprocessing is ``"standard"`` by default, any existing normalization
    method, or ``"off"``/``None``. Statistics use training inputs/targets after
    washout only. Implicit next-step targets share their input's statistics,
    preserving feedback units. **Temporary forecasting boundary:** explicit
    targets with different input/output scaling can be fitted and called, but
    autonomous forecasting rejects them until a feedback scaling bridge is
    available. Feedback currently follows the first input/output with unit delay;
    arbitrary routes and delays are not implemented by this constructor.

    ``state_dict`` includes scaling, ordered port metadata and the observed tail;
    reconstruct the same DAG before loading. Complete artifact loading, tuning
    and diagram exports are later roadmap increments.
    """

    _fitted: torch.Tensor

    def __init__(
        self,
        *,
        inputs: Mapping[str, SymbolicData] | None = None,
        outputs: Mapping[str, SymbolicData] | None = None,
        graph: ESNModel | None = None,
        input_names: Sequence[str] | None = None,
        output_names: Sequence[str] | None = None,
        washout: int = 100,
        preprocessing: Preprocessing = "standard",
    ) -> None:
        super().__init__()
        self.washout = _washout(washout)
        if preprocessing not in (
            None,
            "off",
            "minmax",
            "standard",
            "noncentered",
            "meanpreserving",
        ):
            raise ValueError("preprocessing must be a known normalization method, 'off', or None")
        self._preprocessing = None if preprocessing in (None, "off") else preprocessing
        if graph is not None:
            if inputs is not None or outputs is not None:
                raise ValueError("Supply graph= or named symbolic inputs/outputs, not both")
            if not isinstance(graph, ESNModel):
                raise TypeError(
                    "Model currently supports ESNModel graphs; use other model APIs directly"
                )
            in_names = (
                input_names
                if input_names is not None
                else tuple(
                    "input" if len(graph.inputs) == 1 else f"input_{i}"
                    for i in range(len(graph.inputs))
                )
            )
            out_names = (
                output_names
                if output_names is not None
                else tuple(
                    "output" if len(graph.outputs) == 1 else f"output_{i}"
                    for i in range(len(graph.outputs))
                )
            )
        else:
            if input_names is not None or output_names is not None:
                raise ValueError("input_names/output_names belong to the graph= constructor")
            if (
                not isinstance(inputs, Mapping)
                or not inputs
                or not isinstance(outputs, Mapping)
                or not outputs
            ):
                raise ValueError("Provide non-empty named symbolic inputs and outputs, or graph=")
            if not all(
                isinstance(node, SymbolicData) for node in (*inputs.values(), *outputs.values())
            ):
                raise TypeError("inputs and outputs must map names to symbolic graph nodes")
            in_names, out_names = tuple(inputs), tuple(outputs)
            _port_names(in_names, len(inputs), "inputs")
            _port_names(out_names, len(outputs), "outputs")
            graph = ESNModel(tuple(inputs.values()), tuple(outputs.values()))
        self.input_names = _port_names(in_names, len(graph.inputs), "input_names")
        self.output_names = _port_names(out_names, len(graph.outputs), "output_names")
        self.graph = graph
        self._declared_inputs = tuple(graph.inputs)
        self._declared_outputs = tuple(graph.outputs)
        self._input_widths = tuple(int(node.shape[-1]) for node in graph.inputs)
        self._output_widths = tuple(int(node.shape[-1]) for node in graph.outputs)
        method = cast(NormMethod, self._preprocessing or "standard")
        self.input_scalers = nn.ModuleList(FittedNormalization(method) for _ in self.input_names)
        self.output_scalers = nn.ModuleList(FittedNormalization(method) for _ in self.output_names)
        reference = module_reference(graph)
        if reference is not None:
            self.input_scalers.to(device=reference.device, dtype=reference.dtype)
            self.output_scalers.to(device=reference.device, dtype=reference.dtype)
        for index, width in enumerate(self._input_widths):
            empty = torch.empty(0, width) if reference is None else reference.new_empty(0, width)
            self.register_buffer(f"_history_{index}", empty)
        self.register_buffer(
            "_fitted", torch.tensor(False, device=None if reference is None else reference.device)
        )
        self._fitted_flag = False

    @property
    def model(self) -> ESNModel:
        """Compatibility access to the one registered graph, without another child."""
        return self.graph

    @property
    def preprocessing(self) -> NormMethod | None:
        """The fitted preprocessing policy; ``None`` means raw units."""
        return cast(NormMethod | None, self._preprocessing)

    @property
    def is_fitted(self) -> bool:
        """Whether this workflow has completed fitting or restored a fitted checkpoint."""
        return self._fitted_flag

    def _streams(
        self,
        data: Streams,
        names: tuple[str, ...],
        widths: tuple[int, ...],
        label: str,
        *,
        finite: bool = False,
        allow_empty: bool = False,
    ) -> tuple[tuple[torch.Tensor, ...], bool]:
        values: tuple[torch.Tensor, ...]
        if isinstance(data, torch.Tensor):
            if len(names) != 1:
                raise ValueError(f"{label} requires a mapping with keys {list(names)}")
            values = (data,)
        elif isinstance(data, Mapping):
            if set(data) != set(names):
                raise ValueError(f"{label} keys must be {list(names)}; got {list(data)}")
            values = tuple(data[name] for name in names)
        else:
            raise TypeError(f"{label} must be a torch.Tensor or a named tensor mapping")
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError(f"{label} values must be torch.Tensor instances")
        two_dimensional = values[0].ndim == 2
        if any(value.ndim != values[0].ndim for value in values):
            raise ValueError(f"{label} streams must use the same 2-D or 3-D layout")
        canonical = tuple(value.unsqueeze(0) if two_dimensional else value for value in values)
        validate_sequence_inputs(
            canonical,
            feature_sizes=widths,
            reference=module_reference(self.graph),
            label=label,
            allow_empty=allow_empty,
        )
        if finite and any(not bool(torch.isfinite(value).all()) for value in canonical):
            raise ValueError(f"{label} must contain only finite values")
        return canonical, two_dimensional

    def _require_scaling(self) -> None:
        self._validate_ports()
        if self.preprocessing is not None and not self.is_fitted:
            raise RuntimeError("Model preprocessing is not fitted; call fit(training) first")

    def _validate_ports(self) -> None:
        for actual, declared, widths in (
            (self.graph.inputs, self._declared_inputs, self._input_widths),
            (self.graph.outputs, self._declared_outputs, self._output_widths),
        ):
            if (
                len(actual) != len(declared)
                or any(a is not b for a, b in zip(actual, declared))
                or tuple(int(node.shape[-1]) for node in actual) != widths
            ):
                raise ValueError(
                    "The raw graph's input/output ports changed; construct a new Model "
                    "with the updated named ports before executing it"
                )

    def _scaled_inputs(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if self.preprocessing is None:
            return inputs
        return tuple(scaler(value) for scaler, value in zip(self.input_scalers, inputs))

    def _outputs(self, output: ModelOutput, two_dimensional: bool) -> WorkflowOutput:
        values = output if isinstance(output, tuple) else (output,)
        if len(values) != len(self.output_names):
            raise RuntimeError("Graph output count no longer matches the declared named outputs")
        restored = []
        for scaler, value in zip(self.output_scalers, values):
            value = (
                cast(FittedNormalization, scaler).inverse(value)
                if self.preprocessing is not None
                else value
            )
            restored.append(value.squeeze(0) if two_dimensional else value)
        return restored[0] if len(restored) == 1 else dict(zip(self.output_names, restored))

    def forward(self, inputs: Streams) -> WorkflowOutput:
        """Predict in original units, retaining gradients and carried recurrent state."""
        self._require_scaling()
        values, was_2d = self._streams(inputs, self.input_names, self._input_widths, "inputs")
        return self._outputs(self.graph(*self._scaled_inputs(values)), was_2d)

    def forward_stateless(
        self,
        inputs: Streams,
        *,
        states: Mapping[str, torch.Tensor] | None = None,
        compile: bool = False,
    ) -> tuple[WorkflowOutput, ReservoirStates]:
        """Predict original units with caller-managed recurrent states and gradients."""
        self._require_scaling()
        values, was_2d = self._streams(inputs, self.input_names, self._input_widths, "inputs")
        output, final = self.graph.forward_stateless(
            *self._scaled_inputs(values), states=states, compile=compile
        )
        return self._outputs(output, was_2d), final

    def reset_state(self) -> None:
        """Start the next ordinary forward from the reservoirs' configured reset states."""
        self.graph.reset_reservoirs()

    def get_state(self) -> ReservoirStates:
        """Copy the graph's initialized recurrent states."""
        return self.graph.get_state()

    def set_state(self, states: Mapping[str, torch.Tensor], strict: bool = True) -> None:
        """Restore graph recurrent states; preprocessing has no mutable streaming state."""
        self.graph.set_state(states, strict=strict)

    def _terminal_readouts(self) -> tuple[str, ...]:
        self._validate_ports()
        self.graph._get_execution_plan(require_explicit_state=False)
        terminals = [node.layer for node in self.graph.outputs]
        all_readouts = [
            (path, layer)
            for path, layer in self.graph.named_modules()
            if isinstance(layer, ReadoutLayer)
        ]
        if (
            not all(isinstance(layer, ReadoutLayer) for layer in terminals)
            or len({id(layer) for layer in terminals}) != len(terminals)
            or {id(layer) for layer in terminals} != {id(layer) for _, layer in all_readouts}
        ):
            raise ValueError(
                "Model.fit currently requires one distinct terminal readout per output "
                "and no internal fitted readouts; use ESNTrainer for an explicit staged fit"
            )
        keys = {id(layer): layer.name or path for path, layer in all_readouts}
        names = tuple(keys[id(layer)] for layer in terminals)
        if len(set(names)) != len(names):
            raise ValueError("Terminal readouts must have distinct internal names for fitting")
        return names

    def fit(self, training: Streams, targets: Streams | None = None) -> Model:
        """Fit terminal heads transactionally and retain the original observed tail.

        Implicit next-step supervision excludes the last input observation from
        the regression inputs and includes it in forecast history. Explicit
        targets align with the supplied input indices; no shift is guessed.
        Successful fitting resets streaming state. Failed fitting leaves the
        existing graph, fitted preprocessing and cached history unchanged.
        """
        from resdag.training import ESNTrainer

        readout_names = self._terminal_readouts()
        values, was_2d = self._streams(
            training, self.input_names, self._input_widths, "training", finite=True
        )
        implicit = targets is None
        train_inputs: tuple[torch.Tensor, ...]
        target_values: tuple[torch.Tensor, ...]
        if implicit:
            if (
                len(values) != 1
                or len(self.output_names) != 1
                or self._input_widths != self._output_widths
            ):
                raise ValueError(
                    "Implicit next-step fitting requires one input/output of equal width; "
                    "provide explicit named targets for custom graphs"
                )
            train_inputs, target_values = (values[0][:, :-1],), (values[0][:, 1:],)
        else:
            target_values, targets_2d = self._streams(
                cast(Streams, targets),
                self.output_names,
                self._output_widths,
                "targets",
                finite=True,
            )
            if targets_2d != was_2d or any(
                value.shape[:2] != values[0].shape[:2] for value in target_values
            ):
                raise ValueError("Explicit targets must share training batch/time axes and layout")
            train_inputs = values
        if train_inputs[0].shape[1] <= self.washout:
            raise ValueError("Training must contain washout plus at least one input/target pair")
        # Stage statistics and algebraic fitting on independent objects. Committing
        # weights into the original graph preserves existing optimizer references.
        method = cast(NormMethod, self.preprocessing or "standard")
        with torch.inference_mode(False):
            input_scalers = [FittedNormalization(method) for _ in values]
            output_scalers = [FittedNormalization(method) for _ in target_values]
        if self.preprocessing is not None:
            for scaler, value in zip(input_scalers, train_inputs):
                scaler.fit(value[:, self.washout :])
            if implicit:
                with torch.inference_mode(False):
                    output_scalers[0] = copy.deepcopy(input_scalers[0])
            else:
                for scaler, value in zip(output_scalers, target_values):
                    scaler.fit(value[:, self.washout :])
            scaled = tuple(scaler(value) for scaler, value in zip(input_scalers, train_inputs))
            scaled_targets = tuple(
                scaler(value) for scaler, value in zip(output_scalers, target_values)
            )
        else:
            scaled, scaled_targets = train_inputs, target_values
            reference = module_reference(self.graph)
            if reference is not None:
                for scaler in (*input_scalers, *output_scalers):
                    scaler.to(device=reference.device, dtype=reference.dtype)
        with torch.inference_mode(False), torch.no_grad():
            # Fitting starts from reset state. Exclude only transient carried
            # states, which may retain non-leaf autograd history and therefore
            # cannot be deep-copied. The live graph remains untouched if the
            # staged fit fails, including its still-differentiable states.
            transient_states = {
                id(layer.state): None
                for layer in self.graph.modules()
                if isinstance(layer, BaseReservoirLayer) and layer.state is not None
            }
            candidate = copy.deepcopy(self.graph, transient_states)
            ESNTrainer(candidate).fit(
                warmup_inputs=tuple(value[:, : self.washout] for value in scaled),
                train_inputs=tuple(value[:, self.washout :] for value in scaled),
                targets={
                    name: value[:, self.washout :]
                    for name, value in zip(readout_names, scaled_targets)
                },
            )
        history = []
        with torch.inference_mode(False), torch.no_grad():
            for value in values:
                tail = value[:, -(self.washout + 1) :].detach().clone()
                history.append(tail.squeeze(0) if was_2d else tail)
        with torch.inference_mode(False):
            self.graph.load_state_dict(candidate.state_dict())
        self.graph.reset_reservoirs()
        for scalers, previous in (
            (input_scalers, self.input_scalers),
            (output_scalers, self.output_scalers),
        ):
            for scaler, old in zip(scalers, previous):
                scaler.train(old.training)
        new_input_scalers, new_output_scalers = nn.ModuleList(input_scalers), nn.ModuleList(
            output_scalers
        )
        new_input_scalers.training = self.input_scalers.training
        new_output_scalers.training = self.output_scalers.training
        self.input_scalers, self.output_scalers = new_input_scalers, new_output_scalers
        for index, tail in enumerate(history):
            setattr(self, f"_history_{index}", tail)
        self._fitted_flag = True
        with torch.inference_mode(False):
            self._fitted = torch.ones((), dtype=torch.bool, device=values[0].device)
        return self

    def _feedback_scaling_matches(self) -> bool:
        if self.preprocessing is None:
            return True
        left, right = self.input_scalers[0], self.output_scalers[0]
        return left.method == right.method and all(
            torch.equal(a, b) for a, b in zip(left.buffers(), right.buffers())
        )

    def forecast(
        self,
        history: Streams | None = None,
        *,
        horizon: int,
        future_inputs: Streams | None = None,
        return_warmup: bool = False,
        compile: bool = False,
    ) -> WorkflowOutput:
        """Forecast the immediate unseen sample, then continue the first feedback route.

        ``future_inputs`` names the remaining input ports and requires
        ``horizon - 1`` driver samples. Without history, reuse the observed tail
        retained by fit. Every forecast resets/resynchronizes the graph and
        temporarily disables training noise. Explicit-target normalization with
        differing feedback scales is currently unsupported and fails before
        recurrent state changes; ordinary forward remains available.
        """
        self._require_scaling()
        horizon = positive_step_count(horizon, "horizon")
        self.graph._get_execution_plan()
        if any(
            not layer.is_fitted for layer in self.graph.modules() if isinstance(layer, ReadoutLayer)
        ):
            raise RuntimeError(
                "Forecast requires fitted terminal readouts; call fit(training) first"
            )
        if self._input_widths[0] != self._output_widths[0]:
            raise ValueError(
                "Forecast currently requires the first output width to match the first input"
            )
        if not self._feedback_scaling_matches():
            raise ValueError(
                "Forecast currently requires identical first input/output scaling; "
                "explicit targets with different scaling need a feedback bridge. "
                "Use implicit next-step fit or preprocessing='off'."
            )
        if history is None:
            if not self.is_fitted:
                raise RuntimeError("No cached history; provide history or call fit(training) first")
            history = {
                name: getattr(self, f"_history_{index}")
                for index, name in enumerate(self.input_names)
            }
        values, was_2d = self._streams(
            history, self.input_names, self._input_widths, "history", finite=True
        )
        driver_names, driver_widths = self.input_names[1:], self._input_widths[1:]
        if not driver_names:
            if future_inputs is not None:
                raise ValueError("This graph has no future driver inputs")
            drivers = None
        elif future_inputs is None:
            if horizon != 1:
                raise ValueError(f"future_inputs requires driver keys {list(driver_names)}")
            drivers = tuple(value[:, :0] for value in values[1:])
        else:
            drivers, _ = self._streams(
                future_inputs,
                driver_names,
                driver_widths,
                "future_inputs",
                finite=True,
                allow_empty=True,
            )
            if drivers[0].shape[0] != values[0].shape[0] or drivers[0].shape[1] < horizon - 1:
                raise ValueError(
                    "future_inputs must match the history batch and contain horizon - 1 samples"
                )
            if self.preprocessing is not None and drivers[0].shape[1]:
                drivers = tuple(
                    scaler(value) for scaler, value in zip(self.input_scalers[1:], drivers)
                )
        with evaluating(self.graph):
            output = self.graph.forecast(
                self._scaled_inputs(values),
                drivers,
                horizon=horizon,
                return_warmup=return_warmup,
                compile=compile,
            )
        return self._outputs(output, was_2d)

    def get_extra_state(self) -> dict[str, Any]:
        """Checkpoint metadata prevents silent remapping of equal-width ports."""
        return {
            "version": 1,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "input_widths": self._input_widths,
            "output_widths": self._output_widths,
            "preprocessing": self.preprocessing,
            "washout": self.washout,
        }

    def _validate_metadata(self, state: Any) -> None:
        if not isinstance(state, dict) or state.get("version") != 1:
            raise RuntimeError("Invalid Model checkpoint metadata/version")
        for key, expected in (
            ("input_names", self.input_names),
            ("output_names", self.output_names),
            ("input_widths", self._input_widths),
            ("output_widths", self._output_widths),
        ):
            if state.get(key) != expected:
                raise RuntimeError(
                    f"Model checkpoint {key} differs from this graph; reconstruct matching named ports"
                )
        if state.get("preprocessing") not in (
            None,
            "minmax",
            "standard",
            "noncentered",
            "meanpreserving",
        ):
            raise RuntimeError("Invalid Model checkpoint preprocessing")
        try:
            _washout(cast(int, state.get("washout")))
        except ValueError as error:
            raise RuntimeError("Invalid Model checkpoint washout") from error

    def set_extra_state(self, state: Any) -> None:
        """Restore workflow configuration after validating the named graph schema."""
        self._validate_metadata(state)
        self._preprocessing, self.washout = state["preprocessing"], state["washout"]

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
        metadata = state_dict.get(prefix + "_extra_state")
        self._validate_metadata(metadata)
        metadata = cast(dict[str, Any], metadata)
        fitted = state_dict.get(prefix + "_fitted")
        if (
            not isinstance(fitted, torch.Tensor)
            or fitted.shape != ()
            or fitted.dtype != torch.bool
            or fitted.device.type == "meta"
        ):
            raise RuntimeError("Invalid Model checkpoint fitted flag")
        fitted_flag = bool(fitted.item())
        if fitted_flag:
            for path, layer in self.graph.named_modules():
                if isinstance(layer, ReadoutLayer):
                    flag = state_dict.get(prefix + f"graph.{path}._is_fitted")
                    if (
                        not isinstance(flag, torch.Tensor)
                        or flag.shape != ()
                        or flag.dtype != torch.bool
                        or flag.device.type == "meta"
                        or not bool(flag.item())
                    ):
                        raise RuntimeError(
                            "Invalid Model checkpoint: fitted workflow requires fitted readouts"
                        )
        method = cast(NormMethod, metadata["preprocessing"] or "standard")
        # Standalone normalization restores unknown widths/rules. The surrounding
        # named graph fixes both; preflight all scalers before mutating weights.
        for group, scalers, widths in (
            ("input_scalers", self.input_scalers, self._input_widths),
            ("output_scalers", self.output_scalers, self._output_widths),
        ):
            for index, (scaler, width) in enumerate(zip(scalers, widths)):
                stem = prefix + f"{group}.{index}."
                scaler_state = {
                    key: state_dict[stem + key]
                    for key in ("_method_code", "_stat0", "_stat1")
                    if stem + key in state_dict
                }
                reference = cast(FittedNormalization, scaler)._stat0
                with torch.inference_mode(False):
                    probe = FittedNormalization(method).to(reference)
                    probe.load_state_dict(
                        scaler_state, assign=local_metadata.get("assign_to_params_buffers", False)
                    )
                expected_width = width if fitted_flag and metadata["preprocessing"] else None
                if probe.method != method or probe.n_features != expected_width:
                    raise RuntimeError(
                        f"Invalid Model checkpoint {group}: preprocessing method, feature "
                        "width and fitted statistics must match the declared graph schema"
                    )
        staged = []
        for index, width in enumerate(self._input_widths):
            key = prefix + f"_history_{index}"
            if key not in state_dict:
                raise RuntimeError("Model checkpoint is missing its cached history")
            value = state_dict[key]
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_floating_point()
                or value.ndim not in (2, 3)
                or value.shape[-1] != width
                or value.device.type == "meta"
                or not bool(torch.isfinite(value).all())
            ):
                raise RuntimeError("Invalid Model checkpoint history")
            with torch.inference_mode(False), torch.no_grad():
                destination = getattr(self, f"_history_{index}")
                converted = (
                    value.detach().clone()
                    if local_metadata.get("assign_to_params_buffers", False)
                    else value.to(destination, copy=True)
                )
                if not bool(torch.isfinite(converted).all()):
                    raise RuntimeError("Model checkpoint history overflows the destination dtype")
                staged.append(converted)
        if any(value.ndim != staged[0].ndim for value in staged):
            raise RuntimeError("Invalid Model checkpoint history: input layouts differ")
        canonical = tuple(value.unsqueeze(0) if value.ndim == 2 else value for value in staged)
        try:
            validate_sequence_inputs(
                canonical,
                feature_sizes=self._input_widths,
                label="checkpoint history",
                allow_empty=not fitted_flag,
            )
        except (ValueError, TypeError) as error:
            raise RuntimeError(f"Invalid Model checkpoint history: {error}") from error
        with torch.inference_mode(False):
            state_dict = dict(state_dict)
            for index, value in enumerate(staged):
                setattr(self, f"_history_{index}", torch.empty_like(value))
                state_dict[prefix + f"_history_{index}"] = value
            self._fitted = torch.empty_like(self._fitted)
            state_dict[prefix + "_fitted"] = fitted.detach().clone()
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
        self._fitted_flag = bool(self._fitted.item())
