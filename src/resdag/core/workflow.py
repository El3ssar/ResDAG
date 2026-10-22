"""Named functional models with a fitted, standard PyTorch workflow.

The existing symbolic, parallel and coupled model families share this workflow.
Algebraic symbolic fitting requires one distinct terminal readout per output.
"""

from __future__ import annotations

import copy
import operator
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
from pytorch_symbolic.symbolic_data import SymbolicData
from torch import nn

from resdag._contracts import module_reference, validate_sequence_inputs
from resdag.core._execution import ModelOutput, ReservoirStates
from resdag.core._feedback import (
    Feedback,
    FeedbackPlan,
    FeedbackRoute,
    feedback_plan,
    feedback_rollout,
    feedback_specs,
    prediction_leads,
)
from resdag.core._forecast import positive_step_count
from resdag.core._forecast_capabilities import (
    ForecastSequenceExecutor,
    validate_forecast_execution,
)
from resdag.core._modes import evaluating
from resdag.core._preprocessing import FittedNormalization
from resdag.core._workflow_backends import recurrent_graph, workflow_backend
from resdag.core.model import ESNModel
from resdag.data._trajectory import (
    forecast_trajectory,
    is_trajectory,
    prepare_trajectory,
    trajectory_type,
    validate_resampling_options,
    validate_time_grid,
    validate_trajectory_context,
)
from resdag.data.prepare import NormMethod
from resdag.layers.readouts import ReadoutLayer
from resdag.layers.reservoirs import BaseReservoirLayer

if TYPE_CHECKING:
    from tsdynamics.data import Trajectory

    from resdag.ensemble.coupled import CoupledEnsembleESNModel

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
    return tuple(str(name) for name in result)


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
    or ``graph=existing_model`` with optional ordered ``input_names`` and
    ``output_names``. The graph is registered once as :attr:`graph`; it remains
    available for raw, already-normalized tensor execution. Existing symbolic,
    parallel local-state and coupled ensemble backends are supported. Parallel
    ports describe the full field; coupled members share one aggregate output.
    Coupled fitting trains member readouts only. A custom feed-forward aggregator
    is supplied/configured by the user and can be optimized separately with
    ordinary PyTorch. Reservoirs hidden inside an aggregator require ordinary
    stateful forward rather than explicit-state execution or forecasting.

    ``fit(series)`` learns future prediction for one input/output of equal
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
    washout only. Implicit targets share their input's statistics. Feedback
    automatically converts each output back to original units before applying
    its destination input's scaling.

    ``prediction_lead`` declares the physical target distance for each head:
    input at step ``t`` predicts target at ``t + lead``. The default is one;
    a mapping overrides named heads. ``feedback`` supplies explicit
    :class:`Feedback` connections with independent graph-step delays. Without
    connections, the first output automatically feeds the first input after
    its prediction lead. An empty sequence leaves all inputs externally driven.

    ``state_dict`` includes scaling, ordered port metadata and the observed tail;
    reconstruct the same DAG before loading. Complete artifact loading, tuning
    and diagram exports are later roadmap increments.

    With the optional ``dynamics`` extra, ``fit(trajectory)`` learns from one
    TSDynamics trajectory and forecasts return a trajectory with future times
    and original component names. ``return_type="tensor"`` keeps predictions
    on the model device. Ordinary ``forward`` always uses Torch tensors.
    Uneven timestamps are automatically interpolated with ``resample_method``
    (linear, PCHIP or cubic); ``resample_dt`` controls the physical step.
    """

    _fitted: torch.Tensor

    def __init__(
        self,
        *,
        inputs: Mapping[str, SymbolicData] | None = None,
        outputs: Mapping[str, SymbolicData] | None = None,
        graph: nn.Module | None = None,
        input_names: Sequence[str] | None = None,
        output_names: Sequence[str] | None = None,
        washout: int = 100,
        preprocessing: Preprocessing = "standard",
        prediction_lead: int | Mapping[str, int] = 1,
        feedback: Sequence[Feedback] | None = None,
        resample_dt: float | None = None,
        resample_method: Literal["linear", "pchip", "cubic"] = "linear",
    ) -> None:
        super().__init__()
        self.washout = _washout(washout)
        self.resample_dt, self.resample_method = validate_resampling_options(
            resample_dt, resample_method
        )
        self._trajectory_context: dict[str, Any] | None = None
        object.__setattr__(self, "_trajectory_system", None)
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
            backend = workflow_backend(graph)
            interface = backend.describe(graph)
            in_names = (
                input_names
                if input_names is not None
                else tuple(
                    "input" if len(interface.input_widths) == 1 else f"input_{i}"
                    for i in range(len(interface.input_widths))
                )
            )
            out_names = (
                output_names
                if output_names is not None
                else tuple(
                    "output" if len(interface.output_widths) == 1 else f"output_{i}"
                    for i in range(len(interface.output_widths))
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
            backend = workflow_backend(graph)
            interface = backend.describe(graph)
        self.input_names = _port_names(in_names, len(interface.input_widths), "input_names")
        self.output_names = _port_names(out_names, len(interface.output_widths), "output_names")
        self.graph = graph
        self._backend = backend
        self._backend_interface_schema = interface.schema
        self._backend_schema = backend.checkpoint_schema(graph, interface)
        self._backend_references = interface.references
        self._input_widths = interface.input_widths
        self._output_widths = interface.output_widths
        self._prediction_leads = prediction_leads(prediction_lead, self.output_names)
        self._feedback = feedback_specs(feedback)
        self._feedback_plan(validate_widths=self._feedback is not None)
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
        self.register_load_state_dict_post_hook(self._synchronize_loaded_buffers)

    def _synchronize_loaded_buffers(self, module: nn.Module, incompatible_keys: Any) -> None:
        # PyTorch shares this key list across the whole loading tree. Missing
        # sibling keys under strict=False do not prevent this graph's tensors
        # from being assigned and must not skip its derived-buffer repair.
        self._backend.synchronize_loaded_buffers(self.graph)

    @property
    def model(self) -> nn.Module:
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

    @property
    def prediction_leads(self) -> dict[str, int]:
        """Physical target distances by public output name."""
        return dict(zip(self.output_names, self._prediction_leads))

    @property
    def feedback(self) -> tuple[Feedback, ...] | None:
        """Explicit connections, or ``None`` for the automatic first-port route."""
        return self._feedback

    def _feedback_plan(self, *, validate_widths: bool = True) -> FeedbackPlan:
        return feedback_plan(
            self._feedback,
            self._prediction_leads,
            self.input_names,
            self.output_names,
            self._input_widths,
            self._output_widths,
            validate_widths=validate_widths,
        )

    def _requires_complete_history(self) -> bool:
        if self._backend.kind == "parallel":
            return False
        try:
            validate_forecast_execution(self.graph, "fast")
        except ValueError:
            return True
        return False

    def _fit_complete_history(
        self,
        candidate: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        targets: tuple[torch.Tensor, ...],
    ) -> None:
        """Fit terminal heads from temporal features evaluated on the full prefix."""
        members = (
            (cast(ESNModel, candidate),)
            if self._backend.kind == "symbolic"
            else tuple(cast("CoupledEnsembleESNModel", candidate)._iter_models())
        )
        for member in members:
            member.reset_reservoirs()
            hooks = []
            try:
                for node, target in zip(member.outputs, targets):
                    readout = cast(ReadoutLayer, node.layer)

                    def capture(
                        layer: nn.Module,
                        args: tuple[torch.Tensor, ...],
                        *,
                        head: ReadoutLayer = readout,
                        expected: torch.Tensor = target,
                    ) -> None:
                        if (
                            len(args) != 1
                            or not isinstance(args[0], torch.Tensor)
                            or args[0].shape != (*inputs[0].shape[:2], head.in_features)
                        ):
                            raise ValueError(
                                "Temporal fitting requires terminal features preserving "
                                "the complete input batch/time axes"
                            )
                        head.fit(args[0][:, self.washout :], expected[:, self.washout :])

                    hooks.append(readout.register_forward_pre_hook(capture))
                member(*inputs)
            finally:
                for hook in hooks:
                    hook.remove()

    def _commit_forecast_states(self, states: ReservoirStates) -> None:
        """Store evolved state without turning it into a user-pinned restore."""
        reservoirs = {
            name: layer
            for name, layer in self.graph.named_modules()
            if isinstance(layer, BaseReservoirLayer)
        }
        if set(states) != set(reservoirs):
            raise RuntimeError("Forecast executor returned an inconsistent reservoir state schema")
        staged = {}
        with torch.inference_mode(False), torch.no_grad():
            for name, layer in reservoirs.items():
                layer.cell.validate_state(states[name])
                staged[name] = states[name].detach().clone()
        for name, layer in reservoirs.items():
            layer.reset_state()
            layer.state = staged[name]

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
        try:
            interface = self._backend.describe(self.graph)
            valid = (
                interface.input_widths == self._input_widths
                and interface.output_widths == self._output_widths
                and interface.schema == self._backend_interface_schema
                and len(interface.references) == len(self._backend_references)
                and all(a is b for a, b in zip(interface.references, self._backend_references))
            )
        except (ValueError, TypeError, AttributeError):
            valid = False
        if not valid:
            raise ValueError(
                "The raw graph's input/output ports changed; construct a new Model "
                "with the updated named ports and backend structure before executing it"
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
        self._backend.validate_stateless(self.graph)
        values, was_2d = self._streams(inputs, self.input_names, self._input_widths, "inputs")
        output, final = recurrent_graph(self.graph).forward_stateless(
            *self._scaled_inputs(values), states=states, compile=compile
        )
        return self._outputs(output, was_2d), final

    def reset_state(self) -> None:
        """Start the next ordinary forward from the reservoirs' configured reset states."""
        recurrent_graph(self.graph).reset_reservoirs()

    def get_state(self) -> ReservoirStates:
        """Copy reservoir states, excluding a forecast's ephemeral feedback queues.

        Forecast calls replay history and construct fresh delay queues. These
        tensors alone do not resume a delayed forecast at its previous clock.
        """
        return recurrent_graph(self.graph).get_state()

    def set_state(self, states: Mapping[str, torch.Tensor], strict: bool = True) -> None:
        """Restore ordinary-forward reservoir states, not delayed-rollout queues."""
        recurrent_graph(self.graph).set_state(states, strict=strict)

    def fit(self, training: Streams | Trajectory, targets: Streams | None = None) -> Model:
        """Fit terminal heads transactionally and retain the original observed tail.

        Implicit supervision pairs input ``t`` with observation
        ``t + prediction_lead`` and retains the complete observed tail.
        Explicit targets are already caller-aligned to input indices; declared
        leads describe their physical timestamps without shifting those arrays.
        Successful fitting resets streaming state. Failed fitting leaves the
        existing graph, fitted preprocessing and cached history unchanged.
        A bare TSDynamics trajectory supports implicit single-input/output
        lead-shifted fitting. Its time axis, rather than a metadata ``dt``, defines
        sampling. Tensor refitting clears trajectory output defaults on success.
        """
        self._validate_ports()
        self._backend.validate_fit(self.graph)
        complete_history = self._requires_complete_history()
        trajectory_context = None
        trajectory_system = None
        if is_trajectory(training):
            if (
                targets is not None
                or len(self.input_names) != 1
                or len(self.output_names) != 1
                or self._input_widths != self._output_widths
            ):
                raise ValueError(
                    "Trajectory fitting requires one input/output of equal width with "
                    "implicit lead-shifted targets; use named tensors for other graphs"
                )
            prepared = prepare_trajectory(
                training, dt=self.resample_dt, method=self.resample_method
            )
            reference = module_reference(self.graph)
            training = torch.tensor(
                prepared.values,
                dtype=None if reference is None else reference.dtype,
                device=None if reference is None else reference.device,
            )
            trajectory_context, trajectory_system = prepared.context, prepared.system
        values, was_2d = self._streams(
            cast(Streams, training), self.input_names, self._input_widths, "training", finite=True
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
                    "Implicit fitting requires one input/output of equal width; "
                    "provide explicit named targets for custom graphs"
                )
            lead = self._prediction_leads[0]
            train_inputs, target_values = (values[0][:, :-lead],), (values[0][:, lead:],)
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
            if complete_history:
                self._fit_complete_history(candidate, scaled, scaled_targets)
            else:
                self._backend.fit(
                    candidate,
                    tuple(value[:, : self.washout] for value in scaled),
                    tuple(value[:, self.washout :] for value in scaled),
                    tuple(value[:, self.washout :] for value in scaled_targets),
                )
        history = []
        tail_length = self.washout + self._feedback_plan(validate_widths=False).required_history
        with torch.inference_mode(False), torch.no_grad():
            for value in values:
                tail = (value if complete_history else value[:, -tail_length:]).detach().clone()
                history.append(tail.squeeze(0) if was_2d else tail)
        with torch.inference_mode(False):
            self.graph.load_state_dict(candidate.state_dict())
        recurrent_graph(self.graph).reset_reservoirs()
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
        self._trajectory_context = trajectory_context
        object.__setattr__(self, "_trajectory_system", trajectory_system)
        return self

    def _feedback_scaling_matches(self, input_index: int = 0, output_index: int = 0) -> bool:
        if self.preprocessing is None:
            return True
        left, right = self.input_scalers[input_index], self.output_scalers[output_index]
        return left.method == right.method and all(
            torch.equal(a, b) for a, b in zip(left.buffers(), right.buffers())
        )

    def forecast(
        self,
        history: Streams | Trajectory | None = None,
        *,
        horizon: int,
        future_inputs: Streams | None = None,
        return_warmup: bool = False,
        compile: bool = False,
        execution: Literal["fast", "history"] = "fast",
        return_type: Literal["auto", "tensor", "trajectory"] = "auto",
    ) -> WorkflowOutput | Trajectory:
        """Return named heads aligned to the immediate unseen physical sample.

        ``future_inputs`` names every input without a feedback connection.
        Drivers start immediately after history and require
        ``max(0, horizon - min(prediction_leads))`` samples. With the default
        next-step head this is ``horizon - 1``. History must contain at least
        the largest prediction lead and feedback delay. Warmup predictions
        already provide the first unseen samples; they are never skipped.
        ``return_warmup`` prepends predictions on the shared physical timeline
        starting at history index ``max(prediction_leads)``.

        Every call rebuilds pending feedback from history and temporarily
        disables training noise. ``execution='fast'`` requires pointwise graph
        modules or explicit reservoir adapters. ``'history'`` replays the full
        growing input prefix for pure causal sequence modules, at higher cost.
        Custom modules may declare ``supports_framewise=True`` when appropriate.
        Compiled forecasting is currently available for symbolic graphs only;
        parallel and coupled backends require ``compile=False``.

        ``return_type="auto"`` returns a TSDynamics trajectory after trajectory
        fitting or with trajectory history; otherwise it returns tensors.
        ``"tensor"`` avoids conversion to NumPy/CPU. A selected trajectory result
        needs timestamped history or the cached training context; a new tensor
        history has no absolute clock. Trajectory history is resampled to the
        fitted physical step and must retain fitted component names/order.
        """
        self._require_scaling()
        if return_type not in ("auto", "tensor", "trajectory"):
            raise ValueError("return_type must be 'auto', 'tensor', or 'trajectory'")
        horizon = positive_step_count(horizon, "horizon")
        self._backend.validate_forecast(self.graph, compile=compile)
        validate_forecast_execution(self.graph, execution, compile=compile)
        plan = self._feedback_plan()
        if any(not layer.is_fitted for _, layer in self._backend.fitted_readouts(self.graph)):
            raise RuntimeError(
                "Forecast requires fitted terminal readouts; call fit(training) first"
            )
        context = self._trajectory_context
        system = self._trajectory_system
        history_times = None
        cached_history = history is None
        if is_trajectory(history):
            if len(self.input_names) != 1 or len(self.output_names) != 1:
                raise ValueError("Trajectory history requires a single input/output graph")
            prepared = prepare_trajectory(
                history,
                dt=context["dt"] if context is not None else self.resample_dt,
                method=self.resample_method,
            )
            if context is not None and prepared.context["variables"] != context["variables"]:
                raise ValueError(
                    "History trajectory components/order must match the fitted trajectory"
                )
            context, system, history_times = prepared.context, prepared.system, prepared.times
            reference = module_reference(self.graph)
            history = torch.tensor(
                prepared.values,
                dtype=None if reference is None else reference.dtype,
                device=None if reference is None else reference.device,
            )
        trajectory_output = return_type == "trajectory" or (
            return_type == "auto" and context is not None
        )
        if trajectory_output:
            if context is None or (not cached_history and history_times is None):
                raise ValueError(
                    "Trajectory output needs timestamped trajectory history or cached "
                    "trajectory fitting history; pass a Trajectory or return_type='tensor'"
                )
            trajectory_type()  # Optional dependency must fail before graph hooks execute.
        if history is None:
            if not self.is_fitted:
                raise RuntimeError("No cached history; provide history or call fit(training) first")
            history = {
                name: getattr(self, f"_history_{index}")
                for index, name in enumerate(self.input_names)
            }
        values, was_2d = self._streams(
            cast(Streams, history), self.input_names, self._input_widths, "history", finite=True
        )
        if values[0].shape[1] < plan.required_history:
            raise ValueError(
                f"history needs at least {plan.required_history} samples to seed all "
                "prediction leads and feedback delays"
            )
        driver_names = tuple(self.input_names[index] for index in plan.driver_indices)
        driver_widths = tuple(self._input_widths[index] for index in plan.driver_indices)
        steps = plan.future_steps(horizon)
        drivers: tuple[torch.Tensor, ...]
        if not driver_names:
            if future_inputs is not None:
                raise ValueError("This graph has no future driver inputs")
            drivers = ()
        elif future_inputs is None:
            if steps:
                raise ValueError(f"future_inputs requires driver keys {list(driver_names)}")
            drivers = tuple(values[index][:, :0] for index in plan.driver_indices)
        else:
            drivers, _ = self._streams(
                future_inputs,
                driver_names,
                driver_widths,
                "future_inputs",
                finite=True,
                allow_empty=True,
            )
            if drivers[0].shape[0] != values[0].shape[0] or drivers[0].shape[1] < steps:
                raise ValueError(
                    f"future_inputs must match the history batch and contain at least {steps} samples"
                )
            if self.preprocessing is not None and drivers[0].shape[1]:
                drivers = tuple(
                    self.input_scalers[index](value)
                    for index, value in zip(plan.driver_indices, drivers)
                )
        times = None
        if trajectory_output:
            assert context is not None
            if history_times is None:
                history_times = context["last_time"] - context["dt"] * np.arange(
                    values[0].shape[1] - 1, -1, -1, dtype=np.float64
                )
            with np.errstate(over="ignore", invalid="ignore"):
                times = context["last_time"] + context["dt"] * np.arange(
                    1, horizon + 1, dtype=np.float64
                )
            if not times[0] > context["last_time"]:
                raise ValueError("Forecast timestamps cannot advance at float64 precision")
            validate_time_grid(np.concatenate(([context["last_time"]], times)), context["dt"])
            if return_warmup:
                times = np.concatenate((history_times[max(plan.leads) :], times))
                validate_time_grid(times, context["dt"])
            if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
                raise ValueError("Forecast timestamps must remain finite and strictly increasing")
        default_route = (
            len(plan.routes) == 1
            and plan.routes[0] == FeedbackRoute(0, 0, 1)
            and all(lead == 1 for lead in plan.leads)
        )
        final_states = None
        with evaluating(self.graph):
            if execution == "fast" and default_route and self._feedback_scaling_matches():
                output = self._backend.forecast(
                    self.graph,
                    self._scaled_inputs(values),
                    drivers or None,
                    horizon=horizon,
                    return_warmup=return_warmup,
                    compile=compile,
                )
            else:
                executor = ForecastSequenceExecutor(self.graph, execution, compile=compile)

                def bridge(route: FeedbackRoute, value: torch.Tensor) -> torch.Tensor:
                    if self.preprocessing is None:
                        return value
                    output_scaler = cast(
                        FittedNormalization, self.output_scalers[route.output_index]
                    )
                    input_scaler = cast(FittedNormalization, self.input_scalers[route.input_index])
                    return cast(torch.Tensor, input_scaler(output_scaler.inverse(value)))

                output, final_states = feedback_rollout(
                    executor,
                    self._scaled_inputs(values),
                    drivers,
                    plan,
                    bridge,
                    horizon=horizon,
                    return_warmup=return_warmup,
                )
        result = self._outputs(output, was_2d)
        if not trajectory_output:
            if final_states is not None:
                self._commit_forecast_states(final_states)
            return result
        assert context is not None and isinstance(result, torch.Tensor)
        # NumPy has no bfloat16 dtype. Conversion is confined to this explicit
        # CPU result boundary; tensor forecasts retain their device and dtype.
        numpy_dtype = torch.float32 if result.dtype == torch.bfloat16 else result.dtype
        array = result.detach().to(device="cpu", dtype=numpy_dtype).numpy()
        trajectory = forecast_trajectory(array, context, system=system, times=times)
        if final_states is not None:
            self._commit_forecast_states(final_states)
        return trajectory

    def get_extra_state(self) -> dict[str, Any]:
        """Checkpoint metadata prevents silent remapping of equal-width ports."""
        return {
            "version": 1,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "input_widths": self._input_widths,
            "output_widths": self._output_widths,
            "backend": copy.deepcopy(self._backend_schema),
            "preprocessing": self.preprocessing,
            "washout": self.washout,
            "prediction_lead": self.prediction_leads,
            "feedback": (
                None
                if self._feedback is None
                else [
                    {"output": route.output, "input": route.input, "delay": route.delay}
                    for route in self._feedback
                ]
            ),
            "resample_dt": self.resample_dt,
            "resample_method": self.resample_method,
            "trajectory": copy.deepcopy(self._trajectory_context),
        }

    def _checkpoint_feedback(
        self, state: dict[str, Any]
    ) -> tuple[tuple[int, ...], tuple[Feedback, ...] | None]:
        leads = prediction_leads(state.get("prediction_lead", 1), self.output_names)
        saved = state.get("feedback")
        specs = None
        if saved is not None:
            if not isinstance(saved, (list, tuple)) or any(
                not isinstance(route, dict) or set(route) != {"output", "input", "delay"}
                for route in saved
            ):
                raise ValueError("feedback must contain output/input/delay connection records")
            specs = tuple(
                Feedback(route["output"], route["input"], route["delay"]) for route in saved
            )
        feedback_plan(
            specs,
            leads,
            self.input_names,
            self.output_names,
            self._input_widths,
            self._output_widths,
            validate_widths=specs is not None,
        )
        return leads, specs

    def _validate_metadata(self, state: Any) -> None:
        if not isinstance(state, dict) or state.get("version") != 1:
            raise RuntimeError("Invalid Model checkpoint metadata/version")
        if state.get("backend", {"kind": "symbolic"}) != self._backend_schema:
            raise RuntimeError(
                "Model checkpoint backend schema differs from this graph; reconstruct "
                "matching backend geometry, members and aggregation"
            )
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
        try:
            self._checkpoint_feedback(state)
        except (ValueError, TypeError) as error:
            raise RuntimeError(f"Invalid Model checkpoint feedback: {error}") from error
        try:
            validate_resampling_options(
                state.get("resample_dt"), state.get("resample_method", "linear")
            )
            if state.get("trajectory") is not None:
                context = validate_trajectory_context(state["trajectory"])
                if (
                    len(self.input_names) != 1
                    or len(self.output_names) != 1
                    or self._input_widths != self._output_widths
                    or len(context["variables"]) != self._input_widths[0]
                ):
                    raise ValueError("trajectory schema differs from the declared graph")
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Invalid Model checkpoint trajectory configuration: {error}"
            ) from error

    def set_extra_state(self, state: Any) -> None:
        """Restore workflow configuration after validating the named graph schema."""
        self._validate_metadata(state)
        self._preprocessing, self.washout = state["preprocessing"], state["washout"]
        self._prediction_leads, self._feedback = self._checkpoint_feedback(state)
        self.resample_dt, self.resample_method = validate_resampling_options(
            state.get("resample_dt"), state.get("resample_method", "linear")
        )
        self._trajectory_context = (
            validate_trajectory_context(state["trajectory"])
            if state.get("trajectory") is not None
            else None
        )
        object.__setattr__(self, "_trajectory_system", None)

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
        if metadata.get("trajectory") is not None and not fitted_flag:
            raise RuntimeError("Invalid Model checkpoint: trajectory history requires fitted model")
        if fitted_flag:
            for path, _ in self._backend.fitted_readouts(self.graph):
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
        if metadata.get("trajectory") is not None and (
            staged[0].ndim != 2 or canonical[0].shape[1] > metadata["trajectory"]["n_samples"]
        ):
            raise RuntimeError("Invalid Model checkpoint trajectory history layout/count")
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
