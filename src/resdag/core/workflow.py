"""Named functional models with a fitted, standard PyTorch workflow.

The existing symbolic, parallel and coupled model families share this workflow.
Algebraic symbolic fitting requires one distinct terminal readout per output.
"""

from __future__ import annotations

import copy
import operator
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

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
from resdag.core._workflow_transforms import (
    apply_transform,
    checkpoint_transform,
    commit_transform_state,
    stage_transform,
    transform_schema,
    validate_custom_transform,
    validate_transform_ownership,
)
from resdag.core.model import ESNModel
from resdag.data._trajectory import (
    is_trajectory,
    trajectory_type,
    validate_resampling_options,
)
from resdag.data._trajectory_ports import (
    forecast_trajectory_ports,
    has_trajectory_port,
    prepare_trajectory_port_drivers,
    prepare_trajectory_port_history,
    prepare_trajectory_ports,
    trajectory_port_forecast_times,
    validate_trajectory_ports_context,
)
from resdag.data.prepare import NormMethod
from resdag.layers.readouts import AutoReadoutLayer
from resdag.layers.reservoirs import BaseReservoirLayer

if TYPE_CHECKING:
    from tsdynamics.data import Trajectory


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
    time-aligned targets; all streams include the same washout prefix. Declared intermediate heads may feed
    later reservoirs; ``readout_targets`` supervises internal fitted heads
    without adding them to public outputs.

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
    reconstruct the same DAG before loading. ``tune`` compares candidates and
    installs a refitted winner; its report exposes the Optuna study. Complete
    ``save("run.pt")`` and ``rd.load`` preserve the complete model; diagram
    suffixes export its declared structure without running it.

    With the optional ``dynamics`` extra, bare or named TSDynamics trajectories
    use one shared observed clock. Explicit targets are aligned to each head's
    physical prediction lead. Forecasts return one trajectory or a named
    dictionary, with target component names and shared future times.
    ``return_type="tensor"`` keeps predictions on the model device.
    Ordinary ``forward`` always uses Torch tensors.
    Uneven timestamps are automatically interpolated with ``resample_method``
    (linear, PCHIP or cubic); ``resample_dt`` controls the physical step.
    """

    _fitted: torch.Tensor
    _trajectory_system: tuple[Any, ...] | None

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
        input_transforms: Mapping[str, nn.Module] | None = None,
        output_transforms: Mapping[str, nn.Module] | None = None,
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
        overrides = []
        for mapping, names in (
            (input_transforms, self.input_names),
            (output_transforms, self.output_names),
        ):
            if mapping is not None and (
                not isinstance(mapping, Mapping) or set(mapping) - set(names)
            ):
                raise ValueError(
                    f"Transform override keys must be declared port names: {list(names)}"
                )
            overrides.append(dict(mapping or {}))
        self._custom_inputs = tuple(name in overrides[0] for name in self.input_names)
        self._custom_outputs = tuple(name in overrides[1] for name in self.output_names)
        originals = tuple(module for mapping in overrides for module in mapping.values())
        if any(not isinstance(module, nn.Module) for module in originals):
            raise TypeError("Transform overrides must be torch.nn.Module instances")
        validate_transform_ownership(graph, originals)
        for mapping, names, widths in (
            (overrides[0], self.input_names, self._input_widths),
            (overrides[1], self.output_names, self._output_widths),
        ):
            for name, width in zip(names, widths):
                if name in mapping:
                    validate_custom_transform(mapping[name], width)
        self.input_scalers = nn.ModuleList(
            overrides[0].get(name, FittedNormalization(method)) for name in self.input_names
        )
        self.output_scalers = nn.ModuleList(
            overrides[1].get(name, FittedNormalization(method)) for name in self.output_names
        )
        reference = module_reference(graph)
        if reference is not None:
            self.input_scalers.to(device=reference.device, dtype=reference.dtype)
            self.output_scalers.to(device=reference.device, dtype=reference.dtype)
        self._transform_schema = self._custom_transform_schema()
        for group, masks, widths in (
            ("input", self._custom_inputs, self._input_widths),
            ("output", self._custom_outputs, self._output_widths),
        ):
            for index, (custom, width) in enumerate(zip(masks, widths)):
                if custom:
                    with torch.inference_mode(False):
                        sample = (
                            torch.zeros(1, 3, width)
                            if reference is None
                            else reference.new_zeros(1, 3, width)
                        )
                    self.register_buffer(f"_{group}_transform_sample_{index}", sample)
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
    def preprocessing(self) -> NormMethod | None:
        """The fallback policy; explicit transform overrides apply independently."""
        return cast(NormMethod | None, self._preprocessing)

    def _custom_transform_schema(self) -> tuple[Any, ...]:
        return tuple(
            tuple(
                transform_schema(module, width) if custom else None
                for module, custom, width in zip(modules, masks, widths)
            )
            for modules, masks, widths in (
                (self.input_scalers, self._custom_inputs, self._input_widths),
                (self.output_scalers, self._custom_outputs, self._output_widths),
            )
        )

    def _transform_active(self, *, output: bool, index: int) -> bool:
        return (
            self.preprocessing is not None
            or (self._custom_outputs if output else self._custom_inputs)[index]
        )

    def _validate_transform_ownership(self) -> None:
        validate_transform_ownership(
            self.graph,
            tuple(
                module
                for modules, masks in (
                    (self.input_scalers, self._custom_inputs),
                    (self.output_scalers, self._custom_outputs),
                )
                for module, custom in zip(modules, masks)
                if custom
            ),
        )

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

    @property
    def internal_readout_names(self) -> tuple[str, ...]:
        """Target keys for internal fitted heads, without exposing extra predictions."""
        self._validate_ports()
        self._backend.validate_fit(self.graph)
        return tuple(name for name, _ in self._backend.internal_readouts(self.graph))

    def _requires_complete_history(self) -> bool:
        if self._backend.kind == "parallel":
            return False
        try:
            validate_forecast_execution(self.graph, "fast")
        except ValueError:
            return True
        return False

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
        if (
            self.preprocessing is not None or any((*self._custom_inputs, *self._custom_outputs))
        ) and not self.is_fitted:
            raise RuntimeError("Model preprocessing is not fitted; call fit(training) first")

    def _validate_ports(self) -> None:
        if self._custom_transform_schema() != self._transform_schema:
            raise ValueError(
                "Custom transform registered structure changed; construct a matching Model"
            )
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
        return tuple(
            (
                apply_transform(scaler, value)
                if self._transform_active(output=False, index=index)
                else value
            )
            for index, (scaler, value) in enumerate(zip(self.input_scalers, inputs))
        )

    def _outputs(self, output: ModelOutput, two_dimensional: bool) -> WorkflowOutput:
        values = output if isinstance(output, tuple) else (output,)
        if len(values) != len(self.output_names):
            raise RuntimeError("Graph output count no longer matches the declared named outputs")
        restored: list[torch.Tensor] = []
        for scaler, value in zip(self.output_scalers, values):
            value = (
                apply_transform(scaler, value, inverse=True)
                if self._transform_active(output=True, index=len(restored))
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

    def plot_model(
        self,
        *,
        save_path: str | Path | None = None,
        format: str | None = None,
        show_shapes: bool = True,
        show_trainable: bool = False,
        rankdir: str = "LR",
    ) -> Any:
        """Display a DAG in notebooks or export SVG/PNG/PDF/DOT without running it."""
        from resdag.core._diagram import plot_model

        self._validate_ports()
        return plot_model(
            self,
            save_path=save_path,
            format=format,
            show_shapes=show_shapes,
            show_trainable=show_trainable,
            rankdir=rankdir,
        )

    @property
    def save_formats(self) -> tuple[str, ...]:
        """Supported complete-model checkpoint and diagram suffixes."""
        return ("pt", "pth", "svg", "png", "pdf", "dot")

    def save(
        self,
        path: str | Path,
        *,
        format: str | None = None,
        include_state: bool = True,
        trusted: bool = False,
        metadata: Mapping[str, Any] | None = None,
        **diagram_options: Any,
    ) -> Any:
        """Save a complete loadable model (.pt/.pth) or a diagram by suffix.

        ``rd.load(path)`` reconstructs checkpoints without a prebuilt DAG.
        Current ordinary-forward memory is included by default. Set
        ``trusted=True`` only to bundle notebook-defined Python code; loading
        such a bundle also requires explicit trust. Diagram formats are
        SVG/PNG/PDF/DOT and accept :meth:`plot_model` options.
        """
        destination = Path(path)
        suffix = destination.suffix.lower().lstrip(".")
        if format is not None and not isinstance(format, str):
            raise TypeError("format must be a string or None")
        selected = format.lower() if isinstance(format, str) else suffix or "pt"
        if selected not in self.save_formats or (suffix and suffix != selected):
            raise ValueError(f"Save format/suffix must agree and be one of {self.save_formats}")
        if not suffix:
            destination = destination.with_suffix(f".{selected}")
        if selected not in ("pt", "pth"):
            if trusted or metadata is not None or not include_state:
                raise ValueError(
                    "trusted, metadata and include_state options apply to model checkpoints"
                )
            return self.plot_model(save_path=destination, format=selected, **diagram_options)
        if diagram_options:
            raise TypeError(
                f"Diagram options cannot be used with a model checkpoint: {list(diagram_options)}"
            )
        from resdag.artifacts import save_model

        self._validate_ports()
        return save_model(
            self, destination, include_state=include_state, trusted=trusted, metadata=metadata
        )

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
    ) -> Model:
        """Tune autonomous forecasts, refit the winner and return this model.

        Requires ``resdag[hpo]``. The final 20% is held out while comparing
        candidates, then the winner is refitted on all supplied data. Keep a
        separate final test set. ``factory(params, seed)`` rebuilds custom DAGs;
        without it, sizes and weights stay fixed and alpha/washout can vary.
        ``search_space(trial)`` uses ordinary Optuna suggestions. Advanced
        options and :attr:`tuning_report` are documented in the tuning guide.
        A failure leaves this model unchanged. Tune before creating optimizers.
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

    def fit(
        self,
        training: Streams | Trajectory | Mapping[str, Trajectory],
        targets: Streams | Trajectory | Mapping[str, Trajectory] | None = None,
        *,
        readout_targets: Mapping[str, torch.Tensor] | None = None,
    ) -> Model:
        """Fit public and internal heads transactionally and retain the original observed tail.

        Implicit supervision pairs input ``t`` with observation
        ``t + prediction_lead`` and retains the complete observed tail.
        Explicit targets are already caller-aligned to input indices; declared
        leads describe their physical timestamps without shifting those arrays.
        Successful fitting resets streaming state. Failed fitting leaves the
        existing graph, fitted preprocessing and cached history unchanged.
        Bare or named TSDynamics trajectories share one physical input grid.
        Explicit trajectory targets are sampled at origin + prediction_lead*dt;
        tensors remain caller-aligned. Regression uses jointly supported origins
        after absolute washout. Target source observations after the common input
        endpoint are held out before interpolation. The cached tail includes the
        final observed inputs, even after the last supervised origin. Tensor
        refitting clears trajectory defaults on success.

        ``readout_targets={"stage1": tensor}`` supervises internal readouts
        without exposing them as outputs. Keys use the readout's ``name`` or
        graph path; coupled members use ``models.<index>.<name>``. Tensors share
        the complete input time axis, including washout, and express the desired
        internal output units without automatic output scaling. For trajectory
        inputs, align these tensors to the prepared shared input grid.
        """
        self._validate_ports()
        implicit = targets is None
        if is_trajectory(training) or has_trajectory_port(training):
            prepared = prepare_trajectory_ports(
                training,
                input_names=self.input_names,
                input_widths=self._input_widths,
                output_names=self.output_names,
                output_widths=self._output_widths,
                prediction_leads=self._prediction_leads,
                targets=targets,
                dt=self.resample_dt,
                method=self.resample_method,
            )
            reference = module_reference(self.graph)
            dtype = None if reference is None else reference.dtype
            device = None if reference is None else reference.device
            inputs = {
                name: torch.tensor(data, dtype=dtype, device=device)
                for name, data in zip(self.input_names, prepared.inputs)
            }
            labels = {
                name: torch.tensor(data, dtype=dtype, device=device)
                for name, data in zip(self.output_names, prepared.targets)
            }
            return self._fit_prepared(
                inputs,
                labels,
                supervised_start=prepared.supervised_start,
                supervised_stop=prepared.supervised_stop,
                implicit=implicit,
                trajectory_context=prepared.context,
                trajectory_system=prepared.output_systems,
                readout_targets=readout_targets,
            )
        if is_trajectory(targets) or has_trajectory_port(targets):
            raise TypeError(
                "Trajectory targets require clocked Trajectory training inputs; "
                "Tensor targets remain caller-aligned"
            )
        values, was_2d = self._streams(
            cast(Streams, training), self.input_names, self._input_widths, "training", finite=True
        )
        stop = values[0].shape[1]
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
            stop -= lead
            labels = {
                self.output_names[0]: (
                    values[0][:, lead:].squeeze(0) if was_2d else values[0][:, lead:]
                )
            }
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
            labels = {
                name: value.squeeze(0) if was_2d else value
                for name, value in zip(self.output_names, target_values)
            }
        return self._fit_prepared(
            cast(Streams, training),
            labels,
            supervised_start=0,
            supervised_stop=stop,
            implicit=implicit,
            readout_targets=readout_targets,
        )

    def _fit_prepared(
        self,
        training: Streams,
        targets: Streams,
        *,
        supervised_start: int,
        supervised_stop: int,
        implicit: bool = False,
        readout_targets: Mapping[str, torch.Tensor] | None = None,
        trajectory_context: dict[str, Any] | None = None,
        trajectory_system: tuple[Any, ...] | None = None,
    ) -> Model:
        """Fit an admitted origin window, retaining every observed input row.

        Private adapters supply already aligned labels for exactly
        ``supervised_start:supervised_stop``. Absolute washout is applied once.
        This separates regression support from the complete observed clock;
        purging future labels must not move the cached forecasting boundary.
        """
        self._validate_ports()
        self._backend.validate_fit(self.graph)
        self._validate_transform_ownership()
        complete_history = self._requires_complete_history()
        values, was_2d = self._streams(
            training, self.input_names, self._input_widths, "training", finite=True
        )
        if (
            type(supervised_start) is not int
            or type(supervised_stop) is not int
            or not 0 <= supervised_start < supervised_stop <= values[0].shape[1]
        ):
            raise ValueError(
                "Supervised origins must be a nonempty interval within observed inputs"
            )
        target_values, targets_2d = self._streams(
            targets, self.output_names, self._output_widths, "targets", finite=True
        )
        if targets_2d != was_2d or any(
            value.shape[:2] != (values[0].shape[0], supervised_stop - supervised_start)
            for value in target_values
        ):
            raise ValueError("Prepared targets must match the admitted origin window and layout")
        internal_heads = self._backend.internal_readouts(self.graph)
        internal_targets: dict[str, torch.Tensor] = {}
        if internal_heads:
            if readout_targets is None:
                raise ValueError(
                    "Provide readout_targets for internal fitted readouts; "
                    f"expected {[name for name, _ in internal_heads]}"
                )
            internal_values, internal_2d = self._streams(
                readout_targets,
                tuple(name for name, _ in internal_heads),
                tuple(head.out_features for _, head in internal_heads),
                "readout_targets",
                finite=True,
            )
            if internal_2d != was_2d or any(
                value.shape[:2] != values[0].shape[:2] for value in internal_values
            ):
                raise ValueError("readout_targets must share the complete input batch/time axes")
            internal_targets = dict(zip((name for name, _ in internal_heads), internal_values))
        elif readout_targets is not None and (
            not isinstance(readout_targets, Mapping) or len(readout_targets)
        ):
            raise ValueError("This graph has no internal fitted readouts for readout_targets")
        fit_start = max(self.washout, supervised_start)
        train_inputs = tuple(value[:, :supervised_stop] for value in values)
        target_values = tuple(value[:, fit_start - supervised_start :] for value in target_values)
        if train_inputs[0].shape[1] <= fit_start:
            raise ValueError("Training must contain washout plus at least one input/target pair")
        fitting_inputs = tuple(value[:, fit_start:] for value in train_inputs)
        # Stage statistics and algebraic fitting on independent objects. Committing
        # weights into the original graph preserves existing optimizer references.
        with torch.inference_mode(False):
            input_scalers = [copy.deepcopy(module) for module in self.input_scalers]
            output_scalers = [copy.deepcopy(module) for module in self.output_scalers]
        samples: dict[str, torch.Tensor] = {}
        for group, staged_scalers, live_scalers, masks, fitting in (
            ("input", input_scalers, self.input_scalers, self._custom_inputs, fitting_inputs),
            ("output", output_scalers, self.output_scalers, self._custom_outputs, target_values),
        ):
            for index, (staged_scaler, original, custom, data) in enumerate(
                zip(staged_scalers, live_scalers, masks, fitting)
            ):
                if custom:
                    staged_scalers[index], samples[f"_{group}_transform_sample_{index}"] = (
                        stage_transform(original, data)
                    )
                elif self.preprocessing is not None:
                    # Unspecified implicit outputs retain the original input
                    # supervised window even when their input has a custom map.
                    source = fitting_inputs[0] if group == "output" and implicit else data
                    cast(FittedNormalization, staged_scaler).fit(source)

        def transformed(
            modules: list[nn.Module], streams: tuple[torch.Tensor, ...], *, output: bool
        ) -> tuple[torch.Tensor, ...]:
            result = []
            for index, (module, data) in enumerate(zip(modules, streams)):
                with evaluating(module), torch.no_grad():
                    value = (
                        apply_transform(module, data)
                        if self._transform_active(output=output, index=index)
                        else data
                    )
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("Preprocessing produced nonfinite training values")
                result.append(value)
            return tuple(result)

        scaled = transformed(input_scalers, train_inputs, output=False)
        scaled_targets = transformed(output_scalers, target_values, output=True)
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
            self._backend.fit_prefix(
                candidate,
                scaled,
                scaled_targets,
                fit_start=fit_start,
                readout_targets={
                    name: value[:, fit_start:supervised_stop]
                    for name, value in internal_targets.items()
                },
            )
        history = []
        tail_length = self.washout + self._feedback_plan(validate_widths=False).required_history
        with torch.inference_mode(False), torch.no_grad():
            for value in values:
                tail = (value if complete_history else value[:, -tail_length:]).detach().clone()
                history.append(tail.squeeze(0) if was_2d else tail)
        # Solve reports are intentionally not checkpoint state. Transfer them
        # from the successful staged fit by graph path without replacing live
        # readouts/parameters, or losing diagnostics at the state-dict commit.
        staged_reports = {
            path: layer.last_solve
            for path, layer in candidate.named_modules()
            if isinstance(layer, AutoReadoutLayer)
        }
        live_auto_readouts = {
            path: layer
            for path, layer in self.graph.named_modules()
            if isinstance(layer, AutoReadoutLayer)
        }
        if staged_reports.keys() != live_auto_readouts.keys():
            raise RuntimeError("Staged automatic readout paths do not match the live graph")
        with torch.inference_mode(False):
            self.graph.load_state_dict(candidate.state_dict())
        recurrent_graph(self.graph).reset_reservoirs()
        for staged_scalers, previous, masks in (
            (input_scalers, self.input_scalers, self._custom_inputs),
            (output_scalers, self.output_scalers, self._custom_outputs),
        ):
            for staged_scaler, original, custom in zip(staged_scalers, previous, masks):
                if custom:
                    commit_transform_state(original, staged_scaler)
                else:
                    with torch.inference_mode(False):
                        original.load_state_dict(staged_scaler.state_dict())
        for name, sample in samples.items():
            setattr(self, name, sample)
        for index, tail in enumerate(history):
            setattr(self, f"_history_{index}", tail)
        self._fitted_flag = True
        with torch.inference_mode(False):
            self._fitted = torch.ones((), dtype=torch.bool, device=values[0].device)
        self._trajectory_context = trajectory_context
        object.__setattr__(self, "_trajectory_system", trajectory_system)
        for path, layer in live_auto_readouts.items():
            layer.last_solve = staged_reports[path]
        return self

    def _feedback_scaling_matches(self, input_index: int = 0, output_index: int = 0) -> bool:
        if any((*self._custom_inputs, *self._custom_outputs)):
            return False
        if self.preprocessing is None:
            return True
        left, right = cast(FittedNormalization, self.input_scalers[input_index]), cast(
            FittedNormalization, self.output_scalers[output_index]
        )
        return left.method == right.method and all(
            torch.equal(a, b) for a, b in zip(left.buffers(), right.buffers())
        )

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
    ) -> WorkflowOutput | Trajectory | dict[str, Trajectory]:
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
        cached_history = history is None
        clocked_history = is_trajectory(history) or has_trajectory_port(history)
        if clocked_history:
            prepared = prepare_trajectory_port_history(
                history,
                input_names=self.input_names,
                input_widths=self._input_widths,
                output_names=self.output_names,
                output_widths=self._output_widths,
                context=context,
                dt=self.resample_dt,
                method=self.resample_method,
                output_systems=system,
            )
            context, system = prepared.context, prepared.output_systems
            reference = module_reference(self.graph)
            history = {
                name: torch.tensor(
                    data,
                    dtype=None if reference is None else reference.dtype,
                    device=None if reference is None else reference.device,
                )
                for name, data in zip(self.input_names, prepared.inputs)
            }
        trajectory_output = return_type == "trajectory" or (
            return_type == "auto" and context is not None
        )
        if trajectory_output:
            if context is None or (not cached_history and not clocked_history):
                raise ValueError(
                    "Trajectory output needs timestamped trajectory history or cached "
                    "trajectory fitting history; pass a Trajectory or return_type='tensor'"
                )
            if set(context["outputs"]) != set(self.output_names):
                raise ValueError(
                    "Trajectory output needs per-head target component metadata; "
                    "fit Trajectory targets or choose return_type='tensor'"
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
            if is_trajectory(future_inputs) or has_trajectory_port(future_inputs):
                if context is None or (not cached_history and not clocked_history):
                    raise ValueError(
                        "Trajectory drivers require timestamped or cached trajectory history; "
                        "new Tensor history has no absolute clock"
                    )
                arrays = prepare_trajectory_port_drivers(
                    future_inputs,
                    input_names=driver_names,
                    input_widths=driver_widths,
                    last_time=context["clock"]["last_time"],
                    dt=context["clock"]["dt"],
                    steps=steps,
                    method=self.resample_method,
                    input_records=context["inputs"],
                )
                future_inputs = {
                    name: torch.tensor(data, device=values[0].device, dtype=values[0].dtype)
                    for name, data in zip(driver_names, arrays)
                }
            drivers, _ = self._streams(
                cast(Streams, future_inputs),
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
        times = None
        if trajectory_output:
            assert context is not None
            times = trajectory_port_forecast_times(
                context,
                horizon,
                history_length=values[0].shape[1],
                warmup_lead=max(plan.leads),
                return_warmup=return_warmup,
            )
        default_route = (
            len(plan.routes) == 1
            and plan.routes[0] == FeedbackRoute(0, 0, 1)
            and all(lead == 1 for lead in plan.leads)
        )
        final_states = None
        with evaluating(self):
            if drivers and drivers[0].shape[1]:
                drivers = tuple(
                    (
                        apply_transform(self.input_scalers[index], value)
                        if self._transform_active(output=False, index=index)
                        else value
                    )
                    for index, value in zip(plan.driver_indices, drivers)
                )
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
                    if self._transform_active(output=True, index=route.output_index):
                        value = apply_transform(
                            self.output_scalers[route.output_index], value, inverse=True
                        )
                    if self._transform_active(output=False, index=route.input_index):
                        value = apply_transform(self.input_scalers[route.input_index], value)
                    return value

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
        assert context is not None and times is not None
        # NumPy has no bfloat16 dtype. Conversion is confined to this explicit
        # CPU result boundary; tensor forecasts retain their device and dtype.
        result_values = (
            {self.output_names[0]: result} if isinstance(result, torch.Tensor) else result
        )
        result_arrays = {}
        for name, value in result_values.items():
            numpy_dtype = torch.float32 if value.dtype == torch.bfloat16 else value.dtype
            result_arrays[name] = value.detach().to(device="cpu", dtype=numpy_dtype).numpy()
        trajectory = forecast_trajectory_ports(
            result_arrays,
            context,
            output_names=self.output_names,
            output_systems=system,
            times=times,
        )
        if final_states is not None:
            self._commit_forecast_states(final_states)
        return trajectory

    def get_extra_state(self) -> dict[str, Any]:
        """Checkpoint metadata prevents silent remapping of equal-width ports."""
        return {
            "version": 2,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "input_widths": self._input_widths,
            "output_widths": self._output_widths,
            "backend": copy.deepcopy(self._backend_schema),
            "preprocessing": self.preprocessing,
            "transforms": copy.deepcopy(self._transform_schema),
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
        leads = prediction_leads(state["prediction_lead"], self.output_names)
        saved = state["feedback"]
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
        if (
            not isinstance(state, dict)
            or type(state.get("version")) is not int
            or state["version"] != 2
        ):
            raise RuntimeError(
                "Model checkpoint requires current metadata version 2; "
                "reconstruct and refit models saved with older metadata"
            )
        if set(state) != {
            "version",
            "input_names",
            "output_names",
            "input_widths",
            "output_widths",
            "backend",
            "preprocessing",
            "transforms",
            "washout",
            "prediction_lead",
            "feedback",
            "resample_dt",
            "resample_method",
            "trajectory",
        }:
            raise RuntimeError(
                "Invalid or unsupported older Model checkpoint metadata fields; "
                "reconstruct and refit the model"
            )
        if state["backend"] != self._backend_schema:
            raise RuntimeError(
                "Model checkpoint backend schema differs from this graph; reconstruct "
                "matching backend geometry, members and aggregation"
            )
        if state.get("transforms") != self._transform_schema:
            raise RuntimeError(
                "Model checkpoint custom transform schema differs; reconstruct matching port modules"
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
            validate_resampling_options(state["resample_dt"], state["resample_method"])
            if state.get("trajectory") is not None:
                validate_trajectory_ports_context(
                    state["trajectory"],
                    self.input_names,
                    self._input_widths,
                    self.output_names,
                    self._output_widths,
                    self._checkpoint_feedback(state)[0],
                )
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
            state["resample_dt"], state["resample_method"]
        )
        self._trajectory_context = (
            validate_trajectory_ports_context(
                state["trajectory"],
                self.input_names,
                self._input_widths,
                self.output_names,
                self._output_widths,
                self._prediction_leads,
            )
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
        self._validate_transform_ownership()
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
        custom_state: dict[str, torch.Tensor] = {}
        # Standalone normalization restores unknown widths/rules. The surrounding
        # named graph fixes both; preflight all scalers before mutating weights.
        for group, scalers, widths, masks in (
            ("input_scalers", self.input_scalers, self._input_widths, self._custom_inputs),
            ("output_scalers", self.output_scalers, self._output_widths, self._custom_outputs),
        ):
            for index, (scaler, width, custom) in enumerate(zip(scalers, widths, masks)):
                stem = prefix + f"{group}.{index}."
                if custom:
                    sample_name = f"_{'input' if group == 'input_scalers' else 'output'}_transform_sample_{index}"
                    sample = state_dict.get(prefix + sample_name)
                    destination_sample = getattr(self, sample_name)
                    assign = local_metadata.get("assign_to_params_buffers", False)
                    if (
                        isinstance(sample, torch.Tensor)
                        and not assign
                        and sample.device.type != "meta"
                    ):
                        with torch.inference_mode(False), torch.no_grad():
                            sample = sample.to(destination_sample, copy=True)
                    try:
                        clean, sample = checkpoint_transform(
                            scaler,
                            {
                                key[len(stem) :]: value
                                for key, value in state_dict.items()
                                if key.startswith(stem)
                            },
                            sample,
                            width,
                            fitted=fitted_flag,
                            assign=assign,
                        )
                    except (ValueError, TypeError, RuntimeError) as error:
                        raise RuntimeError(
                            f"Invalid Model checkpoint custom transform {group}.{index}: {error}"
                        ) from error
                    custom_state.update({stem + key: value for key, value in clean.items()})
                    custom_state[prefix + sample_name] = sample
                    continue
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
            staged[0].ndim != 2
            or canonical[0].shape[1] > metadata["trajectory"]["clock"]["n_samples"]
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
            # The recursive Module loader consumes this same local mapping for
            # child tensors. Install only fully preflighted normal tensors.
            state_dict.update(custom_state)
            for name, value in custom_state.items():
                local_name = name.removeprefix(prefix)
                if local_name.startswith(("_input_transform_sample_", "_output_transform_sample_")):
                    setattr(self, local_name, torch.empty_like(value))
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
