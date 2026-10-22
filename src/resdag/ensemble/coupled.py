"""
Coupled Ensemble ESN Model
==========================

This module provides :class:`CoupledEnsembleESNModel`, an ensemble of N
independently-trained ESN models whose forecasts are coupled through a shared
averaged feedback signal at each autoregressive step.

See Also
--------
resdag.models.coupled_ensemble_esn : Factory function to build this ensemble.
resdag.core.ESNModel : Base ESN model used as sub-models.
"""

from collections.abc import Iterator, Mapping
from itertools import chain
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

from resdag.core import ESNModel
from resdag.core._execution import ReservoirStates, _validate_unique_reservoir_invocations
from resdag.core._flat_inference import FlatStep
from resdag.core._forecast import ForecastAlignment, prepare_forecast_inputs
from resdag.core._persistence import (
    load_checkpoint,
    load_full_model,
    save_checkpoint,
    save_full_model,
)
from resdag.core.model import ReservoirModelModule
from resdag.layers.reservoirs import BaseReservoirLayer
from resdag.training import ESNTrainer
from resdag.training.trainer import InputStreams, ReadoutTargets


class CoupledEnsembleESNModel(nn.Module):
    """Ensemble of N independently-trained ESN models with coupled feedback.

    Each sub-model is a complete :class:`~resdag.core.ESNModel` built
    via the ``pytorch_symbolic`` API. The models are trained independently but
    coupled during forecasting: at every autoregressive step every model
    receives the **same** aggregated output (e.g. the mean across models) as
    its next feedback input.

    Parameters
    ----------
    models : list of ESNModel
        N pre-built ESN sub-models.  Diversity comes from their independent
        random reservoir initialization.
    aggregator : str or nn.Module, default ``"mean"``
        How to combine the N per-model outputs into a single feedback tensor.

        - ``"mean"`` — arithmetic mean across models.
        - ``"median"`` — interpolated (statistical) median across models;
          for an even number of models the two central values are averaged.
        - Any ``nn.Module`` that accepts a stacked tensor of shape
          ``(N, batch, timesteps, features)`` and returns
          ``(batch, timesteps, features)``.  E.g.
          :class:`~resdag.ensemble.aggregators.OutliersFilteredMean`.

    Examples
    --------
    Typical use via the factory:

    >>> from resdag.models import coupled_ensemble_esn
    >>> ensemble = coupled_ensemble_esn(n_models=5, reservoir_size=300,
    ...                                feedback_size=3, output_size=3)
    >>> ensemble.fit((warmup,), (train,), {"output": targets})
    >>> ensemble.reset_reservoirs()
    >>> preds = ensemble.forecast(forecast_warmup, horizon=200)
    >>> preds.shape
    torch.Size([1, 200, 3])

    With a custom aggregator:

    >>> from resdag.ensemble.aggregators import OutliersFilteredMean
    >>> ensemble = coupled_ensemble_esn(
    ...     n_models=10, reservoir_size=300, feedback_size=3, output_size=3,
    ...     aggregate=OutliersFilteredMean(method="z_score", threshold=2.0),
    ... )

    See Also
    --------
    resdag.models.coupled_ensemble_esn : Convenience factory function.
    resdag.ensemble.aggregators.OutliersFilteredMean : Outlier-robust aggregation layer.
    """

    def __init__(
        self,
        models: list[ESNModel],
        aggregator: str | nn.Module = "mean",
    ) -> None:
        """Initialize the coupled ensemble.

        Parameters
        ----------
        models : list of ESNModel
            Sub-models.  Must be non-empty.
        aggregator : str or nn.Module, default ``"mean"``
            Aggregation strategy.  See class docstring for options.

        Raises
        ------
        ValueError
            If ``models`` is empty, if ``aggregator`` is an unknown string, if
            any sub-model is multi-output (multi-head), or if any sub-model's
            output dimension does not match its feedback (first input)
            dimension.  Multi-output sub-models are rejected because the
            ensemble aggregates the members into a single shared feedback signal
            and has no meaningful coupled aggregation for auxiliary heads.  The
            dimension check rejects headless / linear sub-models, whose output is
            the raw reservoir state rather than a feedback-dimensioned readout,
            since the coupled autoregressive loop feeds the aggregated output
            back as the next feedback input.
        """
        super().__init__()
        if len(models) == 0:
            raise ValueError("CoupledEnsembleESNModel requires at least one sub-model.")
        if isinstance(aggregator, str) and aggregator not in ("mean", "median"):
            raise ValueError(
                f"aggregator must be 'mean', 'median', or an nn.Module; got '{aggregator}'."
            )

        if not isinstance(aggregator, (str, nn.Module)):
            raise TypeError("aggregator must be 'mean', 'median', or an nn.Module")
        for index, model in enumerate(models):
            if not isinstance(model, ESNModel):
                raise TypeError(
                    f"Sub-model {index} must be an ESNModel, got {type(model).__name__}"
                )
        self._validate_feedback_output_dims(models)
        self._validate_member_interfaces(models)

        reservoir_owners: dict[int, int] = {}
        for index, model in enumerate(models):
            for layer in model.modules():
                if isinstance(layer, BaseReservoirLayer):
                    owner = reservoir_owners.get(id(layer))
                    if owner is not None:
                        raise ValueError(
                            f"Ensemble members {owner} and {index} share a reservoir layer instance. "
                            "Members require independent recurrent state. Use distinct reservoir "
                            "layer instances; their parameter tensors may still share weights."
                        )
                    reservoir_owners[id(layer)] = index

        self.models = nn.ModuleList(models)

        if isinstance(aggregator, nn.Module):
            self.aggregator_module: nn.Module | None = aggregator
            self._aggregator_str: str | None = None
        else:
            self.aggregator_module = None
            self._aggregator_str = aggregator

    @staticmethod
    def _validate_member_interfaces(models: list[ESNModel]) -> None:
        """Reject incompatible shared inputs before running any member."""
        expected = tuple(int(node.shape[-1]) for node in models[0].inputs)
        reference: tuple[torch.device, torch.dtype] | None = None
        for index, model in enumerate(models):
            widths = tuple(int(node.shape[-1]) for node in model.inputs)
            if widths != expected:
                raise ValueError(
                    f"Sub-model {index} expects input feature widths {widths}, but shared "
                    f"ensemble inputs must match sub-model 0's widths {expected}."
                )
            tensor = next(
                (
                    item
                    for item in chain(model.parameters(), model.buffers())
                    if item.is_floating_point()
                ),
                None,
            )
            if tensor is not None:
                placement = (tensor.device, tensor.dtype)
                if reference is None:
                    reference = placement
                elif placement != reference:
                    raise ValueError(
                        f"Sub-model {index} uses device={tensor.device}, dtype={tensor.dtype}; "
                        f"all ensemble members must share device={reference[0]}, "
                        f"dtype={reference[1]}. Move/cast members before constructing the ensemble."
                    )

    @staticmethod
    def _validate_feedback_output_dims(models: list[ESNModel]) -> None:
        """Check that every sub-model can be driven autoregressively.

        Two constraints are enforced, both raising a clear, sub-model-naming
        :class:`ValueError` at construction rather than a cryptic failure deep
        inside :meth:`forecast` / :meth:`forward`:

        1. **Single-output only.** The ensemble aggregates the N members'
           outputs (stack + mean/median across members) into a *single* shared
           feedback signal, and the warmup/forecast/aggregation paths treat each
           member's output as one tensor.  A multi-output (multi-head) sub-model
           has no meaningful ensemble aggregation for its auxiliary heads — the
           mean of unrelated per-model aux readouts is not a coupled quantity —
           so such models are rejected here.  (:class:`~resdag.core.ESNModel`'s
           own ``forecast`` *does* support multi-output by feeding back the
           first head; the ensemble deliberately does not, to keep the coupling
           semantics well-defined.)
        2. **Autoregressive coupleability.** The coupled loop feeds each
           sub-model's aggregated output back as its next feedback input, so the
           output dimension must equal the feedback (first input) dimension.
           Headless / linear sub-models emit the raw reservoir state (e.g.
           ``reservoir_size``) instead of a feedback-dimensioned readout and
           would silently break the loop — those are rejected too.

        Parameters
        ----------
        models : list of ESNModel
            Candidate sub-models.

        Raises
        ------
        ValueError
            If any sub-model is multi-output, or if any sub-model's output
            dimension does not match its feedback (first input) dimension —
            naming the offending sub-model and the mismatched dimensions.
        """
        for i, model in enumerate(models):
            input_shape = model.input_shape
            output_shape = model.output_shape

            # Multi-input / multi-output models expose a tuple of torch.Size;
            # the feedback signal is always the first input.
            multi_input = isinstance(input_shape, tuple) and isinstance(input_shape[0], torch.Size)
            feedback_dim = input_shape[0][-1] if multi_input else input_shape[-1]

            multi_output = isinstance(output_shape, tuple) and isinstance(
                output_shape[0], torch.Size
            )
            if multi_output:
                n_heads = len(output_shape)
                head_dims = [int(s[-1]) for s in output_shape]
                raise ValueError(
                    f"Sub-model {i} is multi-output ({n_heads} heads with output "
                    f"dimensions {head_dims}), which CoupledEnsembleESNModel does not "
                    f"support. The ensemble aggregates the members' outputs into a "
                    f"single shared feedback signal (stack + mean/median across "
                    f"models); there is no well-defined coupled aggregation for a "
                    f"sub-model's auxiliary heads. Use single-output sub-models whose "
                    f"one output matches the feedback dimension (e.g. classic_esn, "
                    f"ott_esn, power_augmented). To forecast a multi-head model, use "
                    f"its own ESNModel.forecast(), which feeds back the first head."
                )

            output_dim = output_shape[-1]

            if output_dim != feedback_dim:
                raise ValueError(
                    f"Sub-model {i} is not autoregressively coupleable: its "
                    f"output dimension ({output_dim}) does not match its feedback "
                    f"(first input) dimension ({feedback_dim}). The coupled ensemble "
                    f"feeds each model's aggregated output back as its next feedback "
                    f"input, so they must be equal. Headless / linear factories "
                    f"(headless_esn, linear_esn) emit raw reservoir states rather "
                    f"than a feedback-dimensioned readout and cannot be used here; "
                    f"use a readout-bearing factory such as classic_esn, ott_esn, or "
                    f"power_augmented (or build a sub-model whose output matches "
                    f"the feedback dimension)."
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_models(self) -> int:
        """Number of sub-models in the ensemble."""
        return len(self.models)

    def _iter_models(self) -> Iterator[ESNModel]:
        """Iterate over the sub-models with their concrete :class:`ESNModel` type.

        ``self.models`` is an :class:`torch.nn.ModuleList` (so the sub-models are
        registered for ``state_dict``/``to``), whose element type is the generic
        ``Module``.  This helper re-exposes them as ``ESNModel`` for static
        type-checking; it does not copy or re-register anything.
        """
        for model in self.models:
            yield cast(ESNModel, model)

    # ------------------------------------------------------------------
    # Forward / warmup
    # ------------------------------------------------------------------

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Run all sub-models on the same inputs and return the aggregated output.

        Parameters
        ----------
        *inputs : torch.Tensor
            Same input tensors passed to every sub-model.

        Returns
        -------
        torch.Tensor
            Aggregated output of shape ``(batch, timesteps, output_size)``.
        """
        outputs = [model(*inputs) for model in self.models]
        return self._aggregate(outputs)

    def forward_stateless(
        self,
        *inputs: torch.Tensor,
        states: Mapping[str, torch.Tensor] | None = None,
        compile: bool = False,
    ) -> tuple[torch.Tensor, ReservoirStates]:
        """Aggregate complete member predictions with explicit recurrent state.

        Omitted states start every member from zero. Otherwise use the named
        dictionary returned by this method, whose keys are
        ``models.<member index>.<reservoir name>``. The model's stored states
        remain untouched; returned states retain gradients across chunks.
        """
        names = {
            f"models.{index}.{name}"
            for index, model in enumerate(self._iter_models())
            for name in model._get_execution_plan().reservoir_names
        }
        if states is not None and set(states) != names:
            raise ValueError(
                "Reservoir state keys must match the model; "
                f"missing={sorted(names - set(states))}, "
                f"unexpected={sorted(set(states) - names)}"
            )
        outputs = []
        final: ReservoirStates = {}
        for index, model in enumerate(self._iter_models()):
            prefix = f"models.{index}."
            member_states = (
                None
                if states is None
                else {
                    key.removeprefix(prefix): value
                    for key, value in states.items()
                    if key.startswith(prefix)
                }
            )
            output, member_final = model.forward_stateless(
                *inputs, states=member_states, compile=compile
            )
            outputs.append(cast(torch.Tensor, output))
            final.update({prefix + key: value for key, value in member_final.items()})
        return self._aggregate(outputs), final

    def as_module(self, *, stateful: bool = False) -> ReservoirModelModule:
        """Share all trained members and their aggregator with a PyTorch pipeline."""
        for model in self._iter_models():
            if stateful:
                _validate_unique_reservoir_invocations(model)
            else:
                model._get_execution_plan()
        return ReservoirModelModule(self, stateful=stateful)

    @torch.no_grad()
    def warmup(self, *inputs: torch.Tensor) -> None:
        """Teacher-forced warmup: synchronize every sub-model's reservoir state.

        Parameters
        ----------
        *inputs : torch.Tensor
            Warmup sequences.  Passed identically to each sub-model.
        """
        for model in self._iter_models():
            model.warmup(*inputs)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_reservoirs(self) -> None:
        """Reset reservoir states in all sub-models to zero / None."""
        for model in self._iter_models():
            model.reset_reservoirs()

    def get_reservoir_states(self) -> list[dict[str, torch.Tensor]]:
        """Return reservoir states for all sub-models.

        Returns
        -------
        list of dict
            One dict per sub-model, mapping layer name to state tensor.
        """
        return [model.get_reservoir_states() for model in self._iter_models()]

    def get_state(self) -> ReservoirStates:
        """Return initialized recurrent states in one named dictionary."""
        return {
            f"models.{index}.{name}": state
            for index, model in enumerate(self._iter_models())
            for name, state in model.get_state().items()
        }

    def set_state(self, states: Mapping[str, torch.Tensor], strict: bool = True) -> None:
        """Restore a named state dictionary matching ``forward_stateless``."""
        names = {
            f"models.{index}.{name}"
            for index, model in enumerate(self._iter_models())
            for name, module in model.named_modules()
            if isinstance(module, BaseReservoirLayer)
        }
        if strict and set(states) != names:
            raise KeyError(
                f"Reservoir state keys must match the model; missing={sorted(names - set(states))}, "
                f"unexpected={sorted(set(states) - names)}"
            )
        for index, model in enumerate(self._iter_models()):
            prefix = f"models.{index}."
            member = {
                key.removeprefix(prefix): value
                for key, value in states.items()
                if key.startswith(prefix)
            }
            model.set_state(member, strict=strict)

    def set_reservoir_states(
        self,
        states: list[dict[str, torch.Tensor]],
        strict: bool = True,
    ) -> None:
        """Restore reservoir states in all sub-models.

        Parameters
        ----------
        states : list of dict
            One dict per sub-model as returned by :meth:`get_reservoir_states`.
        strict : bool, default=True
            Forwarded to :meth:`ESNModel.set_reservoir_states`.  When
            ``True`` (default), each sub-model raises on missing/unexpected
            reservoir keys.

        Raises
        ------
        ValueError
            If the number of dicts in ``states`` does not match
            ``self.n_models``.
        """
        if len(states) != self.n_models:
            raise ValueError(
                f"Expected {self.n_models} state dict(s) (one per sub-model), "
                f"got {len(states)}."
            )
        for model, state_dict in zip(self._iter_models(), states):
            model.set_reservoir_states(state_dict, strict=strict)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        warmup_inputs: InputStreams,
        train_inputs: InputStreams,
        targets: ReadoutTargets,
        n_workers: int = 1,
        coerce: bool = False,
    ) -> None:
        """Train all sub-models independently using :class:`~resdag.training.ESNTrainer`.

        Each sub-model is trained separately on the same warmup/train data.
        Ensemble diversity comes from the different random reservoir
        initialisations of each sub-model.

        Before dispatching to :class:`~resdag.training.ESNTrainer`, every
        ``warmup_inputs`` / ``train_inputs`` tensor and every ``targets`` value
        is checked against the ensemble's reference device/dtype (the first
        sub-model's first floating-point parameter; see
        :meth:`_reference_device_dtype`).  A mismatch raises a clear, named
        :class:`ValueError` — e.g. CPU targets fed to a GPU ensemble — instead
        of a raw cross-device ``RuntimeError`` deep inside the readout solve.
        Pass ``coerce=True`` to ``.to()``-coerce the data to the reference
        instead of raising.

        Parameters
        ----------
        warmup_inputs : tuple of torch.Tensor
            Warmup sequences ``(feedback, driver1, ...)``.
            Shape of each: ``(batch, warmup_steps, features)``.
        train_inputs : tuple of torch.Tensor
            Training sequences ``(feedback, driver1, ...)``.
            Shape of each: ``(batch, train_steps, features)``.
        targets : dict of str to torch.Tensor
            Mapping from readout name to target tensor.
            Shape of each target: ``(batch, train_steps, out_features)``.
        n_workers : int, default ``1``
            Number of worker threads used to fit sub-models concurrently.
            ``1`` (the default) runs sequentially in the calling thread.
            Larger values dispatch fits through a
            :class:`concurrent.futures.ThreadPoolExecutor` — PyTorch releases
            the GIL during BLAS, so threading gives a real speed-up for
            CPU-bound CG ridge solves without the pickling cost of
            multiprocessing.  On GPU, all workers share one device and may
            interfere via the same CUDA stream; benchmark before raising
            ``n_workers`` above 1 in that case.
        coerce : bool, default ``False``
            If ``False`` (default), a device/dtype mismatch between any input or
            target and the ensemble's sub-models raises a clear, named
            :class:`ValueError`.  If ``True``, such tensors are coerced with
            ``.to(device=..., dtype=...)`` to the reference instead.

        Raises
        ------
        ValueError
            If ``n_workers < 1``, or if ``coerce=False`` and any input/target
            tensor's device or dtype does not match the ensemble's sub-models.
        """
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}.")

        warmup_inputs = ESNTrainer._input_tuple(warmup_inputs)
        train_inputs = ESNTrainer._input_tuple(train_inputs)
        if isinstance(targets, torch.Tensor):
            trainer = ESNTrainer(next(self._iter_models()))
            targets = trainer._target_dict(targets, trainer._discover_readouts())
        ref_device, ref_dtype = self._reference_device_dtype()
        warmup_inputs = tuple(
            self._coerce_tensor_to_reference(
                t, ref_device, ref_dtype, coerce=coerce, label=f"warmup_inputs[{i}]"
            )
            for i, t in enumerate(warmup_inputs)
        )
        train_inputs = tuple(
            self._coerce_tensor_to_reference(
                t, ref_device, ref_dtype, coerce=coerce, label=f"train_inputs[{i}]"
            )
            for i, t in enumerate(train_inputs)
        )
        targets = {
            name: self._coerce_tensor_to_reference(
                t, ref_device, ref_dtype, coerce=coerce, label=f"targets[{name!r}]"
            )
            for name, t in targets.items()
        }

        if n_workers == 1 or self.n_models == 1:
            for model in self._iter_models():
                ESNTrainer(model).fit(warmup_inputs, train_inputs, targets)
            return

        # Thread-pool path.  Each thread gets its own ESNTrainer over a
        # distinct sub-model, so there is no shared mutable state to guard.
        from concurrent.futures import ThreadPoolExecutor

        def _fit_one(model: ESNModel) -> None:
            ESNTrainer(model).fit(warmup_inputs, train_inputs, targets)

        with ThreadPoolExecutor(max_workers=min(n_workers, self.n_models)) as pool:
            list(pool.map(_fit_one, self.models))

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forecast(
        self,
        warmup_inputs: tuple[torch.Tensor, ...] | torch.Tensor,
        forecast_inputs: tuple[torch.Tensor, ...] | torch.Tensor | None = None,
        *,
        horizon: int,
        initial_feedback: torch.Tensor | None = None,
        return_warmup: bool = False,
        return_individuals: bool = False,
        reset: bool = True,
        alignment: ForecastAlignment = "next",
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Forecast the immediate unseen samples using shared coupled feedback.

        All members consume the same observed history. The aggregate of their
        final warmup predictions is slot zero, then each additional step consumes
        their previous aggregate. ``forecast_inputs`` contains future drivers
        starting immediately after history and needs ``horizon - 1`` samples.
        A one-step forecast needs no future driver.

        ``initial_feedback`` is a distinct extra input after all warmup inputs,
        shaped ``(batch, 1, feedback_features)``; when supplied, every prediction
        consumes a driver and an empty warmup is supported. ``reset=False``
        continues saved states. ``alignment="legacy"`` reproduces the old
        convention which skipped the final warmup prediction and needs
        ``horizon`` future drivers. These semantics match ``ESNModel.forecast``.

        Results have shape ``(batch, horizon, output_features)``. With
        ``return_warmup=True``, prepend historical predictions: total length is
        ``warmup_steps - 1 + horizon``, or ``warmup_steps + horizon`` with an
        explicit seed or legacy alignment. With ``return_individuals=True``,
        also return a list of member trajectories, each covering only the
        ``horizon`` future samples. Per-member buffers are allocated only when
        requested. ``horizon`` must be a positive integer; zero is rejected.

        The final returned prediction has not been consumed. Continue with
        empty history, that prediction as ``initial_feedback``, and ``reset=False``.
        """
        models = list(self._iter_models())
        for model in models:
            model._get_execution_plan()
        warmup_inputs, drivers, horizon, boundary = prepare_forecast_inputs(
            warmup_inputs,
            forecast_inputs,
            num_inputs=len(models[0].inputs),
            horizon=horizon,
            initial_feedback=initial_feedback,
            alignment=alignment,
        )
        batch_size = warmup_inputs[0].shape[0]
        device, dtype = self._reference_device_dtype()
        warmup_outputs_per_model = [
            cast(torch.Tensor, model.warmup(*warmup_inputs, return_outputs=True, reset=reset))
            for model in models
        ]
        output_size = warmup_outputs_per_model[0].shape[-1]
        forecast_outputs = torch.empty(batch_size, horizon, output_size, device=device, dtype=dtype)
        individual_outputs = (
            [torch.empty_like(forecast_outputs) for _ in models] if return_individuals else None
        )
        if initial_feedback is None:
            last_steps = [out[:, -1:, :] for out in warmup_outputs_per_model]
            current_feedback = self._coerce_step_for_buffer(
                self._aggregate(last_steps), device, dtype
            )
        else:
            last_steps = []
            current_feedback = self._coerce_step_for_buffer(initial_feedback, device, dtype)
        if boundary:
            forecast_outputs[:, :1, :] = current_feedback
            if individual_outputs is not None:
                for buf, out in zip(individual_outputs, last_steps):
                    buf[:, :1, :] = self._coerce_step_for_buffer(out, device, dtype)

        flats: list[FlatStep] = [model._get_flat_step() for model in models]
        states_per_model = [
            self._resolve_member_states(flat, batch_size, device, dtype) for flat in flats
        ]
        for t in range(int(boundary), horizon):
            driver_idx = t - int(boundary)
            step_inputs = (current_feedback,) + tuple(
                driver[:, driver_idx : driver_idx + 1, :] for driver in drivers
            )
            step_outputs = []
            for i, flat in enumerate(flats):
                out_tuple, states_per_model[i] = flat.step(step_inputs, states_per_model[i])
                step_outputs.append(out_tuple[0])
            if individual_outputs is not None:
                for buf, out in zip(individual_outputs, step_outputs):
                    buf[:, t : t + 1, :] = self._coerce_step_for_buffer(out, device, dtype)
            current_feedback = self._coerce_step_for_buffer(
                self._aggregate(step_outputs), device, dtype
            )
            forecast_outputs[:, t : t + 1, :] = current_feedback

        for flat, states in zip(flats, states_per_model):
            for reservoir, state in zip(flat.reservoir_layers, states):
                if reservoir.detach_state_between_calls and state.grad_fn is not None:
                    reservoir.state = state.detach().clone()
                else:
                    reservoir.state = state.clone()
        if return_warmup:
            warmup_avg = self._aggregate_stacked(torch.stack(warmup_outputs_per_model, dim=0))
            if warmup_avg.device != device:
                raise ValueError(
                    f"Aggregated warmup output is on device={warmup_avg.device}, but "
                    f"the forecast output buffer is on device={device}. A custom "
                    "aggregator must keep its output on the sub-models' device."
                )
            warmup_avg = warmup_avg.to(dtype=dtype)
            prefix = warmup_avg[:, :-1, :] if boundary else warmup_avg
            forecast_outputs = torch.cat((prefix, forecast_outputs), dim=1)
        if individual_outputs is not None:
            return forecast_outputs, individual_outputs
        return forecast_outputs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        path: str | Path,
        include_states: bool = False,
        **metadata: Any,
    ) -> None:
        """Save ensemble weights (and optionally reservoir states) to a file.

        Parameters
        ----------
        path : str
            Destination file path.
        include_states : bool, default ``False``
            If ``True``, also save current reservoir states for all sub-models.
        **metadata
            Arbitrary key-value pairs stored alongside the weights.
        """
        save_checkpoint(self, path, include_states, metadata)

    def load(
        self,
        path: str | Path,
        strict: bool = True,
        load_states: bool = False,
        map_location: Any = None,
    ) -> None:
        """Load ensemble weights from a file created by :meth:`save`.

        Parameters
        ----------
        path : str
            Source file path.
        strict : bool, default ``True``
            Passed to ``load_state_dict`` for each sub-model.
        load_states : bool, default ``False``
            If ``True``, also restore reservoir states.
        map_location : optional
            Passed to ``torch.load`` to remap saved storage devices.

        Raises
        ------
        ValueError
            If the checkpoint contains a different number of sub-models.
        """
        load_checkpoint(self, path, strict, load_states, map_location)

    @classmethod
    def load_from_file(
        cls,
        path: str | Path,
        model: "CoupledEnsembleESNModel | None" = None,
        strict: bool = True,
        load_states: bool = False,
        map_location: Any = None,
    ) -> "CoupledEnsembleESNModel":
        """Restore weights into an already constructed matching ensemble."""
        if model is None:
            raise ValueError("model argument is required")
        model.load(path, strict=strict, load_states=load_states, map_location=map_location)
        return model

    def save_full(self, path: str | Path, **metadata: Any) -> None:
        """Serialize the entire ensemble — every sub-model's architecture,
        weights, and reservoir states — to a single file.

        Unlike :meth:`save` (state dicts only, requires rebuilding the ensemble
        before :meth:`load`), this pickles the whole ensemble object and is
        restored with :meth:`load_full` without rebuilding anything.  Relies on
        the pickle support added in ``pytorch-symbolic`` 1.2.

        Parameters
        ----------
        path : str
            Destination file path.
        **metadata
            Arbitrary key-value pairs stored alongside the ensemble.

        Notes
        -----
        Loaded back with ``weights_only=False`` (arbitrary unpickling), so only
        open files you trust.  Custom callable topology/initializer/activation
        specs must be importable (module-level, not lambdas) to be picklable;
        otherwise use the lighter state-dict :meth:`save`.

        See Also
        --------
        load_full : Reconstruct an ensemble saved with this method.
        save : Lighter, state-dict-only persistence (architecture not stored).
        """
        save_full_model(self, path, metadata)

    @classmethod
    def load_full(
        cls,
        path: str | Path,
        return_metadata: bool = False,
        map_location: Any = None,
    ) -> "CoupledEnsembleESNModel | tuple[CoupledEnsembleESNModel, dict[str, Any]]":
        """Reconstruct a complete ensemble saved with :meth:`save_full`.

        No pre-built ensemble is required — every sub-model is restored intact.

        Parameters
        ----------
        path : str
            File path written by :meth:`save_full`.
        return_metadata : bool, default ``False``
            If ``True``, return ``(ensemble, metadata)`` instead of just the
            ensemble.
        map_location : optional
            Passed to ``torch.load`` to remap storage devices.

        Raises
        ------
        ValueError
            If the file does not contain a whole ensemble (e.g. a state-dict
            checkpoint from :meth:`save`).

        Warnings
        --------
        Loads with ``weights_only=False``, which unpickles arbitrary Python
        objects.  Only call this on files from a source you trust.

        See Also
        --------
        save_full : Serialize a complete ensemble.
        """
        return load_full_model(cls, path, return_metadata, map_location)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reference_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        """Resolve the canonical device and dtype of the ensemble's weights.

        The reference is the first floating-point parameter or buffer of the
        first sub-model — the same scan reservoir layers use to derive their
        own reference (see
        :meth:`resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer._reference_device_dtype`)
        — falling back to ``cpu`` / ``float32`` for a (hypothetical) weightless
        sub-model.  This is the device/dtype that ``fit`` inputs/targets and the
        ``forecast`` output buffer must agree with, so a CPU-target-on-GPU-model
        mistake surfaces as a clear, named error rather than a raw cross-device
        ``RuntimeError`` deep inside the readout solve.

        Returns
        -------
        device : torch.device
            Device of the first sub-model's first floating-point tensor, or
            ``cpu``.
        dtype : torch.dtype
            Dtype of the first sub-model's first floating-point tensor, or
            ``float32``.
        """
        first = next(self._iter_models())
        ref = next(
            (t for t in chain(first.parameters(), first.buffers()) if t.is_floating_point()),
            None,
        )
        device = ref.device if ref is not None else torch.device("cpu")
        dtype = ref.dtype if ref is not None else torch.float32
        return device, dtype

    def _coerce_tensor_to_reference(
        self,
        tensor: torch.Tensor,
        ref_device: torch.device,
        ref_dtype: torch.dtype,
        *,
        coerce: bool,
        label: str,
    ) -> torch.Tensor:
        """Validate or coerce one tensor against the ensemble's reference.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to check.
        ref_device, ref_dtype : torch.device, torch.dtype
            The ensemble's reference device/dtype from
            :meth:`_reference_device_dtype`.
        coerce : bool
            If ``True``, return ``tensor.to(device=ref_device, dtype=ref_dtype)``
            (a no-op when already matching).  If ``False``, raise a clear,
            named :class:`ValueError` on any device or dtype mismatch.
        label : str
            Human-readable name of the tensor (e.g. ``"warmup_inputs[0]"`` or
            ``"targets['output']"``) used in the error message.

        Returns
        -------
        torch.Tensor
            ``tensor`` unchanged when it already matches the reference, the
            coerced tensor when ``coerce=True``, otherwise raises.

        Raises
        ------
        ValueError
            When ``coerce=False`` and ``tensor``'s device or dtype differs from
            the reference.
        """
        if tensor.device == ref_device and tensor.dtype == ref_dtype:
            return tensor
        if coerce:
            return tensor.to(device=ref_device, dtype=ref_dtype)
        raise ValueError(
            f"{label} is on device={tensor.device}, dtype={tensor.dtype}, but the "
            f"ensemble's sub-models are on device={ref_device}, dtype={ref_dtype}. "
            f"Move the data to match (e.g. tensor.to(device='{ref_device}', "
            f"dtype={ref_dtype})) or pass coerce=True to fit() to coerce it "
            f"automatically."
        )

    @staticmethod
    def _coerce_step_for_buffer(
        step: torch.Tensor,
        buf_device: torch.device,
        buf_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Deliberately conform one autoregressive step to the output buffer.

        The per-step in-place buffer writes in :meth:`forecast` would otherwise
        *silently* down/upcast a custom aggregator whose output dtype differs
        from the buffer's (and raise an opaque cross-device ``RuntimeError`` on
        a device mismatch).  This makes both explicit: a device mismatch raises
        a clear, named :class:`ValueError`; a dtype mismatch is cast on purpose
        (a no-op in the common matching case).

        Parameters
        ----------
        step : torch.Tensor
            One aggregated (or per-model) step — shape-agnostic; only its
            device and dtype are inspected, the shape is preserved.
        buf_device, buf_dtype : torch.device, torch.dtype
            The pre-allocated output buffer's device/dtype.

        Returns
        -------
        torch.Tensor
            ``step`` unchanged when it already matches, else cast to
            ``buf_dtype`` on ``buf_device``.

        Raises
        ------
        ValueError
            If ``step`` lives on a different device than the buffer (e.g. a
            custom aggregator that moved the output off-device).
        """
        if step.device != buf_device:
            raise ValueError(
                f"Aggregated forecast step is on device={step.device}, but the "
                f"forecast output buffer is on device={buf_device}. A custom "
                f"aggregator must keep its output on the sub-models' device."
            )
        if step.dtype != buf_dtype:
            return step.to(dtype=buf_dtype)
        return step

    @staticmethod
    def _resolve_member_states(
        flat: FlatStep,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Collect a sub-model's reservoir states for explicit threading.

        Warmup leaves each reservoir's state set; this reads them (in the
        ``flat.reservoir_layers`` order the flat step threads them) and lazily
        initialises any that are somehow still ``None`` — mirroring the same
        resolution :meth:`resdag.core.ESNModel.forecast` performs before it hands
        the states to :func:`resdag.core._flat_inference.rollout`.

        Parameters
        ----------
        flat : FlatStep
            The sub-model's flattened single-step executor; its
            ``reservoir_layers`` fix the threading order.
        batch_size : int
            Batch size used to lazily initialise any missing state.
        device, dtype : torch.device, torch.dtype
            Device/dtype used to lazily initialise any missing state.

        Returns
        -------
        list of torch.Tensor
            The reservoir states in ``flat.reservoir_layers`` order, ready to be
            threaded through :meth:`FlatStep.step`.
        """
        states: list[torch.Tensor] = []
        for reservoir in flat.reservoir_layers:
            if reservoir.state is None:
                reservoir._maybe_init_state(batch_size, device, dtype)
            assert reservoir.state is not None  # narrows for mypy
            states.append(reservoir.state)
        return states

    def _aggregate(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        """Stack a list of per-model tensors and aggregate them."""
        stacked = torch.stack(outputs, dim=0)  # (N, batch, T, F)
        return self._aggregate_stacked(stacked)

    def _aggregate_stacked(self, stacked: torch.Tensor) -> torch.Tensor:
        """Aggregate a pre-stacked (N, batch, T, F) tensor."""
        if self.aggregator_module is not None:
            result = self.aggregator_module(stacked)
            if not isinstance(result, torch.Tensor):
                raise TypeError("A custom ensemble aggregator must return a torch.Tensor")
            if result.shape != stacked.shape[1:]:
                raise ValueError(
                    "A custom ensemble aggregator must preserve (batch, time, features): "
                    f"expected {tuple(stacked.shape[1:])}, got {tuple(result.shape)}"
                )
            return result
        if self._aggregator_str == "mean":
            return stacked.mean(dim=0)
        # Interpolated (statistical) median: ``torch.quantile(.., 0.5)`` averages
        # the two central members for even N, unlike ``Tensor.median`` which
        # returns the lower of the two and biases an even-sized ensemble's
        # feedback downward at every autoregressive step.
        return torch.quantile(stacked, 0.5, dim=0)

    def extra_repr(self) -> str:
        agg = (
            repr(self.aggregator_module)
            if self.aggregator_module is not None
            else self._aggregator_str
        )
        return f"n_models={self.n_models}, aggregator={agg}"
