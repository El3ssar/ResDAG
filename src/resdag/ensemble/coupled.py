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

import warnings
from collections.abc import Iterator
from itertools import chain
from typing import Any, cast

import torch
import torch.nn as nn

from resdag.core import ESNModel
from resdag.core._flat_inference import FlatStep
from resdag.training import ESNTrainer


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

        self._validate_feedback_output_dims(models)

        self.models = nn.ModuleList(models)

        if isinstance(aggregator, nn.Module):
            self.aggregator_module: nn.Module | None = aggregator
            self._aggregator_str: str | None = None
        else:
            self.aggregator_module = None
            self._aggregator_str = aggregator

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
        warmup_inputs: tuple[torch.Tensor, ...],
        train_inputs: tuple[torch.Tensor, ...],
        targets: dict[str, torch.Tensor],
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
        forecast_inputs: tuple[torch.Tensor, ...] | None = None,
        *,
        horizon: int,
        return_warmup: bool = False,
        return_individuals: bool = False,
        reset: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Coupled autoregressive forecast.

        **Phase 1 — Warmup**: all sub-models are teacher-forced with the same
        warmup data, advancing their reservoir states independently.

        **Phase 2 — Coupled autoregression**: at every timestep ``t`` every
        sub-model receives the *same* ``current_feedback`` (the aggregation of
        all models' outputs at ``t-1``).  The averaged output couples all models
        through a shared feedback signal.

        Parameters
        ----------
        warmup_inputs : tuple of torch.Tensor or torch.Tensor
            Warmup sequences ``(feedback, driver1, ...)``.  A single tensor
            is accepted for the common feedback-only case.
        forecast_inputs : tuple of torch.Tensor, optional
            Exogenous driver inputs for the autoregressive phase (feedback is
            generated by the ensemble itself).  Autoregressive step ``t``
            consumes ``forecast_inputs[i][:, t, :]`` — pass the driver series
            continuing exactly where the warmup drivers ended.  Each tensor
            must have **at least** ``horizon`` timesteps (one per
            autoregressive step); extra trailing steps are ignored.  This
            matches :meth:`resdag.core.ESNModel.forecast` exactly.
        horizon : int, keyword-only
            Number of autoregressive steps to generate.
        return_warmup : bool, default ``False``
            If ``True``, prepend the averaged warmup outputs to the returned
            aggregated forecast, giving shape
            ``(batch, warmup_steps + horizon, output_size)``.
            Individual trajectories (when ``return_individuals=True``) always
            cover only the autoregressive phase.
        return_individuals : bool, default ``False``
            If ``True``, also return per-sub-model autoregressive trajectories.
            One buffer of shape ``(batch, horizon, output_size)`` is
            pre-allocated per sub-model on the same device at the start of the
            forecast loop — only set this flag when you actually need the
            individual sequences, to avoid allocating N extra GPU buffers.
        reset : bool, default ``True``
            If ``True``, every sub-model's reservoir state is reset before
            warmup.

        Returns
        -------
        torch.Tensor
            Aggregated forecast of shape ``(batch, horizon, output_size)``, or
            ``(batch, warmup_steps + horizon, output_size)`` when
            ``return_warmup=True``.  Returned alone when
            ``return_individuals=False``.
        tuple of (torch.Tensor, list of torch.Tensor)
            When ``return_individuals=True``: a 2-tuple whose first element is
            the aggregated forecast (as above) and whose second element is a
            list of ``n_models`` tensors each of shape
            ``(batch, horizon, output_size)``, in the same order as
            ``self.models``.

        Raises
        ------
        ValueError
            If ``horizon`` is not a positive integer, if driver arguments are
            inconsistent with the number of warmup inputs, or if a
            ``forecast_inputs`` tensor has fewer than ``horizon`` timesteps.
        """
        if isinstance(warmup_inputs, torch.Tensor):
            warmup_inputs = (warmup_inputs,)
        else:
            warmup_inputs = tuple(warmup_inputs)
        if len(warmup_inputs) == 0:
            raise ValueError("At least one warmup input (feedback) is required.")

        # Validate the horizon up front so callers get a clear error instead of
        # a downstream IndexError / negative-dimension RuntimeError.
        if horizon < 1:
            raise ValueError(f"horizon must be a positive integer, got {horizon}")

        num_drivers = len(warmup_inputs) - 1
        has_drivers = num_drivers > 0

        if has_drivers:
            if forecast_inputs is None:
                raise ValueError(
                    f"Model has {num_drivers} driving input(s). "
                    "forecast_inputs must be provided for the autoregressive phase."
                )
            forecast_inputs = tuple(forecast_inputs)
            if len(forecast_inputs) != num_drivers:
                raise ValueError(
                    f"Expected {num_drivers} forecast driver(s), got {len(forecast_inputs)}."
                )
            for i, driver in enumerate(forecast_inputs):
                if driver.shape[1] < horizon:
                    raise ValueError(
                        f"forecast_inputs[{i}] has {driver.shape[1]} steps, expected at "
                        f"least {horizon} (one per autoregressive step). forecast_inputs "
                        f"must hold the driver series for the forecast window, starting "
                        f"right after the warmup window."
                    )

        batch_size = warmup_inputs[0].shape[0]
        # Build the output buffers from the sub-models' own device/dtype, not
        # from ``warmup_inputs`` — the sub-models compute on their parameter
        # device regardless of where the warmup data lives, so this is the
        # device/dtype every aggregated step actually has.  Deriving the buffer
        # from the inputs (the old behaviour) risked a cross-device in-place
        # write or a silent dtype cast when a custom aggregator changed dtype.
        device, dtype = self._reference_device_dtype()

        # Phase 1: warmup — all models independently, same teacher-forced data
        warmup_outputs_per_model: list[torch.Tensor] = []
        for model in self._iter_models():
            # ``return_outputs=True`` yields the warmup outputs. Sub-models are
            # guaranteed single-output by ``_validate_feedback_output_dims`` at
            # construction (multi-output models are rejected there), so this is
            # always a single tensor rather than a tuple of per-head tensors.
            out = cast(torch.Tensor, model.warmup(*warmup_inputs, return_outputs=True, reset=reset))
            warmup_outputs_per_model.append(out)  # (batch, W, output_size)

        output_size = warmup_outputs_per_model[0].shape[-1]

        # Aggregate the last warmup step as the initial autoregressive feedback.
        # This seeds ``current_feedback`` but is *not* written into any output
        # buffer — slot 0 is a genuine forecast step, matching ESNModel.forecast.
        # Conform it to the reference device/dtype so a dtype-changing custom
        # aggregator does not feed a float-mismatched tensor into the sub-models
        # on the very first autoregressive step.
        last_steps = [out[:, -1:, :] for out in warmup_outputs_per_model]
        current_feedback = self._coerce_step_for_buffer(
            self._aggregate(last_steps), device, dtype
        )  # (batch, 1, output_size)

        # Pre-allocate aggregated forecast buffer
        forecast_outputs = torch.empty(batch_size, horizon, output_size, device=device, dtype=dtype)

        # Per-model buffers: only allocated when the caller explicitly requests them
        individual_outputs: list[torch.Tensor] | None = None
        if return_individuals:
            individual_outputs = [
                torch.empty(batch_size, horizon, output_size, device=device, dtype=dtype)
                for _ in self.models
            ]

        # Resolve each sub-model's flattened single-step executor and its
        # reservoir states once, *before* the loop. Stepping every member through
        # its ``FlatStep`` (a straight-line, capture-the-layers step function)
        # instead of re-invoking ``model(*step_inputs)`` sheds the per-step
        # symbolic-graph re-walk, ``nn.Module.__call__`` dispatch, and reservoir
        # sequence-loop bookkeeping — the same overhead the core forecast engine
        # eliminates for single models. The step is numerically identical to the
        # graph path (proven bit-exact upstream), so the coupled forecast output
        # is unchanged. States are threaded explicitly per member rather than
        # mutated on the module buffers mid-loop; they are persisted back to each
        # reservoir after the loop so ``reset=False`` / ``get_reservoir_states``
        # continue to observe the advanced states.
        flats: list[FlatStep] = [m._get_flat_step() for m in self._iter_models()]
        states_per_model: list[list[torch.Tensor]] = [
            self._resolve_member_states(flat, batch_size, device, dtype) for flat in flats
        ]

        # Phase 2: coupled autoregressive loop. Every slot ``0..horizon-1`` is a
        # genuine coupled step; no teacher-forced warmup echo is emitted. The
        # members are coupled through the aggregator: every member is stepped on
        # the *same* ``current_feedback`` (the aggregation of all members' outputs
        # at ``t-1``), and the aggregated output becomes the shared feedback for
        # the next step — so the members cannot be rolled out independently and
        # the aggregation must be interleaved between single steps.
        for t in range(horizon):
            if has_drivers:
                # ``has_drivers`` implies ``forecast_inputs`` was validated as a
                # non-empty tuple above; restate it for the type-checker.
                assert forecast_inputs is not None
                # Same pairing as ESNModel.forecast: step ``t`` consumes the
                # driver at the ``t``-th step after warmup = forecast_inputs[:, t].
                driver_slice = tuple(d[:, t : t + 1, :] for d in forecast_inputs)
                step_inputs: tuple[torch.Tensor, ...] = (current_feedback,) + driver_slice
            else:
                step_inputs = (current_feedback,)

            # Step every member through its flat executor on the shared feedback,
            # threading each member's own reservoir states in and out. Sub-models
            # are single-output (enforced at construction), so ``out_tuple`` holds
            # exactly one ``(batch, 1, output_size)`` slice — identical to what
            # ``model(*step_inputs)`` returned on the graph path.
            step_outputs: list[torch.Tensor] = []
            for i, flat in enumerate(flats):
                out_tuple, states_per_model[i] = flat.step(step_inputs, states_per_model[i])
                step_outputs.append(out_tuple[0])

            if individual_outputs is not None:
                for buf, out in zip(individual_outputs, step_outputs):
                    buf[:, t, :] = self._coerce_step_for_buffer(out.squeeze(1), device, dtype)

            # Conform the aggregated feedback to the reference device/dtype once,
            # deliberately. A custom aggregator that changes dtype would
            # otherwise (a) be silently down/upcast by the in-place buffer write
            # and (b) re-enter the float-mismatched sub-models on the *next*
            # step, raising an opaque ``F.linear`` dtype error. Coercing here
            # fixes both and keeps the autoregressive loop type-stable.
            current_feedback = self._coerce_step_for_buffer(
                self._aggregate(step_outputs), device, dtype
            )  # (batch, 1, output_size)
            forecast_outputs[:, t, :] = current_feedback.squeeze(1)

        # Persist each member's advanced reservoir states back onto its layers so
        # a follow-up ``forecast(reset=False)`` / ``get_reservoir_states`` sees
        # them, mirroring the in-place mutation of the legacy per-step loop and
        # honouring the ``detach_state_between_calls`` contract exactly as
        # ``ESNModel.forecast`` does.
        for flat, states in zip(flats, states_per_model):
            for reservoir, state in zip(flat.reservoir_layers, states):
                if reservoir.detach_state_between_calls and state.grad_fn is not None:
                    reservoir.state = state.detach().clone()
                else:
                    reservoir.state = state.clone()

        if return_warmup:
            warmup_stacked = torch.stack(warmup_outputs_per_model, dim=0)
            warmup_avg = self._aggregate_stacked(warmup_stacked)
            # Conform the aggregated warmup to the forecast buffer before the
            # concat so a custom aggregator that changes dtype/device produces a
            # clear error or deliberate cast rather than a raw ``torch.cat``
            # mismatch.  ``warmup_avg`` is (batch, W, F); reuse the per-step
            # coercion over the flattened time axis by checking the tensor as a
            # whole (device + dtype only, shape is preserved).
            if warmup_avg.device != device:
                raise ValueError(
                    f"Aggregated warmup output is on device={warmup_avg.device}, but "
                    f"the forecast output buffer is on device={device}. A custom "
                    f"aggregator must keep its output on the sub-models' device."
                )
            if warmup_avg.dtype != dtype:
                warmup_avg = warmup_avg.to(dtype=dtype)
            aggregated = torch.cat([warmup_avg, forecast_outputs], dim=1)
        else:
            aggregated = forecast_outputs

        if return_individuals:
            return aggregated, individual_outputs  # type: ignore[return-value]
        return aggregated

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        path: str,
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
        save_dict: dict[str, Any] = {
            "state_dicts": [model.state_dict() for model in self.models],
            "metadata": metadata,
        }
        if include_states:
            save_dict["reservoir_states"] = self.get_reservoir_states()
        torch.save(save_dict, path)

    def load(self, path: str, strict: bool = True, load_states: bool = False) -> None:
        """Load ensemble weights from a file created by :meth:`save`.

        Parameters
        ----------
        path : str
            Source file path.
        strict : bool, default ``True``
            Passed to ``load_state_dict`` for each sub-model.
        load_states : bool, default ``False``
            If ``True``, also restore reservoir states.

        Raises
        ------
        ValueError
            If the checkpoint contains a different number of sub-models.
        """
        checkpoint = torch.load(path, weights_only=False)
        if not isinstance(checkpoint, dict) or "state_dicts" not in checkpoint:
            raise ValueError(
                f"{path} is not a save() state-dict checkpoint. If it was written "
                f"by save_full() (or torch.save(ensemble)), use "
                f"CoupledEnsembleESNModel.load_full()."
            )
        state_dicts: list[dict] = checkpoint["state_dicts"]
        if len(state_dicts) != len(self.models):
            raise ValueError(
                f"Checkpoint has {len(state_dicts)} sub-model(s), "
                f"but this ensemble has {len(self.models)}."
            )
        for model, sd in zip(self.models, state_dicts):
            model.load_state_dict(sd, strict=strict)

        if load_states:
            if "reservoir_states" in checkpoint:
                self.set_reservoir_states(checkpoint["reservoir_states"])
            else:
                warnings.warn(
                    "load_states=True but checkpoint has no 'reservoir_states' key.",
                    UserWarning,
                    stacklevel=2,
                )

    def save_full(self, path: str, **metadata: Any) -> None:
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
        torch.save({"resdag_full_ensemble": self, "metadata": metadata}, path)

    @classmethod
    def load_full(
        cls,
        path: str,
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
        payload = torch.load(path, weights_only=False, map_location=map_location)
        ensemble: CoupledEnsembleESNModel
        metadata: dict[str, Any]
        if isinstance(payload, dict) and "resdag_full_ensemble" in payload:
            ensemble = payload["resdag_full_ensemble"]
            metadata = payload.get("metadata", {})
        elif isinstance(payload, CoupledEnsembleESNModel):
            # A bare ``torch.save(ensemble)`` file, with no metadata wrapper.
            ensemble = payload
            metadata = {}
        else:
            raise ValueError(
                f"{path} does not contain a full ensemble. Write one with "
                f"save_full() (or torch.save(ensemble)); for a state-dict "
                f"checkpoint from save(), rebuild the ensemble and use load()."
            )
        if return_metadata:
            return ensemble, metadata
        return ensemble

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
            # nn.Module.__call__ is typed to return Any; the aggregator contract
            # is to return a single aggregated tensor.
            return cast(torch.Tensor, self.aggregator_module(stacked))
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
