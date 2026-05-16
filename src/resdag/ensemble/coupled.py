"""
Coupled Ensemble ESN Model
==========================

This module provides :class:`CoupledEnsembleESNModel`, an ensemble of N
independently-trained ESN models whose forecasts are coupled through a shared
averaged feedback signal at each autoregressive step.

See Also
--------
resdag.models.coupled_ensemble_esn : Factory function to build this ensemble.
resdag.composition.ESNModel : Base ESN model used as sub-models.
"""

import warnings
from typing import Any

import torch
import torch.nn as nn

from resdag.composition import ESNModel
from resdag.training import ESNTrainer


class CoupledEnsembleESNModel(nn.Module):
    """Ensemble of N independently-trained ESN models with coupled feedback.

    Each sub-model is a complete :class:`~resdag.composition.ESNModel` built
    via the symbolic composition API. The models are trained independently but
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
        - ``"median"`` — median across models.
        - Any ``nn.Module`` that accepts a stacked tensor of shape
          ``(N, batch, timesteps, features)`` and returns
          ``(batch, timesteps, features)``.  E.g.
          :class:`~resdag.layers.OutliersFilteredMean`.

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

    >>> from resdag.layers import OutliersFilteredMean
    >>> ensemble = coupled_ensemble_esn(
    ...     n_models=10, reservoir_size=300, feedback_size=3, output_size=3,
    ...     aggregate=OutliersFilteredMean(method="z_score", threshold=2.0),
    ... )

    See Also
    --------
    resdag.models.coupled_ensemble_esn : Convenience factory function.
    resdag.layers.OutliersFilteredMean : Outlier-robust aggregation layer.
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
            If ``models`` is empty or ``aggregator`` is an unknown string.
        """
        super().__init__()
        if len(models) == 0:
            raise ValueError("CoupledEnsembleESNModel requires at least one sub-model.")
        if isinstance(aggregator, str) and aggregator not in ("mean", "median"):
            raise ValueError(
                f"aggregator must be 'mean', 'median', or an nn.Module; got '{aggregator}'."
            )

        self.models = nn.ModuleList(models)

        if isinstance(aggregator, nn.Module):
            self.aggregator_module: nn.Module | None = aggregator
            self._aggregator_str: str | None = None
        else:
            self.aggregator_module = None
            self._aggregator_str = aggregator

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_models(self) -> int:
        """Number of sub-models in the ensemble."""
        return len(self.models)

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

    def warmup(self, *inputs: torch.Tensor) -> None:
        """Teacher-forced warmup: synchronize every sub-model's reservoir state.

        Parameters
        ----------
        *inputs : torch.Tensor
            Warmup sequences.  Passed identically to each sub-model.
        """
        for model in self.models:
            model.warmup(*inputs)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_reservoirs(self) -> None:
        """Reset reservoir states in all sub-models to zero / None."""
        for model in self.models:
            model.reset_reservoirs()

    def get_reservoir_states(self) -> list[dict[str, torch.Tensor]]:
        """Return reservoir states for all sub-models.

        Returns
        -------
        list of dict
            One dict per sub-model, mapping layer name to state tensor.
        """
        return [model.get_reservoir_states() for model in self.models]

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
        for model, state_dict in zip(self.models, states):
            model.set_reservoir_states(state_dict, strict=strict)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        warmup_inputs: tuple[torch.Tensor, ...],
        train_inputs: tuple[torch.Tensor, ...],
        targets: dict[str, torch.Tensor],
    ) -> None:
        """Train all sub-models independently using :class:`~resdag.training.ESNTrainer`.

        Each sub-model is trained separately on the same warmup/train data.
        Ensemble diversity comes from the different random reservoir
        initializations of each sub-model.

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
        """
        for model in self.models:
            ESNTrainer(model).fit(warmup_inputs, train_inputs, targets)

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

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
            generated by the ensemble itself).  Each tensor must have shape
            ``(batch, horizon, driver_features)``.
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
            If driver arguments are inconsistent with the number of warmup
            inputs or if ``forecast_inputs`` timesteps do not match
            ``horizon``.
        """
        if isinstance(warmup_inputs, torch.Tensor):
            warmup_inputs = (warmup_inputs,)
        else:
            warmup_inputs = tuple(warmup_inputs)
        if len(warmup_inputs) == 0:
            raise ValueError("At least one warmup input (feedback) is required.")

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
                if driver.shape[1] != horizon:
                    raise ValueError(
                        f"forecast_inputs[{i}] has {driver.shape[1]} timesteps, "
                        f"expected {horizon}."
                    )

        batch_size = warmup_inputs[0].shape[0]
        device = warmup_inputs[0].device
        dtype = warmup_inputs[0].dtype

        # Phase 1: warmup — all models independently, same teacher-forced data
        warmup_outputs_per_model: list[torch.Tensor] = []
        for model in self.models:
            out = model.warmup(*warmup_inputs, return_outputs=True, reset=reset)
            warmup_outputs_per_model.append(out)  # (batch, W, output_size)

        output_size = warmup_outputs_per_model[0].shape[-1]

        # Aggregate the last warmup step as the initial autoregressive feedback
        last_steps = [out[:, -1:, :] for out in warmup_outputs_per_model]
        current_feedback = self._aggregate(last_steps)  # (batch, 1, output_size)

        # Pre-allocate aggregated forecast buffer
        forecast_outputs = torch.empty(
            batch_size, horizon, output_size, device=device, dtype=dtype
        )
        forecast_outputs[:, 0, :] = current_feedback.squeeze(1)

        # Per-model buffers: only allocated when the caller explicitly requests them
        individual_outputs: list[torch.Tensor] | None = None
        if return_individuals:
            individual_outputs = [
                torch.empty(batch_size, horizon, output_size, device=device, dtype=dtype)
                for _ in self.models
            ]
            for buf in individual_outputs:
                buf[:, 0, :] = current_feedback.squeeze(1)

        # Phase 2: coupled autoregressive loop
        for t in range(1, horizon):
            if has_drivers:
                driver_slice = tuple(d[:, t : t + 1, :] for d in forecast_inputs)
                step_inputs: tuple[torch.Tensor, ...] = (current_feedback,) + driver_slice
            else:
                step_inputs = (current_feedback,)

            step_outputs = [model(*step_inputs) for model in self.models]

            if individual_outputs is not None:
                for buf, out in zip(individual_outputs, step_outputs):
                    buf[:, t, :] = out.squeeze(1)

            current_feedback = self._aggregate(step_outputs)  # (batch, 1, output_size)
            forecast_outputs[:, t, :] = current_feedback.squeeze(1)

        if return_warmup:
            warmup_stacked = torch.stack(warmup_outputs_per_model, dim=0)
            warmup_avg = self._aggregate_stacked(warmup_stacked)
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        """Stack a list of per-model tensors and aggregate them."""
        stacked = torch.stack(outputs, dim=0)  # (N, batch, T, F)
        return self._aggregate_stacked(stacked)

    def _aggregate_stacked(self, stacked: torch.Tensor) -> torch.Tensor:
        """Aggregate a pre-stacked (N, batch, T, F) tensor."""
        if self.aggregator_module is not None:
            return self.aggregator_module(stacked)
        if self._aggregator_str == "mean":
            return stacked.mean(dim=0)
        # median
        return stacked.median(dim=0).values

    def extra_repr(self) -> str:
        agg = (
            repr(self.aggregator_module)
            if self.aggregator_module is not None
            else self._aggregator_str
        )
        return f"n_models={self.n_models}, aggregator={agg}"
