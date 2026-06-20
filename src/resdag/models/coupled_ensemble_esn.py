"""
Coupled Ensemble ESN Factory
=============================

This module provides :func:`coupled_ensemble_esn`, which builds a
:class:`~resdag.ensemble.CoupledEnsembleESNModel` from N independently
initialized sub-models produced by any ESN factory function.

See Also
--------
resdag.ensemble.CoupledEnsembleESNModel : The ensemble class itself.
resdag.models.ott_esn : Default sub-model factory.
resdag.models.classic_esn : Alternative sub-model factory.
"""

import inspect
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from resdag.core import ESNModel
from resdag.ensemble import CoupledEnsembleESNModel
from resdag.models.ott_esn import ott_esn


def coupled_ensemble_esn(
    n_models: int,
    model_factory: Callable[..., ESNModel] = ott_esn,
    aggregate: str | nn.Module = "mean",
    seed: int | None = None,
    **model_kwargs: Any,
) -> CoupledEnsembleESNModel:
    """Build a coupled ensemble of N independently-initialized ESN models.

    Each sub-model is created by calling ``model_factory(**model_kwargs)``
    with an independent random reservoir initialization, providing the
    diversity needed for ensemble averaging to be beneficial.

    During autoregressive forecasting all N sub-models receive the **same
    aggregated output** (e.g. the mean across models) as their next feedback
    input — they are coupled through the shared signal.

    Parameters
    ----------
    n_models : int
        Number of sub-models in the ensemble.
    model_factory : callable, default :func:`ott_esn`
        Factory that creates one :class:`~resdag.core.ESNModel`.
        All keyword arguments not consumed by ``coupled_ensemble_esn`` itself
        (i.e. ``model_kwargs``) are forwarded verbatim to every call of this
        factory.  The factory must return a **readout-bearing** model whose
        first output dimension equals the feedback input dimension, so the
        aggregated output can be fed back autoregressively.  The readout-bearing
        premade factories — ``classic_esn``, ``ott_esn`` and ``power_augmented``
        — satisfy this.  The headless factories (``headless_esn``,
        ``linear_esn``) emit raw reservoir states rather than a
        feedback-dimensioned output and are rejected by
        :class:`~resdag.ensemble.CoupledEnsembleESNModel` (see its ``__init__``
        for the exact check).
    aggregate : str or nn.Module, default ``"mean"``
        Aggregation strategy applied to the N model outputs at each
        autoregressive step:

        - ``"mean"`` — arithmetic mean.
        - ``"median"`` — median.
        - ``nn.Module`` — any module accepting a stacked tensor of shape
          ``(N, batch, timesteps, features)`` and returning
          ``(batch, timesteps, features)``.  E.g.
          :class:`~resdag.ensemble.aggregators.OutliersFilteredMean`.
    seed : int, optional
        Master seed for deterministic sub-model construction.  Sub-model
        ``i`` is seeded with ``seed + i`` immediately before the factory call.
        If ``model_factory`` itself accepts a ``seed`` keyword argument,
        ``seed + i`` is also forwarded as that argument so initialisers that
        take an explicit seed (e.g. graph topologies) become reproducible.

        The process-global default RNG is **not** clobbered: its state is
        captured before the construction loop and restored on exit (even if a
        factory raises), so building a seeded ensemble does not perturb a
        subsequent global ``torch.randn`` draw.  Ensemble construction is
        therefore composable inside an otherwise-reproducible pipeline.

    **model_kwargs
        All remaining keyword arguments are forwarded verbatim to each
        ``model_factory`` call.  This includes architecture parameters such as
        ``reservoir_size``, ``feedback_size``, ``output_size``,
        ``spectral_radius``, ``readout_alpha``, ``readout_name``, etc.
        The exact set of valid keys depends on the chosen ``model_factory``.

    Returns
    -------
    CoupledEnsembleESNModel
        Ensemble ready for training and forecasting.

    Examples
    --------
    Quick start with default Ott's ESN sub-models:

    >>> import resdag as rd
    >>>
    >>> ensemble = rd.coupled_ensemble_esn(
    ...     n_models=5,
    ...     reservoir_size=300,
    ...     feedback_size=3,
    ...     output_size=3,
    ... )
    >>> ensemble.fit((warmup,), (train,), {"output": targets})
    >>> ensemble.reset_reservoirs()
    >>> preds = ensemble.forecast(forecast_warmup, horizon=200)
    >>> preds.shape
    torch.Size([1, 200, 3])

    Using ``classic_esn`` as the base architecture:

    >>> from resdag.models import classic_esn
    >>> ensemble = rd.coupled_ensemble_esn(
    ...     n_models=8,
    ...     model_factory=classic_esn,
    ...     reservoir_size=500,
    ...     feedback_size=3,
    ...     output_size=3,
    ...     spectral_radius=0.95,
    ...     readout_alpha=1e-7,
    ... )

    Outlier-robust aggregation:

    >>> from resdag.ensemble.aggregators import OutliersFilteredMean
    >>> ensemble = rd.coupled_ensemble_esn(
    ...     n_models=10,
    ...     reservoir_size=300,
    ...     feedback_size=3,
    ...     output_size=3,
    ...     aggregate=OutliersFilteredMean(method="z_score", threshold=2.0),
    ... )

    Recovering individual sub-model trajectories for post-hoc analysis:

    >>> ensemble.reset_reservoirs()
    >>> preds, individuals = ensemble.forecast(
    ...     forecast_warmup, horizon=200, return_individuals=True
    ... )
    >>> preds.shape          # averaged forecast
    torch.Size([1, 200, 3])
    >>> len(individuals)     # one tensor per sub-model
    10
    >>> individuals[0].shape
    torch.Size([1, 200, 3])

    See Also
    --------
    CoupledEnsembleESNModel : The ensemble class with full API documentation.
    resdag.ensemble.aggregators.OutliersFilteredMean : Outlier-robust aggregation layer.
    resdag.models.ott_esn : Default sub-model factory.
    """
    # Detect whether the factory accepts a ``seed`` kwarg so we can forward
    # the per-model seed to initialisers that take an explicit one (graph
    # topologies, etc.). When it doesn't, the per-model RNG seed below still
    # controls every non-seeded random draw.
    factory_accepts_seed = (
        seed is not None and "seed" in inspect.signature(model_factory).parameters
    )

    models: list[ESNModel] = []
    if seed is None:
        # No seeding requested: nothing touches the global RNG beyond what the
        # factory itself draws, so no save/restore dance is needed.
        for _ in range(n_models):
            models.append(model_factory(**model_kwargs))
        return CoupledEnsembleESNModel(models, aggregator=aggregate)

    # Seeded path. Most reservoir/feedback initialisers draw from PyTorch's
    # process-global default generator, so we still seed it per sub-model with
    # ``seed + i``. To keep ensemble construction composable inside a larger
    # reproducible pipeline we snapshot the global RNG state up front and
    # restore it in ``finally`` — building the ensemble leaves the global
    # generator exactly where it was, even if a factory call raises.
    global_rng_state = torch.get_rng_state()
    try:
        for i in range(n_models):
            sub_seed = seed + i
            torch.manual_seed(sub_seed)
            kwargs = dict(model_kwargs)
            if factory_accepts_seed:
                kwargs.setdefault("seed", sub_seed)
            models.append(model_factory(**kwargs))
    finally:
        torch.set_rng_state(global_rng_state)
    return CoupledEnsembleESNModel(models, aggregator=aggregate)
