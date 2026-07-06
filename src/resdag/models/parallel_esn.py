"""
Parallel Reservoir with Local States (Pathak et al. 2018)
=========================================================

This module provides :func:`parallel_esn` and :class:`ParallelESNModel`, the
**parallel / local-state** reservoir architecture of Pathak et al. 2018 for
forecasting large spatiotemporally chaotic systems (PDEs / weather / coupled
lattices) that a single global reservoir cannot scale to.

The idea: a spatially-extended state of dimension ``D`` is split into ``P``
overlapping local **patches** of core width ``w`` (``P = D / w``), each with a
halo of ``g`` neighbouring cells on either side (so each patch *reads* width
``w + 2g`` but *predicts* only its core width ``w``).  Every patch has its **own
reservoir state**, but they all **share one set of reservoir weights** — so the
``P`` patches advance as a **single batched matmul** with the patch axis playing
the role of the reservoir batch dimension.  Each patch's readout predicts only
its core region; the per-patch core predictions are stitched back into the full
``D``-dimensional next state, re-windowed into patches, and the loop repeats
(closed-loop autoregressive forecast).

Because the reservoir size ``N`` and the input/output widths (``w + 2g`` / ``w``)
stay fixed as ``D`` grows — only the batch count ``P`` grows — the per-step cost
is *flat in ``D`` on a GPU* (one batched op over ``P``).  This is what lets the
architecture forecast ``D`` in the hundreds or thousands where a single
``D``-wide global reservoir would need an unaffordably large ``N`` (and would
lose to a comparable-total-size parallel model on valid-prediction-time).

References
----------
J. Pathak, B. Hunt, M. Girvan, Z. Lu, and E. Ott, "Model-Free Prediction of
Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing
Approach," Phys. Rev. Lett. 120, 024102 (2018).

See Also
--------
resdag.datasets.kuramoto_sivashinsky : The canonical spatiotemporal benchmark.
resdag.models.classic_esn : Single global reservoir (the baseline it beats at
    scale).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.layers import ESNLayer
from resdag.layers.readouts import ReadoutLayer

from ._builder import ReadoutSpec, _resolve_readout_cls


def _patch_gather_index(input_dim: int, core_width: int, halo: int) -> torch.Tensor:
    """Build the periodic per-patch gather index for windowing a full state.

    Partitions ``input_dim`` into ``P = input_dim / core_width`` contiguous
    cores and returns, for each patch, the ``core_width + 2 * halo`` **wrapped**
    spatial indices it reads (its core plus ``halo`` neighbours on each side,
    with periodic boundaries — the standard convention for the Kuramoto-
    Sivashinsky and other periodic-domain benchmarks).

    A single ``state[..., idx]`` with the returned index then windows the whole
    field into all ``P`` patches at once (no Python loop over patches), which is
    what lets the patches advance as one batched reservoir op.

    Parameters
    ----------
    input_dim : int
        Full spatial dimension ``D`` of the state.
    core_width : int
        Core width ``w`` predicted by each patch.  Must divide ``input_dim``.
    halo : int
        Number of neighbouring cells ``g`` added on each side of the core.

    Returns
    -------
    torch.Tensor
        Long tensor of shape ``(P, core_width + 2 * halo)`` where row ``p`` holds
        the periodic indices for patch ``p`` (``P = input_dim // core_width``).

    Raises
    ------
    ValueError
        If ``core_width`` does not divide ``input_dim``, or if any argument is
        out of range (non-positive core width, negative halo).
    """
    if core_width <= 0:
        raise ValueError(f"core_width must be a positive integer, got {core_width}.")
    if halo < 0:
        raise ValueError(f"halo must be a non-negative integer, got {halo}.")
    if input_dim % core_width != 0:
        raise ValueError(
            f"core_width ({core_width}) must divide input_dim ({input_dim}) evenly "
            f"so the domain partitions into equal patches."
        )
    n_patches = input_dim // core_width
    # Patch p's core starts at p*w; the windowed read spans [p*w - g, p*w + w + g).
    patch_starts = torch.arange(n_patches).unsqueeze(1) * core_width  # (P, 1)
    offsets = torch.arange(-halo, core_width + halo).unsqueeze(0)  # (1, w + 2g)
    # Periodic wrap so patches near the boundary read across the seam.
    return (patch_starts + offsets) % input_dim  # (P, w + 2g)


class ParallelESNModel(torch.nn.Module):
    """Parallel local-state reservoir for spatiotemporal forecasting.

    Holds one shared-weight :class:`~resdag.layers.ESNLayer` (advanced batched
    over patches) and one weight-tied :class:`~resdag.layers.readouts.ReadoutLayer`
    (fitted once over the pooled per-patch data), plus the patch geometry
    (``input_dim``, ``core_width``, ``halo``).  Build it with
    :func:`parallel_esn` rather than directly.

    The public workflow mirrors the rest of the library — :meth:`fit` then
    :meth:`forecast` — but the forecasting is spatial-autoregressive: it stitches
    per-patch core predictions into the full field and re-windows every step,
    rather than feeding a single output straight back.

    Parameters
    ----------
    reservoir : ESNLayer
        The shared reservoir.  Its ``feedback_size`` must equal
        ``core_width + 2 * halo`` (the per-patch input width).
    readout : ReadoutLayer
        The weight-tied readout.  Its ``in_features`` must equal the reservoir
        size and its ``out_features`` must equal ``core_width``.
    input_dim : int
        Full spatial dimension ``D`` of the modelled field.
    core_width : int
        Core width ``w`` each patch predicts (``P = input_dim // core_width``).
    halo : int
        Halo width ``g`` (neighbouring cells read on each side of the core).

    Attributes
    ----------
    reservoir : ESNLayer
        The shared reservoir layer.
    readout : ReadoutLayer
        The weight-tied readout layer.
    n_patches : int
        Number of patches ``P = input_dim // core_width``.

    See Also
    --------
    parallel_esn : Factory that builds and wires this class.
    """

    #: Non-persistent buffer holding the ``(P, w + 2g)`` periodic gather index.
    patch_index: torch.Tensor

    def __init__(
        self,
        reservoir: ESNLayer,
        readout: ReadoutLayer,
        input_dim: int,
        core_width: int,
        halo: int,
    ) -> None:
        super().__init__()
        patch_width = core_width + 2 * halo
        if reservoir.feedback_size != patch_width:
            raise ValueError(
                f"reservoir.feedback_size ({reservoir.feedback_size}) must equal the "
                f"per-patch input width core_width + 2*halo = {patch_width}."
            )
        if readout.out_features != core_width:
            raise ValueError(
                f"readout.out_features ({readout.out_features}) must equal core_width "
                f"({core_width}) — each patch predicts only its core region."
            )
        if readout.in_features != reservoir.reservoir_size:
            raise ValueError(
                f"readout.in_features ({readout.in_features}) must equal the reservoir "
                f"size ({reservoir.reservoir_size})."
            )

        self.reservoir = reservoir
        self.readout = readout
        self.input_dim = input_dim
        self.core_width = core_width
        self.halo = halo
        self.n_patches = input_dim // core_width
        # Non-persistent: rebuilt cheaply from geometry, moves with ``.to()``.
        self.register_buffer(
            "patch_index",
            _patch_gather_index(input_dim, core_width, halo),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Windowing / stitching (batched over patches — no Python loop)
    # ------------------------------------------------------------------

    def window(self, series: torch.Tensor) -> torch.Tensor:
        """Window a full ``(batch, T, D)`` field into per-patch inputs.

        Gathers, for every patch, its ``core_width + 2 * halo`` periodic cells in
        one indexing op, then folds the patch axis into the batch axis so the
        result drops straight into the shared reservoir as a batched sequence.

        Parameters
        ----------
        series : torch.Tensor
            Full field of shape ``(batch, T, input_dim)``.

        Returns
        -------
        torch.Tensor
            Windowed input of shape ``(batch * P, T, core_width + 2 * halo)`` —
            the patch axis is folded into the leading (reservoir-batch) axis.
        """
        # series: (B, T, D) -> gather patch cells -> (B, T, P, w+2g).
        windowed = series[..., self.patch_index]
        b, t, n_patches, patch_width = windowed.shape
        # Move patches next to the batch axis and fold them in: (B*P, T, w+2g).
        windowed = windowed.permute(0, 2, 1, 3).reshape(b * n_patches, t, patch_width)
        return windowed

    def _cores_from_index(self) -> torch.Tensor:
        """Column offsets of the core region inside each patch window (``(w,)``)."""
        return torch.arange(self.halo, self.halo + self.core_width, device=self.patch_index.device)

    def stitch(self, patch_cores: torch.Tensor) -> torch.Tensor:
        """Assemble per-patch core predictions into the full field.

        Inverse of :meth:`window` for the core regions: patch ``p`` owns cores
        ``[p*w : (p+1)*w]`` of the full field, so concatenating the patches' core
        predictions along the spatial axis reconstructs the whole ``D``-wide
        state.  With non-overlapping cores this is a contiguous reshape (no
        averaging needed).

        Parameters
        ----------
        patch_cores : torch.Tensor
            Per-patch core predictions of shape ``(batch, P, core_width)``.

        Returns
        -------
        torch.Tensor
            Full field of shape ``(batch, input_dim)``.
        """
        b = patch_cores.shape[0]
        return patch_cores.reshape(b, self.input_dim)

    # ------------------------------------------------------------------
    # Reservoir-state management (delegates to the shared reservoir)
    # ------------------------------------------------------------------

    def reset_reservoirs(self) -> None:
        """Reset the shared reservoir's per-patch states (lazy re-init)."""
        self.reservoir.reset_state()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        warmup: torch.Tensor,
        train: torch.Tensor,
        targets: torch.Tensor,
        *,
        reset: bool = True,
    ) -> None:
        """Fit the weight-tied readout over all patches in one pooled solve.

        Windows ``warmup`` / ``train`` into per-patch inputs, runs them through
        the shared reservoir (batched over ``batch * P``), and fits the single
        readout on the pooled ``(reservoir_state -> core_target)`` pairs from
        every patch and timestep — the weight-tied readout of Pathak et al.

        Parameters
        ----------
        warmup : torch.Tensor
            Teacher-forced warmup field of shape ``(batch, warmup_steps, D)``
            used only to synchronise the reservoir states.
        train : torch.Tensor
            Training field of shape ``(batch, train_steps, D)`` whose reservoir
            states are regressed onto ``targets``.
        targets : torch.Tensor
            Next-step field of shape ``(batch, train_steps, D)`` — typically
            ``train`` shifted one step ahead.  Each patch is supervised on its
            own core slice of this field.
        reset : bool, default=True
            If ``True``, reset the reservoir states before warmup.

        Raises
        ------
        ValueError
            If any input is not 3-D, if their feature dimension is not
            ``input_dim``, or if ``train`` and ``targets`` disagree on
            batch/length.
        """
        for name, arr in (("warmup", warmup), ("train", train), ("targets", targets)):
            if arr.dim() != 3:
                raise ValueError(f"{name} must be 3-D (batch, T, D); got shape {tuple(arr.shape)}.")
            if arr.shape[-1] != self.input_dim:
                raise ValueError(
                    f"{name} has feature dim {arr.shape[-1]}, expected input_dim={self.input_dim}."
                )
        if train.shape[:2] != targets.shape[:2]:
            raise ValueError(
                f"train and targets must share (batch, T); got {tuple(train.shape[:2])} "
                f"vs {tuple(targets.shape[:2])}."
            )

        if reset:
            self.reset_reservoirs()

        with torch.no_grad():
            # Warmup: synchronise the per-patch reservoir states.
            self.reservoir(self.window(warmup))
            # Train pass: collect reservoir states for every patch and step.
            states = self.reservoir(self.window(train))  # (B*P, train_steps, N)

        # Per-patch core targets: window the target field and keep only the core
        # columns (offsets [halo, halo + core_width)).  Same (B*P, T) folding as
        # the states, so state[i] and target[i] are the same patch/step pair.
        windowed_targets = self.window(targets)  # (B*P, T, w+2g)
        core_targets = windowed_targets[..., self._cores_from_index()]  # (B*P, T, w)

        # One pooled ridge solve over all patches (weight-tied readout).  fit()
        # flattens (B*P, T, .) -> (B*P*T, .) internally.
        self.readout.fit(states, core_targets)

    # ------------------------------------------------------------------
    # Forecasting (spatial-autoregressive, closed loop)
    # ------------------------------------------------------------------

    def forecast(
        self,
        warmup: torch.Tensor,
        *,
        horizon: int,
        reset: bool = True,
        return_warmup: bool = False,
    ) -> torch.Tensor:
        """Closed-loop autoregressive forecast of the full field.

        Warms the reservoir on ``warmup``, then autonomously advances the field:
        each step reads the ``P`` reservoir states, applies the shared readout to
        get per-patch core predictions, **stitches** them into the full next
        field, **re-windows** that field into patches, and steps the reservoir —
        the whole batch of patches advancing as one op per step.

        Parameters
        ----------
        warmup : torch.Tensor
            Teacher-forced warmup field of shape ``(batch, warmup_steps, D)``.
        horizon : int, keyword-only
            Number of autonomous steps to generate (``>= 1``).
        reset : bool, default=True
            If ``True``, reset the reservoir states before warmup.
        return_warmup : bool, default=False
            If ``True``, prepend the (teacher-forced) warmup field to the result.

        Returns
        -------
        torch.Tensor
            Forecast field of shape ``(batch, horizon, D)`` (or
            ``(batch, warmup_steps + horizon, D)`` if ``return_warmup``).

        Raises
        ------
        ValueError
            If ``warmup`` is not 3-D with feature dim ``input_dim``, if
            ``horizon < 1``, or if the readout has not been fitted.
        """
        if warmup.dim() != 3 or warmup.shape[-1] != self.input_dim:
            raise ValueError(
                f"warmup must be 3-D (batch, T, input_dim={self.input_dim}); "
                f"got shape {tuple(warmup.shape)}."
            )
        if horizon < 1:
            raise ValueError(f"horizon must be a positive integer, got {horizon}.")
        if not self.readout.is_fitted:
            raise ValueError("forecast() requires a fitted readout; call fit() first.")

        batch = warmup.shape[0]

        with torch.no_grad():
            if reset:
                self.reset_reservoirs()
            # Warmup: teacher-force every step *except the last* so the closed
            # loop can feed that last real field itself.  This mirrors training
            # exactly — the reservoir state that consumed field ``k`` is the one
            # regressed onto field ``k + 1`` — so there is no double-feed of, or
            # gap before, the seam.  (A warmup of length 1 skips this pass and
            # seeds the loop directly from that single observed field.)
            if warmup.shape[1] > 1:
                self.reservoir(self.window(warmup[:, :-1, :]))

            # Seed the closed loop from the last observed full field.
            current = warmup[:, -1, :]  # (batch, D)
            outputs = torch.empty(
                batch, horizon, self.input_dim, dtype=warmup.dtype, device=warmup.device
            )

            for t in range(horizon):
                # Window the current field into (batch*P, 1, w+2g) and step the
                # reservoir once (all patches advance in one batched op); the
                # readout then predicts each patch's core of the *next* field.
                windowed = self.window(current.unsqueeze(1))  # (batch*P, 1, w+2g)
                state_seq = self.reservoir(windowed)  # (batch*P, 1, N)
                cores = self.readout(state_seq[:, 0, :])  # (batch*P, w)
                # Unfold the patch axis back out and stitch into the full field.
                cores = cores.reshape(batch, self.n_patches, self.core_width)
                current = self.stitch(cores)  # (batch, D)
                outputs[:, t, :] = current

            if return_warmup:
                return torch.cat([warmup, outputs], dim=1)
            return outputs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_full(self, path: str | Path, **metadata: Any) -> None:
        """Serialize the whole model (architecture + weights + states) to a file.

        Mirrors :meth:`resdag.core.ESNModel.save_full`: pickles the entire module
        so :meth:`load_full` reconstructs it with no pre-built architecture.  The
        file is a ``torch.save`` payload loaded back with ``weights_only=False`` —
        only open files you trust.

        Parameters
        ----------
        path : str or Path
            File path to save to. Parent directories are created if needed.
        **metadata
            Additional metadata stored alongside the model.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"resdag_parallel_model": self, "metadata": metadata}, path)

    @classmethod
    def load_full(
        cls,
        path: str | Path,
        return_metadata: bool = False,
        map_location: Any = None,
    ) -> "ParallelESNModel | tuple[ParallelESNModel, dict[str, Any]]":
        """Reconstruct a model saved with :meth:`save_full`.

        Parameters
        ----------
        path : str or Path
            File path written by :meth:`save_full`.
        return_metadata : bool, default=False
            If ``True``, return ``(model, metadata)`` instead of just the model.
        map_location : optional
            Passed to ``torch.load`` to remap storage devices.

        Returns
        -------
        ParallelESNModel or tuple of (ParallelESNModel, dict)
            The reconstructed model, optionally paired with its metadata.

        Raises
        ------
        ValueError
            If the file does not contain a parallel model.

        Warnings
        --------
        Loads with ``weights_only=False`` (unpickles arbitrary objects); only
        call on trusted files.
        """
        path = Path(path)
        payload = torch.load(path, weights_only=False, map_location=map_location)
        model: ParallelESNModel
        if isinstance(payload, dict) and "resdag_parallel_model" in payload:
            model = payload["resdag_parallel_model"]
            metadata = payload.get("metadata", {})
        elif isinstance(payload, ParallelESNModel):
            model, metadata = payload, {}
        else:
            raise ValueError(
                f"{path} does not contain a ParallelESNModel. Write one with save_full()."
            )
        if return_metadata:
            return model, metadata
        return model

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"ParallelESNModel(input_dim={self.input_dim}, core_width={self.core_width}, "
            f"halo={self.halo}, n_patches={self.n_patches}, "
            f"reservoir_size={self.reservoir.reservoir_size})"
        )


def parallel_esn(
    input_dim: int,
    core_width: int,
    halo: int,
    reservoir_size: int = 500,
    *,
    # Reservoir params
    topology: TopologySpec | None = None,
    spectral_radius: float = 0.6,
    leak_rate: float = 1.0,
    input_scaling: float = 0.1,
    feedback_initializer: InitializerSpec | None = None,
    activation: str = "tanh",
    bias: bool = True,
    trainable: bool = False,
    seed: Any = None,
    # Readout params
    readout: ReadoutSpec = "ridge",
    readout_alpha: float = 1e-4,
    readout_bias: bool = True,
    readout_name: str = "output",
    # Extra reservoir kwargs
    **reservoir_kwargs: Any,
) -> ParallelESNModel:
    r"""Build a parallel local-state reservoir (Pathak et al. 2018).

    Splits a spatially-extended field of dimension ``input_dim`` into ``P =
    input_dim / core_width`` overlapping patches (core width ``core_width``, halo
    ``halo`` on each side, periodic boundaries), all sharing **one** reservoir
    (advanced batched over patches) and **one** weight-tied readout (core width
    ``core_width``).  See :class:`ParallelESNModel` and the module docstring for
    the full architecture and the forecasting loop.

    The defaults (``spectral_radius=0.6``, ``input_scaling=0.1``,
    ``readout_alpha=1e-4``) are tuned for the local-patch regime, which uses a
    smaller, better-conditioned reservoir per patch than a single global ESN —
    the classic Pathak KS settings.  ``input_scaling=0.1`` in particular is
    decisive: the per-patch input width (``core_width + 2*halo``) is large enough
    that an unscaled ``uniform(-1, 1)`` input map drives ``tanh`` deep into
    saturation, collapsing the reservoir's memory; scaling the input weights down
    keeps the dynamics responsive (on the canonical L=22 KS field it is the
    difference between a fraction of a Lyapunov time and ~4 Lyapunov times of
    valid prediction).

    Parameters
    ----------
    input_dim : int
        Full spatial dimension ``D`` of the field (e.g. the KS grid size).
    core_width : int
        Core width ``w`` each patch predicts.  Must divide ``input_dim``.
    halo : int
        Halo width ``g`` — neighbouring cells read on each side of the core
        (periodic).  The per-patch reservoir input width is ``core_width + 2 *
        halo``.
    reservoir_size : int, default=500
        Number of units ``N`` in the shared reservoir (per patch).
    topology : TopologySpec, optional
        Topology for the recurrent weights.  Accepts a registry name, a
        ``(name, params)`` tuple, a callable, or a configured
        :class:`~resdag.init.topology.TopologyInitializer`.
    spectral_radius : float, default=0.6
        Target spectral radius for the recurrent weights.
    leak_rate : float, default=1.0
        Leaky-integration rate (1.0 = no leak).
    input_scaling : float, default=0.1
        Scale of the per-patch input (feedback) weights when
        ``feedback_initializer`` is left at its default.  The weights are drawn
        from ``uniform(-input_scaling, input_scaling)`` (via the ``"random"``
        initializer), keeping the ``tanh`` reservoir out of saturation on a
        wide per-patch input.  Ignored when an explicit ``feedback_initializer``
        is supplied (it then owns the input-weight scale).
    feedback_initializer : InitializerSpec, optional
        Initializer for the per-patch input weights.  Same accepted forms as
        ``topology``.  When given, it overrides ``input_scaling``.  Left at its
        default (``None``), the factory uses ``("random", {"input_scaling":
        input_scaling})``.
    activation : str, default="tanh"
        Activation function (``"tanh"``, ``"relu"``, ``"sigmoid"``,
        ``"identity"``).
    bias : bool, default=True
        Whether the reservoir uses a bias term.
    trainable : bool, default=False
        Whether the reservoir weights are trainable.
    seed : int or torch.Generator, optional
        Reproducibility seed fixing every reservoir parameter.
    readout : str or type[ReadoutLayer], default="ridge"
        Which readout to attach.  ``"ridge"`` (default) is the exact Cholesky
        ridge solve; ``"cg"`` is the legacy iterative solver; a
        :class:`~resdag.layers.readouts.ReadoutLayer` subclass may be passed.
    readout_alpha : float, default=1e-4
        Ridge regularization for the readout.
    readout_bias : bool, default=True
        Whether the readout uses a bias term.
    readout_name : str, default="output"
        Name for the readout layer.
    **reservoir_kwargs : Any
        Additional keyword arguments forwarded to :class:`~resdag.layers.ESNLayer`
        (e.g. ``sparse``, ``noise``, ``input_initializer``).

    Returns
    -------
    ParallelESNModel
        The assembled parallel reservoir, ready for :meth:`~ParallelESNModel.fit`
        and :meth:`~ParallelESNModel.forecast`.

    Raises
    ------
    ValueError
        If ``core_width`` does not divide ``input_dim``, or if a size argument is
        out of range.

    Examples
    --------
    Forecast a Kuramoto-Sivashinsky field with local patches::

    >>> import torch
    >>> from resdag import kuramoto_sivashinsky
    >>> from resdag.models import parallel_esn
    >>> data = kuramoto_sivashinsky(3000, n_grid=64, dtype=torch.float64)
    >>> model = parallel_esn(input_dim=64, core_width=4, halo=6, reservoir_size=400,
    ...                      seed=0)
    >>> warmup, train = data[:, :200], data[:, 200:2200]
    >>> model.fit(warmup, train[:, :-1], train[:, 1:])   # doctest: +SKIP
    >>> pred = model.forecast(train[:, -100:], horizon=400)  # doctest: +SKIP

    See Also
    --------
    ParallelESNModel : The model class this factory builds.
    resdag.datasets.kuramoto_sivashinsky : The canonical spatiotemporal benchmark.
    resdag.models.classic_esn : Single global reservoir baseline.
    """
    if input_dim % core_width != 0:
        raise ValueError(f"core_width ({core_width}) must divide input_dim ({input_dim}) evenly.")
    if halo < 0:
        raise ValueError(f"halo must be a non-negative integer, got {halo}.")
    patch_width = core_width + 2 * halo

    # Default the per-patch input map to a *scaled* random init: on a wide
    # per-patch input an unscaled ``uniform(-1, 1)`` saturates the tanh
    # reservoir (see the ``input_scaling`` docstring).  An explicit
    # ``feedback_initializer`` overrides this and owns the input-weight scale.
    if feedback_initializer is None:
        feedback_initializer = ("random", {"input_scaling": input_scaling})

    reservoir = ESNLayer(
        reservoir_size=reservoir_size,
        feedback_size=patch_width,
        topology=topology,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        feedback_initializer=feedback_initializer,
        activation=activation,
        bias=bias,
        trainable=trainable,
        seed=seed,
        **reservoir_kwargs,
    )

    readout_cls = _resolve_readout_cls(readout)
    readout_layer = readout_cls(
        in_features=reservoir_size,
        out_features=core_width,
        bias=readout_bias,
        alpha=readout_alpha,  # type: ignore[call-arg]
        name=readout_name,
    )

    return ParallelESNModel(
        reservoir=reservoir,
        readout=readout_layer,
        input_dim=input_dim,
        core_width=core_width,
        halo=halo,
    )
