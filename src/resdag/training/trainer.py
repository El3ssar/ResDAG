"""
ESN Trainer
===========

This module provides :class:`ESNTrainer`, which trains ESN models by
fitting readout layers algebraically in topological order using a
single forward pass.

The trainer uses PyTorch forward hooks to fit each readout layer just
before its forward method executes, ensuring downstream layers receive
outputs from already-fitted readouts.

For sequences too long to materialise in memory, :meth:`ESNTrainer.fit_stream`
fits :class:`~resdag.layers.IncrementalRidgeReadoutLayer` readouts over a stream of
contiguous chunks (e.g. a :class:`torch.utils.data.DataLoader`), accumulating
ridge sufficient statistics per chunk and solving once at the end.

See Also
--------
resdag.ESNModel : ESN model class.
resdag.layers.readouts.CGReadoutLayer : Conjugate gradient readout layer.
resdag.layers.readouts.IncrementalRidgeReadoutLayer : Streaming partial_fit readout.
"""

import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager

import torch

from resdag.core import ESNModel
from resdag.layers import IncrementalRidgeReadoutLayer, ReadoutLayer, RLSReadout

# A streaming chunk: a tuple of contiguous input tensors (feedback, driver1,
# ...) paired with a dict mapping each readout name to its target tensor for
# that chunk.
StreamChunk = tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]


class ESNTrainer:
    """
    Trainer for ESN models with algebraic readout fitting.

    This trainer fits all :class:`ReadoutLayer` instances in an ESN model
    using algebraic methods (e.g., ridge regression) rather than gradient
    descent. The fitting is performed efficiently in a single forward pass
    using pre-hooks that intercept inputs to each readout.

    The training process:

    1. Reset reservoir states
    2. Warmup phase: synchronize reservoir states with input dynamics
    3. Single forward pass with pre-hooks that fit each readout in
       topological order before it processes its input

    Each readout handles its own fitting hyperparameters (e.g., ``alpha``
    for ridge regression is set during layer construction).

    Parameters
    ----------
    model : ESNModel
        ESN model to train. Must contain at least one :class:`ReadoutLayer`.

    Attributes
    ----------
    model : ESNModel
        The ESN model being trained.

    Examples
    --------
    Basic training workflow:

    >>> from resdag.training import ESNTrainer
    >>> from resdag.core import ESNModel
    >>>
    >>> trainer = ESNTrainer(model)
    >>> trainer.fit(
    ...     warmup_inputs=(warmup_data,),
    ...     train_inputs=(train_data,),
    ...     targets={"output": train_targets},
    ... )

    Training with driving inputs:

    >>> trainer.fit(
    ...     warmup_inputs=(warmup_feedback, warmup_driver),
    ...     train_inputs=(train_feedback, train_driver),
    ...     targets={"output": targets},
    ... )

    Multi-readout training:

    >>> trainer.fit(
    ...     warmup_inputs=(warmup_data,),
    ...     train_inputs=(train_data,),
    ...     targets={
    ...         "position": position_targets,
    ...         "velocity": velocity_targets,
    ...     },
    ... )

    See Also
    --------
    ESNModel : ESN model class.
    CGReadoutLayer : Conjugate gradient readout layer.
    ESNModel.forecast : Forecasting after training.

    Notes
    -----
    - Warmup and training data must have the same number of input tensors.
    - Training data and targets must have the same sequence length.
    - Target keys must match readout names (user-defined or auto-generated).
    """

    def __init__(self, model: ESNModel) -> None:
        self.model = model

    def fit(
        self,
        warmup_inputs: tuple[torch.Tensor, ...],
        train_inputs: tuple[torch.Tensor, ...],
        targets: dict[str, torch.Tensor],
    ) -> None:
        """
        Fit all readout layers in a single forward pass.

        Uses pre-hooks to fit each readout layer just before its forward
        method executes. This ensures downstream layers receive outputs
        from already-fitted readouts.

        Parameters
        ----------
        warmup_inputs : tuple of torch.Tensor
            Warmup sequences for state synchronization.
            Format: ``(feedback, driver1, driver2, ...)``.
            Each tensor shape: ``(batch, warmup_steps, features)``.
        train_inputs : tuple of torch.Tensor
            Training sequences for fitting.
            Format: ``(feedback, driver1, driver2, ...)``.
            Each tensor shape: ``(batch, train_steps, features)``.
            Must have same sequence length as targets.
        targets : dict of str to torch.Tensor
            Dictionary mapping readout name to target tensor.
            Each target shape: ``(batch, train_steps, out_features)``.
            Names are either user-defined (via ``name="output"`` in readout
            constructor) or auto-generated module names (e.g., ``"CGReadoutLayer_1"``).

        Raises
        ------
        ValueError
            If no warmup or training inputs provided.
            If number of warmup and training inputs don't match.
            If any readout is missing from targets.
            If target sequence length doesn't match training inputs.

        Notes
        -----
        After calling ``fit()``, all readouts will have ``is_fitted=True`` and the model is ready for inference or forecasting.

        Emits a ``UserWarning`` if targets dict contains names not matching any readout.

        Examples
        --------
        >>> trainer = ESNTrainer(model)
        >>> trainer.fit(
        ...     warmup_inputs=(warmup_data,),
        ...     train_inputs=(train_data,),
        ...     targets={"output": targets},
        ... )
        >>> print(model.CGReadoutLayer_1.is_fitted)
        True
        """
        if len(warmup_inputs) == 0:
            raise ValueError("At least one warmup input is required")
        if len(train_inputs) == 0:
            raise ValueError("At least one training input is required")
        if len(warmup_inputs) != len(train_inputs):
            raise ValueError(
                f"warmup_inputs has {len(warmup_inputs)} tensors, "
                f"but train_inputs has {len(train_inputs)} tensors. Must match."
            )

        # Resolve readouts once; reuse for target validation and hook setup.
        readouts = self._discover_readouts()
        self._validate_targets(targets, readouts)

        train_steps = train_inputs[0].shape[1]

        # Validate target shapes
        readout_names = [name for name, _ in readouts]
        for name in readout_names:
            target = targets[name]
            if target.shape[1] != train_steps:
                raise ValueError(
                    f"Target for '{name}' has {target.shape[1]} timesteps, "
                    f"but train_inputs has {train_steps} timesteps. Must match."
                )

        # Single warmup to sync reservoir states (warmup resets by default).
        # Readouts are unfitted during warmup, so let any guarded readout (e.g.
        # IncrementalRidgeReadoutLayer) flow a value while the reservoir synchronises.
        with self._allow_unfitted_warmup(readouts):
            self.model.warmup(*warmup_inputs)

        # Register pre-hooks that fit each readout when its forward fires.
        # We don't need explicit topological order: ``self.model(*train_inputs)``
        # walks the graph in execution order, so each hook runs exactly when
        # its readout would naturally execute.
        hooks = []
        for name, readout in readouts:
            target = targets[name]

            def make_fit_hook(
                layer: ReadoutLayer, tgt: torch.Tensor
            ) -> Callable[[torch.nn.Module, tuple[torch.Tensor, ...]], None]:
                def hook(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
                    layer.fit(args[0], tgt)

                return hook

            handle = readout.register_forward_pre_hook(make_fit_hook(readout, target))
            hooks.append(handle)

        try:
            # Single forward pass - hooks fit each readout as it executes
            with torch.no_grad():
                self.model(*train_inputs)
        finally:
            # Always remove hooks
            for h in hooks:
                h.remove()

    def fit_stream(
        self,
        warmup_inputs: tuple[torch.Tensor, ...],
        chunks: Iterable[StreamChunk],
    ) -> None:
        """Fit :class:`IncrementalRidgeReadoutLayer` readouts over a stream of chunks.

        The streaming counterpart of :meth:`fit`. Instead of fitting from one
        in-memory ``(B, T, F)`` block, it warms the reservoir once and then
        consumes ``chunks`` one at a time — e.g. windows yielded by a
        :class:`torch.utils.data.DataLoader` — accumulating each readout's ridge
        sufficient statistics with
        :meth:`~resdag.layers.IncrementalRidgeReadoutLayer.partial_fit` and solving
        once at the end with
        :meth:`~resdag.layers.IncrementalRidgeReadoutLayer.finalize`. No more than a
        single chunk's states are ever held in memory, which is what lets a
        sequence too long to materialise — or one streamed off disk — be fitted.

        Because the reservoir is stateful, the chunks must be **contiguous in
        time and in order**: state flows from the end of one chunk into the
        start of the next, exactly as it would in a single long forward pass.
        Shuffling the chunks would desynchronise the reservoir and is not
        supported.

        Every readout in the model must be an
        :class:`~resdag.layers.IncrementalRidgeReadoutLayer` (the only readout with a
        ``partial_fit`` / ``finalize`` interface). The accumulators are reset at
        the start, so calling ``fit_stream`` again re-fits from scratch.

        Parameters
        ----------
        warmup_inputs : tuple of torch.Tensor
            Warmup sequences for state synchronization, format
            ``(feedback, driver1, ...)``, each of shape
            ``(batch, warmup_steps, features)``. The reservoir is reset before
            this pass (as in :meth:`fit`).
        chunks : iterable of (tuple of torch.Tensor, dict of str to torch.Tensor)
            An iterable yielding ``(inputs, targets)`` per chunk. ``inputs`` is
            an input tuple ``(feedback, driver1, ...)`` of the same arity as
            ``warmup_inputs``; ``targets`` maps each readout name to its target
            tensor of shape ``(batch, chunk_steps, out_features)``. Consumed
            lazily, so a generator or ``DataLoader`` never materialises the whole
            sequence.

        Raises
        ------
        ValueError
            If no warmup inputs are provided, if a chunk's input arity does not
            match the warmup arity, if any readout is missing from a chunk's
            targets, or if a target's sequence length does not match its chunk's
            input length.
        TypeError
            If any readout in the model is not an ``IncrementalRidgeReadoutLayer``.
        RuntimeError
            If ``chunks`` is empty (no statistics were accumulated, so there is
            nothing to finalize).

        Notes
        -----
        After ``fit_stream`` returns, every readout has ``is_fitted=True`` and
        the model is ready for inference or forecasting.

        Examples
        --------
        >>> from resdag.layers import IncrementalRidgeReadoutLayer
        >>> trainer = ESNTrainer(model)  # model uses IncrementalRidgeReadoutLayer
        >>> def chunk_stream():
        ...     for x, y in dataloader:  # contiguous windows  # doctest: +SKIP
        ...         yield (x,), {"output": y}
        >>> trainer.fit_stream(  # doctest: +SKIP
        ...     warmup_inputs=(warmup_data,),
        ...     chunks=chunk_stream(),
        ... )

        See Also
        --------
        fit : Single-pass in-memory fitting.
        resdag.layers.IncrementalRidgeReadoutLayer : The streaming readout.
        """
        if len(warmup_inputs) == 0:
            raise ValueError("At least one warmup input is required")

        readouts = self._discover_readouts()
        non_incremental = [
            name for name, ro in readouts if not isinstance(ro, IncrementalRidgeReadoutLayer)
        ]
        if non_incremental:
            raise TypeError(
                f"fit_stream() requires every readout to be an IncrementalRidgeReadoutLayer, "
                f"but these are not: {non_incremental}. Use fit() for single-pass readouts, "
                f"or rebuild the model with IncrementalRidgeReadoutLayer."
            )

        # Reset accumulators so a re-fit starts clean, then warm up once. The
        # readouts are unfitted during warmup too, so let their forward flow a
        # value while the reservoir synchronises (shared with ``fit``).
        for _, readout in readouts:
            assert isinstance(readout, IncrementalRidgeReadoutLayer)
            readout.reset_accumulators()
        with self._allow_unfitted_warmup(readouts):
            self.model.warmup(*warmup_inputs)

        # Per-chunk: register pre-hooks that accumulate sufficient statistics
        # for each readout, then run a forward pass that carries the reservoir
        # state forward into the next chunk.
        n_chunks = 0
        for inputs, targets in chunks:
            if len(inputs) != len(warmup_inputs):
                raise ValueError(
                    f"Chunk {n_chunks} has {len(inputs)} input tensors, but warmup_inputs "
                    f"has {len(warmup_inputs)}. Must match."
                )
            self._validate_targets(targets, readouts)
            chunk_steps = inputs[0].shape[1]

            hooks: list[torch.utils.hooks.RemovableHandle] = []
            for name, readout in readouts:
                target = targets[name]
                if target.shape[1] != chunk_steps:
                    for h in hooks:
                        h.remove()
                    raise ValueError(
                        f"Target for '{name}' in chunk {n_chunks} has {target.shape[1]} "
                        f"timesteps, but the chunk inputs have {chunk_steps}. Must match."
                    )

                def make_partial_fit_hook(
                    layer: IncrementalRidgeReadoutLayer, tgt: torch.Tensor
                ) -> Callable[[torch.nn.Module, tuple[torch.Tensor, ...]], None]:
                    def hook(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
                        layer.partial_fit(args[0], tgt)

                    return hook

                assert isinstance(readout, IncrementalRidgeReadoutLayer)
                handle = readout.register_forward_pre_hook(make_partial_fit_hook(readout, target))
                hooks.append(handle)
                # The readout is unfitted during accumulation; its forward only
                # needs to flow a value downstream, so bypass the fitted guard.
                readout._allow_unfitted_forward = True

            try:
                with torch.no_grad():
                    self.model(*inputs)
            finally:
                for h in hooks:
                    h.remove()
                for _, readout in readouts:
                    assert isinstance(readout, IncrementalRidgeReadoutLayer)
                    readout._allow_unfitted_forward = False
            n_chunks += 1

        if n_chunks == 0:
            raise RuntimeError(
                "fit_stream() received no chunks; nothing was accumulated to finalize."
            )

        # Solve every readout once from its accumulated statistics.
        for _, readout in readouts:
            assert isinstance(readout, IncrementalRidgeReadoutLayer)
            readout.finalize()

    def fit_force(
        self,
        warmup_inputs: tuple[torch.Tensor, ...],
        targets: torch.Tensor,
        *,
        forgetting: float | None = None,
        delta: float | None = None,
        initial_feedback: torch.Tensor | None = None,
        reset: bool = True,
    ) -> torch.Tensor:
        r"""FORCE-train the readout to autonomously *generate* a target pattern.

        Implements FORCE learning (Sussillo & Abbott 2009, "Generating Coherent
        Patterns of Activity from Chaotic Neural Networks"): the reservoir is run
        **closed-loop** — driven at every step by the network's *own* readout
        output fed back as the next input — while the single
        :class:`~resdag.layers.RLSReadout` is updated online (one rank-1 RLS step
        per timestep) toward the target signal ``d_t`` at that step.  Because the
        readout is trained *inside* the feedback loop it will actually run in, the
        fed-back output progressively locks onto the target; after training the
        network **free-runs** and reproduces the pattern autonomously.

        When to use FORCE (honest scope)
        --------------------------------
        FORCE's value is training the readout *inside the closed loop*, so its
        weights are fit against the states the loop actually visits rather than
        the teacher-forced states an open-loop fit sees. Its classic advantage is
        **stabilising autonomous generation in regimes where an open-loop
        teacher-forced ridge fit would diverge in closed loop** — high-dimensional
        or strongly chaotic reservoirs where teacher-forced and free-running state
        trajectories pull apart and open-loop errors compound.

        It is **not** a free win over open-loop ridge in general. For a *clean,
        low-dimensional periodic* target trained over a long window, an open-loop
        teacher-forced ridge readout (:meth:`fit`) is already an excellent
        closed-loop generator — the smooth target leaves little error to compound
        — and FORCE will *match* it rather than beat it (empirically, across
        seeds, FORCE reproduces such a target to a small fraction of its scale and
        stays competitive with ridge, but any seed-specific win is not a
        guaranteed property). Reach for FORCE when the reservoir is chaotic enough
        that the open-loop free run is unstable, or when the target is
        non-stationary and you want the online / forgetting machinery of
        :class:`RLSReadout`; for a stationary, easily-generated target, the
        simpler :meth:`fit` is often sufficient.

        Per step ``t`` (after an optional teacher-forced warmup that settles the
        reservoir on the target):

        1. Feed the current feedback ``z_t`` (the network's own last output, or
           the warmup's for the first step) through the model, reading the
           readout's input feature vector ``r_t`` (reservoir state, possibly
           concatenated with the input, exactly as the graph presents it).
        2. RLS-update the readout toward ``d_t`` using ``r_t`` — the a-priori
           error ``d_t - W_t^\top r_t`` drives a rank-1 update of ``W`` and the
           inverse-correlation matrix ``P`` (see :class:`RLSReadout`).
        3. Recompute the output ``z_{t+1} = W_{t+1}^\top r_t`` with the *updated*
           weights and feed it back as the next step's input.

        The readout's RLS state (``P`` and the weights) is reset at the start of
        the FORCE pass (via :meth:`RLSReadout.reset_state`) so a re-run trains
        from scratch; pass ``forgetting`` / ``delta`` to override the readout's
        recursion hyperparameters for the pass.

        Parameters
        ----------
        warmup_inputs : tuple of torch.Tensor
            Teacher-forced warmup driving the reservoir *open-loop* with the true
            signal to settle its state before the closed-loop FORCE phase.
            Format ``(feedback, driver1, ...)``; the feedback is typically the
            leading slice of the target series.  Each tensor has shape
            ``(batch, warmup_steps, features)``.  May be empty-length in time
            (``warmup_steps == 0``) to start FORCE from the reset state, but at
            least the feedback tensor must be present (its dtype/device/batch and
            the driver arity are read from it).
        targets : torch.Tensor
            The pattern to generate, shape ``(batch, T, feedback_dim)``.  Its
            feature dimension **must** equal the model's feedback input dimension
            (and the readout's ``out_features``) — the autoregression contract,
            since the readout output is fed back as the next feedback.  ``T`` is
            the number of closed-loop FORCE steps.
        forgetting : float, optional
            Overrides the readout's forgetting factor :math:`\lambda \in (0, 1]`
            for this pass.  ``None`` (default) keeps the readout's configured
            value.  FORCE is classically run with ``1.0`` (growing window).
        delta : float, optional
            Overrides the readout's :math:`P_0 = I / \delta` prior for this pass.
            ``None`` (default) keeps the readout's configured value.  A moderate
            ``delta`` (around ``1.0``) is the standard FORCE choice; a very small
            ``delta`` (huge initial ``P``) makes the early closed-loop updates
            explode.
        initial_feedback : torch.Tensor, optional
            Feedback of shape ``(batch, 1, feedback_dim)`` seeding the first
            closed-loop step.  If ``None`` (default) the last warmup target slice
            is used (or a zero seed when the warmup is empty).
        reset : bool, default=True
            If ``True`` (default), reset the reservoir states before the
            teacher-forced warmup.  Set ``False`` to continue from the current
            (already warmed) reservoir state.

        Returns
        -------
        torch.Tensor
            The closed-loop training outputs of shape ``(batch, T, feedback_dim)``
            — the network's own generated signal at each FORCE step, *after* that
            step's RLS update.  By the end of training these track ``targets``
            closely; free-run generation with :meth:`ESNModel.forecast` continues
            the pattern from the final reservoir state.

        Raises
        ------
        TypeError
            If the model's readout is not an :class:`RLSReadout` (FORCE needs the
            online RLS ``partial_fit`` machinery).
        ValueError
            If the model has more than one readout (FORCE trains a single
            generator readout), if ``warmup_inputs`` is empty, if ``targets`` is
            not 3-D, if the readout's ``out_features`` (and hence the fed-back
            output dimension) does not match ``targets``'s feature dimension (the
            autoregression contract), if the driver arity of ``warmup_inputs``
            does not match the model, or if ``initial_feedback`` has the wrong
            shape.

        Notes
        -----
        - **Single-readout, self-generating only.**  FORCE here trains exactly one
          readout whose output is fed straight back as the model's feedback, so
          the readout must be an ``RLSReadout`` and its ``out_features`` must equal
          the feedback dimension.  Multi-readout or driven-generation setups are
          out of scope and rejected with a clear error.
        - **Driven models.**  If the model has driver inputs, pass the driver
          warmup slices in ``warmup_inputs``; during the closed-loop phase the
          drivers are held at their last warmup value (FORCE classically targets
          autonomous generation, not driven mapping).  A feedback-only model is
          the common case.
        - **Reservoir regime.**  FORCE works best with a *chaotic* reservoir
          (spectral radius ``> 1``, e.g. ``1.5``); the online readout tames the
          chaos into the target pattern.  See the module example.

        Examples
        --------
        >>> from resdag.models import classic_esn
        >>> from resdag import RLSReadout
        >>> from resdag.training import ESNTrainer
        >>> model = classic_esn(  # doctest: +SKIP
        ...     600, feedback_size=1, output_size=1, spectral_radius=1.5,
        ...     readout=RLSReadout, readout_name="output", readout_bias=False,
        ... )
        >>> trainer = ESNTrainer(model)  # doctest: +SKIP
        >>> trainer.fit_force(  # doctest: +SKIP
        ...     warmup_inputs=(target[:, :200, :],),
        ...     targets=target[:, 200:, :],
        ...     delta=1.0,
        ... )
        >>> generated = model.forecast(target, horizon=1000)  # doctest: +SKIP

        See Also
        --------
        fit : Open-loop teacher-forced algebraic readout fitting.
        resdag.layers.RLSReadout : The online RLS readout FORCE drives.
        resdag.core.ESNModel.forecast : Free-run generation after FORCE training.
        """
        if len(warmup_inputs) == 0:
            raise ValueError(
                "fit_force() requires at least one warmup input (the feedback tensor), "
                "even if it is length-0 in time; the model's dtype/device/batch and "
                "driver arity are read from it."
            )
        if targets.dim() != 3:
            raise ValueError(
                f"fit_force() targets must be 3-D (batch, T, feedback_dim); "
                f"got shape {tuple(targets.shape)}."
            )

        # FORCE trains exactly one online RLS readout whose output is the
        # generated signal fed straight back as feedback.
        readouts = self._discover_readouts()
        if len(readouts) != 1:
            names = [name for name, _ in readouts]
            raise ValueError(
                f"fit_force() trains a single self-generating readout, but the model has "
                f"{len(readouts)} readout(s): {names}. FORCE feeds one readout's output "
                f"straight back as the model feedback; multi-readout generation is out of "
                f"scope."
            )
        readout_name, readout = readouts[0]
        if not isinstance(readout, RLSReadout):
            raise TypeError(
                f"fit_force() requires the readout ('{readout_name}') to be an RLSReadout "
                f"(the online recursive-least-squares readout whose partial_fit drives the "
                f"FORCE update), but it is a {type(readout).__name__}. Rebuild the model "
                f"with readout=RLSReadout."
            )

        feedback_dim = targets.shape[-1]
        if readout.out_features != feedback_dim:
            raise ValueError(
                f"fit_force(): the readout ('{readout_name}') has out_features="
                f"{readout.out_features}, but targets have feedback_dim={feedback_dim}. "
                f"FORCE feeds the readout output back as the next feedback, so the readout "
                f"output dimension must equal the feedback (target) dimension."
            )

        num_drivers = len(warmup_inputs) - 1
        expected_drivers = len(self.model.inputs) - 1
        if num_drivers != expected_drivers:
            raise ValueError(
                f"fit_force(): the model has {expected_drivers} driver input(s) but "
                f"warmup_inputs carries {num_drivers}. Pass (feedback, driver1, ...) matching "
                f"the model's inputs."
            )

        # Driven models hold each driver at its last warmup value during the
        # closed-loop phase, so an empty warmup leaves no value to hold.
        if num_drivers and warmup_inputs[0].shape[1] == 0:
            raise ValueError(
                "fit_force(): a driven model needs a non-empty warmup so each driver has a "
                "last value to hold during the autonomous closed-loop phase; the warmup "
                "feedback has 0 timesteps. Provide at least one warmup step."
            )

        batch_size = warmup_inputs[0].shape[0]
        device = warmup_inputs[0].device
        dtype = warmup_inputs[0].dtype

        if initial_feedback is not None:
            expected = (batch_size, 1, feedback_dim)
            if initial_feedback.dim() != 3 or tuple(initial_feedback.shape) != expected:
                raise ValueError(
                    f"fit_force(): initial_feedback must have shape {expected} "
                    f"(batch, 1, feedback_dim); got {tuple(initial_feedback.shape)}."
                )

        # Apply per-pass overrides, then reset the RLS recursion (P = I/delta,
        # zero weights) so a re-run trains from scratch.
        if forgetting is not None:
            if not 0.0 < forgetting <= 1.0:
                raise ValueError(
                    f"fit_force(): forgetting (lambda) must be in (0, 1], got {forgetting}."
                )
            readout.forgetting = forgetting
        if delta is not None:
            if delta <= 0.0:
                raise ValueError(f"fit_force(): delta must be positive, got {delta}.")
            readout.delta = delta
        readout.reset_state()

        # Capture the readout's input feature vector each step via a pre-hook.
        # The normal graph forward (unlike the flat forecast engine) fires hooks,
        # so this sees exactly what the readout consumes — reservoir state,
        # concatenated input, etc. — regardless of the architecture upstream.
        captured: dict[str, torch.Tensor] = {}

        def _capture_hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            captured["features"] = args[0]

        handle = readout.register_forward_pre_hook(_capture_hook)

        total_steps = targets.shape[1]
        outputs = torch.empty(batch_size, total_steps, feedback_dim, dtype=dtype, device=device)

        # Drivers are held at their last warmup value during the closed-loop
        # phase (FORCE generates autonomously). Feedback-only models skip this.
        held_drivers: tuple[torch.Tensor, ...] = ()
        try:
            with torch.no_grad():
                # Phase 1: teacher-forced warmup, driving the reservoir open-loop
                # with the true signal. ``RLSReadout.forward`` is plain linear
                # (no fitted-guard, weights zero after reset_state), so it flows a
                # harmless zero downstream while the reservoir synchronises — the
                # warmup only needs the reservoir state, not the readout output.
                if warmup_inputs[0].shape[1] > 0:
                    self.model.warmup(*warmup_inputs, reset=reset)
                    seed = warmup_inputs[0][:, -1:, :]
                elif reset:
                    self.model.reset_reservoirs()
                    seed = torch.zeros(batch_size, 1, feedback_dim, dtype=dtype, device=device)
                else:
                    seed = torch.zeros(batch_size, 1, feedback_dim, dtype=dtype, device=device)

                if num_drivers:
                    held_drivers = tuple(d[:, -1:, :] for d in warmup_inputs[1:])

                if initial_feedback is not None:
                    z = initial_feedback
                else:
                    z = seed

                # Phase 2: closed-loop FORCE. Each step drives the model with the
                # network's own feedback, reads the readout feature vector, applies
                # one RLS update toward the target, and feeds the *updated* output
                # back into the next step.
                for t in range(total_steps):
                    d = targets[:, t : t + 1, :]
                    self.model(z, *held_drivers)
                    features = captured["features"]
                    readout.partial_fit(features, d)
                    z = readout(features)
                    outputs[:, t, :] = z.squeeze(1)
        finally:
            handle.remove()

        return outputs

    @staticmethod
    @contextmanager
    def _allow_unfitted_warmup(
        readouts: list[tuple[str, ReadoutLayer]],
    ) -> Iterator[None]:
        """Bypass the unfitted-forward guard for the duration of a warmup pass.

        The warmup pass in :meth:`fit` and :meth:`fit_stream` runs *before* any
        readout is fitted, so an :class:`IncrementalRidgeReadoutLayer`'s forward guard
        would otherwise raise ``RuntimeError('... is not fitted ...')`` while the
        reservoir merely synchronises. Inside this context, every
        :class:`IncrementalRidgeReadoutLayer` (the only readout carrying the
        ``_allow_unfitted_forward`` flag) has it set to
        ``True`` so its forward flows a value downstream; the flag is reset to
        ``False`` on exit (including on exception, via ``try/finally``), so the
        guard is only bypassed during warmup — never during the actual fit or
        during normal inference. Readouts without the flag (e.g.
        :class:`CGReadoutLayer`) are left untouched.

        Parameters
        ----------
        readouts : list of (str, ReadoutLayer)
            The ``(name, readout)`` pairs from :meth:`_discover_readouts`.

        Yields
        ------
        None
        """
        guarded = [ro for _, ro in readouts if isinstance(ro, IncrementalRidgeReadoutLayer)]
        for readout in guarded:
            readout._allow_unfitted_forward = True
        try:
            yield
        finally:
            for readout in guarded:
                readout._allow_unfitted_forward = False

    def _discover_readouts(self) -> list[tuple[str, ReadoutLayer]]:
        """Return ``(resolved_name, readout)`` pairs for every readout in the model.

        Walks ``self.model.named_modules()`` (a stable PyTorch API) instead of
        reaching into ``pytorch_symbolic``'s private ``_execution_order_*`` /
        ``_node_to_layer_name`` attributes.  Readouts keep their
        user-supplied ``name`` when set; otherwise the module path returned
        by ``named_modules()`` is used as the fallback name.

        The list order matches ``named_modules`` traversal, which is not the
        graph's topological order — that's fine because each hook fires at
        the moment its own forward executes during ``self.model(...)``.

        Raises
        ------
        ValueError
            If two or more distinct readouts resolve to the same name. Because
            ``targets`` and all downstream bookkeeping are keyed by name, a
            collision would silently map one target onto several readouts (and
            drop the others), training a readout on the wrong data with no
            error. Detecting it here — the single point every fitting path
            funnels through — turns that silent data loss into a clear failure.
        """
        readouts: list[tuple[str, ReadoutLayer]] = []
        seen: set[int] = set()
        for module_name, module in self.model.named_modules():
            if isinstance(module, ReadoutLayer) and id(module) not in seen:
                resolved_name = module.name if module.name else module_name
                readouts.append((resolved_name, module))
                seen.add(id(module))

        name_counts = Counter(name for name, _ in readouts)
        duplicates = sorted(name for name, count in name_counts.items() if count > 1)
        if duplicates:
            raise ValueError(
                f"Duplicate readout name(s) {duplicates}: the model has multiple "
                "readout layers sharing a name, but targets and fitted-readout "
                "bookkeeping are keyed by name, so they would silently collapse "
                "onto one target. Give each readout a unique 'name'."
            )

        return readouts

    def _validate_targets(
        self,
        targets: dict[str, torch.Tensor],
        readouts: list[tuple[str, ReadoutLayer]] | None = None,
    ) -> None:
        """Validate that all readouts have targets."""
        if readouts is None:
            readouts = self._discover_readouts()
        readout_names = [name for name, _ in readouts]
        missing = [name for name in readout_names if name not in targets]

        if missing:
            raise ValueError(
                f"Missing targets for readouts: {missing}. "
                f"Available readouts: {readout_names}. "
                f"Provided targets: {list(targets.keys())}."
            )

        extra = [name for name in targets if name not in readout_names]
        if extra:
            warnings.warn(
                f"Targets provided for non-existent readouts: {extra}. These will be ignored.",
                UserWarning,
            )
