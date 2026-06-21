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
fits :class:`~resdag.layers.IncrementalRidgeReadout` readouts over a stream of
contiguous chunks (e.g. a :class:`torch.utils.data.DataLoader`), accumulating
ridge sufficient statistics per chunk and solving once at the end.

See Also
--------
resdag.ESNModel : ESN model class.
resdag.layers.readouts.CGReadoutLayer : Conjugate gradient readout layer.
resdag.layers.readouts.IncrementalRidgeReadout : Streaming partial_fit readout.
"""

from collections.abc import Callable, Iterable

import torch

from resdag.core import ESNModel
from resdag.layers import IncrementalRidgeReadout, ReadoutLayer

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
        for name, _ in readouts:
            target = targets[name]
            if target.shape[1] != train_steps:
                raise ValueError(
                    f"Target for '{name}' has {target.shape[1]} timesteps, "
                    f"but train_inputs has {train_steps} timesteps. Must match."
                )

        # Single warmup to sync reservoir states (warmup resets by default)
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
        """Fit :class:`IncrementalRidgeReadout` readouts over a stream of chunks.

        The streaming counterpart of :meth:`fit`. Instead of fitting from one
        in-memory ``(B, T, F)`` block, it warms the reservoir once and then
        consumes ``chunks`` one at a time — e.g. windows yielded by a
        :class:`torch.utils.data.DataLoader` — accumulating each readout's ridge
        sufficient statistics with
        :meth:`~resdag.layers.IncrementalRidgeReadout.partial_fit` and solving
        once at the end with
        :meth:`~resdag.layers.IncrementalRidgeReadout.finalize`. No more than a
        single chunk's states are ever held in memory, which is what lets a
        sequence too long to materialise — or one streamed off disk — be fitted.

        Because the reservoir is stateful, the chunks must be **contiguous in
        time and in order**: state flows from the end of one chunk into the
        start of the next, exactly as it would in a single long forward pass.
        Shuffling the chunks would desynchronise the reservoir and is not
        supported.

        Every readout in the model must be an
        :class:`~resdag.layers.IncrementalRidgeReadout` (the only readout with a
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
            If any readout in the model is not an ``IncrementalRidgeReadout``.
        RuntimeError
            If ``chunks`` is empty (no statistics were accumulated, so there is
            nothing to finalize).

        Notes
        -----
        After ``fit_stream`` returns, every readout has ``is_fitted=True`` and
        the model is ready for inference or forecasting.

        Examples
        --------
        >>> from resdag.layers import IncrementalRidgeReadout
        >>> trainer = ESNTrainer(model)  # model uses IncrementalRidgeReadout
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
        resdag.layers.IncrementalRidgeReadout : The streaming readout.
        """
        if len(warmup_inputs) == 0:
            raise ValueError("At least one warmup input is required")

        readouts = self._discover_readouts()
        non_incremental = [
            name for name, ro in readouts if not isinstance(ro, IncrementalRidgeReadout)
        ]
        if non_incremental:
            raise TypeError(
                f"fit_stream() requires every readout to be an IncrementalRidgeReadout, "
                f"but these are not: {non_incremental}. Use fit() for single-pass readouts, "
                f"or rebuild the model with IncrementalRidgeReadout."
            )

        # Reset accumulators so a re-fit starts clean, then warm up once. The
        # readouts are unfitted during warmup too, so let their forward flow a
        # value while the reservoir synchronises.
        for _, readout in readouts:
            assert isinstance(readout, IncrementalRidgeReadout)
            readout.reset_accumulators()
            readout._allow_unfitted_forward = True
        try:
            self.model.warmup(*warmup_inputs)
        finally:
            for _, readout in readouts:
                assert isinstance(readout, IncrementalRidgeReadout)
                readout._allow_unfitted_forward = False

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
                    layer: IncrementalRidgeReadout, tgt: torch.Tensor
                ) -> Callable[[torch.nn.Module, tuple[torch.Tensor, ...]], None]:
                    def hook(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
                        layer.partial_fit(args[0], tgt)

                    return hook

                assert isinstance(readout, IncrementalRidgeReadout)
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
                    assert isinstance(readout, IncrementalRidgeReadout)
                    readout._allow_unfitted_forward = False
            n_chunks += 1

        if n_chunks == 0:
            raise RuntimeError(
                "fit_stream() received no chunks; nothing was accumulated to finalize."
            )

        # Solve every readout once from its accumulated statistics.
        for _, readout in readouts:
            assert isinstance(readout, IncrementalRidgeReadout)
            readout.finalize()

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
        """
        readouts: list[tuple[str, ReadoutLayer]] = []
        seen: set[int] = set()
        for module_name, module in self.model.named_modules():
            if isinstance(module, ReadoutLayer) and id(module) not in seen:
                resolved_name = module.name if module.name else module_name
                readouts.append((resolved_name, module))
                seen.add(id(module))
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
            import warnings

            warnings.warn(
                f"Targets provided for non-existent readouts: {extra}. These will be ignored.",
                UserWarning,
            )
