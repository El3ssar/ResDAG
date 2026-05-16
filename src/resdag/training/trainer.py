"""
ESN Trainer
===========

This module provides :class:`ESNTrainer`, which trains ESN models by
fitting readout layers algebraically in topological order using a
single forward pass.

The trainer uses PyTorch forward hooks to fit each readout layer just
before its forward method executes, ensuring downstream layers receive
outputs from already-fitted readouts.

See Also
--------
resdag.ESNModel : ESN model class.
resdag.layers.readouts.CGReadoutLayer : Conjugate gradient readout layer.
"""

import torch

from resdag.core import ESNModel
from resdag.layers import ReadoutLayer


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

            def make_fit_hook(layer: ReadoutLayer, tgt: torch.Tensor):
                def hook(module, args):
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
