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
resdag.composition.ESNModel : ESN model class.
resdag.layers.readouts.CGReadoutLayer : Conjugate gradient readout layer.
"""

import torch

from ..composition.symbolic import ESNModel
from ..layers.readouts.base import ReadoutLayer


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
    >>> from resdag.composition import ESNModel
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

        Warns
        -----
        UserWarning
            If targets dict contains names not matching any readout.

        Notes
        -----
        After calling ``fit()``, all readouts will have ``is_fitted=True``
        and the model is ready for inference or forecasting.

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

        # Validate all readouts have targets
        self._validate_targets(targets)

        train_steps = train_inputs[0].shape[1]

        # Get readouts in topological order
        readouts = self._get_readouts_in_order()

        # Validate target shapes
        for name, _, _ in readouts:
            target = targets[name]
            if target.shape[1] != train_steps:
                raise ValueError(
                    f"Target for '{name}' has {target.shape[1]} timesteps, "
                    f"but train_inputs has {train_steps} timesteps. Must match."
                )

        # Single warmup to sync reservoir states
        self.model.reset_reservoirs()
        self.model.warmup(*warmup_inputs)

        # Register pre-hooks that fit each readout before its forward
        hooks = []
        for name, readout, _ in readouts:
            target = targets[name]

            def make_fit_hook(layer: ReadoutLayer, tgt: torch.Tensor):
                def hook(module, args):
                    layer.fit(args[0], tgt)

                return hook

            handle = readout.register_forward_pre_hook(make_fit_hook(readout, target))
            hooks.append(handle)

        try:
            # Single forward pass - hooks fit each readout in topological order
            with torch.no_grad():
                self.model(*train_inputs)
        finally:
            # Always remove hooks
            for h in hooks:
                h.remove()

    def _get_readouts_in_order(self) -> list[tuple[str, ReadoutLayer, object]]:
        """Return readouts in topological order."""
        readouts = []
        for node, layer in zip(
            self.model._execution_order_nodes,
            self.model._execution_order_layers,
        ):
            if isinstance(layer, ReadoutLayer):
                module_name = self.model._node_to_layer_name[node]
                resolved_name = layer.name if layer.name else module_name
                readouts.append((resolved_name, layer, node))
        return readouts

    def _validate_targets(self, targets: dict[str, torch.Tensor]) -> None:
        """Validate that all readouts have targets."""
        readouts = self._get_readouts_in_order()
        readout_names = [name for name, _, _ in readouts]
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
