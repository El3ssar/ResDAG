"""ESN Trainer for algebraic readout fitting.

This module provides the ESNTrainer class that trains ESN models by fitting
each ReadoutLayer algebraically in topological order using a single forward pass.
"""

import torch

from ..composition.symbolic import ESNModel
from ..layers.readouts.base import ReadoutLayer


class ESNTrainer:
    """Trainer for ESN models with algebraic readout fitting.

    Traverses the model DAG in topological order and fits each ReadoutLayer
    using pre-hooks during a single forward pass. This is efficient for models
    with multiple stacked readouts.

    Each ReadoutLayer handles its own fitting hyperparameters (e.g., alpha
    for ridge regression is set during layer construction).

    Example:
        >>> from torch_rc.training import ESNTrainer
        >>> trainer = ESNTrainer(model)
        >>> trainer.fit(
        ...     warmup_inputs=(warmup_data,),
        ...     train_inputs=(train_data,),
        ...     targets={"output": train_targets},
        ... )
    """

    def __init__(self, model: ESNModel) -> None:
        """Initialize trainer.

        Args:
            model: ESNModel to train
        """
        self.model = model

    def fit(
        self,
        warmup_inputs: tuple[torch.Tensor, ...],
        train_inputs: tuple[torch.Tensor, ...],
        targets: dict[str, torch.Tensor],
    ) -> None:
        """Train all readout layers in a single forward pass.

        Uses pre-hooks to fit each readout layer just before its forward
        method executes. This ensures downstream layers receive outputs
        from already-fitted readouts.

        Args:
            warmup_inputs: Warmup sequences (feedback, driver1, ...).
                           Each tensor shape: (B, warmup_steps, features).
            train_inputs: Training sequences (feedback, driver1, ...).
                          Each tensor shape: (B, train_steps, features).
                          Must have same length as targets.
            targets: Dict mapping readout name -> target tensor.
                     Target shape: (B, train_steps, out_features).
                     Name is either user-defined (name="output") or auto-generated
                     module name ("CGReadoutLayer_1").

        Raises:
            ValueError: If any ReadoutLayer is missing a target or shapes mismatch.
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
        """Return [(resolved_name, layer, node), ...] in topological order.

        Returns:
            List of tuples containing:
            - resolved_name: User-defined name or auto-generated module name
            - layer: The ReadoutLayer instance
            - node: The SymbolicTensor node
        """
        readouts = []
        for node, layer in zip(
            self.model._execution_order_nodes,
            self.model._execution_order_layers,
        ):
            if isinstance(layer, ReadoutLayer):
                module_name = self.model._node_to_layer_name[node]
                # Use user-defined name if set, else module name
                resolved_name = layer.name if layer.name else module_name
                readouts.append((resolved_name, layer, node))
        return readouts

    def _validate_targets(self, targets: dict[str, torch.Tensor]) -> None:
        """Raise error if any readout is missing a target.

        Args:
            targets: Dict of readout name -> target tensor

        Raises:
            ValueError: If any readout is missing from targets dict
        """
        readouts = self._get_readouts_in_order()
        readout_names = [name for name, _, _ in readouts]
        missing = [name for name in readout_names if name not in targets]

        if missing:
            raise ValueError(
                f"Missing targets for readouts: {missing}. "
                f"Available readouts: {readout_names}. "
                f"Provided targets: {list(targets.keys())}."
            )

        # Also warn about extra targets
        extra = [name for name in targets if name not in readout_names]
        if extra:
            import warnings

            warnings.warn(
                f"Targets provided for non-existent readouts: {extra}. These will be ignored.",
                UserWarning,
            )
