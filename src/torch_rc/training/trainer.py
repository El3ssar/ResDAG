"""ESN Trainer for algebraic readout fitting.

This module provides the ESNTrainer class that trains ESN models by fitting
each ReadoutLayer algebraically in topological order.
"""

from typing import Dict, List, Optional, Tuple

import torch

from ..composition.symbolic import ESNModel
from ..layers.readouts.base import ReadoutLayer


class ESNTrainer:
    """Trainer for ESN models with algebraic readout fitting.

    Traverses the model DAG in topological order, captures activations
    at each readout's parent layer, and calls readout.fit() with the
    user-provided targets.

    Each ReadoutLayer handles its own fitting hyperparameters (e.g., alpha
    for ridge regression is set during layer construction).

    For stacked architectures (readout1 -> reservoir2 -> readout2), training
    requires a fresh warmup + forward pass per readout to ensure each layer
    sees correct activations from previously-fitted layers.

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
        warmup_inputs: Tuple[torch.Tensor, ...],
        train_inputs: Tuple[torch.Tensor, ...],
        targets: Dict[str, torch.Tensor],
    ) -> None:
        """Train all readout layers in topological order.

        For each readout:
        1. Reset reservoir states
        2. Run warmup to synchronize reservoir states
        3. Run forward on training data
        4. Capture parent layer output
        5. Call readout.fit(captured_input, target)

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

        for name, readout, node in readouts:
            # Reset states for clean slate
            self.model.reset_reservoirs()

            # Warmup to sync reservoir states
            self.model.warmup(*warmup_inputs)

            # Find parent layer to capture its output
            parent_name = self._get_parent_layer_name(node)

            if parent_name is None:
                raise ValueError(
                    f"Could not find parent layer for readout '{name}'. "
                    f"Readout must have a parent layer."
                )

            # Capture parent output during forward pass
            captured: Dict[str, torch.Tensor] = {}

            def make_hook(storage: Dict[str, torch.Tensor]):
                def hook(module, input, output):
                    storage["output"] = output

                return hook

            parent_layer = getattr(self.model, parent_name)
            handle = parent_layer.register_forward_hook(make_hook(captured))

            try:
                # Forward pass (no grad needed for reservoir computation)
                with torch.no_grad():
                    self.model(*train_inputs)

                # Get target for this readout
                target = targets[name]

                # Validate target shape
                if target.shape[1] != train_steps:
                    raise ValueError(
                        f"Target for '{name}' has {target.shape[1]} timesteps, "
                        f"but train_inputs has {train_steps} timesteps. Must match."
                    )

                # Fit this readout
                readout.fit(captured["output"], target)

            finally:
                handle.remove()

    def _get_readouts_in_order(self) -> List[Tuple[str, ReadoutLayer, object]]:
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

    def _get_parent_layer_name(self, node) -> Optional[str]:
        """Get module name of parent node.

        Args:
            node: SymbolicTensor node

        Returns:
            Module name of parent, or None if parent is an Input
        """
        if not node.parents:
            return None

        parent_node = node.parents[0]  # Readout typically has single parent
        return self.model._node_to_layer_name.get(parent_node)

    def _validate_targets(self, targets: Dict[str, torch.Tensor]) -> None:
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
