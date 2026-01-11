"""Simplified model composition using pytorch_symbolic.

This module provides a cleaner, more Pythonic API for building ESN models
by leveraging the pytorch_symbolic library. It replaces our custom DAG and
ModelBuilder implementations with a battle-tested library.

Example:
    >>> import pytorch_symbolic as ps
    >>> from torch_rc.layers import ReservoirLayer
    >>> from torch_rc.layers.readouts import CGReadoutLayer
    >>>
    >>> # Simple ESN
    >>> inp = ps.Input((100, 1))  # seq_len, features
    >>> reservoir = ReservoirLayer(50, 1, 0)(inp)
    >>> readout = CGReadoutLayer(50, 1)(reservoir)
    >>> model = ESNModel(inp, readout)
    >>>
    >>> # Get summary
    >>> model.summary()
    >>>
    >>> # Forecast
    >>> predictions = model.forecast(warmup_data, forecast_steps=100)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pytorch_symbolic as ps
import torch

from ..layers import ReservoirLayer

# Re-export for convenience
Input = ps.Input


class ESNModel(ps.SymbolicModel):
    """Extended SymbolicModel with ESN-specific methods.

    This class wraps pytorch_symbolic.SymbolicModel and adds:
    - forecast() method for time series prediction
    - Reservoir state management (reset, get, set)
    - Model save/load functionality
    - Visualization using torchvista

    The model automatically inherits:
    - summary() - Keras-style model summary
    - All standard nn.Module functionality
    - Automatic parameter tracking
    """

    def reset_reservoirs(self) -> None:
        """Reset all reservoir states to zero."""
        for module in self.modules():
            if isinstance(module, ReservoirLayer):
                module.reset_state()

    def get_reservoir_states(self) -> Dict[str, torch.Tensor]:
        """Get current states of all reservoir layers.

        Returns:
            Dictionary mapping layer names to state tensors.
        """
        states = {}
        for name, module in self.named_modules():
            if isinstance(module, ReservoirLayer) and module.state is not None:
                states[name] = module.state.clone()
        return states

    def set_reservoir_states(self, states: Dict[str, torch.Tensor]) -> None:
        """Set states of reservoir layers.

        Args:
            states: Dictionary mapping layer names to state tensors.
        """
        for name, module in self.named_modules():
            if isinstance(module, ReservoirLayer) and name in states:
                module.state = states[name].clone()

    def save(
        self,
        path: Union[str, Path],
        include_states: bool = False,
        **metadata: Any,
    ) -> None:
        """Save model to file.

        Args:
            path: Path to save the model
            include_states: Whether to save reservoir states
            **metadata: Additional metadata to save with the model
        """
        path = Path(path)

        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "state_dict": self.state_dict(),
            "metadata": metadata,
        }

        if include_states:
            save_dict["reservoir_states"] = self.get_reservoir_states()

        torch.save(save_dict, path)

    def load(
        self,
        path: Union[str, Path],
        strict: bool = True,
        load_states: bool = False,
    ) -> None:
        """Load model from file.

        Args:
            path: Path to load the model from
            strict: Whether to strictly enforce state_dict keys match
            load_states: Whether to load reservoir states
        """
        import warnings

        path = Path(path)
        checkpoint = torch.load(path, weights_only=False)

        self.load_state_dict(checkpoint["state_dict"], strict=strict)

        if load_states:
            if "reservoir_states" in checkpoint:
                self.set_reservoir_states(checkpoint["reservoir_states"])
            else:
                warnings.warn(
                    "load_states=True but no reservoir states found in checkpoint", UserWarning
                )

    @classmethod
    def load_from_file(
        cls,
        path: Union[str, Path],
        model: Optional["ESNModel"] = None,
        strict: bool = True,
        load_states: bool = False,
    ) -> "ESNModel":
        """Load state dict into an existing model instance.

        Args:
            path: Path to load from
            model: Model instance to load into (required)
            strict: Whether to strictly enforce key matching
            load_states: Whether to load reservoir states

        Returns:
            The model instance (for convenience)

        Raises:
            ValueError: If model is None
        """
        if model is None:
            raise ValueError("model argument is required")

        model.load(path, strict=strict, load_states=load_states)
        return model

    def plot_model(
        self,
        save_path: Optional[Union[str, Path]] = None,
        format: str = "svg",
        show_shapes: bool = True,
        rankdir: str = "TB",
        **kwargs: Any,
    ) -> Any:
        """Visualize model architecture using the symbolic graph.

        This method uses the pytorch_symbolic graph structure directly,
        providing accurate visualization of complex model topologies
        including multi-input models and branching architectures.

        Args:
            save_path: Path to save the visualization (e.g., "model.svg", "model.png").
                       If None, displays inline in notebooks or prints DOT source.
            format: Output format when saving ("svg", "png", "pdf"). Default: "svg".
            show_shapes: Show tensor shapes on edges. Default: True.
            rankdir: Graph direction ("TB"=top-bottom, "LR"=left-right). Default: "TB".
            **kwargs: Additional arguments passed to graphviz.Digraph.

        Returns:
            In Jupyter: SVG display object
            Otherwise: DOT source string

        Example:
            >>> model.plot_model()  # Display in notebook
            >>> model.plot_model(save_path="model.svg")  # Save to file
            >>> model.plot_model(rankdir="LR")  # Left-to-right layout
        """
        # Build graph from symbolic tensors
        node_to_name = getattr(self, "_node_to_layer_name", {})

        def get_node_label(node: Any) -> str:
            """Get display label for a symbolic tensor node."""
            if node in node_to_name:
                return node_to_name[node]
            # Check if it's an input
            for i, inp in enumerate(self.inputs):
                if node is inp:
                    return f"Input_{i + 1}"
            return f"node_{id(node)}"

        def get_node_shape(node: Any) -> str:
            """Get tensor shape string for a node."""
            if hasattr(node, "shape"):
                shape = node.shape
                if isinstance(shape, torch.Size):
                    return str(tuple(shape))
            return ""

        # Collect all nodes and edges
        nodes = {}  # name -> (label, shape, is_input, is_output)
        edges = []  # (from_name, to_name, shape_label)

        # Add input nodes
        for i, inp in enumerate(self.inputs):
            name = f"Input_{i + 1}"
            shape = get_node_shape(inp)
            nodes[name] = (name, shape, True, False)

        # Add layer nodes from the symbolic graph
        for node, layer_name in node_to_name.items():
            shape = get_node_shape(node)
            nodes[layer_name] = (layer_name, shape, False, False)

            # Add edges from parents
            parents = getattr(node, "_parents", [])
            for parent in parents:
                parent_name = get_node_label(parent)
                edge_shape = get_node_shape(parent) if show_shapes else ""
                edges.append((parent_name, layer_name, edge_shape))

        # Mark output nodes
        for out in self.outputs:
            out_name = get_node_label(out)
            if out_name in nodes:
                label, shape, is_input, _ = nodes[out_name]
                nodes[out_name] = (label, shape, is_input, True)

        # Generate DOT source
        dot_lines = [
            "digraph ESNModel {",
            f"  rankdir={rankdir};",
            "  node [shape=box, style=filled];",
            "  edge [fontsize=10];",
        ]

        # Add nodes with styling
        for name, (label, shape, is_input, is_output) in nodes.items():
            if is_input:
                style = 'fillcolor="#FFB6C1", shape=ellipse'  # Pink for inputs
            elif is_output:
                style = 'fillcolor="#90EE90", shape=ellipse'  # Green for outputs
            else:
                style = 'fillcolor="#87CEEB"'  # Light blue for layers

            # Extract layer type for display
            layer_type = label.rsplit("_", 1)[0] if "_" in label else label
            display_label = f"{layer_type}\\n{shape}" if shape and show_shapes else layer_type

            dot_lines.append(f'  "{name}" [label="{display_label}", {style}];')

        # Add edges
        for from_name, to_name, shape_label in edges:
            if shape_label and show_shapes:
                dot_lines.append(f'  "{from_name}" -> "{to_name}" [label="{shape_label}"];')
            else:
                dot_lines.append(f'  "{from_name}" -> "{to_name}";')

        dot_lines.append("}")
        dot_source = "\n".join(dot_lines)

        # Try to render with graphviz
        try:
            import graphviz

            graph = graphviz.Source(dot_source)

            if save_path is not None:
                save_path = Path(save_path)
                # graphviz adds extension automatically
                graph.render(
                    str(save_path.with_suffix("")),
                    format=format,
                    cleanup=True,
                )
                print(f"Saved to {save_path.with_suffix('.' + format)}")
                return graph

            # Try to display in notebook
            try:
                from IPython.display import SVG, display

                svg_data = graph.pipe(format="svg").decode("utf-8")
                display(SVG(svg_data))
                return svg_data
            except ImportError:
                # Not in notebook, return the graph object
                return graph

        except ImportError:
            # graphviz not installed, return DOT source
            print("Note: Install 'graphviz' package for visual rendering.")
            print("      pip install graphviz")
            print("      Also install graphviz system package (apt install graphviz)")
            print("\nDOT source (can be rendered at https://dreampuf.github.io/GraphvizOnline/):")
            print(dot_source)
            return dot_source

    @torch.no_grad()
    def warmup(
        self,
        *inputs: torch.Tensor,
        return_outputs: bool = False,
    ) -> Optional[torch.Tensor]:
        """Teacher-forced warmup to synchronize reservoir states.

        Runs the model with provided inputs, updating internal reservoir states
        to achieve the Echo State Property (synchronization with input dynamics).

        Convention: input0 is feedback, input1+ are driving inputs.

        Args:
            *inputs: Tensors of shape (B, T, features) for each model input.
                    First input is feedback, remaining are drivers.
            return_outputs: Whether to return model outputs during warmup.

        Returns:
            If return_outputs=True: Tensor of shape (B, T, output_dim)
            Otherwise: None (only internal state is updated)

        Example:
            >>> model.warmup(feedback, driving_input)  # Just sync states
            >>> outputs = model.warmup(feedback, driving_input, return_outputs=True)
        """
        if len(inputs) == 0:
            raise ValueError("At least one input (feedback) is required")

        # Simply run forward pass - ReservoirLayer handles the sequence internally
        output = self(*inputs)

        return output if return_outputs else None

    @torch.no_grad()
    def forecast(
        self,
        *warmup_inputs: torch.Tensor,
        horizon: int,
        forecast_drivers: Optional[Tuple[torch.Tensor, ...]] = None,
        initial_feedback: Optional[torch.Tensor] = None,
        return_warmup: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Two-phase forecast: teacher-forced warmup + autoregressive generation.

        Phase 1 (Warmup): Run model with provided inputs to synchronize
        reservoir states with the input dynamics (Echo State Property).

        Phase 2 (Forecast): Run autoregressive generation where the feedback
        input comes from the model's own output, while driving inputs (if any)
        are provided via forecast_drivers.

        Convention: First input (warmup_inputs[0]) is always feedback.
        Remaining inputs are driving inputs for the reservoirs.

        For multi-output models, the first output tensor is used as feedback.
        All outputs are stored and returned.

        Args:
            *warmup_inputs: Warmup tensors (feedback, driver1, driver2, ...)
                            Each of shape (B, warmup_steps, features).
            horizon: Number of autoregressive steps to generate.
            forecast_drivers: Optional tuple of driving inputs for forecast phase.
                              Each tensor should be (B, horizon, features).
                              Required if model has driving inputs.
            initial_feedback: Optional custom initial feedback (B, 1, feedback_dim).
                              If None, uses last warmup output.
            return_warmup: If True, prepend warmup outputs to result.

        Returns:
            Single-output model: Tensor (B, horizon, output_dim) or
                                 (B, warmup_steps + horizon, output_dim) if return_warmup
            Multi-output model: Tuple of tensors with same structure

        Example:
            >>> # Simple feedback-only model
            >>> predictions = model.forecast(warmup_data, horizon=100)
            >>>
            >>> # Input-driven model
            >>> predictions = model.forecast(
            ...     warmup_feedback, warmup_driver,
            ...     horizon=100,
            ...     forecast_drivers=(future_driver,),
            ... )

        Raises:
            ValueError: If forecast_drivers is required but not provided,
                        or if dimensions don't match.
        """
        if len(warmup_inputs) == 0:
            raise ValueError("At least one warmup input (feedback) is required")

        # Determine if model has driving inputs
        num_drivers = len(warmup_inputs) - 1
        has_drivers = num_drivers > 0

        # Validate forecast_drivers
        if has_drivers:
            if forecast_drivers is None:
                raise ValueError(
                    f"Model has {num_drivers} driving input(s). "
                    f"forecast_drivers must be provided for forecast phase."
                )
            if len(forecast_drivers) != num_drivers:
                raise ValueError(
                    f"Expected {num_drivers} forecast drivers, got {len(forecast_drivers)}"
                )
            for i, driver in enumerate(forecast_drivers):
                if driver.shape[1] != horizon:
                    raise ValueError(
                        f"forecast_drivers[{i}] has {driver.shape[1]} steps, expected {horizon}"
                    )

        batch_size = warmup_inputs[0].shape[0]
        feedback_dim = warmup_inputs[0].shape[-1]
        device = warmup_inputs[0].device
        dtype = warmup_inputs[0].dtype

        # Phase 1: Warmup - get all outputs to sync state AND get initial feedback
        warmup_outputs = self.warmup(*warmup_inputs, return_outputs=True)

        # Determine output structure from model.output_shape
        output_shape = self.output_shape
        multi_output = isinstance(output_shape, tuple) and isinstance(output_shape[0], torch.Size)

        # Validate feedback dimension matches output dimension
        if multi_output:
            feedback_output_dim = output_shape[0][-1]
        else:
            feedback_output_dim = output_shape[-1]

        if feedback_output_dim != feedback_dim:
            raise ValueError(
                f"Model design error: feedback input expects {feedback_dim} features, "
                f"but model output (used as feedback) has {feedback_output_dim} features. "
                f"For forecasting, the first output must match the feedback input dimension."
            )

        # Get initial feedback from last warmup output
        if initial_feedback is not None:
            current_feedback = initial_feedback
        else:
            if multi_output:
                current_feedback = warmup_outputs[0][:, -1:, :]
            else:
                current_feedback = warmup_outputs[:, -1:, :]

        # Pre-allocate forecast output storage based on model.output_shape
        # Note: horizon outputs = warmup's last prediction + (horizon-1) autoregressive steps
        if multi_output:
            forecast_outputs = tuple(
                torch.empty(batch_size, horizon, shape[-1], dtype=dtype, device=device)
                for shape in output_shape
            )
        else:
            forecast_outputs = torch.empty(
                batch_size, horizon, output_shape[-1], dtype=dtype, device=device
            )

        # Store warmup's last output as first forecast step (prediction for val[0])
        if multi_output:
            for i, out in enumerate(warmup_outputs):
                forecast_outputs[i][:, 0, :] = out[:, -1, :]
        else:
            forecast_outputs[:, 0, :] = warmup_outputs[:, -1, :]

        # Phase 2: Autoregressive forecast for remaining (horizon - 1) steps
        for t in range(1, horizon):
            if has_drivers:
                driver_inputs_t = tuple(driver[:, t : t + 1, :] for driver in forecast_drivers)
                step_inputs = (current_feedback,) + driver_inputs_t
            else:
                step_inputs = (current_feedback,)

            output = self(*step_inputs)

            if multi_output:
                for i, out in enumerate(output):
                    forecast_outputs[i][:, t, :] = out.squeeze(1)
                current_feedback = output[0]
            else:
                forecast_outputs[:, t, :] = output.squeeze(1)
                current_feedback = output

        # Combine warmup and forecast if requested
        if return_warmup:
            if multi_output:
                return tuple(
                    torch.cat([warmup_outputs[i], forecast_outputs[i]], dim=1)
                    for i in range(len(output_shape))
                )
            else:
                return torch.cat([warmup_outputs, forecast_outputs], dim=1)
        else:
            return forecast_outputs


__all__ = ["Input", "ESNModel"]
