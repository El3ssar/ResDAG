"""
Model Composition with pytorch_symbolic
========================================

This module provides the main model class for building ESN architectures
using the ``pytorch_symbolic`` library for symbolic tensor computation.

The :class:`ESNModel` class extends ``pytorch_symbolic.SymbolicModel`` with
ESN-specific functionality including forecasting, reservoir state management,
and model persistence.

Examples
--------
Simple ESN model:

>>> import pytorch_symbolic as ps
>>> from resdag.layers import ReservoirLayer
>>> from resdag.layers.readouts import CGReadoutLayer
>>> from resdag.composition import ESNModel
>>>
>>> inp = ps.Input((100, 1))
>>> reservoir = ReservoirLayer(50, feedback_size=1)(inp)
>>> readout = CGReadoutLayer(50, 1)(reservoir)
>>> model = ESNModel(inp, readout)
>>> model.summary()

Multi-input model with driving signal:

>>> feedback = ps.Input((100, 3))
>>> driver = ps.Input((100, 5))
>>> reservoir = ReservoirLayer(100, feedback_size=3, input_size=5)(feedback, driver)
>>> readout = CGReadoutLayer(100, 3)(reservoir)
>>> model = ESNModel([feedback, driver], readout)

See Also
--------
resdag.layers.ReservoirLayer : Reservoir layer with recurrent dynamics.
resdag.layers.readouts.CGReadoutLayer : Conjugate gradient readout layer.
resdag.training.ESNTrainer : Trainer for fitting readout layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_symbolic as ps
import torch

from ..layers import ReservoirLayer

# Re-export for convenience
Input = ps.Input


class ESNModel(ps.SymbolicModel):
    """
    Echo State Network model with forecasting and state management.

    This class extends ``pytorch_symbolic.SymbolicModel`` with ESN-specific
    functionality including:

    - Time series forecasting with warmup and autoregressive generation
    - Reservoir state management (reset, get, set)
    - Model persistence (save/load)
    - Architecture visualization

    The model inherits all standard ``torch.nn.Module`` functionality and
    the ``summary()`` method from pytorch_symbolic.

    Parameters
    ----------
    inputs : Input or list of Input
        Model input(s) created with ``pytorch_symbolic.Input()``.
    outputs : SymbolicTensor or list of SymbolicTensor
        Model output(s) from the computational graph.

    Attributes
    ----------
    inputs : list
        List of model input tensors.
    outputs : list
        List of model output tensors.
    output_shape : torch.Size or tuple of torch.Size
        Shape(s) of model outputs.

    Examples
    --------
    Create and use a simple ESN:

    >>> import pytorch_symbolic as ps
    >>> from resdag.composition import ESNModel
    >>> from resdag.layers import ReservoirLayer
    >>> from resdag.layers.readouts import CGReadoutLayer
    >>>
    >>> inp = ps.Input((100, 3))
    >>> reservoir = ReservoirLayer(200, feedback_size=3)(inp)
    >>> readout = CGReadoutLayer(200, 3)(reservoir)
    >>> model = ESNModel(inp, readout)
    >>>
    >>> # Forward pass
    >>> x = torch.randn(4, 100, 3)
    >>> y = model(x)
    >>> print(y.shape)
    torch.Size([4, 100, 3])

    Forecasting with warmup:

    >>> warmup_data = torch.randn(1, 50, 3)
    >>> predictions = model.forecast(warmup_data, horizon=100)
    >>> print(predictions.shape)
    torch.Size([1, 100, 3])

    See Also
    --------
    pytorch_symbolic.SymbolicModel : Parent class.
    ReservoirLayer : Reservoir layer component.
    ESNTrainer : Trainer for fitting readout layers.
    """

    def reset_reservoirs(self) -> None:
        """
        Reset all reservoir layer states to zero.

        This clears the internal hidden states of all :class:`ReservoirLayer`
        modules in the model, preparing it for a new sequence.

        Examples
        --------
        >>> model.reset_reservoirs()
        >>> output = model(new_sequence)
        """
        for module in self.modules():
            if isinstance(module, ReservoirLayer):
                module.reset_state()

    def set_random_reservoir_states(self) -> None:
        """
        Set random states of all reservoir layers.

        Examples
        --------
        >>> model.set_random_reservoir_states()
        """
        for module in self.modules():
            if isinstance(module, ReservoirLayer):
                module.set_random_state()

    def get_reservoir_states(self) -> dict[str, torch.Tensor]:
        """
        Get current states of all reservoir layers.

        Returns
        -------
        dict of str to torch.Tensor
            Dictionary mapping layer names to their state tensors.
            Only includes reservoirs with non-None states.

        Examples
        --------
        >>> states = model.get_reservoir_states()
        >>> for name, state in states.items():
        ...     print(f"{name}: {state.shape}")
        """
        states = {}
        for name, module in self.named_modules():
            if isinstance(module, ReservoirLayer) and module.state is not None:
                states[name] = module.state.clone()
        return states

    def set_reservoir_states(self, states: dict[str, torch.Tensor]) -> None:
        """
        Set states of reservoir layers.

        Parameters
        ----------
        states : dict of str to torch.Tensor
            Dictionary mapping layer names to state tensors.
            Names should match those returned by :meth:`get_reservoir_states`.

        Examples
        --------
        >>> states = model.get_reservoir_states()
        >>> # ... do something ...
        >>> model.set_reservoir_states(states)  # Restore states
        """
        for name, module in self.named_modules():
            if isinstance(module, ReservoirLayer) and name in states:
                module.state = states[name].clone()

    def save(
        self,
        path: str | Path,
        include_states: bool = False,
        **metadata: Any,
    ) -> None:
        """
        Save model weights and optionally reservoir states.

        Parameters
        ----------
        path : str or Path
            File path to save the model. Parent directories are created
            if they don't exist.
        include_states : bool, default=False
            If True, also save current reservoir states.
        **metadata
            Additional metadata to store with the model (e.g., training info).

        Examples
        --------
        Save model weights only:

        >>> model.save("model.pt")

        Save with states and metadata:

        >>> model.save(
        ...     "checkpoint.pt",
        ...     include_states=True,
        ...     epoch=10,
        ...     loss=0.05
        ... )

        See Also
        --------
        load : Load model from file.
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
        path: str | Path,
        strict: bool = True,
        load_states: bool = False,
    ) -> None:
        """
        Load model weights from file.

        Parameters
        ----------
        path : str or Path
            File path to load from.
        strict : bool, default=True
            If True, strictly enforce that state_dict keys match.
        load_states : bool, default=False
            If True, also load reservoir states if available.

        Warns
        -----
        UserWarning
            If ``load_states=True`` but no states are found in checkpoint.

        Examples
        --------
        >>> model.load("model.pt")
        >>> model.load("checkpoint.pt", load_states=True)

        See Also
        --------
        save : Save model to file.
        load_from_file : Class method for loading.
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
        path: str | Path,
        model: "ESNModel" | None = None,
        strict: bool = True,
        load_states: bool = False,
    ) -> "ESNModel":
        """
        Load weights into an existing model instance.

        This is a convenience class method that loads state dict into
        a pre-constructed model.

        Parameters
        ----------
        path : str or Path
            File path to load from.
        model : ESNModel
            Model instance to load weights into. Required.
        strict : bool, default=True
            If True, strictly enforce state_dict key matching.
        load_states : bool, default=False
            If True, also load reservoir states.

        Returns
        -------
        ESNModel
            The model instance with loaded weights.

        Raises
        ------
        ValueError
            If ``model`` is None.

        Examples
        --------
        >>> model = create_my_model()  # Create architecture
        >>> model = ESNModel.load_from_file("weights.pt", model=model)
        """
        if model is None:
            raise ValueError("model argument is required")

        model.load(path, strict=strict, load_states=load_states)
        return model

    def plot_model(
        self,
        save_path: str | Path | None = None,
        format: str = "svg",
        show_shapes: bool = True,
        rankdir: str = "TB",
        **kwargs: Any,
    ) -> Any:
        """
        Visualize model architecture as a graph.

        Uses the pytorch_symbolic graph structure to generate an accurate
        visualization of the model topology, including multi-input models
        and branching architectures.

        Parameters
        ----------
        save_path : str or Path, optional
            Path to save the visualization. If None, displays inline
            in notebooks or prints DOT source.
        format : {'svg', 'png', 'pdf'}, default='svg'
            Output format when saving.
        show_shapes : bool, default=True
            If True, show tensor shapes on graph edges.
        rankdir : {'TB', 'LR', 'BT', 'RL'}, default='TB'
            Graph layout direction:
            - 'TB': top to bottom
            - 'LR': left to right
            - 'BT': bottom to top
            - 'RL': right to left
        **kwargs
            Additional arguments passed to ``graphviz.Digraph``.

        Returns
        -------
        str or graphviz.Source
            In Jupyter: displays SVG and returns SVG string.
            Otherwise: returns graphviz.Source or DOT source string.

        Notes
        -----
        Requires the ``graphviz`` Python package and system installation.
        Install with: ``pip install graphviz`` and ``apt install graphviz``.

        Examples
        --------
        Display in notebook:

        >>> model.plot_model()

        Save to file:

        >>> model.plot_model(save_path="model.svg")

        Left-to-right layout:

        >>> model.plot_model(rankdir="LR")
        """
        # Build graph from symbolic tensors
        node_to_name = getattr(self, "_node_to_layer_name", {})

        def _get_node_label(node: Any) -> str:
            """Get display label for a symbolic tensor node."""
            if node in node_to_name:
                return node_to_name[node]
            for i, inp in enumerate(self.inputs):
                if node is inp:
                    return f"Input_{i + 1}"
            return f"node_{id(node)}"

        def _get_node_shape(node: Any) -> str:
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
            shape = _get_node_shape(inp)
            nodes[name] = (name, shape, True, False)

        # Add layer nodes from the symbolic graph
        for node, layer_name in node_to_name.items():
            shape = _get_node_shape(node)
            nodes[layer_name] = (layer_name, shape, False, False)

            # Add edges from parents
            parents = getattr(node, "_parents", [])
            for parent in parents:
                parent_name = _get_node_label(parent)
                edge_shape = _get_node_shape(parent) if show_shapes else ""
                edges.append((parent_name, layer_name, edge_shape))

        # Mark output nodes
        for out in self.outputs:
            out_name = _get_node_label(out)
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
                return graph

        except ImportError:
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
    ) -> torch.Tensor | None:
        """
        Teacher-forced warmup to synchronize reservoir states.

        Runs the model forward with provided inputs, updating internal
        reservoir states to achieve the Echo State Property (synchronization
        with input dynamics).

        Parameters
        ----------
        *inputs : torch.Tensor
            Input tensors of shape ``(batch, timesteps, features)``.
            Convention: first input is feedback, remaining are drivers.
        return_outputs : bool, default=False
            If True, return model outputs during warmup.

        Returns
        -------
        torch.Tensor or None
            If ``return_outputs=True``: output tensor(s) of shape
            ``(batch, timesteps, output_dim)``.
            Otherwise: None (only internal state is updated).

        Raises
        ------
        ValueError
            If no inputs are provided.

        Examples
        --------
        Synchronize states without capturing output:

        >>> model.warmup(feedback_data)

        Synchronize and capture output:

        >>> outputs = model.warmup(feedback_data, return_outputs=True)

        With driving input:

        >>> model.warmup(feedback, driving_signal)

        See Also
        --------
        forecast : Two-phase forecasting with warmup and generation.
        reset_reservoirs : Reset all reservoir states.
        """
        if len(inputs) == 0:
            raise ValueError("At least one input (feedback) is required")

        output = self(*inputs)

        return output if return_outputs else None

    @torch.no_grad()
    def forecast(
        self,
        *warmup_inputs: torch.Tensor,
        horizon: int,
        forecast_drivers: tuple[torch.Tensor, ...] | None = None,
        initial_feedback: torch.Tensor | None = None,
        return_warmup: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Two-phase forecast: teacher-forced warmup + autoregressive generation.

        Phase 1 (Warmup): Runs model with provided inputs to synchronize
        reservoir states with input dynamics (Echo State Property).

        Phase 2 (Forecast): Autoregressive generation where feedback comes
        from the model's own output while driving inputs (if any) are
        provided via ``forecast_drivers``.

        Parameters
        ----------
        *warmup_inputs : torch.Tensor
            Warmup tensors of shape ``(batch, warmup_steps, features)``.
            Convention: first input is feedback, remaining are drivers.
        horizon : int
            Number of autoregressive steps to generate.
        forecast_drivers : tuple of torch.Tensor, optional
            Driving inputs for forecast phase. Each tensor should have
            shape ``(batch, horizon, features)``. Required if model has
            driving inputs.
        initial_feedback : torch.Tensor, optional
            Custom initial feedback of shape ``(batch, 1, feedback_dim)``.
            If None, uses last warmup output.
        return_warmup : bool, default=False
            If True, prepend warmup outputs to the result.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            For single-output models: tensor of shape
            ``(batch, horizon, output_dim)`` or
            ``(batch, warmup_steps + horizon, output_dim)`` if ``return_warmup``.

            For multi-output models: tuple of tensors with same structure.

        Raises
        ------
        ValueError
            If no warmup inputs provided, if forecast_drivers is required
            but not provided, or if dimensions don't match.

        Notes
        -----
        - Convention: first input is always feedback (used for autoregression).
        - For multi-output models, first output is used as feedback.
        - Feedback output dimension must match feedback input dimension.

        Examples
        --------
        Simple feedback-only model:

        >>> warmup_data = torch.randn(1, 50, 3)
        >>> predictions = model.forecast(warmup_data, horizon=100)
        >>> print(predictions.shape)
        torch.Size([1, 100, 3])

        Input-driven model:

        >>> predictions = model.forecast(
        ...     warmup_feedback,
        ...     warmup_driver,
        ...     horizon=100,
        ...     forecast_drivers=(future_driver,),
        ... )

        Include warmup in output:

        >>> full_output = model.forecast(
        ...     warmup_data,
        ...     horizon=100,
        ...     return_warmup=True
        ... )
        >>> print(full_output.shape)  # warmup_steps + horizon
        torch.Size([1, 150, 3])

        See Also
        --------
        warmup : Teacher-forced warmup only.
        reset_reservoirs : Reset reservoir states before forecasting.
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

        # Phase 1: Warmup
        warmup_outputs = self.warmup(*warmup_inputs, return_outputs=True)

        # Determine output structure
        output_shape = self.output_shape
        multi_output = isinstance(output_shape, tuple) and isinstance(output_shape[0], torch.Size)

        # Validate feedback dimension
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

        # Get initial feedback
        if initial_feedback is not None:
            current_feedback = initial_feedback
        else:
            if multi_output:
                current_feedback = warmup_outputs[0][:, -1:, :]
            else:
                current_feedback = warmup_outputs[:, -1:, :]

        # Pre-allocate forecast output storage
        if multi_output:
            forecast_outputs = tuple(
                torch.empty(batch_size, horizon, shape[-1], dtype=dtype, device=device)
                for shape in output_shape
            )
        else:
            forecast_outputs = torch.empty(
                batch_size, horizon, output_shape[-1], dtype=dtype, device=device
            )

        # Store warmup's last output as first forecast step
        if multi_output:
            for i, out in enumerate(warmup_outputs):
                forecast_outputs[i][:, 0, :] = out[:, -1, :]
        else:
            forecast_outputs[:, 0, :] = warmup_outputs[:, -1, :]

        # Phase 2: Autoregressive forecast
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
