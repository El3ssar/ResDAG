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

from resdag.layers import ReservoirLayer

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
        show_shapes: bool = True,
        show_params: bool = False,
        show_trainable: bool = False,
        rankdir: str = "TB",
        save_path: str | Path | None = None,
        format: str = "svg",
        **kwargs: Any,
    ) -> Any:
        """
        Visualize model architecture as a graph.

        Uses the pytorch_symbolic graph structure to generate an accurate
        visualization of the model topology, including multi-input models
        and branching architectures.

        Each layer type gets a distinct color; identical types share the
        same color. A legend is appended to the diagram automatically.

        In a Jupyter environment, an interactive widget is displayed with
        toggle checkboxes for ``show_shapes``, ``show_params``, and
        ``show_trainable``. Outside Jupyter the diagram is opened with the
        system viewer; pass ``save_path`` to write a file instead.

        Parameters
        ----------
        show_shapes : bool, default=True
            Show tensor shapes on edges.
        show_params : bool, default=False
            Show key hyperparameters inside each layer node.
        show_trainable : bool, default=False
            Show a padlock indicator (🔒 frozen / 🔓 trainable) on nodes
            that have learnable parameters.
        rankdir : {'TB', 'LR', 'BT', 'RL'}, default='TB'
            Graph layout direction (top-to-bottom, left-to-right, …).
        save_path : str or Path, optional
            If provided, render and save to this path instead of displaying.
        format : {'svg', 'png', 'pdf'}, default='svg'
            Output format used when ``save_path`` is given.
        **kwargs
            Ignored (kept for backward compatibility).

        Returns
        -------
        widget, str, or graphviz.Source
            - Jupyter: ``ipywidgets.VBox`` widget (or SVG string if
              ``ipywidgets`` is not installed).
            - Script / REPL: ``graphviz.Source`` (viewer opened
              automatically, or file saved when ``save_path`` is set).
            - No graphviz installed: DOT source string.

        Notes
        -----
        Requires the ``graphviz`` Python package and system binary.
        Install with ``pip install graphviz`` and ``apt install graphviz``.
        Interactive toggles require ``pip install ipywidgets``.

        Examples
        --------
        Display in notebook with all details:

        >>> model.plot_model(show_params=True, show_trainable=True)

        Save to file:

        >>> model.plot_model(save_path="model.png", format="png")

        Left-to-right layout:

        >>> model.plot_model(rankdir="LR")
        """
        # ── Color palette (light theme) ───────────────────────────────────────
        _BG = "white"
        _FONT = "#1E293B"   # dark slate — primary label text
        _DIM = "#64748B"    # slate gray — secondary text (shapes, params)
        _EDGE = "#94A3B8"   # edge lines
        _FONTS = "Helvetica Neue,Helvetica,Arial,sans-serif"

        # Known layer types → (fill, border)
        _KNOWN_COLORS: dict[str, tuple[str, str]] = {
            "Input":                   ("#EFF6FF", "#3B82F6"),
            "ReservoirLayer":          ("#FFFBEB", "#D97706"),
            "CGReadoutLayer":          ("#F0FDF4", "#16A34A"),
            "ReadoutLayer":            ("#F0FDF4", "#16A34A"),
            "Concatenate":             ("#F5F3FF", "#7C3AED"),
            "SelectiveExponentiation": ("#FFF1F2", "#E11D48"),
            "Power":                   ("#FFF7ED", "#EA580C"),
            "FeaturePartitioner":      ("#F0FDFA", "#0D9488"),
        }
        _FALLBACK_PALETTE = [
            ("#F8F0FF", "#9333EA"),
            ("#F0F8FF", "#0369A1"),
            ("#FFF8F0", "#B45309"),
            ("#F0FFF8", "#047857"),
            ("#FFF0F8", "#BE185D"),
            ("#F8FFF0", "#4D7C0F"),
        ]

        # ── Helpers ───────────────────────────────────────────────────────────
        node_to_name = getattr(self, "_node_to_layer_name", {})

        def _is_jupyter() -> bool:
            try:
                return get_ipython().__class__.__name__ == "ZMQInteractiveShell"  # type: ignore[name-defined]
            except NameError:
                return False

        def _get_node_label(node: Any) -> str:
            if node in node_to_name:
                return node_to_name[node]
            for i, inp in enumerate(self.inputs):
                if node is inp:
                    return f"Input_{i + 1}"
            return f"node_{id(node)}"

        def _get_node_shape_str(node: Any) -> str:
            if hasattr(node, "shape") and isinstance(node.shape, torch.Size):
                return str(tuple(node.shape))
            return ""

        def _get_type_color(cls_name: str) -> tuple[str, str]:
            if cls_name in _KNOWN_COLORS:
                return _KNOWN_COLORS[cls_name]
            return _FALLBACK_PALETTE[hash(cls_name) % len(_FALLBACK_PALETTE)]

        def _get_trainable_status(module: Any) -> bool | None:
            """True = trainable, False = frozen, None = no parameters."""
            if hasattr(module, "trainable"):
                return bool(module.trainable)
            params = list(module.parameters())
            if not params:
                return None
            return any(p.requires_grad for p in params)

        def _get_layer_params(module: Any) -> list[str]:
            cls = type(module).__name__
            if cls == "ReservoirLayer":
                return [
                    f"N={module.reservoir_size} | sr={module.spectral_radius}",
                    f"leak={module.leak_rate} | act={module.activation}",
                ]
            if cls == "CGReadoutLayer":
                return [
                    f"in={module.in_features} | out={module.out_features}",
                    f"\u03b1={module.alpha}",
                ]
            if cls == "ReadoutLayer":
                return [f"in={module.in_features} | out={module.out_features}"]
            if cls == "SelectiveExponentiation":
                parity = "even" if module.index % 2 == 0 else "odd"
                return [f"exp={module.exponent} | {parity} idx"]
            if cls == "Power":
                return [f"exp={module.exponent}"]
            if cls == "FeaturePartitioner":
                return [f"parts={module.partitions} | overlap={module.overlap}"]
            return []

        def _html_label(
            cls_name: str,
            shape_str: str,
            module: Any,
            show_s: bool,
            show_p: bool,
            show_t: bool,
        ) -> str:
            """Build an HTML-like graphviz label for a node."""
            # Padlock indicator
            lock_str = ""
            if show_t and module is not None:
                status = _get_trainable_status(module)
                if status is not None:
                    lock_str = " \U0001f513" if status else " \U0001f512"

            rows = [
                '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" CELLPADDING="4">',
                f'<TR><TD ALIGN="CENTER"><B><FONT COLOR="{_FONT}">'
                f"{cls_name}{lock_str}</FONT></B></TD></TR>",
            ]

            has_details = (show_s and shape_str) or (
                show_p and module is not None and bool(_get_layer_params(module))
            )
            if has_details:
                rows.append("<TR><TD><HR/></TD></TR>")

            if show_s and shape_str:
                rows.append(
                    f'<TR><TD ALIGN="CENTER">'
                    f'<FONT POINT-SIZE="9" COLOR="{_DIM}">{shape_str}</FONT>'
                    f"</TD></TR>"
                )

            if show_p and module is not None:
                for line in _get_layer_params(module):
                    rows.append(
                        f'<TR><TD ALIGN="CENTER">'
                        f'<FONT POINT-SIZE="9" COLOR="{_DIM}">{line}</FONT>'
                        f"</TD></TR>"
                    )

            rows.append("</TABLE>>")
            return "".join(rows)

        def _build_dot(show_s: bool, show_p: bool, show_t: bool) -> str:
            """Generate DOT source for the current flag combination."""
            # Collect nodes: name -> (cls_name, shape_str, module, is_input, is_output)
            nodes: dict[str, tuple[str, str, Any, bool, bool]] = {}
            edges: list[tuple[str, str, str]] = []

            for i, inp in enumerate(self.inputs):
                name = f"Input_{i + 1}"
                nodes[name] = ("Input", _get_node_shape_str(inp), None, True, False)

            for node, layer_name in node_to_name.items():
                module = getattr(node, "layer", None)
                cls_name = (
                    type(module).__name__
                    if module is not None
                    else layer_name.rsplit("_", 1)[0]
                )
                nodes[layer_name] = (
                    cls_name,
                    _get_node_shape_str(node),
                    module,
                    False,
                    False,
                )
                for parent in getattr(node, "_parents", []):
                    parent_name = _get_node_label(parent)
                    edge_shape = _get_node_shape_str(parent) if show_s else ""
                    edges.append((parent_name, layer_name, edge_shape))

            for out in self.outputs:
                out_name = _get_node_label(out)
                if out_name in nodes:
                    cls_name, shape_str, module, is_input, _ = nodes[out_name]
                    nodes[out_name] = (cls_name, shape_str, module, is_input, True)

            # Gather class types seen (for legend)
            seen_classes: dict[str, tuple[str, str]] = {}
            for cls_name, _, _, _, _ in nodes.values():
                if cls_name not in seen_classes:
                    seen_classes[cls_name] = _get_type_color(cls_name)

            lines = [
                "digraph ESNModel {",
                f'  bgcolor="{_BG}";',
                f"  rankdir={rankdir};",
                "  splines=true;",
                "  nodesep=0.7;",
                "  ranksep=0.9;",
                "  pad=0.5;",
                f'  graph [fontname="{_FONTS}"];',
                f'  node [fontname="{_FONTS}", penwidth=1.5];',
                f'  edge [color="{_EDGE}", penwidth=1.5, arrowsize=0.75,'
                f' arrowhead=vee, fontname="{_FONTS}",'
                f' fontcolor="{_DIM}", fontsize=9];',
            ]

            for name, (cls_name, shape_str, module, is_input, is_output) in nodes.items():
                html = _html_label(cls_name, shape_str, module, show_s, show_p, show_t)
                fill, border = _get_type_color(cls_name)
                node_shape = "ellipse" if is_input else "box"
                style = "filled" if is_input else (
                    "filled,rounded,bold" if is_output else "filled,rounded"
                )
                lines.append(
                    f'  "{name}" [label={html}, shape={node_shape},'
                    f' style="{style}", fillcolor="{fill}", color="{border}"];'
                )

            for from_name, to_name, shape_label in edges:
                if shape_label:
                    lines.append(
                        f'  "{from_name}" -> "{to_name}" [label="{shape_label}"];'
                    )
                else:
                    lines.append(f'  "{from_name}" -> "{to_name}";')

            # Legend subgraph
            legend_items = list(seen_classes.items())  # [(cls_name, (fill, border))]
            if legend_items:
                lnames = [f"__legend_{i}__" for i in range(len(legend_items))]
                lines.append("  subgraph cluster_legend {")
                lines.append('    label = "Legend";')
                lines.append('    style = "filled";')
                lines.append('    fillcolor = "#F8FAFC";')
                lines.append('    color = "#CBD5E1";')
                lines.append(f'    fontname = "{_FONTS}";')
                lines.append("    fontsize = 10;")
                for lname, (cls_name, (fill, border)) in zip(lnames, legend_items):
                    lines.append(
                        f'    "{lname}" [label="{cls_name}", shape=box,'
                        f' style="filled,rounded", fillcolor="{fill}",'
                        f' color="{border}", fontsize=9, fontname="{_FONTS}"];'
                    )
                rank_nodes = "; ".join(f'"{n}"' for n in lnames)
                lines.append(f"    {{rank=same; {rank_nodes};}}")
                lines.append("  }")

            lines.append("}")
            return "\n".join(lines)

        # ── Rendering ─────────────────────────────────────────────────────────
        try:
            import graphviz

            if save_path is not None:
                save_path = Path(save_path)
                dot_src = _build_dot(show_shapes, show_params, show_trainable)
                graphviz.Source(dot_src).render(
                    str(save_path.with_suffix("")), format=format, cleanup=True
                )
                print(f"Saved to {save_path.with_suffix('.' + format)}")
                return graphviz.Source(dot_src)

            if _is_jupyter():
                try:
                    import ipywidgets as widgets
                    from IPython.display import SVG, display as ipy_display

                    cb_shapes = widgets.Checkbox(
                        value=show_shapes,
                        description="Show shapes",
                        layout=widgets.Layout(width="150px"),
                    )
                    cb_params = widgets.Checkbox(
                        value=show_params,
                        description="Show params",
                        layout=widgets.Layout(width="150px"),
                    )
                    cb_trainable = widgets.Checkbox(
                        value=show_trainable,
                        description="Show trainable",
                        layout=widgets.Layout(width="160px"),
                    )
                    out = widgets.Output()

                    def _render(*_: Any) -> None:
                        dot_src = _build_dot(cb_shapes.value, cb_params.value, cb_trainable.value)
                        svg_data = graphviz.Source(dot_src).pipe(format="svg").decode("utf-8")
                        out.clear_output(wait=True)
                        with out:
                            ipy_display(SVG(svg_data))

                    cb_shapes.observe(_render, names="value")
                    cb_params.observe(_render, names="value")
                    cb_trainable.observe(_render, names="value")
                    _render()

                    widget = widgets.VBox(
                        [widgets.HBox([cb_shapes, cb_params, cb_trainable]), out]
                    )
                    ipy_display(widget)
                    return widget

                except ImportError:
                    # ipywidgets not available — static display
                    from IPython.display import SVG, display as ipy_display

                    dot_src = _build_dot(show_shapes, show_params, show_trainable)
                    svg_data = graphviz.Source(dot_src).pipe(format="svg").decode("utf-8")
                    ipy_display(SVG(svg_data))
                    return svg_data

            # Non-Jupyter: open system viewer
            dot_src = _build_dot(show_shapes, show_params, show_trainable)
            graph = graphviz.Source(dot_src)
            try:
                graph.view(cleanup=True)
            except Exception:
                pass
            return graph

        except ImportError:
            print("Note: Install 'graphviz' package for visual rendering.")
            print("      pip install graphviz")
            print("      Also install graphviz system package (apt install graphviz)")
            print("\nDOT source (can be rendered at https://dreampuf.github.io/GraphvizOnline/):")
            dot_src = _build_dot(show_shapes, show_params, show_trainable)
            print(dot_src)
            return dot_src

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
