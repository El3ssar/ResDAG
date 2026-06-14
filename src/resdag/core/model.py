"""
ESNModel — core model class for reservoir computing.

This module provides the main model class for building ESN architectures
using the ``pytorch_symbolic`` library for symbolic tensor computation.

The :class:`ESNModel` class extends ``pytorch_symbolic.SymbolicModel`` with
ESN-specific functionality including forecasting, reservoir state management,
and model persistence.

Examples
--------
Simple ESN model:

>>> import pytorch_symbolic as ps
>>> from resdag.layers import ESNLayer
>>> from resdag.layers.readouts import CGReadoutLayer
>>> from resdag.core import ESNModel
>>>
>>> inp = ps.Input((100, 1))
>>> reservoir = ESNLayer(50, feedback_size=1)(inp)
>>> readout = CGReadoutLayer(50, 1)(reservoir)
>>> model = ESNModel(inp, readout)
>>> model.summary()

Multi-input model with driving signal:

>>> feedback = ps.Input((100, 3))
>>> driver = ps.Input((100, 5))
>>> reservoir = ESNLayer(100, feedback_size=3, input_size=5)(feedback, driver)
>>> readout = CGReadoutLayer(100, 3)(reservoir)
>>> model = ESNModel([feedback, driver], readout)

See Also
--------
resdag.layers.ESNLayer : Reservoir layer with recurrent dynamics.
resdag.layers.readouts.CGReadoutLayer : Conjugate gradient readout layer.
resdag.training.ESNTrainer : Trainer for fitting readout layers.
"""

from __future__ import annotations

import colorsys
import copy
from pathlib import Path
from typing import Any

import pytorch_symbolic as ps
import torch
from pytorch_symbolic.symbolic_data import SymbolicData

from resdag.layers.reservoirs import BaseReservoirLayer

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

    Pass symbolic ``inputs`` and ``outputs`` to the constructor (see
    ``pytorch_symbolic.SymbolicModel``).

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
    >>> from resdag.core import ESNModel
    >>> from resdag.layers import ESNLayer
    >>> from resdag.layers.readouts import CGReadoutLayer
    >>>
    >>> inp = ps.Input((100, 3))
    >>> reservoir = ESNLayer(200, feedback_size=3)(inp)
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
    BaseReservoirLayer : Reservoir layer component.
    ESNTrainer : Trainer for fitting readout layers.
    """

    def __deepcopy__(self, memo: dict[int, Any]) -> "ESNModel":
        """
        Deep copy preserving the model subclass and its symbolic graph.

        ``pytorch_symbolic.SymbolicModel.__deepcopy__`` rebuilds the copy as a
        ``DetachedSymbolicModel``, silently dropping every subclass method
        (``forecast``, ``reset_reservoirs``, ``save``, ...).  This override
        instead produces a faithful, fully independent :class:`ESNModel`:
        parameters, buffers and reservoir states are copied, and the symbolic
        graph (inputs, outputs, execution order) is duplicated so that
        ``summary()`` and ``plot_model()`` keep working on the copy.

        Returns
        -------
        ESNModel
            An independent copy sharing no tensors or graph nodes with
            ``self``.  Subclasses of :class:`ESNModel` are preserved as well.

        Notes
        -----
        Non-input graph nodes cache a ``_value`` that may be an autograd-tracked
        non-leaf tensor, which :func:`copy.deepcopy` cannot duplicate.  Each node
        is therefore pre-seeded into ``memo`` as an empty shell whose ``__dict__``
        is filled by hand with ``_value`` cleared, so the copy machinery never
        tries to copy those caches — they are recomputed on demand from the
        retained input placeholders.

        Since ``pytorch-symbolic`` 1.2 the whole model is also picklable, so
        :meth:`save_full` / :meth:`load_full` (or ``torch.save(model)``) offer an
        alternative way to clone or persist a model, graph and all.

        Examples
        --------
        >>> import copy
        >>> clone = copy.deepcopy(model)
        >>> type(clone) is type(model)
        True
        >>> clone.reset_reservoirs()  # ESN API intact on the copy
        """
        cls = self.__class__
        obj = cls.__new__(cls)
        memo[id(self)] = obj

        shells = [(node, object.__new__(type(node))) for node in self._graph_nodes()]
        for node, shell in shells:
            memo[id(node)] = shell

        input_ids = {id(node) for node in self.inputs}
        for node, shell in shells:
            node_dict = dict(node.__dict__)
            if id(node) not in input_ids:
                # Non-input values are lazy caches (possibly autograd-tracked,
                # hence not deep-copyable); they are recomputed on demand from
                # the retained input placeholders.
                node_dict["_value"] = None
            shell.__dict__.update(copy.deepcopy(node_dict, memo))

        for key, value in self.__dict__.items():
            obj.__dict__[key] = copy.deepcopy(value, memo)
        return obj

    def _graph_nodes(self) -> list[SymbolicData]:
        """Collect every symbolic graph node reachable from inputs/outputs."""
        seen: set[int] = set()
        nodes: list[SymbolicData] = []
        stack: list[SymbolicData] = [*self.inputs, *self.outputs]
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            nodes.append(node)
            stack.extend(node._parents)
            stack.extend(node._children)
            stack.extend(node._layer_full_siblings)
        return nodes

    def reset_reservoirs(self) -> None:
        """
        Reset all reservoir layer states to zero.

        This clears the internal hidden states of all :class:`BaseReservoirLayer`
        modules in the model, preparing it for a new sequence.

        Examples
        --------
        >>> model.reset_reservoirs()
        >>> output = model(new_sequence)
        """
        for module in self.modules():
            if isinstance(module, BaseReservoirLayer):
                module.reset_state()

    def set_random_reservoir_states(
        self,
        batch_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Set random states of all reservoir layers.

        Parameters
        ----------
        batch_size : int, optional
            If provided, lazily initialise each reservoir's state with this
            batch size before filling it with random values.
        device : torch.device, optional
            Target device for lazy initialisation.
        dtype : torch.dtype, optional
            Target dtype for lazy initialisation.

        Examples
        --------
        >>> model.set_random_reservoir_states()                # state must exist
        >>> model.set_random_reservoir_states(batch_size=4)    # lazy
        """
        for module in self.modules():
            if isinstance(module, BaseReservoirLayer):
                module.set_random_state(batch_size=batch_size, device=device, dtype=dtype)

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
            if isinstance(module, BaseReservoirLayer) and module.state is not None:
                states[name] = module.state.clone()
        return states

    def set_reservoir_states(
        self,
        states: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        """
        Set states of reservoir layers.

        Parameters
        ----------
        states : dict of str to torch.Tensor
            Dictionary mapping layer names to state tensors.
            Names should match those returned by :meth:`get_reservoir_states`.
        strict : bool, default=True
            If ``True``, raise a ``KeyError`` when the provided ``states``
            dict is missing entries for any reservoir in the model or
            contains keys that don't match any reservoir.  Set to ``False``
            to silently ignore both kinds of mismatch (legacy behaviour).

        Raises
        ------
        KeyError
            If ``strict=True`` and the keys of ``states`` do not exactly
            match the set of reservoir layer names in the model.

        Examples
        --------
        >>> states = model.get_reservoir_states()
        >>> # ... do something ...
        >>> model.set_reservoir_states(states)  # Restore states
        """
        reservoir_names = {
            name for name, module in self.named_modules() if isinstance(module, BaseReservoirLayer)
        }
        provided = set(states.keys())

        if strict:
            missing = reservoir_names - provided
            extra = provided - reservoir_names
            if missing or extra:
                parts: list[str] = []
                if missing:
                    parts.append(f"missing keys for reservoirs: {sorted(missing)}")
                if extra:
                    parts.append(f"unexpected keys not matching any reservoir: {sorted(extra)}")
                raise KeyError("set_reservoir_states(strict=True): " + "; ".join(parts))

        for name, module in self.named_modules():
            if isinstance(module, BaseReservoirLayer) and name in states:
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

        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError(
                f"{path} is not a save() state-dict checkpoint. If it was written "
                f"by save_full() (or torch.save(model)), use ESNModel.load_full()."
            )

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
        model: "ESNModel | None" = None,
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

    def save_full(
        self,
        path: str | Path,
        **metadata: Any,
    ) -> None:
        """
        Serialize the *entire* model — architecture, weights, and reservoir
        states — to a single file.

        Unlike :meth:`save`, which stores only the ``state_dict`` (so the
        architecture must be re-created before :meth:`load`), this pickles the
        whole model object, the ``pytorch_symbolic`` graph included.  Restore it
        with :meth:`load_full` without rebuilding anything.  This relies on the
        pickle support added in ``pytorch-symbolic`` 1.2.

        Parameters
        ----------
        path : str or Path
            File path to save to. Parent directories are created if needed.
        **metadata
            Additional metadata stored alongside the model (e.g. training info).

        Notes
        -----
        The file is a regular ``torch.save`` payload, loaded back with
        ``weights_only=False`` — only open files you trust.  Current reservoir
        states are captured as-is; reset or warm up beforehand to control what
        is persisted.

        Any custom callable passed as a ``topology``, ``*_initializer``, or
        ``activation`` spec must be importable (a module-level ``def``, not a
        ``lambda`` or locally-defined function) for the model to pickle.  Specs
        given as strings, ``(name, kwargs)`` tuples, or registered objects
        always serialize.  If a callable is not picklable, use the lighter
        state-dict :meth:`save` instead.

        Examples
        --------
        >>> model.save_full("model_full.pt", epoch=10)
        >>> restored = ESNModel.load_full("model_full.pt")  # no rebuild needed
        >>> predictions = restored.forecast(warmup, horizon=100)

        See Also
        --------
        load_full : Reconstruct a model saved with this method.
        save : Lighter, state-dict-only persistence (architecture not stored).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"resdag_full_model": self, "metadata": metadata}, path)

    @classmethod
    def load_full(
        cls,
        path: str | Path,
        return_metadata: bool = False,
        map_location: Any = None,
    ) -> "ESNModel | tuple[ESNModel, dict[str, Any]]":
        """
        Reconstruct a complete model saved with :meth:`save_full`.

        No pre-built architecture is required — the model object, its symbolic
        graph, weights, and reservoir states are all restored from the file.

        Parameters
        ----------
        path : str or Path
            File path written by :meth:`save_full`.
        return_metadata : bool, default=False
            If True, return ``(model, metadata)`` instead of just the model.
        map_location : optional
            Passed to ``torch.load`` to remap storage devices — e.g.
            ``"cpu"`` to load a GPU-saved model on a CPU-only machine.

        Returns
        -------
        ESNModel or tuple of (ESNModel, dict)
            The reconstructed model, optionally paired with its metadata dict.

        Raises
        ------
        ValueError
            If the file does not contain a whole model — e.g. it is a
            state-dict checkpoint from :meth:`save` (rebuild the architecture
            and use :meth:`load` / :meth:`load_from_file` for those).

        Warnings
        --------
        Loads with ``weights_only=False``, which unpickles arbitrary Python
        objects.  Only call this on files from a source you trust.

        Notes
        -----
        Accepts both :meth:`save_full` files (a metadata wrapper) and bare
        ``torch.save(model)`` files (the model object on its own); the latter
        carry no metadata.

        Examples
        --------
        >>> model.save_full("model_full.pt")
        >>> restored = ESNModel.load_full("model_full.pt")
        >>> model_cpu = ESNModel.load_full("gpu_model.pt", map_location="cpu")

        See Also
        --------
        save_full : Serialize a complete model.
        load_from_file : Load a state-dict checkpoint into an existing model.
        """
        path = Path(path)
        payload = torch.load(path, weights_only=False, map_location=map_location)
        model: ESNModel
        metadata: dict[str, Any]
        if isinstance(payload, dict) and "resdag_full_model" in payload:
            model = payload["resdag_full_model"]
            metadata = payload.get("metadata", {})
        elif isinstance(payload, ESNModel):
            # A bare ``torch.save(model)`` file, with no metadata wrapper.
            model = payload
            metadata = {}
        else:
            raise ValueError(
                f"{path} does not contain a full ESNModel. Write one with "
                f"save_full() (or torch.save(model)); for a state-dict checkpoint "
                f"from save(), rebuild the architecture and use load()/load_from_file()."
            )
        if return_metadata:
            return model, metadata
        return model

    def plot_model(
        self,
        show_shapes: bool = False,
        show_trainable: bool = False,
        rankdir: str = "TB",
        save_path: str | Path | None = None,
        format: str = "svg",
        **kwargs: Any,
    ) -> Any:
        """
        Visualize model architecture as a graph.

        Parameters
        ----------
        show_shapes : bool, default=False
            Show tensor shapes on edges.
        show_trainable : bool, default=False
            Show a padlock indicator (🔒 frozen / 🔓 trainable) on nodes
            that have learnable parameters.
        rankdir : {'TB', 'LR', 'BT', 'RL'}, default='TB'
            Graph layout direction.
        save_path : str or Path, optional
            Render and save to this path instead of displaying.
        format : {'svg', 'png', 'pdf'}, default='svg'
            Output format when ``save_path`` is given.

        Returns
        -------
        graphviz.Source or None
            ``None`` in Jupyter (diagram already displayed). ``graphviz.Source``
            in script/REPL (system viewer opened). DOT string if graphviz is
            not installed.

        Notes
        -----
        Requires the ``graphviz`` Python package and system binary.
        ``pip install graphviz`` and ``apt install graphviz``.
        """
        # ── Palette ───────────────────────────────────────────────────────────
        _FONT = "#1A2332"
        _EDGE = "#8B9CB6"
        _FONTS = "Helvetica Neue,Helvetica,Arial,sans-serif"

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

        def _get_shape_str(node: Any) -> str:
            if hasattr(node, "shape") and isinstance(node.shape, torch.Size):
                return str(tuple(node.shape))
            return ""

        def _trainable_status(module: Any) -> bool | None:
            if hasattr(module, "trainable"):
                return bool(module.trainable)
            params = list(module.parameters())
            return None if not params else any(p.requires_grad for p in params)

        def _build_dot() -> str:
            nodes: dict[str, tuple[str, str, Any, bool, bool]] = {}
            edges: list[tuple[str, str, str]] = []

            for i, inp in enumerate(self.inputs):
                nodes[f"Input_{i + 1}"] = ("Input", _get_shape_str(inp), None, True, False)

            for node, layer_name in node_to_name.items():
                module = getattr(node, "layer", None)
                cls_name = (
                    type(module).__name__ if module is not None else layer_name.rsplit("_", 1)[0]
                )
                nodes[layer_name] = (cls_name, _get_shape_str(node), module, False, False)
                for parent in getattr(node, "_parents", []):
                    edge_label = _get_shape_str(parent) if show_shapes else ""
                    edges.append((_get_node_label(parent), layer_name, edge_label))

            for out in self.outputs:
                out_name = _get_node_label(out)
                if out_name in nodes:
                    cls_name, shape_str, module, is_input, _ = nodes[out_name]
                    nodes[out_name] = (cls_name, shape_str, module, is_input, True)

            # Assign hues via golden ratio over sorted unique class names.
            # Golden ratio on sequential *indices* guarantees maximal perceptual
            # separation — the only correct way to avoid adjacent colors.
            # Sorting makes assignment deterministic regardless of graph order.
            sorted_classes = sorted({cls for cls, *_ in nodes.values()})
            _hues = {cls: (i * 0.618033988749895) % 1.0 for i, cls in enumerate(sorted_classes)}

            def _hex(r: float, g: float, b: float) -> str:
                return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"

            def _color_for(cls_name: str) -> tuple[str, str]:
                hue = _hues.get(cls_name, 0.0)
                fill = _hex(*colorsys.hsv_to_rgb(hue, 0.20, 0.97))
                border = _hex(*colorsys.hsv_to_rgb(hue, 0.82, 0.52))
                return fill, border

            lines = [
                "digraph ESNModel {",
                "  bgcolor=white;",
                f"  rankdir={rankdir};",
                "  splines=true;",
                "  nodesep=0.6;",
                "  ranksep=0.8;",
                "  pad=0.4;",
                f'  graph [fontname="{_FONTS}"];',
                f'  node [fontname="{_FONTS}", penwidth=1.5];',
                f'  edge [color="{_EDGE}", penwidth=1.2, arrowsize=0.7, arrowhead=vee,'
                f' fontname="{_FONTS}", fontcolor="{_EDGE}", fontsize=9];',
            ]

            for name, (cls_name, shape_str, module, is_input, is_output) in nodes.items():
                lock = ""
                if show_trainable and module is not None:
                    status = _trainable_status(module)
                    if status is not None:
                        lock = " \U0001f513" if status else " \U0001f512"
                label = f'"{cls_name}{lock}"'
                if show_shapes and shape_str:
                    label = f'"{cls_name}{lock}\\n{shape_str}"'

                fill, border = _color_for(cls_name)
                node_shape = "ellipse" if is_input else "box"
                style = (
                    "filled"
                    if is_input
                    else ("filled,rounded,bold" if is_output else "filled,rounded")
                )
                lines.append(
                    f'  "{name}" [label={label}, shape={node_shape},'
                    f' style="{style}", fillcolor="{fill}", color="{border}",'
                    f' fontcolor="{_FONT}"];'
                )

            for from_name, to_name, edge_label in edges:
                if edge_label:
                    lines.append(f'  "{from_name}" -> "{to_name}" [label="{edge_label}"];')
                else:
                    lines.append(f'  "{from_name}" -> "{to_name}";')

            lines.append("}")
            return "\n".join(lines)

        # ── Rendering ─────────────────────────────────────────────────────────
        def _print_dot_fallback(reason: str) -> str:
            print(reason)
            print("DOT source (paste at https://dreampuf.github.io/GraphvizOnline/):")
            dot_src = _build_dot()
            print(dot_src)
            return dot_src

        try:
            import graphviz

            dot_src = _build_dot()
            src = graphviz.Source(dot_src)

            if save_path is not None:
                save_path = Path(save_path)
                try:
                    src.render(str(save_path.with_suffix("")), format=format, cleanup=True)
                except graphviz.ExecutableNotFound:
                    return _print_dot_fallback(
                        "graphviz system binary not found: install it "
                        "(e.g. apt install graphviz) or render the DOT source below."
                    )
                print(f"Saved to {save_path.with_suffix('.' + format)}")
                return src

            if _is_jupyter():
                from IPython.display import SVG
                from IPython.display import display as ipy_display

                try:
                    svg = src.pipe(format="svg").decode("utf-8")
                except graphviz.ExecutableNotFound:
                    return _print_dot_fallback(
                        "graphviz system binary not found: install it "
                        "(e.g. apt install graphviz) or render the DOT source below."
                    )
                ipy_display(SVG(svg))
                return None  # prevent double-display from cell output

            try:
                src.view(cleanup=True)
            except Exception:
                pass
            return src

        except ImportError:
            return _print_dot_fallback(
                "graphviz not installed: pip install graphviz && apt install graphviz"
            )

    @torch.no_grad()
    def warmup(
        self,
        *inputs: torch.Tensor,
        return_outputs: bool = False,
        reset: bool = True,
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
        reset : bool, default=True
            If True, reservoir states are reset to ``None`` before the
            warmup pass.  Set to ``False`` only if you want to continue
            from a previously saved state — typical workflows always reset.

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

        Continue warming from a saved state:

        >>> model.set_reservoir_states(saved_states)
        >>> model.warmup(more_data, reset=False)

        See Also
        --------
        forecast : Two-phase forecasting with warmup and generation.
        reset_reservoirs : Reset all reservoir states.
        """
        if len(inputs) == 0:
            raise ValueError("At least one input (feedback) is required")

        if reset:
            self.reset_reservoirs()

        output = self(*inputs)

        return output if return_outputs else None

    @torch.no_grad()
    def forecast(
        self,
        warmup_inputs: tuple[torch.Tensor, ...] | torch.Tensor,
        forecast_inputs: tuple[torch.Tensor, ...] | None = None,
        *,
        horizon: int,
        initial_feedback: torch.Tensor | None = None,
        return_warmup: bool = False,
        reset: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Two-phase forecast: teacher-forced warmup + autoregressive generation.

        Phase 1 (Warmup): Runs model with provided inputs to synchronize
        reservoir states with input dynamics (Echo State Property).

        Phase 2 (Forecast): Autoregressive generation where feedback comes
        from the model's own output while driving inputs (if any) are
        supplied through ``forecast_inputs``.

        Parameters
        ----------
        warmup_inputs : tuple of torch.Tensor or torch.Tensor
            Warmup tensors of shape ``(batch, warmup_steps, features)``.
            Convention: first element is feedback, remaining are drivers.
            A single tensor is accepted for the common feedback-only case.
        forecast_inputs : tuple of torch.Tensor, optional
            Driver inputs for the autoregressive phase (feedback is provided
            by the model's own output, so it is not part of this tuple).
            ``forecast_inputs[i][:, t, :]`` is the driver value at the
            ``t``-th timestep *after* the warmup window — i.e. pass the
            driver series sliced over the forecast window, continuing
            exactly where the warmup drivers ended.  Each tensor must have
            ``horizon - 1`` timesteps (or ``horizon``; the last step is then
            unused, which lets you slice drivers over the same window as
            your validation targets).  Required when the model has driver
            inputs.
        horizon : int, keyword-only
            Number of autoregressive steps to generate.
        initial_feedback : torch.Tensor, optional
            Custom initial feedback of shape ``(batch, 1, feedback_dim)``.
            If None, uses last warmup output.
        return_warmup : bool, default=False
            If True, prepend warmup outputs to the result.
        reset : bool, default=True
            If True, reservoir states are reset to ``None`` before warmup.
            Set to ``False`` only if you want to continue from a previously
            saved state.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            For single-output models: tensor of shape
            ``(batch, horizon, output_dim)`` or
            ``(batch, warmup_steps + horizon, output_dim)`` if ``return_warmup``.

            For multi-output models: tuple of tensors with the same structure.

        Raises
        ------
        ValueError
            If no warmup inputs are provided, if ``forecast_inputs`` is required
            but missing, or if dimensions don't match.

        Notes
        -----
        - Convention: first warmup element is always feedback (used for autoregression).
        - For multi-output models, first output is used as feedback.
        - Feedback output dimension must match feedback input dimension.

        **Driver time alignment.** Training pairs ``(feedback_t, driver_t)``
        with target ``feedback_{t+1}``.  The forecast loop keeps that pairing:
        prediction ``i`` (for the ``(i+1)``-th step after warmup) is produced
        from the previous prediction together with ``forecast_inputs[:, i-1]``,
        the driver at that same timestep.  Prediction ``0`` is produced during
        warmup from the last warmup feedback/driver pair, so the forecast
        drivers start at the first step *after* the warmup window — no overlap
        with the warmup drivers.

        Examples
        --------
        Simple feedback-only model:

        >>> warmup_data = torch.randn(1, 50, 3)
        >>> predictions = model.forecast(warmup_data, horizon=100)
        >>> print(predictions.shape)
        torch.Size([1, 100, 3])

        Input-driven model:

        >>> predictions = model.forecast(
        ...     (warmup_feedback, warmup_driver),
        ...     forecast_inputs=(future_driver,),
        ...     horizon=100,
        ... )

        Include warmup in output:

        >>> full_output = model.forecast(
        ...     warmup_data,
        ...     horizon=100,
        ...     return_warmup=True,
        ... )
        >>> print(full_output.shape)  # warmup_steps + horizon
        torch.Size([1, 150, 3])

        See Also
        --------
        warmup : Teacher-forced warmup only.
        reset_reservoirs : Reset reservoir states before forecasting.
        """
        # Normalise warmup_inputs into a tuple
        if isinstance(warmup_inputs, torch.Tensor):
            warmup_inputs = (warmup_inputs,)
        else:
            warmup_inputs = tuple(warmup_inputs)
        if len(warmup_inputs) == 0:
            raise ValueError("At least one warmup input (feedback) is required")

        # Determine if model has driving inputs
        num_drivers = len(warmup_inputs) - 1
        has_drivers = num_drivers > 0

        # Validate forecast_inputs (drivers only)
        if has_drivers:
            if forecast_inputs is None:
                raise ValueError(
                    f"Model has {num_drivers} driving input(s). "
                    f"forecast_inputs must be provided for the autoregressive phase."
                )
            forecast_inputs = tuple(forecast_inputs)
            if len(forecast_inputs) != num_drivers:
                raise ValueError(
                    f"Expected {num_drivers} forecast driver(s), got {len(forecast_inputs)}"
                )
            for i, driver in enumerate(forecast_inputs):
                if driver.shape[1] not in (horizon - 1, horizon):
                    raise ValueError(
                        f"forecast_inputs[{i}] has {driver.shape[1]} steps, expected "
                        f"{horizon - 1} (or {horizon}; the last step is unused). "
                        f"forecast_inputs must hold the driver series for the "
                        f"forecast window, starting right after the warmup window."
                    )

        batch_size = warmup_inputs[0].shape[0]
        feedback_dim = warmup_inputs[0].shape[-1]
        device = warmup_inputs[0].device
        dtype = warmup_inputs[0].dtype

        # Phase 1: Warmup (handles reset internally)
        warmup_outputs = self.warmup(*warmup_inputs, return_outputs=True, reset=reset)

        # Determine output structure
        output_shape = self.output_shape
        multi_output = isinstance(output_shape, tuple) and isinstance(output_shape[0], torch.Size)

        # Validate feedback dimension
        if multi_output:
            feedback_output_dim = output_shape[0][-1]
        else:
            feedback_output_dim = output_shape[-1]

        if feedback_output_dim != feedback_dim:
            # Collect readout layer names for a helpful hint.
            readout_names = []
            for name, module in self.named_modules():
                # ReadoutLayer subclasses always have an out_features attribute.
                if hasattr(module, "out_features") and hasattr(module, "_is_fitted"):
                    label = getattr(module, "_name", None) or name
                    readout_names.append(f"{label} (out_features={module.out_features})")
            readouts_hint = f" Model readouts: {readout_names}." if readout_names else ""
            raise ValueError(
                f"forecast(): feedback dimension mismatch. "
                f"Feedback input has {feedback_dim} features but the model output "
                f"used as feedback has {feedback_output_dim} features. "
                f"For autoregressive forecasting the first output must match the "
                f"feedback input dimension.{readouts_hint}"
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
                # Prediction t pairs the previous prediction (feedback at the
                # t-th step after warmup) with the driver at that same step,
                # which is forecast_inputs[:, t-1] (drivers start right after
                # the warmup window; the step-0 driver was consumed in warmup).
                driver_inputs_t = tuple(driver[:, t - 1 : t, :] for driver in forecast_inputs)
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
