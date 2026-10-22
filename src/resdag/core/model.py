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
import contextlib
import copy
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, cast

import pytorch_symbolic as ps
import torch
from pytorch_symbolic.symbolic_data import SymbolicData

from resdag.core._execution import (
    ExecutionPlan,
    ModelOutput,
    ReservoirModelProtocol,
    ReservoirStates,
    build_execution_plan,
)
from resdag.core._flat_inference import (
    FlatStep,
    build_flat_step,
    rollout,
)
from resdag.core._forecast import ForecastAlignment, positive_step_count, prepare_forecast_inputs
from resdag.core._persistence import (
    load_checkpoint,
    load_full_model,
    save_checkpoint,
    save_full_model,
)
from resdag.layers.reservoirs import BaseReservoirLayer

# Re-export for convenience
Input = ps.Input


def _grad_context(no_grad: bool) -> AbstractContextManager[Any]:
    """
    Select the autograd context for a rollout helper.

    Parameters
    ----------
    no_grad : bool
        If ``True``, return :func:`torch.no_grad` so the rollout builds no
        autograd graph (the default — preserves behaviour and memory cost of
        the historical ``@torch.no_grad()`` decorators).  If ``False``, return
        :class:`contextlib.nullcontext` so the per-step calls build a
        backpropagatable graph (BPTT through the rollout).

    Returns
    -------
    contextlib.AbstractContextManager
        ``torch.no_grad()`` when ``no_grad=True``, else ``contextlib.nullcontext()``.
    """
    return torch.no_grad() if no_grad else contextlib.nullcontext()


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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Build the symbolic model and clear reservoir trace residue.

        ``pytorch_symbolic.SymbolicModel`` propagates shapes by running each
        layer's real ``forward`` on a ``torch.rand`` placeholder (batch size 1).
        For a stateful :class:`~resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer`
        that leaves the layer holding a *saturated* trace state at ``batch=1``.
        A fresh model used directly at ``batch=1`` (e.g. ``y = model(x)`` or
        standard forecasting) would silently continue from this junk instead of a
        clean zero state — batch sizes ``!= 1`` auto-resize to zeros and escape
        it, but ``batch=1`` direct calls do not. Resetting here, once, right after
        the graph is built, makes every freshly constructed model start from a
        clean state regardless of how it is first called.
        """
        super().__init__(*args, **kwargs)
        self.reset_reservoirs()

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
            if key in {"_flat_step_cache", "_execution_plan_cache"}:
                # The flattened-step cache holds an ``exec``-built closure that
                # ``copy.deepcopy`` treats as atomic (it would keep closing over
                # the *original* layers). Skip it; the copy rebuilds lazily on
                # its next forecast against its own layers.
                continue
            obj.__dict__[key] = copy.deepcopy(value, memo)
        return obj

    def __getstate__(self) -> dict[str, Any]:
        """Drop the unpicklable flattened-step cache before pickling.

        The cache holds an ``exec``-built step closure (and layer-bound methods)
        that cannot be pickled — like the generated ``forward`` that
        ``pytorch_symbolic.SymbolicModel.__getstate__`` already removes.  It is
        rebuilt lazily on the next :meth:`forecast`, so nothing is lost; this
        keeps :meth:`save_full` / ``torch.save(model)`` working after a forecast
        has populated the cache.
        """
        state: dict[str, Any] = super().__getstate__()
        state.pop("_flat_step_cache", None)
        state.pop("_execution_plan_cache", None)
        return state

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

    def _get_execution_plan(self) -> ExecutionPlan:
        """Share one graph description across sequence and forecast backends."""
        cache: tuple[tuple[SymbolicData, ...], ExecutionPlan] | None = getattr(
            self, "_execution_plan_cache", None
        )
        if cache is None or cache[0] is not self.outputs:
            plan = build_execution_plan(self)
            self._execution_plan_cache = (self.outputs, plan)
            return plan
        return cache[1]

    def forward_stateless(
        self,
        *inputs: torch.Tensor,
        states: Mapping[str, torch.Tensor] | None = None,
        compile: bool = False,
    ) -> tuple[ModelOutput, ReservoirStates]:
        """Run the complete DAG with explicit recurrent states.

        Inputs use ``(batch, time, features)``. Omitting ``states`` starts each
        reservoir from its configured reset state (zero by default). Otherwise
        pass the complete state dictionary returned by an earlier call; keys
        match :meth:`get_reservoir_states`.
        The returned output includes every original transform and readout.
        Stored reservoir states and supplied state tensors are not changed.

        Returned states retain gradients across chunks. Detach their tensors
        explicitly when truncated backpropagation is desired. Ordinary layer
        behavior (such as BatchNorm running statistics in training mode) still
        applies to non-reservoir modules in the graph.

        Each graph position must use a distinct reservoir layer instance.
        Separate instances may share weights, but sharing their recurrent
        state would make predictions depend on sequence chunk boundaries.
        """
        return self._get_execution_plan().run_sequence(inputs, states, compile=compile)

    def as_module(self, *, stateful: bool = False) -> "ReservoirModelModule":
        """Wrap the complete trained graph for a larger PyTorch pipeline.

        The wrapper shares this model's parameters and readouts. The default
        processes independent batches without touching stored reservoir state.
        Set ``stateful=True`` to carry state between calls, as ``model(x)``
        does, or use the wrapper's ``forward_stateless`` for explicit state.
        """
        self._get_execution_plan()
        return ReservoirModelModule(self, stateful=stateful)

    def reset_reservoirs(self) -> None:
        """
        Reset all reservoir layers for lazy initialization from their configured state.

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

    def get_state(self) -> ReservoirStates:
        """Return named copies of all currently initialized recurrent states."""
        return self.get_reservoir_states()

    def set_state(self, states: Mapping[str, torch.Tensor], strict: bool = True) -> None:
        """Restore named recurrent states; names match ``forward_stateless``."""
        self.set_reservoir_states(dict(states), strict=strict)

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

        Notes
        -----
        Each restored tensor is routed through
        :meth:`~resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer.set_state`,
        which clones the tensor and validates its shape against the target
        cell's contract (e.g. the 2-D ``(batch, reservoir_size)`` layout of an
        ESN).  Before validation the tensor is coerced to the target
        reservoir's reference device and dtype (its first floating-point
        parameter/buffer) so that a state saved on one device/dtype is not
        silently re-zeroed by ``_maybe_init_state`` on the next forward pass —
        the canonical save-on-GPU / load-on-CPU round-trip therefore preserves
        the warmed-up state values.

        Raises
        ------
        KeyError
            If ``strict=True`` and the keys of ``states`` do not exactly
            match the set of reservoir layer names in the model.
        ValueError
            If a restored tensor does not match the target cell's state-shape
            contract.  The error names the cell class and the offending shape.

        Warns
        -----
        UserWarning
            If a restored state had to be moved to a different device or cast
            to a different dtype to match its target reservoir.

        Examples
        --------
        >>> states = model.get_reservoir_states()
        >>> # ... do something ...
        >>> model.set_reservoir_states(states)  # Restore states
        """
        import warnings

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
                state = states[name]
                ref_device, ref_dtype = module.reference_device_dtype()
                if state.device != ref_device or state.dtype != ref_dtype:
                    warnings.warn(
                        f"set_reservoir_states: state for reservoir '{name}' was "
                        f"coerced from (device={state.device}, dtype={state.dtype}) to "
                        f"(device={ref_device}, dtype={ref_dtype}) to match the "
                        f"target reservoir.",
                        UserWarning,
                        stacklevel=2,
                    )
                    state = state.to(device=ref_device, dtype=ref_dtype)
                # set_state clones and validates the tensor via the cell's
                # state-shape contract.
                module.set_state(state)

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
            If True, also save current reservoir states. Only reservoirs whose
            state is non-``None`` are persisted (see
            :meth:`get_reservoir_states`); reservoirs that are still
            lazily-uninitialised at save time — e.g. a model saved before any
            warmup — are omitted from the checkpoint. :meth:`load` restores the
            saved states after resetting current recurrent state, so omitted
            reservoirs resume from an uninitialized state.
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
        save_checkpoint(self, path, include_states, metadata)

    def load(
        self,
        path: str | Path,
        strict: bool = True,
        load_states: bool = False,
        map_location: Any = None,
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
            If True, also load reservoir states if available. States are
            restored tolerantly (``strict=False``): because
            :meth:`save` only persists reservoirs whose state is non-``None``,
            a checkpoint may legitimately omit some reservoir keys (e.g. one
            saved before warmup). Current states are reset first; reservoirs
            absent from the checkpoint start uninitialized.
        map_location : optional
            Passed to ``torch.load``. Use ``"cpu"`` to load a checkpoint
            written on a GPU on a machine without that GPU.

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
        load_checkpoint(self, path, strict, load_states, map_location)

    @classmethod
    def load_from_file(
        cls,
        path: str | Path,
        model: "ESNModel | None" = None,
        strict: bool = True,
        load_states: bool = False,
        map_location: Any = None,
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
        map_location : optional
            Storage-device remapping forwarded to :meth:`load`.

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

        model.load(path, strict=strict, load_states=load_states, map_location=map_location)
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
        save_full_model(self, path, metadata)

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
        return load_full_model(cls, path, return_metadata, map_location)

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
        node_to_name: dict[Any, str] = getattr(self, "_node_to_layer_name", {})

        def _is_jupyter() -> bool:
            try:
                return bool(get_ipython().__class__.__name__ == "ZMQInteractiveShell")  # type: ignore[name-defined]
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
            try:
                import graphviz
            except ImportError as exc:
                # graphviz is an optional dependency; the Python package may be
                # absent. Re-raise with an actionable install hint (the outer
                # handler turns any ImportError into the printable DOT fallback).
                raise ImportError(
                    "graphviz Python package not installed; install it with "
                    "`pip install resdag[viz]` or `pip install graphviz` "
                    "(the system binary is also required: e.g. apt install graphviz)."
                ) from exc

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

        except ImportError as exc:
            return _print_dot_fallback(
                f"{exc} pip install resdag[viz] (or pip install graphviz) "
                f"&& apt install graphviz, or render the DOT source below."
            )

    def warmup(
        self,
        *inputs: torch.Tensor,
        return_outputs: bool = False,
        reset: bool = True,
        no_grad: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
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
        no_grad : bool, default=True
            If True (default), the warmup pass runs under :func:`torch.no_grad`
            — no autograd graph is built and behaviour/performance are identical
            to the historical ``@torch.no_grad()`` decorator.  Set ``False`` to
            keep the graph so the teacher-forced pass is part of a wider
            backpropagation (e.g. BPTT over the warmup→forecast seam).

            For gradients to flow *across* the warmup→forecast boundary the
            reservoir must also keep its carried state attached: set
            ``reservoir.detach_state_between_calls = False`` (it defaults to
            ``True``, which detaches the stored state at every forward call —
            see :class:`~resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer`).
            Note that ``no_grad=False`` retains the full warmup graph in memory,
            so cost grows with the warmup length.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor or None
            If ``return_outputs=True``: the model output of shape
            ``(batch, timesteps, output_dim)`` for a single-output model, or a
            tuple of such tensors for a multi-output model.
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

        with _grad_context(no_grad):
            output = self(*inputs)

        return output if return_outputs else None

    def _get_flat_step(self) -> FlatStep:
        """
        Return the cached flattened single-step executor, building it on demand.

        The flattened step (see :mod:`resdag.core._flat_inference`) compiles the
        symbolic graph into a straight-line per-step function that drives the
        autoregressive :meth:`forecast` loop without re-walking the graph each
        step.  It is cached and reused; the cache is keyed on the identity of
        ``self.outputs`` so it is transparently rebuilt if the graph's outputs
        change (e.g. via :meth:`add_output`).
        """
        cache: tuple[tuple[SymbolicData, ...], FlatStep] | None = getattr(
            self, "_flat_step_cache", None
        )
        if cache is not None and cache[0] is self.outputs:
            return cache[1]
        flat = build_flat_step(self)
        self._flat_step_cache: tuple[tuple[SymbolicData, ...], FlatStep] = (self.outputs, flat)
        return flat

    def forecast(
        self,
        warmup_inputs: tuple[torch.Tensor, ...] | torch.Tensor,
        forecast_inputs: tuple[torch.Tensor, ...] | torch.Tensor | None = None,
        *,
        horizon: int,
        initial_feedback: torch.Tensor | None = None,
        return_warmup: bool = False,
        reset: bool = True,
        compile: bool = False,
        no_grad: bool = True,
        alignment: ForecastAlignment = "next",
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Predict the next ``horizon`` samples after an observed history.

        Training pairs input ``x[t]`` with target ``x[t+1]``. Accordingly, the
        final teacher-forced warmup output is already the prediction of the
        first unseen sample. It becomes forecast slot zero; the remaining
        ``horizon - 1`` slots feed predictions back into the model.

        Parameters
        ----------
        warmup_inputs : torch.Tensor or tuple of torch.Tensor
            Observed history, shaped ``(batch, time, features)``. For a driven
            model pass ``(feedback, driver1, ...)`` with matching batch/time axes.
            Empty history is allowed only with ``initial_feedback``; with
            ``reset=False`` this continues saved reservoir states.
        forecast_inputs : torch.Tensor or tuple of torch.Tensor, optional
            Future drivers beginning immediately after the warmup drivers.
            Driver ``d[t]`` pairs with input feedback ``x[t]`` to predict
            ``x[t+1]``. The default needs ``horizon - 1`` driver samples because
            warmup already supplied the driver for the first prediction.
            ``horizon=1`` needs no future drivers. Explicit ``initial_feedback``
            and ``alignment="legacy"`` need ``horizon`` future driver samples.
            Extra trailing driver samples are ignored.
        horizon : int
            Number of predictions, a positive integer. Zero is rejected; use a
            normal forward/warmup call when no future predictions are needed.
        initial_feedback : torch.Tensor, optional
            A distinct additional observed input of shape ``(batch, 1, features)``
            consumed *after* the complete warmup. Slot zero predicts the sample
            after this seed. Do not pass the last warmup observation again: that
            would consume it twice. This also seeds an empty warmup continuation.
        return_warmup : bool, default=False
            Prepend historical predictions. With default alignment, the last
            warmup prediction is already forecast slot zero, so the full result
            has ``warmup_steps - 1 + horizon`` samples with no duplicate seam.
            With an explicit seed or legacy alignment it has
            ``warmup_steps + horizon`` samples.
        reset : bool, default=True
            Reset reservoir states before warmup. False continues current states.
        compile : bool, default=False
            Compile the flat autoregressive executor when supported, falling
            back to eager execution with a warning if compilation fails.
            Incompatible with ``no_grad=False``.
        no_grad : bool, default=True
            Disable autograd for inference. False makes the warmup and rollout
            differentiable. For gradients through carried warmup state, set
            each reservoir's ``detach_state_between_calls=False``. Memory grows
            with history and horizon length.
        alignment : {"next", "legacy"}, default="next"
            ``"next"`` predicts the immediate unseen sample. ``"legacy"``
            reproduces the old low-level convention, which consumed the final
            warmup prediction before returning slot zero and therefore skipped
            the immediate unseen sample. Pair it with
            ``prepare_esn_data(..., alignment="legacy")`` to reproduce old runs.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            ``(batch, horizon, output_features)`` per output head, or the longer
            sequence described above when ``return_warmup=True``. The first
            output head supplies autoregressive feedback and must match the
            feedback input width.

        Notes
        -----
        Returned predictions have not yet been consumed as inputs. After a
        forecast, the stored state has consumed the history and all predictions
        except the final one. To continue without replaying history, pass empty
        warmup streams, ``initial_feedback=prediction[:, -1:]`` and ``reset=False``
        (plus the corresponding future drivers).

        Autoregressive steps use a cached flat execution plan. Module forward
        hooks run during warmup but do not fire for these flat steps.

        Examples
        --------
        >>> predictions = model.forecast(observed, horizon=100)
        >>> # predictions[:, 0] estimates the sample immediately after observed.
        """
        self._get_execution_plan()
        warmup_inputs, drivers, horizon, boundary = prepare_forecast_inputs(
            warmup_inputs,
            forecast_inputs,
            num_inputs=len(self.inputs),
            horizon=horizon,
            initial_feedback=initial_feedback,
            alignment=alignment,
        )
        if compile and not no_grad:
            raise ValueError(
                "forecast(compile=True) is incompatible with no_grad=False: the "
                "reduce-overhead/cudagraph step path does not support autograd. "
                "Use no_grad=False with compile=False for a differentiable rollout."
            )

        output_shape = self.output_shape
        multi_output = isinstance(output_shape, tuple) and isinstance(output_shape[0], torch.Size)
        feedback_dim = warmup_inputs[0].shape[-1]
        feedback_output_dim = output_shape[0][-1] if multi_output else output_shape[-1]
        if feedback_output_dim != feedback_dim:
            raise ValueError(
                "forecast(): feedback dimension mismatch. "
                f"Feedback input has {feedback_dim} features but the model output "
                f"used as feedback has {feedback_output_dim} features. "
                "For autoregressive forecasting the first output must match the feedback input dimension."
            )

        warmup_outputs = self.warmup(
            *warmup_inputs, return_outputs=True, reset=reset, no_grad=no_grad
        )
        assert warmup_outputs is not None
        outputs = warmup_outputs if isinstance(warmup_outputs, tuple) else (warmup_outputs,)
        assert all(isinstance(out, torch.Tensor) for out in outputs)
        with _grad_context(no_grad):
            seed = initial_feedback if initial_feedback is not None else outputs[0][:, -1:, :]
            steps = horizon - int(boundary)
            if steps:
                flat = self._get_flat_step()
                states: list[torch.Tensor] = []
                for reservoir in flat.reservoir_layers:
                    if reservoir.state is None:
                        reservoir._maybe_init_state(
                            warmup_inputs[0].shape[0], seed.device, seed.dtype
                        )
                    assert reservoir.state is not None
                    states.append(reservoir.state)
                forecast_buffers, states = rollout(
                    flat,
                    seed,
                    drivers,
                    states,
                    steps,
                    [out.shape[-1] for out in outputs],
                    no_grad=no_grad,
                    compile=compile,
                    output_templates=outputs,
                )
                for reservoir, state in zip(flat.reservoir_layers, states):
                    if reservoir.detach_state_between_calls and state.grad_fn is not None:
                        reservoir.state = state.detach().clone()
                    else:
                        reservoir.state = state.clone()
            else:
                forecast_buffers = [out[:, :0, :] for out in outputs]

            predictions = []
            for warm, generated in zip(outputs, forecast_buffers):
                if boundary:
                    generated = torch.cat((warm[:, -1:, :], generated), dim=1)
                if return_warmup:
                    prefix = warm[:, :-1, :] if boundary else warm
                    generated = torch.cat((prefix, generated), dim=1)
                predictions.append(generated)
            return tuple(predictions) if multi_output else predictions[0]

    def windowed_forecast(
        self,
        series: torch.Tensor,
        *driver_series: torch.Tensor,
        predict_len: int,
        teacher_len: int,
        warmup_len: int | None = None,
        reset: bool = True,
        return_mask: bool = False,
        no_grad: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Gap-filling reconstruction by alternating re-sync and free-running.

        Reconstruct a long trajectory that is only *observed* in periodic
        windows: the reservoir is repeatedly re-synchronized on a short window
        of real data (teacher forcing) and then left to forecast autonomously
        across the unobserved gap.  Because the Echo State Property gives the
        reservoir fading memory, every teacher-forced window pulls its state
        back onto the true trajectory, no matter how far the previous free run
        had drifted — so per-window error does not compound across windows
        (accuracy *within* a gap still depends on keeping ``predict_len`` short
        relative to the system's predictability horizon).

        The reservoir state is carried across the whole pass (it is reset only
        once, at the start, and only when ``reset=True``).  Each cycle is one
        :meth:`forecast` call: its warmup phase is the re-sync window and its
        autoregressive phase is the gap.

        Timeline (``W = warmup_len``, ``F = teacher_len``, ``P = predict_len``)::

            [== warmup W ==][~~ gap P ~~][= teacher F =][~~ gap P ~~][= F =]...
              observed         filled       observed       filled      obs
            (teacher-forced)  (forecast)  (teacher-forced) (forecast)

        Observed segments are copied verbatim from ``series``; gap segments are
        replaced by the model's autonomous forecast.  Any trailing steps that
        cannot host a full re-sync window plus at least one forecast step are
        left as observed.

        Parameters
        ----------
        series : torch.Tensor
            Ground-truth feedback series of shape ``(batch, T, feedback_dim)``.
            It is both the source of every teacher-forced window and the value
            returned on observed segments.
        *driver_series : torch.Tensor
            Exogenous driver series, one per driver input of the model, each of
            shape ``(batch, T, driver_dim)`` spanning the full timeline.  Omit
            for feedback-only models.  Unlike :meth:`forecast` (which takes
            *separate* warmup and ``forecast_inputs`` driver tuples), here each
            driver covers the whole ``T``-step timeline and is sliced internally
            per cycle.
        predict_len : int, keyword-only
            Length ``P`` of each autonomous forecast window (the gap to fill).
            Must be ``>= 1``.
        teacher_len : int, keyword-only
            Length ``F`` of each teacher-forced re-synchronization window.
            Must be ``>= 1``.  Longer windows re-sync more reliably; shorter
            windows observe less of the signal.
        warmup_len : int, optional
            Length ``W`` of the initial teacher-forced window.  Defaults to
            ``teacher_len``.
        reset : bool, default=True
            If True, reservoir states are reset before the first window.  Set
            False to continue from the current (already warmed) state.
        return_mask : bool, default=False
            If True, also return a 1-D boolean mask of length ``T`` that is
            True where the reconstruction is a real value and False where it is
            a model forecast (see Returns).
        no_grad : bool, default=True
            If True (default), every cycle runs under :func:`torch.no_grad` —
            graph-free and identical to the historical ``@torch.no_grad()``
            decorator.  Set ``False`` to make the gap-fill reconstruction
            **differentiable**: the value threaded into the reconstruction on
            every forecast segment carries an autograd graph, so a loss over the
            gaps (``reconstruction[:, ~mask]``) can be back-propagated through the
            BPTT unroll of each window into the trainable reservoir and readout
            parameters.  The flag is threaded into the per-cycle :meth:`forecast`
            calls; the same memory cost and
            ``reservoir.detach_state_between_calls`` caveats documented on
            :meth:`forecast` apply, multiplied across cycles (state is carried
            across windows, so detaching it severs gradients window-to-window).

        Returns
        -------
        reconstruction : torch.Tensor
            Reconstructed series of shape ``(batch, T, feedback_dim)``: real
            values on observed segments, forecasts on the gaps.
        mask : torch.Tensor
            Only if ``return_mask=True``.  Boolean tensor of shape ``(T,)``,
            True where the step is a real value (a teacher-forced window, or the
            untouched trailing remainder copied verbatim from ``series``) and
            False where it is a model forecast.

        Raises
        ------
        ValueError
            If ``series`` is not 3-D, if ``predict_len``/``teacher_len``/
            ``warmup_len`` are not positive, if the number of ``driver_series``
            does not match the model's drivers, if any driver does not span
            ``T`` steps, or if ``series`` is too short for even one cycle.

        Notes
        -----
        - The forecast steps to score are exactly ``reconstruction[:, ~mask]``
          against ``series[:, ~mask]`` — the values the model had to infer
          without ever seeing them.
        - For multi-output models the first output is fed back (as in
          :meth:`forecast`) and only that channel is reconstructed.

        Examples
        --------
        Reconstruct a Lorenz trajectory observed 40 steps out of every 240,
        filling 200-step gaps:

        >>> recon, mask = model.windowed_forecast(
        ...     series,                 # (1, T, 3) ground truth
        ...     predict_len=200,
        ...     teacher_len=40,
        ...     return_mask=True,
        ... )
        >>> gap_rmse = ((recon[:, ~mask] - series[:, ~mask]) ** 2).mean().sqrt()

        With an exogenous driver:

        >>> recon = model.windowed_forecast(
        ...     feedback_series, driver_series,
        ...     predict_len=100, teacher_len=50,
        ... )

        See Also
        --------
        forecast : A single warmup + autoregressive forecast cycle.
        warmup : Teacher-forced state synchronization only.
        """
        if series.dim() != 3:
            raise ValueError(
                f"series must be 3-D (batch, T, feedback_dim); got shape {tuple(series.shape)}."
            )
        predict_len = positive_step_count(predict_len, "predict_len")
        teacher_len = positive_step_count(teacher_len, "teacher_len")
        if warmup_len is None:
            warmup_len = teacher_len
        warmup_len = positive_step_count(warmup_len, "warmup_len")

        num_drivers = len(driver_series)
        expected_drivers = len(self.inputs) - 1
        if num_drivers != expected_drivers:
            raise ValueError(
                f"model has {expected_drivers} driver input(s) but {num_drivers} "
                f"driver_series were given. Pass one full-timeline driver series per "
                f"driver input (the feedback is supplied by `series`)."
            )

        _, total_steps, _ = series.shape
        for i, driver in enumerate(driver_series):
            if driver.dim() != 3:
                raise ValueError(
                    f"driver_series[{i}] must be 3-D (batch, T, driver_dim); "
                    f"got shape {tuple(driver.shape)}."
                )
            if driver.shape[1] != total_steps:
                raise ValueError(
                    f"driver_series[{i}] spans {driver.shape[1]} steps but series spans "
                    f"{total_steps}; drivers must cover the full timeline."
                )

        # ``no_grad`` is also threaded into each per-cycle ``forecast`` below; the
        # context here additionally covers the in-place reconstruction writes so
        # that, when differentiable, the forecast segments graft their autograd
        # graph onto the (cloned) reconstruction via ``CopySlices``.
        reconstruction = series.clone()
        mask = torch.ones(total_steps, dtype=torch.bool, device=series.device)

        pos = 0
        window_len = warmup_len
        ran_a_cycle = False
        with _grad_context(no_grad):
            while True:
                teacher_end = pos + window_len
                # Stop once a full re-sync window leaves no room to forecast. The
                # guard guarantees ``teacher_end < total_steps`` below, so the gap
                # (clamped to whatever remains) is always at least one step.
                if teacher_end >= total_steps:
                    break
                gap = min(predict_len, total_steps - teacher_end)

                teacher_window = series[:, pos:teacher_end, :]
                if num_drivers:
                    warmup_inputs: tuple[torch.Tensor, ...] = (
                        teacher_window,
                        *(d[:, pos:teacher_end, :] for d in driver_series),
                    )
                    forecast_inputs: tuple[torch.Tensor, ...] | None = tuple(
                        d[:, teacher_end : teacher_end + gap, :] for d in driver_series
                    )
                else:
                    warmup_inputs = (teacher_window,)
                    forecast_inputs = None

                # Generate one additional prediction so the final gap sample is
                # consumed too. Its value is discarded, but this advances state
                # through the entire gap before the next observed window.
                full = self.forecast(
                    warmup_inputs,
                    forecast_inputs=forecast_inputs,
                    horizon=gap + 1,
                    reset=reset and not ran_a_cycle,
                    no_grad=no_grad,
                )
                feedback_full = full[0] if isinstance(full, tuple) else full
                gap_preds = feedback_full[:, :gap, :]
                reconstruction[:, teacher_end : teacher_end + gap, :] = gap_preds
                mask[teacher_end : teacher_end + gap] = False

                ran_a_cycle = True
                pos = teacher_end + gap
                window_len = teacher_len

        if not ran_a_cycle:
            raise ValueError(
                f"series length ({total_steps}) is too short for even one cycle: the "
                f"initial warmup window (warmup_len={warmup_len}) leaves no room to "
                f"forecast (need at least warmup_len + 1 = {warmup_len + 1} steps; the "
                f"final gap is clamped to whatever remains)."
            )

        if return_mask:
            return reconstruction, mask
        return reconstruction


__all__ = ["Input", "ESNModel", "ReservoirModelModule"]


class ReservoirModelModule(torch.nn.Module):
    """A standard module preserving a trained model's full DAG and parameters.

    Construct with ``model.as_module()``. The source model is registered as a
    child module, so ``parameters()``, ``state_dict()``, ``train()/eval()`` and
    ``to(device, dtype)`` behave like ordinary PyTorch composition. Parameters
    are shared with the source, not copied.
    """

    def __init__(self, model: torch.nn.Module, *, stateful: bool = False) -> None:
        super().__init__()
        if not all(
            callable(getattr(model, name, None))
            for name in ("forward_stateless", "reset_reservoirs", "get_state", "set_state")
        ):
            raise TypeError("model must implement reservoir model execution and state methods")
        self.model = model
        self.stateful = stateful

    @property
    def _model_api(self) -> ReservoirModelProtocol:
        return cast(ReservoirModelProtocol, self.model)

    def forward(self, *inputs: torch.Tensor) -> ModelOutput:
        """Predict independent sequences from each reservoir's configured reset state."""
        if self.stateful:
            output: ModelOutput = self.model(*inputs)
            return output
        output, _ = self._model_api.forward_stateless(*inputs)
        return output

    def forward_stateless(
        self,
        *inputs: torch.Tensor,
        states: Mapping[str, torch.Tensor] | None = None,
        compile: bool = False,
    ) -> tuple[ModelOutput, ReservoirStates]:
        """Predict with explicitly supplied and returned reservoir states."""
        return self._model_api.forward_stateless(*inputs, states=states, compile=compile)

    def reset_state(self) -> None:
        """Reset the source model's stored states for stateful streaming."""
        self._model_api.reset_reservoirs()

    def get_state(self) -> ReservoirStates:
        """Return copies of the wrapped model's initialized streaming states."""
        return self._model_api.get_state()

    def set_state(self, states: Mapping[str, torch.Tensor], strict: bool = True) -> None:
        """Restore the wrapped model's streaming states."""
        self._model_api.set_state(states, strict=strict)

    def extra_repr(self) -> str:
        return f"stateful={self.stateful}"
