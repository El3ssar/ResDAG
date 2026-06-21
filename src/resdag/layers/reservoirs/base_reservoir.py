"""
Base Reservoir Class
======================

This module provides abstract base class for the layer ESN reservoir:

- :class:`BaseReservoirLayer` — abstract sequence loop with full state-management API.

See Also
--------
resdag.layers.reservoir : Concrete implementations.
"""

from abc import ABC
from itertools import chain

import torch
import torch.nn as nn

from resdag.layers.cells import ReservoirCell


class BaseReservoirLayer(nn.Module, ABC):
    """
    Abstract base that owns the sequence loop and all state-management methods.

    Subclasses create a :class:`ReservoirCell` and pass it to this
    constructor.  The sequence loop (iterating over timesteps and calling the
    cell) lives here; the cell handles the per-step computation.

    Parameters
    ----------
    cell : ReservoirCell
        Concrete cell instance that performs the single-step update.

    Attributes
    ----------
    cell : ReservoirCell
        The wrapped single-step cell.
    state : torch.Tensor or None
        Current reservoir state of shape ``(batch, cell.state_size)``, or
        ``None`` if not yet initialized.
    detach_state_between_calls : bool
        If ``True`` (default), the stored state is detached from the autograd
        graph at the end of each forward call (truncated BPTT at call
        boundaries).  Gradients still flow through the *returned* states
        within a call — this only severs gradient flow *across* calls, which
        otherwise raises "backward through the graph a second time" when a
        model is trained with SGD over consecutive batches without resetting
        the reservoir.  Set to ``False`` only if you intentionally backprop
        through state carried across multiple forward calls (and manage the
        retained graphs yourself).

    See Also
    --------
    resdag.layers.esn.ESNLayer : Concrete ESN layer built on this base.
    resdag.layers.base.ReservoirCell : Abstract cell interface.
    """

    def __init__(self, cell: ReservoirCell) -> None:
        super().__init__()
        self.cell = cell
        # Register the reservoir state as a *non-persistent* buffer so that
        # ``nn.Module._apply`` (driven by ``.to()`` / ``.cuda()`` / ``.double()``)
        # moves a warmed-up state along with the module's parameters, instead of
        # leaving it on the original device/dtype and triggering a silent
        # zero-reinit on the next forward.  ``persistent=False`` keeps it out of
        # ``state_dict()`` so the ``save`` / ``include_states`` split is
        # untouched.  The buffer is initialised to ``None`` and reassigned with a
        # concrete tensor on first use (buffers accept reassignment, including
        # back to ``None`` for lazy re-init).
        self.register_buffer("state", None, persistent=False)
        self.state: torch.Tensor | None
        self.detach_state_between_calls: bool = True
        # Distinguishes a state the user restored on purpose (via ``set_state``)
        # from one that merely accumulated through forward passes or lazy
        # zero-init.  Only the former is protected against a silent batch-size
        # re-init in ``_maybe_init_state`` — the latter is free to auto-resize.
        self._state_user_set: bool = False

    def forward_stateless(
        self,
        inputs: list[torch.Tensor],
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the time loop purely: thread ``state`` in and out, touch no ``self`` state.

        This is the functional core of the reservoir.  It takes the initial
        state as an explicit argument, threads it through the per-timestep
        loop, and returns the per-step outputs together with the final state.
        It never reads or writes :attr:`state`, performs no in-place tensor
        writes (per-step outputs are collected with :func:`torch.stack`), and
        the loop body contains no data-dependent Python branch.  Those three
        properties are what make it ``torch.compile``-scan, :func:`torch.func.vmap`,
        and TorchScript/ONNX-export friendly, unlike the stateful
        :meth:`forward` wrapper that drives it.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            Feedback tensor first, then at most one driving-input tensor, each
            of shape ``(batch, timesteps, features)``.
        state : torch.Tensor
            Initial reservoir state, shape ``(batch, ...)`` as produced by
            ``self.cell.init_state``.

        Returns
        -------
        outputs : torch.Tensor
            Per-step outputs, shape ``(batch, timesteps, cell.output_size)``.
        new_state : torch.Tensor
            Final reservoir state after the last timestep.

        Notes
        -----
        The cross-call detach applied by :meth:`forward` (truncated BPTT) lives
        in that wrapper, *outside* this method — gradients flow through the
        returned ``new_state`` here, so callers that backpropagate through state
        carried across calls are free to do so.
        """
        feedback = inputs[0]
        seq_len = feedback.shape[1]

        # Fast path: cells that split their update into input-dependent and
        # state-dependent parts precompute the former for the whole sequence
        # in one batched matmul, leaving only the recurrent term in the loop.
        # Choosing the path is a per-cell static decision made once, outside
        # the loop, so the loop body itself stays branch-free.
        projected = self.cell.project_inputs(inputs)

        outputs: list[torch.Tensor] = []
        if projected is not None:
            for t in range(seq_len):
                output, state = self.cell.step(projected[:, t, :], state)
                outputs.append(output)
        else:
            for t in range(seq_len):
                inputs_t = [stream[:, t, :] for stream in inputs]
                output, state = self.cell(inputs_t, state)
                outputs.append(output)

        if not outputs:
            # Degenerate empty sequence (T == 0): torch.stack rejects an empty
            # list, so return the (batch, 0, output_size) tensor the previous
            # preallocating loop produced, with the input ``state`` untouched.
            empty = feedback.new_empty(feedback.shape[0], 0, self.cell.output_size)
            return empty, state

        return torch.stack(outputs, dim=1), state

    def step_stateless(
        self,
        inputs: list[torch.Tensor],
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single-timestep analogue of :meth:`forward_stateless`: state in, state out.

        Where :meth:`forward_stateless` consumes whole
        ``(batch, timesteps, features)`` sequences and runs the time loop, this
        consumes a single ``(batch, features)`` slice per stream and performs
        exactly one cell update.  It is the per-step primitive that the
        flattened autoregressive engine
        (:meth:`resdag.core.ESNModel.forecast`) drives, sidestepping the
        sequence-loop bookkeeping (``torch.stack``, ``shape[1]`` reads) that
        :meth:`forward` / :meth:`forward_stateless` pay even for a length-1
        sequence.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            Feedback slice first, then at most one driving-input slice, each of
            shape ``(batch, features)``.
        state : torch.Tensor
            Current reservoir state, shape ``(batch, ...)`` as produced by
            ``self.cell.init_state``.

        Returns
        -------
        output : torch.Tensor
            Per-step output, shape ``(batch, cell.output_size)``.
        new_state : torch.Tensor
            Updated reservoir state.

        Notes
        -----
        The project-vs-fallback choice is the same per-cell static decision made
        in :meth:`forward_stateless` (no data-dependent branch in the hot path),
        so this method stays ``torch.compile``-fullgraph friendly.  Like
        :meth:`forward_stateless` it never reads or writes :attr:`state`, and it
        applies no cross-call detach — the forecast engine runs under
        :func:`torch.no_grad`, so the threaded state never carries a graph.
        """
        projected = self.cell.project_inputs(inputs)
        if projected is not None:
            return self.cell.step(projected, state)
        # ``nn.Module.__call__`` is typed ``Any``; unpack then return a tuple
        # literal so the declared ``tuple[Tensor, Tensor]`` is honoured (mirrors
        # ``forward_stateless``).
        output, new_state = self.cell(inputs, state)
        return output, new_state

    def forward(
        self,
        feedback: torch.Tensor,
        *driving_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process an input sequence through the reservoir.

        Computes reservoir states for each timestep using the feedback
        signal and optional driving inputs.  This is a thin stateful wrapper
        over the pure :meth:`forward_stateless`: it validates inputs, lazily
        initialises :attr:`state`, runs the functional loop, stores the
        returned state, and applies the cross-call detach.

        Parameters
        ----------
        feedback : torch.Tensor
            Feedback signal of shape ``(batch, timesteps, feedback_size)``.
        *driving_inputs : torch.Tensor
            Optional driving input of shape ``(batch, timesteps, input_size)``.
            At most one driving input tensor is supported.

        Returns
        -------
        torch.Tensor
            Reservoir states for all timesteps, shape
            ``(batch, timesteps, cell.output_size)``.

        Raises
        ------
        ValueError
            If ``feedback`` is not 3-D, if more than one driving input is
            supplied, or if the driving input batch/sequence dimensions do not
            match ``feedback``.

        Notes
        -----
        The layer maintains internal state across forward calls.  Use
        :meth:`reset_state` to clear the state between independent sequences.
        For a pure, ``self``-free variant (used by the compile-scan, vmap, and
        export paths) call :meth:`forward_stateless` directly.
        """
        if feedback.dim() != 3:
            raise ValueError(f"Feedback must be 3D (B, T, F), got shape {feedback.shape}")

        batch_size, seq_len, _ = feedback.shape

        if len(driving_inputs) > 0:
            if len(driving_inputs) > 1:
                raise ValueError("Only one driving input tensor allowed")
            driving_input = driving_inputs[0]
            if driving_input.shape[0] != batch_size or driving_input.shape[1] != seq_len:
                raise ValueError(
                    f"Driving input must match feedback dimensions. "
                    f"Feedback: {feedback.shape}, Driving: {driving_input.shape}"
                )

        self._maybe_init_state(batch_size, feedback.device, feedback.dtype)
        assert self.state is not None  # guaranteed by _maybe_init_state; narrows for mypy

        outputs, new_state = self.forward_stateless([feedback, *driving_inputs], self.state)

        # Truncated BPTT at call boundaries: gradients flow through the returned
        # states within this call, but the *stored* state must not keep the
        # graph alive — a later forward + backward would otherwise try to
        # backprop through this call's already-freed graph.  This data-dependent
        # ``grad_fn`` branch lives here, in the wrapper, never inside the pure
        # loop of ``forward_stateless``.
        if self.detach_state_between_calls and new_state.grad_fn is not None:
            new_state = new_state.detach()

        self.state = new_state
        # The stored state has now evolved one or more steps past whatever the
        # user restored; it is no longer the as-set tensor, so drop the
        # user-set protection — a subsequent batch-size change is back to the
        # ordinary auto-resize regime.
        self._state_user_set = False
        return outputs

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _maybe_init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """
        Lazily allocate a zero state only when it is genuinely missing.

        A re-init happens **only** when the state is ``None`` or its batch size
        no longer matches the incoming batch.  Device and dtype are *not* a
        trigger: the state is a registered buffer, so ``.to()`` / ``.cuda()`` /
        ``.double()`` already move it together with the module's parameters, and
        re-zeroing on a device/dtype mismatch would silently discard a warmed-up
        trajectory — the very bug this guard used to cause.

        When the state was restored **on purpose** via :meth:`set_state`, a
        batch-size mismatch is treated as a user error and raises rather than
        silently zero-reinitialising: discarding a deliberately restored state
        would otherwise produce plausible-looking but wrong forecasts (e.g. the
        common train-with-``batch>1`` / forecast-with-``batch=1`` pattern).  A
        state that merely accumulated through forward passes or lazy zero-init
        still auto-resizes without complaint.

        Parameters
        ----------
        batch_size : int
            Batch size of the incoming sequence.
        device : torch.device
            Target device for a freshly allocated state.
        dtype : torch.dtype
            Target dtype for a freshly allocated state.

        Raises
        ------
        RuntimeError
            If a state restored via :meth:`set_state` has a batch size that does
            not match the incoming ``batch_size``.  Re-restore at the right batch
            size or call :meth:`reset_state` to opt back into auto-resizing.
        """
        if self.state is None or self.state.shape[0] != batch_size:
            if self._state_user_set and self.state is not None:
                set_batch = self.state.shape[0]
                raise RuntimeError(
                    f"{type(self).__name__}: a state restored via set_state has "
                    f"batch size {set_batch}, but this forward pass uses batch "
                    f"size {batch_size}. Silently zero-reinitialising would "
                    f"discard the state you restored on purpose. Restore a state "
                    f"with batch size {batch_size}, or call reset_state() to opt "
                    f"back into automatic batch-size re-initialisation."
                )
            self.state = self.cell.init_state(batch_size, device, dtype)
            # A fresh zero state is a lazy/natural allocation, never a user-set
            # one: clear the flag so the new state is free to auto-resize next.
            self._state_user_set = False

    def reference_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        """
        Resolve the canonical device and dtype of this reservoir's weights.

        The reference is the first floating-point parameter or buffer of the
        inner cell (the same scan used when lazily allocating a zero state),
        falling back to ``cpu`` / ``float32`` for weightless cells such as the
        NG-RC reservoir.  This is the device/dtype an incoming forward pass is
        expected to use, so callers restoring a saved state should coerce it to
        this reference to avoid a silent re-initialisation in
        :meth:`_maybe_init_state`.

        Returns
        -------
        device : torch.device
            Device of the cell's first floating-point tensor, or ``cpu``.
        dtype : torch.dtype
            Dtype of the cell's first floating-point tensor, or ``float32``.
        """
        ref = next(
            (
                t
                for t in chain(self.cell.parameters(), self.cell.buffers())
                if t.is_floating_point()
            ),
            None,
        )
        device = ref.device if ref is not None else torch.device("cpu")
        dtype = ref.dtype if ref is not None else torch.float32
        return device, dtype

    def reset_state(self, batch_size: int | None = None) -> None:
        """
        Reset internal state to zero.

        Parameters
        ----------
        batch_size : int, optional
            If provided, initialize state with this batch size using the
            cell's device and dtype.  If ``None``, state is set to ``None``
            and will be lazily initialized on the next forward pass.

        Examples
        --------
        >>> layer.reset_state()          # Lazy initialization
        >>> layer.reset_state(batch_size=4)  # Explicit zero state
        """
        if batch_size is not None:
            if self.state is not None:
                device, dtype = self.state.device, self.state.dtype
            else:
                device, dtype = self.reference_device_dtype()
            self.state = self.cell.init_state(batch_size, device, dtype)
        else:
            self.state = None
        # A reset (zeros or back-to-None) is a deliberate opt-in to automatic
        # batch-size re-initialisation: drop any user-set protection.
        self._state_user_set = False

    def get_state(self) -> torch.Tensor | None:
        """
        Get a copy of the current internal state.

        Returns
        -------
        torch.Tensor or None
            Clone of the current state tensor of shape
            ``(batch, cell.state_size)``, or ``None`` if not yet initialized.

        Examples
        --------
        >>> state = layer.get_state()
        >>> if state is not None:
        ...     print(f"State shape: {state.shape}")
        """
        return self.state.clone() if self.state is not None else None

    def set_state(self, state: torch.Tensor) -> None:
        """
        Set the internal state to a specific tensor.

        Validation is delegated to ``self.cell.validate_state(state)`` so
        each cell type owns its own state-shape contract (2-D for ESNCell,
        3-D delay buffer for NGCell, etc.).

        Batch-size contract
        -------------------
        The restored state's batch size (``state.shape[0]``) pins the batch size
        the **next forward pass must use**.  A restored state is treated as
        deliberate: if a subsequent ``forward`` is called with a different batch
        size, :meth:`_maybe_init_state` raises a :class:`RuntimeError` instead of
        silently zero-reinitialising and discarding the state you restored.  To
        forecast at a different batch size, restore (or :meth:`reset_state` and
        re-warm) a state of the matching batch size; to opt back into automatic
        batch-size re-initialisation, call :meth:`reset_state`.  Device and dtype
        are not pinned — the state moves with the module under ``.to()`` and is
        cast to match an incoming forward.

        Autograd contract
        -----------------
        The stored state honours :attr:`detach_state_between_calls`, exactly as
        the cross-call truncated-BPTT detach in :meth:`forward` does.  Under the
        default (``True``) a restored state is treated as a fresh initial
        condition: it is stored as ``state.detach().clone()`` so it carries no
        ``grad_fn`` and a later forward + backward cannot try to backprop
        through ``state``'s already-freed graph (the "backward through the graph
        a second time" / retained-memory failure that
        :attr:`detach_state_between_calls` exists to prevent).  Set
        ``detach_state_between_calls=False`` to store ``state.clone()`` with its
        graph intact, when you intentionally backprop through state carried in
        across calls (and manage the retained graph yourself).

        Parameters
        ----------
        state : torch.Tensor
            New state tensor.  Its batch size (``shape[0]``) becomes the
            required batch size of the next forward pass.

        Raises
        ------
        ValueError
            If the cell's :meth:`ReservoirCell.validate_state` rejects the
            tensor.  The error includes the cell class name and the
            offending shape.

        See Also
        --------
        reset_state : Clear the state and opt back into automatic batch-size
            re-initialisation.

        Examples
        --------
        >>> saved = layer.get_state()      # captured at batch size B
        >>> # ... process data ...
        >>> layer.set_state(saved)         # next forward must also use batch B
        """
        try:
            self.cell.validate_state(state)
        except ValueError as exc:
            # Re-raise with a richer message including a recovery hint.
            raise ValueError(
                f"{type(self).__name__}.set_state failed: {exc}. "
                f"Tip: call reset_state(batch_size=...) before set_state() "
                f"to materialise an empty state of the expected layout."
            ) from None
        # Honour the truncated-BPTT contract on restore, just as ``forward``
        # does for the evolved state (see lines around the cross-call detach
        # there): a restored state is a fresh initial condition by default, so
        # under ``detach_state_between_calls`` strip any incoming ``grad_fn``
        # before storing it — otherwise restoring a graph-bearing tensor leaks
        # an autograd graph across calls (#146).  ``clone`` always runs so the
        # stored state is an independent copy regardless of the detach choice.
        if self.detach_state_between_calls and state.grad_fn is not None:
            self.state = state.detach().clone()
        else:
            self.state = state.clone()
        # Mark this as a deliberate restore: a later forward at a mismatched
        # batch size must raise rather than silently zero-reinit (see #145).
        self._state_user_set = True

    def set_random_state(
        self,
        batch_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Set the internal state to random (standard-normal) values.

        Parameters
        ----------
        batch_size : int, optional
            If provided, lazily initialise the state with this batch size
            before filling it with random values.  Uses ``device`` / ``dtype``
            kwargs when given, otherwise the cell's first floating-point
            parameter/buffer (falling back to CPU/float32).
        device : torch.device, optional
            Target device when lazily initialising.  Ignored if the state is
            already initialised.
        dtype : torch.dtype, optional
            Target dtype when lazily initialising.  Ignored if the state is
            already initialised.

        Raises
        ------
        RuntimeError
            If the state has not been initialised and ``batch_size`` is not
            provided.

        Examples
        --------
        >>> # Lazy: initialise then randomise in one call
        >>> layer.set_random_state(batch_size=4)
        >>> # Already-initialised state: just randomise in place
        >>> layer.set_random_state()
        """
        if self.state is None:
            if batch_size is None:
                raise RuntimeError(
                    "Reservoir state is not initialised. "
                    "Pass batch_size= to initialise it before randomising."
                )
            self.reset_state(batch_size=batch_size)
            # ``reset_state(batch_size=...)`` always assigns a real tensor here
            # (batch_size is non-None past the guard above); narrows for mypy.
            assert self.state is not None
            # ``reset_state`` honours device/dtype of existing state or cell;
            # apply overrides if explicitly requested.
            if device is not None or dtype is not None:
                self.state = self.state.to(device=device, dtype=dtype)
        # State is non-None on both branches: either just initialised above or
        # already non-None at method entry.
        assert self.state is not None
        self.state = torch.randn_like(self.state)
