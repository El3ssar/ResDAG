"""
Reservoir Feature Extractor
===========================

This module provides :class:`ReservoirFeatureExtractor`, an ``nn.Module``
adapter that packages one or more reservoir layers as a drop-in feature
extractor for ordinary ``torch.nn`` pipelines.

The "reservoir as a frozen feature extractor feeding a trainable head"
pattern is the cleanest expression of the torch-pipeline philosophy: a fixed
random recurrent map turns a sequence into a rich, high-dimensional feature
stream, and any ``torch.nn`` head learns to read it out with a single
optimizer.  :class:`ReservoirFeatureExtractor` makes that pattern a one-liner
that composes directly inside :class:`torch.nn.Sequential`.

See Also
--------
resdag.layers.reservoirs.ESNLayer : Underlying stateful reservoir layer.
resdag.core.ESNModel : Symbolic-graph model the extractor can be built from.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch
import torch.nn as nn

from resdag.layers.reservoirs import BaseReservoirLayer, ESNLayer


class ReservoirFeatureExtractor(nn.Module):
    """
    ``nn.Sequential``-friendly reservoir feature extractor.

    Wraps one reservoir layer (or a stack of them) as a plain
    :class:`torch.nn.Module` that maps a feedback sequence
    ``(batch, timesteps, feedback_size)`` (plus an optional driving input) to
    reservoir features ``(batch, timesteps, reservoir_size)``.  Because the
    feedback-only case takes a single positional input, the extractor drops
    straight into a :class:`torch.nn.Sequential` ahead of any trainable head::

        model = nn.Sequential(
            ReservoirFeatureExtractor(reservoir_size=300, feedback_size=3),
            nn.Linear(300, 3),
        )

    The reservoir is **frozen by default** (its parameters do not require
    gradients), so a single optimizer over ``model.parameters()`` trains only
    the head — the canonical reservoir-computing setup.  Pass ``trainable=True``
    (or call :meth:`unfreeze`) to backpropagate through the recurrence as well.

    Stacking is supported: passing several feedback dimensions, or constructing
    the extractor from pre-built layers, chains the reservoirs so each consumes
    the previous reservoir's features as its feedback signal.  A driving input,
    when supplied, is passed to the **first** reservoir only.

    Parameters
    ----------
    reservoir_size : int or sequence of int, optional
        Reservoir width.  An ``int`` builds a single reservoir; a sequence of
        ``int`` builds a stack (one reservoir per entry, in order).  Required
        unless ``layers`` is given.
    feedback_size : int, optional
        Dimension of the feedback signal entering the **first** reservoir.
        Required unless ``layers`` is given.  For stacked reservoirs only the
        first layer's feedback size is taken from this argument; the feedback
        size of each subsequent layer is the previous layer's reservoir size.
    input_size : int, optional
        Dimension of an optional driving input fed to the first reservoir.  If
        ``None`` (default), the extractor operates in the feedback-only,
        single-positional-input mode required by :class:`torch.nn.Sequential`.
    trainable : bool, default=False
        If ``True``, the reservoir parameters require gradients (full BPTT
        through the recurrence).  The default ``False`` freezes them, the
        standard reservoir-computing configuration.
    layers : ESNLayer or sequence of BaseReservoirLayer, optional
        Pre-built reservoir layer(s) to wrap directly instead of constructing
        new ones.  When given, ``reservoir_size`` / ``feedback_size`` /
        ``input_size`` must be omitted (the layers already define them), but
        ``trainable`` still applies as a freeze/unfreeze toggle over the
        provided layers.  The layers are wrapped **by reference** — their
        parameters are shared, not copied.
    **layer_kwargs
        Extra keyword arguments forwarded to every :class:`ESNLayer` that this
        constructor builds (for example ``spectral_radius``, ``leak_rate``,
        ``activation``, ``topology``, ``seed``).  Ignored when ``layers`` is
        provided.

    Attributes
    ----------
    reservoirs : torch.nn.ModuleList
        The wrapped reservoir layer(s), in feed order.
    output_size : int
        Feature dimension of the extractor's output (the last reservoir's
        ``reservoir_size``).

    Raises
    ------
    ValueError
        If neither ``layers`` nor ``(reservoir_size, feedback_size)`` is given,
        if both forms are mixed, or if ``layers`` is empty.

    Examples
    --------
    Drop straight into a :class:`torch.nn.Sequential` and train the head with a
    single optimizer:

    >>> import torch
    >>> import torch.nn as nn
    >>> from resdag import ReservoirFeatureExtractor
    >>> model = nn.Sequential(
    ...     ReservoirFeatureExtractor(reservoir_size=64, feedback_size=3),
    ...     nn.Linear(64, 3),
    ... )
    >>> x = torch.randn(2, 50, 3)  # (batch, time, features)
    >>> y = model(x)
    >>> y.shape
    torch.Size([2, 50, 3])

    The reservoir is frozen by default, so only the head receives gradients:

    >>> extractor = model[0]
    >>> all(not p.requires_grad for p in extractor.parameters())
    True

    Re-zero the stateful reservoir between epochs:

    >>> for epoch in range(3):
    ...     extractor.on_epoch_start()  # alias of reset_state()
    ...     # ... train one epoch ...

    Reuse the reservoir of an existing :class:`~resdag.core.ESNModel` (shared,
    not copied):

    >>> from resdag import ESNModel, ESNLayer, reservoir_input
    >>> inp = reservoir_input(3)
    >>> states = ESNLayer(64, feedback_size=3)(inp)
    >>> esn = ESNModel(inp, states)
    >>> extractor = ReservoirFeatureExtractor.from_model(esn)

    See Also
    --------
    resdag.layers.reservoirs.ESNLayer : The reservoir layer being wrapped.
    resdag.core.ESNModel : Build an extractor from an existing model.
    """

    def __init__(
        self,
        reservoir_size: int | Iterable[int] | None = None,
        feedback_size: int | None = None,
        input_size: int | None = None,
        trainable: bool = False,
        layers: ESNLayer | Iterable[BaseReservoirLayer] | None = None,
        **layer_kwargs: object,
    ) -> None:
        super().__init__()

        if layers is not None:
            if reservoir_size is not None or feedback_size is not None or input_size is not None:
                raise ValueError(
                    "Pass either pre-built `layers` or "
                    "`reservoir_size`/`feedback_size`/`input_size`, not both."
                )
            if layer_kwargs:
                raise ValueError(
                    "Extra layer keyword arguments are not supported when "
                    "wrapping pre-built `layers`."
                )
            built = self._wrap_layers(layers)
        else:
            built = self._build_layers(
                reservoir_size=reservoir_size,
                feedback_size=feedback_size,
                input_size=input_size,
                **layer_kwargs,
            )

        self.reservoirs: nn.ModuleList = nn.ModuleList(built)
        # Apply the freeze/unfreeze toggle uniformly over the wrapped layers.
        self.unfreeze() if trainable else self.freeze()

    @property
    def _layers(self) -> list[BaseReservoirLayer]:
        """Typed view of :attr:`reservoirs` (``nn.ModuleList`` erases the type)."""
        return [cast(BaseReservoirLayer, layer) for layer in self.reservoirs]

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_layers(
        layers: ESNLayer | Iterable[BaseReservoirLayer],
    ) -> list[BaseReservoirLayer]:
        """Validate and normalise pre-built layers into a non-empty list."""
        if isinstance(layers, BaseReservoirLayer):
            return [layers]
        wrapped = list(layers)
        if not wrapped:
            raise ValueError("`layers` must contain at least one reservoir layer.")
        for layer in wrapped:
            if not isinstance(layer, BaseReservoirLayer):
                raise ValueError(
                    f"Every entry of `layers` must be a BaseReservoirLayer, "
                    f"got {type(layer).__name__}."
                )
        return wrapped

    @staticmethod
    def _build_layers(
        reservoir_size: int | Iterable[int] | None,
        feedback_size: int | None,
        input_size: int | None,
        **layer_kwargs: object,
    ) -> list[BaseReservoirLayer]:
        """Construct a stack of :class:`ESNLayer` from sizes."""
        if reservoir_size is None or feedback_size is None:
            raise ValueError(
                "Provide `reservoir_size` and `feedback_size` (or pass pre-built "
                "`layers`) to construct a ReservoirFeatureExtractor."
            )
        sizes = [reservoir_size] if isinstance(reservoir_size, int) else list(reservoir_size)
        if not sizes:
            raise ValueError("`reservoir_size` sequence must be non-empty.")

        built: list[BaseReservoirLayer] = []
        in_dim = feedback_size
        for depth, size in enumerate(sizes):
            built.append(
                ESNLayer(
                    reservoir_size=size,
                    feedback_size=in_dim,
                    # The driving input is only meaningful for the first layer;
                    # deeper layers consume the previous reservoir's features.
                    input_size=input_size if depth == 0 else None,
                    **layer_kwargs,  # type: ignore[arg-type]
                )
            )
            in_dim = size
        return built

    @classmethod
    def from_model(
        cls,
        esn_model: nn.Module,
        trainable: bool = False,
    ) -> ReservoirFeatureExtractor:
        """
        Build an extractor that reuses the reservoir layers of an existing model.

        The reservoir layers of ``esn_model`` are wrapped **by reference** — the
        returned extractor shares their parameters (and stateful buffers) with
        the source model, it does not copy them.  Training (or freezing) one
        therefore affects the other.

        Parameters
        ----------
        esn_model : torch.nn.Module
            A model containing one or more
            :class:`~resdag.layers.reservoirs.BaseReservoirLayer` submodules
            (typically an :class:`~resdag.core.ESNModel`).  Reservoirs are
            collected in :func:`~torch.nn.Module.named_modules` order.
        trainable : bool, default=False
            Freeze/unfreeze toggle applied to the shared reservoir layers after
            wrapping.  Because the layers are shared, this also changes
            ``requires_grad`` on the source model's reservoir parameters.

        Returns
        -------
        ReservoirFeatureExtractor
            Extractor wrapping the model's reservoir layers by reference.

        Raises
        ------
        ValueError
            If ``esn_model`` contains no reservoir layers.

        Examples
        --------
        >>> from resdag import ESNModel, ESNLayer, reservoir_input
        >>> from resdag import ReservoirFeatureExtractor
        >>> inp = reservoir_input(3)
        >>> states = ESNLayer(64, feedback_size=3)(inp)
        >>> esn = ESNModel(inp, states)
        >>> extractor = ReservoirFeatureExtractor.from_model(esn)
        >>> src = next(m for m in esn.modules() if hasattr(m, "weight_hh"))
        >>> extractor.reservoirs[0].weight_hh is src.weight_hh
        True
        """
        reservoirs = [
            module
            for _, module in esn_model.named_modules()
            if isinstance(module, BaseReservoirLayer)
        ]
        if not reservoirs:
            raise ValueError(
                "`esn_model` contains no reservoir layers (BaseReservoirLayer) "
                "to reuse. Build the extractor from sizes instead."
            )
        return cls(layers=reservoirs, trainable=trainable)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        feedback: torch.Tensor,
        *driving_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map a feedback sequence to reservoir features.

        Parameters
        ----------
        feedback : torch.Tensor
            Feedback signal of shape ``(batch, timesteps, feedback_size)``.
        *driving_inputs : torch.Tensor
            Optional driving input of shape ``(batch, timesteps, input_size)``,
            passed to the **first** reservoir only.  At most one is supported.

        Returns
        -------
        torch.Tensor
            Reservoir features of shape ``(batch, timesteps, output_size)``.

        Raises
        ------
        ValueError
            If more than one driving input is supplied.

        Notes
        -----
        The wrapped reservoirs are stateful: state carries across forward calls
        until :meth:`reset_state` (or its :meth:`on_epoch_start` alias) is
        invoked.  When the extractor sits inside a :class:`torch.nn.Sequential`
        only ``feedback`` is passed, satisfying the single-positional-input
        contract.
        """
        if len(driving_inputs) > 1:
            raise ValueError("Only one driving input tensor allowed")

        # The driving input enters the first reservoir; deeper reservoirs
        # consume the previous layer's features as their (sole) feedback.
        layers = self._layers
        features: torch.Tensor = layers[0](feedback, *driving_inputs)
        for reservoir in layers[1:]:
            features = reservoir(features)
        return features

    # ------------------------------------------------------------------
    # State and grad management
    # ------------------------------------------------------------------

    def reset_state(self, batch_size: int | None = None) -> None:
        """
        Reset every wrapped reservoir's internal state.

        Parameters
        ----------
        batch_size : int, optional
            If given, materialise a zero state with this batch size on each
            reservoir's device/dtype.  If ``None`` (default), states are set to
            ``None`` and lazily re-initialised on the next forward pass.

        See Also
        --------
        on_epoch_start : Alias intended for use as an epoch-reset hook.
        """
        for reservoir in self._layers:
            reservoir.reset_state(batch_size=batch_size)

    def on_epoch_start(self, batch_size: int | None = None) -> None:
        """
        Epoch-reset hook: re-zero the stateful reservoir between epochs.

        A thin, intent-revealing alias of :meth:`reset_state` meant to be
        called at the top of each training epoch so a trajectory left over from
        the previous epoch never bleeds into the next one.

        Parameters
        ----------
        batch_size : int, optional
            Forwarded to :meth:`reset_state`.

        Examples
        --------
        >>> for epoch in range(num_epochs):  # doctest: +SKIP
        ...     extractor.on_epoch_start()
        ...     for batch in loader:
        ...         ...  # train the head
        """
        self.reset_state(batch_size=batch_size)

    def freeze(self) -> ReservoirFeatureExtractor:
        """
        Freeze the reservoir: set ``requires_grad=False`` on all its parameters.

        Returns
        -------
        ReservoirFeatureExtractor
            ``self``, to allow chaining.
        """
        for param in self.reservoirs.parameters():
            param.requires_grad_(False)
        return self

    def unfreeze(self) -> ReservoirFeatureExtractor:
        """
        Unfreeze the reservoir: set ``requires_grad=True`` on all its parameters.

        Returns
        -------
        ReservoirFeatureExtractor
            ``self``, to allow chaining.
        """
        for param in self.reservoirs.parameters():
            param.requires_grad_(True)
        return self

    @property
    def is_frozen(self) -> bool:
        """bool: ``True`` when no wrapped reservoir parameter requires gradients."""
        return all(not p.requires_grad for p in self.reservoirs.parameters())

    @property
    def output_size(self) -> int:
        """int: Feature dimension of the extractor output (last reservoir width)."""
        return int(self._layers[-1].cell.output_size)

    def extra_repr(self) -> str:
        """Return a one-line summary of the wrapped reservoir stack."""
        widths = ", ".join(str(int(r.cell.output_size)) for r in self._layers)
        return f"reservoirs=[{widths}], frozen={self.is_frozen}"
