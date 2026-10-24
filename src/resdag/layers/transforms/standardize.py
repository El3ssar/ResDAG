"""Per-feature standardization transform with buffer-stored statistics.

Provides `Standardize`, an `nn.Module` that centers and scales every
feature to zero mean and unit standard deviation. The per-feature `mean` and
`std` are stored as registered buffers, so they travel with `.to(device)`,
`.double()`, `state_dict`/`load_state_dict`, and the
`resdag.core.ESNModel.save_full`/`load_full` round trip — letting
normalization live inside the model graph rather than ad-hoc user code.
"""

import math

import torch
import torch.nn as nn


class Standardize(nn.Module):
    r"""Per-feature standardization (z-score) with a learnable-free inverse.

    Centers and scales each feature of the last dimension using stored
    statistics, applying `(x - mean) / std` in `forward` and the exact
    inverse `x * std + mean` in `inverse`. The `mean` and `std`
    vectors are registered buffers of shape `(num_features,)`, so they are
    part of the `state_dict` and move with `.to()`/dtype casts and the
    `resdag.core.ESNModel.save_full` round trip.

    Statistics may be supplied at construction or estimated from data with
    `fit`. Until either happens the layer initializes to `mean = 0` and
    `std = 1`. Supplied floating-point tensors preserve their precision and
    device; both buffers use their common promoted dtype. A missing statistic
    is created on the supplied tensor's device in the same dtype. Lists and
    integer tensors use the default floating dtype. Two tensor statistics on
    different devices are rejected rather than moved implicitly.

    A finite, nonnegative `eps` is added to `std` before division to avoid
    blow-up on constant (zero-variance) features; the same padded scale is used by
    `inverse`, so `inverse(forward(x))` reconstructs `x` to within
    floating-point tolerance when the padded scale is nonzero. Setting
    `eps=0` disables stabilization. In float16, a positive `eps` is floored
    at the dtype's smallest normal value, keeping the reciprocal scale
    representable for constant features. Both methods are available in TorchScript.

    Parameters
    ----------
    num_features : int
        Size of the feature (last) dimension this layer standardizes.
    mean : torch.Tensor, optional
        Initial per-feature mean of shape `(num_features,)`. Defaults to
        zeros (no centering until `fit` is called).
    std : torch.Tensor, optional
        Finite, nonnegative initial standard deviation of shape `(num_features,)`.
        Defaults to ones (no scaling until `fit` is called).
    eps : float, default=1e-8
        Numerical-stability floor added to `std` before dividing. Guards
        against division by zero on constant features.

    Attributes
    ----------
    mean : torch.Tensor
        Registered buffer holding the per-feature mean `(num_features,)`.
    std : torch.Tensor
        Registered buffer holding the per-feature standard deviation
        `(num_features,)`.

    Examples
    --------
    Fit statistics from a batch, then standardize and invert:

    >>> import torch
    >>> from resdag.layers.transforms import Standardize
    >>>
    >>> layer = Standardize(num_features=3)
    >>> x = torch.randn(8, 100, 3) * 5.0 + 2.0  # (batch, time, features)
    >>> layer.fit(x)
    Standardize(num_features=3, eps=1e-08)
    >>> z = layer(x)  # ~zero mean, ~unit std per feature
    >>> torch.allclose(layer.inverse(z), x, atol=1e-5)
    True

    Inside a composable pipeline (statistics travel with `save_full`/`.to`):

    >>> import pytorch_symbolic as ps
    >>> from resdag import ESNModel, ESNLayer, CGReadoutLayer, Standardize
    >>>
    >>> norm = Standardize(num_features=3)
    >>> norm.fit(torch.randn(4, 200, 3))
    Standardize(num_features=3, eps=1e-08)
    >>> inp = ps.Input((200, 3))
    >>> normed = norm(inp)
    >>> reservoir = ESNLayer(100, feedback_size=3)(normed)
    >>> readout = CGReadoutLayer(100, 3, name="output")(reservoir)
    >>> model = ESNModel(inp, readout)

    See Also
    --------
    Power : Per-feature exponentiation.
    SelectiveExponentiation : Parity-selective exponentiation.
    """

    # Class-level annotations so static checkers see the buffers as tensors
    # (``register_buffer`` otherwise widens them to ``Tensor | Module``).
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        num_features: int,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        eps: float = 1e-8,
    ) -> None:
        """Register the per-feature mean/std buffers and the `eps` floor.

        Parameters
        ----------
        num_features : int
            Size of the feature dimension to standardize.
        mean : torch.Tensor, optional
            Initial per-feature mean of shape `(num_features,)` (default
            zeros).
        std : torch.Tensor, optional
            Initial per-feature standard deviation of shape
            `(num_features,)` (default ones).
        eps : float, default=1e-8
            Stability floor added to `std` before dividing.

        Raises
        ------
        ValueError
            If `num_features` is not positive, or if a supplied `mean`/
            `std` has the wrong shape, nonfinite values, or conflicting
            devices; also if `std` or `eps` is negative, or `eps` is
            nonfinite.
        TypeError
            If a supplied statistic is complex or `eps` is not a real number.
        """
        super().__init__()

        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        try:
            finite_eps = math.isfinite(eps)
        except TypeError as error:
            raise TypeError("eps must be a finite, nonnegative real number") from error
        if not finite_eps or eps < 0:
            raise ValueError("eps must be finite and nonnegative")
        if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
            if mean.device != std.device:
                raise ValueError("mean and std must be on the same device; move them explicitly")

        self.num_features = num_features
        self.eps = float(eps)

        # Constructor inputs, including inference tensors, must become ordinary
        # detached buffers so a later differentiable forward can save them.
        with torch.inference_mode(False), torch.no_grad():
            mean_stat = self._validate_stat(mean, "mean")
            std_stat = self._validate_stat(std, "std")
            provided = [stat for stat in (mean_stat, std_stat) if stat is not None]
            dtype = provided[0].dtype if provided else torch.get_default_dtype()
            for stat in provided[1:]:
                dtype = torch.promote_types(dtype, stat.dtype)
            tensor_stats = [stat for stat in (mean, std) if isinstance(stat, torch.Tensor)]
            device = tensor_stats[0].device if tensor_stats else torch.get_default_device()
            mean_buffer = (
                torch.zeros(num_features, dtype=dtype, device=device)
                if mean_stat is None
                else mean_stat.to(device=device, dtype=dtype).detach().clone()
            )
            std_buffer = (
                torch.ones(num_features, dtype=dtype, device=device)
                if std_stat is None
                else std_stat.to(device=device, dtype=dtype).detach().clone()
            )

        self.register_buffer("mean", mean_buffer)
        self.register_buffer("std", std_buffer)

    def _validate_stat(self, value: torch.Tensor | None, name: str) -> torch.Tensor | None:
        """Validate a real statistic without narrowing floating tensor inputs.

        Parameters
        ----------
        value : torch.Tensor or None
            Candidate statistic. `None` leaves the statistic unspecified.
        name : str
            Statistic name, used in error messages.

        Returns
        -------
        torch.Tensor or None
            A floating 1-D tensor of length `num_features`, or `None` for a
            missing statistic. Lists and integer tensors use the default dtype.

        Raises
        ------
        ValueError
            If *value* has the wrong shape, contains nonfinite values, is a
            meta tensor, or is a negative standard deviation.
        TypeError
            If *value* is complex.
        """
        if value is None:
            return None

        # Non-tensor inputs have no placement contract. Materialize them on CPU
        # before validation; the constructor places both final buffers together.
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device="cpu")
        if tensor.is_complex():
            raise TypeError(f"{name} must contain real values")
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be 1D, but got shape {tuple(tensor.shape)}")
        if tensor.shape[0] != self.num_features:
            raise ValueError(
                f"{name} length ({tensor.shape[0]}) does not match "
                f"num_features ({self.num_features})"
            )
        if tensor.device.type == "meta":
            raise ValueError(f"{name} must contain concrete values, not a meta tensor")
        if not isinstance(value, torch.Tensor) and tensor.is_floating_point():
            # Validate Python/list floats before default-dtype rounding can
            # erase a tiny negative standard deviation or overflow a value.
            tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu")
        if name == "std" and bool((tensor < 0).any()):
            raise ValueError("std must be nonnegative")
        if not isinstance(value, torch.Tensor) or not tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.get_default_dtype())
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain finite values")
        return tensor

    @torch.no_grad()
    def fit(self, input: torch.Tensor) -> "Standardize":
        """Estimate per-feature mean and std from data and store them.

        Statistics are reduced over every dimension except the last, so an
        input of shape `(batch, time, features)` (or any number of leading
        dims) yields per-feature vectors of length `num_features`. The
        population standard deviation (`unbiased=False`) is used so a single
        observation does not produce a `nan`. The fitted buffers inherit the
        device/dtype of *input*. Validation and statistics calculation finish
        before either previous buffer is replaced. Fitting inside
        `torch.inference_mode` still permits later differentiable calls.

        Parameters
        ----------
        input : torch.Tensor
            Data of shape `(..., num_features)` whose last dimension matches
            `num_features`.

        Returns
        -------
        Standardize
            `self`, so calls can be chained (e.g. `layer.fit(x)(x)`).

        Raises
        ------
        ValueError
            If *input* is empty, has the wrong feature width, contains
            nonfinite values, or overflows while computing statistics.
        TypeError
            If *input* is not a real floating-point tensor.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("Standardize.fit requires a torch.Tensor")
        if not input.is_floating_point():
            raise TypeError("Standardize.fit requires a real floating-point tensor")
        if input.ndim == 0 or input.numel() == 0:
            raise ValueError("Standardize.fit requires non-empty data with a feature dimension")
        if input.shape[-1] != self.num_features:
            raise ValueError(
                f"input feature dim ({input.shape[-1]}) does not match "
                f"num_features ({self.num_features})"
            )

        if input.device.type == "meta":
            raise ValueError("Standardize.fit requires concrete data, not a meta tensor")
        if not bool(torch.isfinite(input).all()):
            raise ValueError("Standardize.fit requires finite training values")
        with torch.inference_mode(False), torch.no_grad():
            flat = input.detach().reshape(-1, self.num_features)
            mean = flat.mean(dim=0)
            std = flat.std(dim=0, unbiased=False)
            if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
                raise ValueError("Standardize statistics overflowed; use a higher training dtype")
            mean, std = mean.detach().clone(), std.detach().clone()
        self.mean, self.std = mean, std
        return self

    def _scale(self) -> torch.Tensor:
        """Keep a positive float16 stability floor representable with finite slope."""
        eps = self.eps
        if self.std.dtype == torch.float16 and eps > 0:
            # IEEE float16's smallest normal value; torch.finfo is not scriptable.
            eps = max(eps, 2.0**-14)
        return self.std + eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Standardize the last dimension as `(x - mean) / (std + eps)`.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of shape `(..., num_features)`.

        Returns
        -------
        torch.Tensor
            Same shape as *input*, centered and scaled per feature. Gradients
            propagate through the input using the constant stored statistics.

        Raises
        ------
        ValueError
            If the last dimension of *input* does not equal `num_features`.
        """
        if input.shape[-1] != self.num_features:
            raise ValueError(
                f"input feature dim ({input.shape[-1]}) does not match "
                f"num_features ({self.num_features})"
            )
        return (input - self.mean) / self._scale()

    @torch.jit.export
    def inverse(self, input: torch.Tensor) -> torch.Tensor:
        """Invert the transform as `x * (std + eps) + mean`.

        The exact algebraic inverse of `forward`, using the same
        `eps`-padded scale so that `inverse(forward(x)) == x` up to
        floating-point error. Use it to map standardized model outputs (e.g.
        forecasts) back to the original data scale.

        Parameters
        ----------
        input : torch.Tensor
            Standardized tensor of shape `(..., num_features)`.

        Returns
        -------
        torch.Tensor
            Same shape as *input*, mapped back to the original feature scale.

        Raises
        ------
        ValueError
            If the last dimension of *input* does not equal `num_features`.
        """
        if input.shape[-1] != self.num_features:
            raise ValueError(
                f"input feature dim ({input.shape[-1]}) does not match "
                f"num_features ({self.num_features})"
            )
        return input * self._scale() + self.mean

    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        return f"num_features={self.num_features}, eps={self.eps}"
