"""Internal, checkpointable feature normalization for the model lifecycle.

This module does not select where preprocessing belongs in a public workflow.
It only packages the existing normalization mathematics as a PyTorch module.
Statistics are fitted explicitly, detached from autograd and registered as
buffers. Transforms never refit or update them.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from resdag.data.prepare import NormMethod, _compute_stats, denormalize_data, normalize_data

_METHODS: tuple[NormMethod, ...] = ("minmax", "standard", "noncentered", "meanpreserving")
_STAT_KEYS: dict[NormMethod, tuple[str, ...]] = {
    "minmax": ("min", "range"),
    "standard": ("mean", "std"),
    "noncentered": ("scale",),
    "meanpreserving": ("mean", "maxdev"),
}


class FittedNormalization(nn.Module):
    """A fitted, differentiable wrapper around the existing normalizers.

    `fit` accepts `(time, features)` or `(batch, time, features)` floating
    tensors and adopts their device and dtype. It is the caller's responsibility
    to pass only the training portion. `forward` and `inverse` preserve the
    input shape, dtype and device without changing the fitted buffers. Ordinary
    module conversions move/cast the buffers; as with other PyTorch modules,
    load into the desired dtype to control checkpoint conversion precision.

    Finite-value checks happen during fitting/loading, not every inference
    step. Inference otherwise follows ordinary Torch NaN/Inf propagation.
    """

    _stat0: torch.Tensor
    _stat1: torch.Tensor
    _method_code: torch.Tensor

    def __init__(self, method: NormMethod = "standard") -> None:
        super().__init__()
        if method not in _METHODS:
            raise ValueError(f"Unknown normalization method {method!r}; expected one of {_METHODS}")
        self._method_name = method
        self.register_buffer("_method_code", torch.tensor(_METHODS.index(method), dtype=torch.long))
        self.register_buffer("_stat0", torch.empty(1, 1, 0))
        self.register_buffer("_stat1", torch.empty(1, 1, 0))

    @property
    def method(self) -> NormMethod:
        """The normalization rule restored with the checkpoint's statistics."""
        return self._method_name

    @property
    def is_fitted(self) -> bool:
        """Whether statistics are available, without reading a device scalar."""
        return self._stat0.shape[-1] > 0

    @property
    def n_features(self) -> int | None:
        """Fitted feature width, or `None` before fitting."""
        return int(self._stat0.shape[-1]) if self.is_fitted else None

    @staticmethod
    def _canonical(data: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if not isinstance(data, torch.Tensor):
            raise TypeError("Normalization requires a torch.Tensor")
        if not data.is_floating_point():
            raise TypeError(f"Normalization requires a real floating tensor, got {data.dtype}")
        if data.ndim not in (2, 3) or any(size == 0 for size in data.shape):
            raise ValueError(
                "Normalization requires non-empty (time, features) or "
                f"(batch, time, features) data, got {tuple(data.shape)}"
            )
        return (data.unsqueeze(0), True) if data.ndim == 2 else (data, False)

    def fit(self, training: torch.Tensor) -> FittedNormalization:
        """Replace statistics with those of the supplied training tensor only.

        Validation completes before the old fitted statistics are replaced.
        This also keeps a fit performed inside `inference_mode` usable later
        in a differentiable pipeline.
        """
        data, _ = self._canonical(training)
        if data.device.type == "meta":
            raise ValueError("Fitting normalization requires concrete data, not a meta tensor")
        if not bool(torch.isfinite(data).all()):
            raise ValueError("Normalization training data must contain only finite values")
        with torch.inference_mode(False), torch.no_grad():
            stats = _compute_stats(data.detach(), self.method)
            if not all(bool(torch.isfinite(value).all()) for value in stats.values()):
                raise ValueError("Normalization statistics overflowed; use a higher training dtype")
            values = [stats[key].detach().clone() for key in _STAT_KEYS[self.method]]
            second = values[1] if len(values) == 2 else values[0].new_empty(1, 1, 0)
            code = torch.tensor(_METHODS.index(self.method), dtype=torch.long, device=data.device)
        self._stat0, self._stat1, self._method_code = values[0], second, code
        return self

    def _prepared(self, data: torch.Tensor) -> tuple[torch.Tensor, bool, dict[str, torch.Tensor]]:
        if not self.is_fitted:
            raise RuntimeError("Normalization is not fitted; call fit(training_data) first")
        canonical, was_2d = self._canonical(data)
        if canonical.shape[-1] != self.n_features:
            raise ValueError(
                f"Normalization was fitted with {self.n_features} features, "
                f"got {canonical.shape[-1]}"
            )
        values = (self._stat0, self._stat1)
        stats = {key: values[index] for index, key in enumerate(_STAT_KEYS[self.method])}
        return canonical, was_2d, stats

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize with fixed training statistics, preserving upstream gradients."""
        canonical, was_2d, stats = self._prepared(data)
        transformed, _ = normalize_data(canonical, method=self.method, stats=stats)
        return transformed.squeeze(0) if was_2d else transformed

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        """Return to original units without changing fitted statistics."""
        canonical, was_2d, stats = self._prepared(data)
        restored = denormalize_data(canonical, method=self.method, stats=stats)
        return restored.squeeze(0) if was_2d else restored

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        names = ("_method_code", "_stat0", "_stat1")
        present = [prefix + name in state_dict for name in names]
        if any(present) and not all(present):
            error_msgs.append(f"{prefix}normalization checkpoint has incomplete statistics")
            return
        if all(present):
            try:
                code, first, second = (state_dict[prefix + name] for name in names)
                if (
                    not isinstance(code, torch.Tensor)
                    or code.ndim != 0
                    or code.dtype != torch.long
                    or code.device.type == "meta"
                ):
                    raise ValueError("invalid method identifier")
                index = int(code.item())
                if not 0 <= index < len(_METHODS):
                    raise ValueError("unknown normalization method identifier")
                method = _METHODS[index]
                for value in (first, second):
                    if (
                        not isinstance(value, torch.Tensor)
                        or not value.is_floating_point()
                        or value.ndim != 3
                        or value.shape[:2] != (1, 1)
                        or value.device.type == "meta"
                        or not bool(torch.isfinite(value).all())
                    ):
                        raise ValueError(
                            "statistics must be finite floating (1, 1, features) tensors"
                        )
                width = first.shape[-1]
                second_width = width if len(_STAT_KEYS[method]) == 2 else 0
                if second.shape[-1] != second_width:
                    raise ValueError("statistic widths do not match the normalization method")
                scale = first if method == "noncentered" else second
                if width and not bool((scale > 0).all()):
                    raise ValueError("normalization scales must be positive")
                # Validate the values that will actually be installed. A valid
                # float64 scale can become zero or infinity in a float32 module.
                # Stage everything before replacing any existing fitted state.
                with torch.inference_mode(False), torch.no_grad():
                    assign = local_metadata.get("assign_to_params_buffers", False)
                    staged = []
                    for source, destination in zip(
                        (code, first, second), (self._method_code, self._stat0, self._stat1)
                    ):
                        if assign:
                            # Assignment preserves checkpoint dtype/device. Only
                            # inference/gradient tensors and Parameters need a
                            # detached buffer rather than an autograd dependency.
                            value = (
                                source.detach().clone()
                                if source.is_inference()
                                or source.requires_grad
                                or isinstance(source, nn.Parameter)
                                else source
                            )
                        else:
                            value = source.to(
                                dtype=destination.dtype, device=destination.device, copy=True
                            )
                        staged.append(value)
                    installed_scale = staged[1] if method == "noncentered" else staged[2]
                    if not all(bool(torch.isfinite(value).all()) for value in staged[1:]) or (
                        width and not bool((installed_scale > 0).all())
                    ):
                        raise ValueError(
                            "statistics cannot be represented in the destination dtype; "
                            "load into the checkpoint's dtype (for example, call .double() "
                            "before loading a float64 checkpoint) or use assign=True"
                        )
                    materialized = [value.new_empty(value.shape) for value in staged]
            except (ValueError, TypeError, RuntimeError) as error:
                error_msgs.append(f"{prefix}invalid normalization checkpoint: {error}")
                return
            # Materialization outside the caller's inference mode leaves future
            # upstream autograd usable. Let Module perform its ordinary copy or
            # assignment using the already validated dtype/device conversion.
            self._method_code, self._stat0, self._stat1 = materialized
            self._method_name = method
            state_dict = dict(state_dict)
            state_dict.update({prefix + name: value for name, value in zip(names, staged)})
        with torch.inference_mode(False):
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

    def extra_repr(self) -> str:
        return f"method={self.method!r}, n_features={self.n_features}"
