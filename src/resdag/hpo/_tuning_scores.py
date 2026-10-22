"""Fixed-training-scale forecast scores for the common tuning workflow."""

import math
from dataclasses import dataclass
from typing import Literal

import torch

__all__ = ["ScoreResult", "score_forecast", "StreamingForecastScore"]


@dataclass(frozen=True)
class ScoreResult:
    """Accurate forecast steps and dimension-weighted normalized error.

    Larger ``score`` is better; smaller ``nrmse`` is better. ``head_horizons``
    contains each output's median contiguous accurate prefix across the batch.
    ``head_errors`` contains each output's RMS normalized error over its entire
    supplied forecast. The dictionaries use the model's output names.
    """

    score: float
    nrmse: float
    head_horizons: dict[str, float]
    head_errors: dict[str, float]


def _stable_rms(values: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    """Compute an RMS without squaring the original, potentially large values."""
    maximum = values.abs().amax(dim=dim, keepdim=True)
    safe_maximum = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    mean_square = (values / safe_maximum).square().mean(dim=dim, keepdim=True)
    result = mean_square.sqrt() * maximum
    if isinstance(dim, int):
        return result.squeeze(dim)
    for axis in sorted((axis % values.ndim for axis in dim), reverse=True):
        result = result.squeeze(axis)
    return result


@torch.no_grad()
def score_forecast(
    predictions: dict[str, torch.Tensor],
    truth: dict[str, torch.Tensor],
    scales: dict[str, torch.Tensor],
    *,
    threshold: float = 0.2,
    aggregation: Literal["weakest", "mean"] = "weakest",
) -> ScoreResult:
    """Score original-unit forecasts using fixed scales from admitted training.

    Each output must provide matching ``(time, features)`` or
    ``(batch, time, features)`` tensors. Outputs share the same batch/time
    layout and device, but may have different feature counts and dtypes.
    Scales are positive finite tensors on that device: a scalar, a feature
    vector, or a feature vector with singleton leading dimensions. They must
    come from training; this function never estimates a scale from validation
    data, the prediction, or a growing forecast prefix.

    At each sample, the error is the RMS over normalized feature errors. A
    batch item's horizon counts consecutive samples strictly below
    ``threshold`` from the start, stopping at the first failure. Each head's
    horizon is the conventional batch median, averaging both middle values for
    an even batch. ``aggregation='weakest'`` takes the minimum head horizon;
    ``'mean'`` takes their arithmetic mean. Scores are positive sample counts.

    Per-head NRMSE is the RMS over all supplied batch/time/feature values. The
    reported ``nrmse`` is the arithmetic mean of those per-head errors weighted
    by each head's feature count. It is a diagnostic alongside the horizon,
    rather than a second optimization objective.

    Reductions stay on the tensor device. Only scalar validation flags and
    final summaries reach the host. Half-precision inputs use float32 for
    scoring; other floating inputs retain their promoted working precision.
    Non-finite inputs, invalid scales and normalized errors outside that
    precision's finite range raise instead of publishing misleading scores.
    """
    if not all(isinstance(value, dict) for value in (predictions, truth, scales)):
        raise TypeError("predictions, truth and scales must be dictionaries keyed by output name")
    if not predictions:
        raise ValueError("At least one named output is required to score a forecast")
    if predictions.keys() != truth.keys() or predictions.keys() != scales.keys():
        raise ValueError("predictions, truth and scales must have exactly the same output names")
    if not all(isinstance(name, str) and name for name in predictions):
        raise ValueError("Output names must be nonempty strings")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    if aggregation not in ("weakest", "mean"):
        raise ValueError("aggregation must be 'weakest' or 'mean'")

    head_horizons = {}
    head_errors = {}
    widths = []
    common_layout = None
    common_device = None

    for name, predicted in predictions.items():
        target = truth[name]
        scale = scales[name]
        if not all(isinstance(value, torch.Tensor) for value in (predicted, target, scale)):
            raise TypeError(f"Output {name!r} predictions, truth and scales must be tensors")
        if predicted.ndim not in (2, 3) or predicted.shape != target.shape:
            raise ValueError(
                f"Output {name!r} needs matching (time, features) or "
                "(batch, time, features) predictions and truth"
            )
        if any(size == 0 for size in predicted.shape):
            raise ValueError(f"Output {name!r} cannot have an empty batch, time or feature axis")
        if not all(value.is_floating_point() for value in (predicted, target, scale)):
            raise TypeError(
                f"Output {name!r} predictions, truth and scales must be floating tensors"
            )
        if predicted.device != target.device or predicted.device != scale.device:
            raise ValueError(f"Output {name!r} predictions, truth and scales must share a device")
        layout = tuple(predicted.shape[:-1])
        if common_layout is not None and layout != common_layout:
            raise ValueError("All outputs must share the same batch/time layout")
        if common_device is not None and predicted.device != common_device:
            raise ValueError("All outputs must share the same device")
        common_layout = layout
        common_device = predicted.device

        width = predicted.shape[-1]
        if scale.ndim and (
            scale.ndim > 3
            or scale.shape[-1] != width
            or any(size != 1 for size in scale.shape[:-1])
        ):
            raise ValueError(
                f"Output {name!r} scales must be a scalar or a fixed feature vector "
                "with optional singleton leading axes"
            )
        valid = (
            torch.isfinite(predicted).all()
            & torch.isfinite(target).all()
            & torch.isfinite(scale).all()
            & (scale > 0).all()
        )
        if not bool(valid):
            raise ValueError(f"Output {name!r} requires finite data and positive finite scales")

        dtype = torch.promote_types(torch.promote_types(predicted.dtype, target.dtype), scale.dtype)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        predicted = predicted.to(dtype)
        target = target.to(dtype)
        scale = scale.to(dtype).reshape(-1)
        if predicted.ndim == 2:
            predicted = predicted.unsqueeze(0)
            target = target.unsqueeze(0)

        difference = predicted - target
        # Opposite large finite values can overflow before division by a large
        # training scale. Divide first only for those overflowing differences;
        # the ordinary path retains accuracy for close, large-valued signals.
        normalized = torch.where(
            torch.isfinite(difference),
            difference / scale,
            predicted / scale - target / scale,
        )
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError(
                f"Output {name!r} normalized errors exceed the working precision; "
                "rescale extreme input units"
            )
        errors = _stable_rms(normalized, dim=2)
        horizons = (errors < threshold).to(torch.int64).cumprod(dim=1).sum(dim=1)
        ordered = horizons.sort().values.to(dtype)
        middle = len(ordered) // 2
        median = (
            ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
        )
        head_horizons[name] = median
        head_errors[name] = _stable_rms(normalized, dim=(0, 1, 2))
        widths.append(width)

    horizons_tensor = torch.stack(list(head_horizons.values()))
    score = horizons_tensor.amin() if aggregation == "weakest" else horizons_tensor.mean()
    errors_tensor = torch.stack(list(head_errors.values()))
    weights = errors_tensor.new_tensor(widths) / sum(widths)
    maximum_error = errors_tensor.amax()
    safe_maximum = torch.where(maximum_error > 0, maximum_error, torch.ones_like(maximum_error))
    # A convex mean cannot exceed its largest term. Normalize first and cap
    # rounding at one so summation cannot overflow near the dtype maximum.
    relative_error = ((errors_tensor / safe_maximum) * weights).sum().clamp(max=1)
    nrmse = relative_error * maximum_error
    return ScoreResult(
        score=float(score.item()),
        nrmse=float(nrmse.item()),
        head_horizons={name: float(value.item()) for name, value in head_horizons.items()},
        head_errors={name: float(value.item()) for name, value in head_errors.items()},
    )


@dataclass
class _HeadScore:
    horizons: torch.Tensor
    alive: torch.Tensor
    maximum: torch.Tensor
    squares: torch.Tensor
    count: int
    layout: tuple[int, int, int]
    invalid: torch.Tensor


class StreamingForecastScore:
    """Bounded device-local score state for successive aligned forecast chunks.

    ``update`` performs tensor operations only: finite-value checks accumulate
    on-device flags. ``summary`` checks those flags and converts final scalars.
    Its NRMSE always describes exactly the prefix admitted so far, using the
    original fixed training scales. No predictions or truth rows are retained.
    """

    def __init__(
        self,
        scales: dict[str, torch.Tensor],
        *,
        threshold: float = 0.2,
        aggregation: Literal["weakest", "mean"] = "weakest",
    ) -> None:
        if not isinstance(scales, dict) or not scales:
            raise ValueError("At least one named output scale is required")
        if not all(isinstance(name, str) and name for name in scales):
            raise ValueError("Output names must be nonempty strings")
        if not all(isinstance(value, torch.Tensor) for value in scales.values()):
            raise TypeError("Scales must be tensors keyed by output name")
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be positive and finite")
        if aggregation not in ("weakest", "mean"):
            raise ValueError("aggregation must be 'weakest' or 'mean'")
        self.scales = {name: value.detach().clone() for name, value in scales.items()}
        self.threshold = threshold
        self.aggregation = aggregation
        self.steps = 0
        self._heads: dict[str, _HeadScore] = {}
        self._device: torch.device | None = None
        self._batch_layout: tuple[int, int] | None = None

    @torch.no_grad()
    def update(self, predictions: dict[str, torch.Tensor], truth: dict[str, torch.Tensor]) -> None:
        """Admit one nonempty chunk without synchronizing tensor values to CPU."""
        if not isinstance(predictions, dict) or not isinstance(truth, dict):
            raise TypeError("predictions and truth must be named dictionaries")
        if predictions.keys() != self.scales.keys() or truth.keys() != self.scales.keys():
            raise ValueError(
                "predictions, truth and scales must have exactly the same output names"
            )
        chunk_length = None
        for name, scale in self.scales.items():
            predicted, target = predictions[name], truth[name]
            if not all(isinstance(value, torch.Tensor) for value in (predicted, target)):
                raise TypeError("Predictions and truth must be tensors")
            if predicted.ndim not in (2, 3) or predicted.shape != target.shape:
                raise ValueError(
                    "Forecast chunks need matching 2-D/3-D prediction and truth layouts"
                )
            if any(size == 0 for size in predicted.shape):
                raise ValueError("Forecast chunks cannot have empty batch, time or feature axes")
            if not all(value.is_floating_point() for value in (predicted, target, scale)):
                raise TypeError("Predictions, truth and scales must be floating tensors")
            if predicted.device != target.device or predicted.device != scale.device:
                raise ValueError("Predictions, truth and scales must share a device")
            width = predicted.shape[-1]
            if scale.ndim and (
                scale.ndim > 3
                or scale.shape[-1] != width
                or any(size != 1 for size in scale.shape[:-1])
            ):
                raise ValueError("Scales must be a scalar or a fixed feature vector")
            batch = predicted.shape[0] if predicted.ndim == 3 else 1
            layout = (predicted.ndim, batch)
            if self._batch_layout is not None and layout != self._batch_layout:
                raise ValueError("All chunks and heads must share their original batch layout")
            if self._device is not None and predicted.device != self._device:
                raise ValueError("All forecast chunks must share a device")
            if chunk_length is not None and predicted.shape[-2] != chunk_length:
                raise ValueError("All heads must share chunk time axes")
            self._device, self._batch_layout = predicted.device, layout
            chunk_length = predicted.shape[-2]
            dtype = torch.promote_types(
                torch.promote_types(predicted.dtype, target.dtype), scale.dtype
            )
            if dtype in (torch.float16, torch.bfloat16):
                dtype = torch.float32
            predicted, target, scale = (
                predicted.to(dtype),
                target.to(dtype),
                scale.to(dtype).reshape(-1),
            )
            if predicted.ndim == 2:
                predicted, target = predicted.unsqueeze(0), target.unsqueeze(0)
            difference = predicted - target
            normalized = torch.where(
                torch.isfinite(difference), difference / scale, predicted / scale - target / scale
            )
            invalid = ~(
                torch.isfinite(predicted).all()
                & torch.isfinite(target).all()
                & torch.isfinite(scale).all()
                & (scale > 0).all()
                & torch.isfinite(normalized).all()
            )
            # Keep accumulators well-defined until summary raises the stored
            # error; replacement values never turn invalid data into a score.
            normalized = torch.where(
                torch.isfinite(normalized), normalized, torch.zeros_like(normalized)
            )
            if name not in self._heads:
                self._heads[name] = _HeadScore(
                    torch.zeros(batch, dtype=torch.int64, device=predicted.device),
                    torch.ones(batch, dtype=torch.bool, device=predicted.device),
                    predicted.new_zeros(()),
                    predicted.new_zeros(()),
                    0,
                    (*layout, width),
                    invalid,
                )
            head = self._heads[name]
            if head.layout != (*layout, width) or head.maximum.dtype != dtype:
                raise ValueError("Forecast chunk widths and working dtypes must remain unchanged")
            head.invalid = head.invalid | invalid
            good = _stable_rms(normalized, dim=2) < self.threshold
            contiguous = good.to(torch.int64).cumprod(dim=1)
            head.horizons = head.horizons + (contiguous * head.alive[:, None]).sum(dim=1)
            head.alive = head.alive & good.all(dim=1)
            maximum = torch.maximum(head.maximum, normalized.abs().amax())
            safe = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
            # Scaled sum of squares (the same idea as BLAS LASSQ) avoids
            # overflow even when finite normalized errors approach dtype max.
            head.squares = (
                head.squares * (head.maximum / safe).square() + (normalized / safe).square().sum()
            )
            head.maximum = maximum
            head.count += normalized.numel()
        assert chunk_length is not None
        self.steps += chunk_length

    @torch.no_grad()
    def summary(self) -> ScoreResult:
        """Return the current prefix's score; only summaries scalarize tensors."""
        if not self.steps:
            raise ValueError("At least one forecast chunk is required before scoring")
        head_horizons, head_errors, widths = {}, {}, []
        for name, head in self._heads.items():
            if bool(head.invalid):
                raise ValueError(f"Output {name!r} requires finite data and positive finite scales")
            ordered = head.horizons.sort().values.to(head.maximum.dtype)
            middle = len(ordered) // 2
            median = (
                ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
            )
            head_horizons[name] = median
            head_errors[name] = (head.squares / head.count).clamp(max=1).sqrt() * head.maximum
            widths.append(head.layout[-1])
        horizons = torch.stack(list(head_horizons.values()))
        score = horizons.amin() if self.aggregation == "weakest" else horizons.mean()
        errors = torch.stack(list(head_errors.values()))
        weights = errors.new_tensor(widths) / sum(widths)
        maximum_error = errors.amax()
        safe_maximum = torch.where(maximum_error > 0, maximum_error, torch.ones_like(maximum_error))
        nrmse = ((errors / safe_maximum) * weights).sum().clamp(max=1) * maximum_error
        return ScoreResult(
            float(score.item()),
            float(nrmse.item()),
            {name: float(value.item()) for name, value in head_horizons.items()},
            {name: float(value.item()) for name, value in head_errors.items()},
        )
