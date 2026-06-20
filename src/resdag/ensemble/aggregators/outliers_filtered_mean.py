"""Outlier-filtered mean aggregator for ensemble forecasts."""

import torch
import torch.nn as nn

# Conventional cutoff for the modified (median/MAD) z-score, per
# Iglewicz & Hoaglin (1993). The 0.6745 constant rescales the MAD so that the
# modified z-score is comparable to a standard z-score for normal data.
_MODIFIED_Z_CONST: float = 0.6745
_DEFAULT_Z_THRESHOLD: float = 3.5
# Conventional Tukey fence multiplier for the IQR rule.
_DEFAULT_IQR_THRESHOLD: float = 1.5


class OutliersFilteredMean(nn.Module):
    """Mean over ensemble members after removing outlier norms.

    For each ``(batch, timestep)`` location, computes the L2 norm of each
    member's feature vector, flags outliers via a robust modified Z-score or
    the IQR rule, then averages the inlier members. If every member is an
    outlier at a location, falls back to the plain mean.

    The ``"z_score"`` method uses the **modified Z-score**
    ``0.6745 * (x - median) / MAD`` (Iglewicz & Hoaglin, 1993), where ``MAD``
    is the median absolute deviation. Unlike the classic mean/standard-deviation
    Z-score, the median and MAD are robust estimators: a single extreme member
    does not inflate the location/scale used to judge it, so a genuine outlier
    can be flagged even in small ensembles. (The classic non-robust Z-score
    saturates at ``sqrt(N - 1)`` for a lone outlier among ``N`` members, so a
    ``threshold=3.0`` cutoff could never flag anything until ``N >= 11``.)

    Parameters
    ----------
    method : {"z_score", "iqr"}, default="z_score"
        Outlier detection rule applied to per-member norms.
    threshold : float or None, default=None
        Detection sensitivity. For ``"z_score"`` it is the modified-Z-score
        cutoff; for ``"iqr"`` it is the Tukey fence multiplier. When ``None``,
        a method-appropriate default is used: ``3.5`` for ``"z_score"`` (the
        conventional modified-Z-score cutoff) and ``1.5`` for ``"iqr"`` (the
        conventional Tukey fence). Pass an explicit value to override.

    Examples
    --------
    >>> layer = OutliersFilteredMean(method="z_score")
    >>> x = torch.randn(10, 3, 5, 4)  # samples, batch, time, features
    >>> layer(x).shape
    torch.Size([3, 5, 4])

    Raises
    ------
    ValueError
        If *method* is not ``"z_score"`` or ``"iqr"``.

    References
    ----------
    Iglewicz, B. and Hoaglin, D. C. (1993). *How to Detect and Handle
    Outliers*. ASQC Quality Press.
    """

    def __init__(self, method: str = "z_score", threshold: float | None = None) -> None:
        """Configure outlier detection.

        Parameters
        ----------
        method : {"z_score", "iqr"}, default="z_score"
            How to label outlier members along the samples axis.
        threshold : float or None, default=None
            Detection sensitivity (modified-Z-score bound or IQR factor). When
            ``None``, defaults to ``3.5`` for ``"z_score"`` and ``1.5`` for
            ``"iqr"``.

        Raises
        ------
        ValueError
            If *method* is not supported.
        """
        super().__init__()

        if method not in ["z_score", "iqr"]:
            raise ValueError(f"Unsupported method: {method}. Choose 'z_score' or 'iqr'.")

        self.method = method
        if threshold is None:
            threshold = _DEFAULT_Z_THRESHOLD if method == "z_score" else _DEFAULT_IQR_THRESHOLD
        self.threshold = threshold

    def forward(self, input: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Aggregate ensemble members with outlier filtering.

        Parameters
        ----------
        input : torch.Tensor or list of torch.Tensor
            Either shape ``(samples, batch, timesteps, features)`` or a list
            of length *samples* with each tensor ``(batch, timesteps, features)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, timesteps, features)``.
        """
        if isinstance(input, list):
            input = torch.stack(input, dim=0)
        elif input.dim() == 3:
            input = input.unsqueeze(0)

        norms = torch.norm(input, p=2, dim=-1)

        if self.method == "z_score":
            mask = self._mask_modified_z_score(norms)
        else:
            mask = self._mask_iqr(norms)

        mask_expanded = mask.unsqueeze(-1)
        plain_mean = input.mean(dim=0)

        masked_input = input * mask_expanded
        sum_inliers = masked_input.sum(dim=0)
        count_inliers = mask_expanded.float().sum(dim=0).expand_as(sum_inliers)

        mean_result = torch.where(
            count_inliers > 0,
            sum_inliers / count_inliers.clamp(min=1),
            plain_mean,
        )

        return mean_result

    def _mask_modified_z_score(self, norms: torch.Tensor) -> torch.Tensor:
        """Flag inliers via the robust modified (median/MAD) Z-score.

        Parameters
        ----------
        norms : torch.Tensor
            Per-member norms of shape ``(samples, batch, timesteps)``.

        Returns
        -------
        torch.Tensor
            Boolean mask of shape ``(samples, batch, timesteps)`` that is
            ``True`` for inlier members.
        """
        median = norms.median(dim=0, keepdim=True).values
        abs_dev = torch.abs(norms - median)
        mad = abs_dev.median(dim=0, keepdim=True).values

        # Normal case: rescale the deviation by the (robust) MAD-based scale and
        # compare to the modified-Z-score cutoff.
        scale = mad / _MODIFIED_Z_CONST
        modified_z = abs_dev / scale.clamp(min=torch.finfo(scale.dtype).eps)
        mask = modified_z < self.threshold

        # Degenerate case: a zero MAD means the majority of members are
        # identical (zero dispersion), so any member deviating from the median
        # is unambiguously an outlier. Rescaling by a non-robust scale here
        # would reintroduce the very saturation this estimator avoids, so flag
        # by raw deviation instead: keep only members exactly at the median.
        mad_zero = mad <= 0
        mask = torch.where(mad_zero, abs_dev <= 0, mask)
        return mask

    def _mask_iqr(self, norms: torch.Tensor) -> torch.Tensor:
        """Flag inliers via the Tukey IQR fence.

        Parameters
        ----------
        norms : torch.Tensor
            Per-member norms of shape ``(samples, batch, timesteps)``.

        Returns
        -------
        torch.Tensor
            Boolean mask of shape ``(samples, batch, timesteps)`` that is
            ``True`` for inlier members.
        """
        q1 = torch.quantile(norms, 0.25, dim=0)
        q3 = torch.quantile(norms, 0.75, dim=0)
        iqr = q3 - q1
        lower_bound = q1 - self.threshold * iqr
        upper_bound = q3 + self.threshold * iqr
        return (norms >= lower_bound) & (norms <= upper_bound)

    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        return f"method='{self.method}', threshold={self.threshold}"
