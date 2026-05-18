"""Outlier-filtered mean aggregator for ensemble forecasts."""

import torch
import torch.nn as nn


class OutliersFilteredMean(nn.Module):
    """Mean over ensemble members after removing outlier norms.

    For each ``(batch, timestep)`` location, computes the L2 norm of each
    member's feature vector, flags outliers via Z-score or IQR, then averages
    inlier members. If every member is an outlier at a location, falls back to
    the plain mean.

    Parameters
    ----------
    method : {"z_score", "iqr"}, default="z_score"
        Outlier detection rule applied to per-member norms.
    threshold : float, default=3.0
        Z-score cutoff (standard deviations) or IQR multiplier.

    Examples
    --------
    >>> layer = OutliersFilteredMean(method="z_score", threshold=2.0)
    >>> x = torch.randn(10, 3, 5, 4)  # samples, batch, time, features
    >>> layer(x).shape
    torch.Size([3, 5, 4])

    Raises
    ------
    ValueError
        If *method* is not ``"z_score"`` or ``"iqr"``.
    """

    def __init__(self, method: str = "z_score", threshold: float = 3.0) -> None:
        """Configure outlier detection.

        Parameters
        ----------
        method : {"z_score", "iqr"}, default="z_score"
            How to label outlier members along the samples axis.
        threshold : float, default=3.0
            Detection sensitivity (Z-score bound or IQR factor).

        Raises
        ------
        ValueError
            If *method* is not supported.
        """
        super().__init__()
        self.method = method
        self.threshold = threshold

        if self.method not in ["z_score", "iqr"]:
            raise ValueError(f"Unsupported method: {self.method}. Choose 'z_score' or 'iqr'.")

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
            mean_norm = norms.mean(dim=0)
            std_norm = norms.std(dim=0, unbiased=False)
            std_norm = torch.where(std_norm > 0, std_norm, torch.ones_like(std_norm))
            z_scores = torch.abs((norms - mean_norm) / std_norm)
            mask = z_scores < self.threshold

        else:
            q1 = torch.quantile(norms, 0.25, dim=0)
            q3 = torch.quantile(norms, 0.75, dim=0)
            iqr = q3 - q1
            lower_bound = q1 - self.threshold * iqr
            upper_bound = q3 + self.threshold * iqr
            mask = (norms >= lower_bound) & (norms <= upper_bound)

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

    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        return f"method='{self.method}', threshold={self.threshold}"
