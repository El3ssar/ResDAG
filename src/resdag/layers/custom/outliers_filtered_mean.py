"""Outlier-filtered mean layer for resdag.

Removes outliers from ensembles and computes the mean, useful for denoising
temporal data by filtering extreme values.
"""

import torch
import torch.nn as nn


class OutliersFilteredMean(nn.Module):
    """Layer that removes outliers and returns the mean of remaining elements.

    This layer removes outliers along the samples dimension independently at each
    (batch, timestep) location, based on either Z-score or IQR method, then returns
    the mean of the remaining elements. Useful for ensemble denoising.

    Input Shape:
        - Single tensor: (samples, batch, timesteps, features)
        - List of tensors: List of length `samples`, each (batch, timesteps, features)

    Output Shape:
        (batch, timesteps, features)

    Procedure:
        1. Compute L2 norm over features dimension -> (samples, batch, timesteps)
        2. For each (batch, timestep), compute outlier thresholds using Z-score or IQR
        3. Build mask indicating inlier vs outlier samples
        4. Compute mean of inlier samples
        5. If all samples are outliers at some location, use all samples (fallback)

    Args:
        method: Outlier detection method. Options: "z_score" or "iqr"
        threshold: Threshold for outlier removal (e.g., 3.0 for ±3 std in Z-score)

    Example:
        >>> layer = OutliersFilteredMean(method="z_score", threshold=2.0)
        >>> # 10 samples, batch=3, timesteps=5, features=4
        >>> x = torch.randn(10, 3, 5, 4)
        >>> y = layer(x)
        >>> y.shape
        torch.Size([3, 5, 4])
    """

    def __init__(self, method: str = "z_score", threshold: float = 3.0) -> None:
        """Initialize the OutliersFilteredMean layer.

        Args:
            method: Outlier detection method ("z_score" or "iqr")
            threshold: Threshold value for outlier detection

        Raises:
            ValueError: If method is not "z_score" or "iqr"
        """
        super().__init__()
        self.method = method
        self.threshold = threshold

        if self.method not in ["z_score", "iqr"]:
            raise ValueError(f"Unsupported method: {self.method}. Choose 'z_score' or 'iqr'.")

    def forward(self, input: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Remove outliers and compute mean.

        Args:
            input: Either a tensor of shape (samples, batch, timesteps, features)
                  or a list of tensors each of shape (batch, timesteps, features)

        Returns:
            Tensor of shape (batch, timesteps, features) with outlier-filtered mean
        """
        # Convert list to stacked tensor if needed
        if isinstance(input, list):
            input = torch.stack(input, dim=0)  # (samples, batch, timesteps, features)
        elif input.dim() == 3:
            # Single tensor without samples dim, add it
            input = input.unsqueeze(0)  # (1, batch, timesteps, features)

        # Compute L2 norm over features -> (samples, batch, timesteps)
        norms = torch.norm(input, p=2, dim=-1)

        # Compute outlier mask based on method
        if self.method == "z_score":
            # Compute mean and std over samples dimension
            mean_norm = norms.mean(dim=0)  # (batch, timesteps)
            std_norm = norms.std(dim=0, unbiased=False)  # (batch, timesteps)

            # Avoid division by zero
            std_norm = torch.where(std_norm > 0, std_norm, torch.ones_like(std_norm))

            # Compute Z-scores
            z_scores = torch.abs((norms - mean_norm) / std_norm)

            # True = inlier, False = outlier
            mask = z_scores < self.threshold

        else:  # method == "iqr"
            # Compute quartiles over samples dimension
            q1 = torch.quantile(norms, 0.25, dim=0)  # (batch, timesteps)
            q3 = torch.quantile(norms, 0.75, dim=0)  # (batch, timesteps)
            iqr = q3 - q1

            # Compute bounds
            lower_bound = q1 - self.threshold * iqr
            upper_bound = q3 + self.threshold * iqr

            # True = inlier, False = outlier
            mask = (norms >= lower_bound) & (norms <= upper_bound)

        # Expand mask to broadcast over features dimension
        mask_expanded = mask.unsqueeze(-1)  # (samples, batch, timesteps, 1)

        # Apply mask (zero out outliers)
        masked_input = input * mask_expanded

        # Sum over samples and count valid samples
        sum_inliers = masked_input.sum(dim=0)  # (batch, timesteps, features)
        count_inliers = mask_expanded.float().sum(dim=0)  # (batch, timesteps, 1)

        # Broadcast count to match features dimension
        count_inliers = count_inliers.expand_as(sum_inliers)

        # Avoid division by zero: if no inliers, use all samples
        count_inliers = torch.where(
            count_inliers > 0, count_inliers, torch.ones_like(count_inliers)
        )

        # Compute mean
        mean_result = sum_inliers / count_inliers

        return mean_result

    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        return f"method='{self.method}', threshold={self.threshold}"
