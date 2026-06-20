"""Feature partitioner layer for resdag.

Splits the feature dimension into multiple overlapping slices with optional
circular wrapping at the boundaries.
"""

import torch
import torch.nn as nn


class FeaturePartitioner(nn.Module):
    """A layer that partitions the feature dimension into overlapping slices.

    This layer is useful for dividing input features into structured regions while
    maintaining smooth transitions between partitions. Commonly used in parallel
    reservoir architectures where different reservoirs process different feature
    subspaces.

    Behavior:
        - Splits the feature dimension into `partitions` groups
        - Each partition overlaps with its neighbors by `overlap` units
        - Applies circular wrapping: last `overlap` features wrap to start, and vice versa

    Args:
        partitions: Number of partitions to divide the feature dimension into
        overlap: Overlap size (in feature units) for each partition

    Input Shape:
        (..., features) — rank-agnostic on the feature (last) dimension. Both
        (batch, features) and (batch, sequence_length, features) are accepted;
        the leading dimensions are preserved. The 2-D rank lets the layer sit in
        the autoregressive ``forecast`` path, where the flattened engine feeds
        single-step slices.

    Output:
        List of `partitions` tensors, each with the same leading dimensions as
        the input and a last dimension of
        partition_width = features // partitions + 2 * overlap

    Raises:
        ValueError: If features % partitions != 0 (unless partitions == 1)
        ValueError: If overlap >= features // partitions (invalid overlap size)

    Example:
        >>> partitioner = FeaturePartitioner(partitions=2, overlap=1)
        >>> x = torch.arange(12).reshape(1, 1, 12).float()
        >>> outputs = partitioner(x)
        >>> len(outputs)
        2
        >>> outputs[0].shape
        torch.Size([1, 1, 8])  # 12//2 + 2*1 = 8
    """

    def __init__(self, partitions: int, overlap: int) -> None:
        """Initialize the FeaturePartitioner.

        Args:
            partitions: Number of partitions
            overlap: Overlap size between adjacent partitions
        """
        super().__init__()
        self.partitions = partitions
        self.overlap = overlap

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        """Split the feature dimension into overlapping partitions with circular wrapping.

        Operates purely on the feature (last) dimension, so it is rank-agnostic:
        both 2-D ``(batch, features)`` and 3-D ``(batch, sequence_length, features)``
        inputs are accepted (the latter is the usual sequence layout; the former
        is what the flattened single-step forecast engine feeds per step). The
        leading dimensions are preserved unchanged.

        Args:
            input: Input tensor whose last dimension is the feature dimension,
                e.g. (batch, features) or (batch, sequence_length, features)

        Returns:
            List of length `self.partitions`, each with the same leading
            dimensions as `input` and a last dimension of `partition_width`

        Raises:
            ValueError: If feature dimension is not divisible by partitions
            ValueError: If overlap is too large relative to partition size
        """
        # If partitions == 1, just return the entire input as a single partition
        if self.partitions == 1:
            return [input]

        features = input.shape[-1]

        # Validate shape
        if features % self.partitions != 0:
            raise ValueError(
                f"Feature dimension ({features}) must be divisible by "
                f"number of partitions ({self.partitions})"
            )

        partition_base_width = features // self.partitions

        if self.overlap >= partition_base_width:
            raise ValueError(
                f"Overlap ({self.overlap}) must be smaller than the base partition "
                f"width ({partition_base_width})"
            )

        # Width of each partition including overlap
        partition_width = partition_base_width + 2 * self.overlap

        # Circular wrapping
        if self.overlap > 0:
            # Concatenate: [last overlap features | all features | first overlap features]
            wrapped_input = torch.cat(
                [input[..., -self.overlap :], input, input[..., : self.overlap]],
                dim=-1,
            )
        else:
            wrapped_input = input

        # Extract partitions
        partitions_out = []
        for i in range(self.partitions):
            start = i * partition_base_width
            end = start + partition_width
            partitions_out.append(wrapped_input[..., start:end])

        return partitions_out

    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        return f"partitions={self.partitions}, overlap={self.overlap}"
