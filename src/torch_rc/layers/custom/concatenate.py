import torch.nn as nn


class Concatenate(nn.Module):
    """Concatenates inputs along feature dimension."""

    def forward(self, *inputs):
        import torch

        return torch.cat(inputs, dim=-1)
