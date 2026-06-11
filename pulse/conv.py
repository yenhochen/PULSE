import torch.nn as nn
from einops import rearrange


class Conv1D(nn.Module):
    """Two-layer 1D conv block with channel-last I/O."""

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        kernel_size=3,
        dilation=1,
        groups=1,
        padding_mode="reflect",
    ):
        super(Conv1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_dim, hidden_dim, kernel_size, padding="same", padding_mode=padding_mode
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            hidden_dim,
            out_dim,
            kernel_size,
            padding="same",
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
        )

    def forward(self, x):
        """Args: x (batch, time, in_dim). Returns (batch, time, out_dim)."""
        x = rearrange(x, "b t c -> b c t")
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = rearrange(x, "b c t -> b t c")
        return x
