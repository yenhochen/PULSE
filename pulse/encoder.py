"""Shared time-series encoder backbone (adapted from REBAR).

TSEncoder wraps a dilated conv stack with optional LayerNorm and temporal pooling.
All SSL baselines and PULSE use this module via trainers/base.py.
"""

import torch
import numpy as np
import torch.nn as nn

from einops import rearrange


class TSEncoder(nn.Module):
    """Encoder with layer norm and max/avg pooling across time."""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.norm_bool = config.encoder_args.norm_last_layer
        self.ts_encoder = TSEncoder_(
            config.data_args.input_dims,
            config.encoder_args.emb_dim,
            hidden_dims=config.encoder_args.hidden_dim,
            depth=config.encoder_args.num_layers,
        )

        if self.norm_bool:
            self.context_norm = nn.LayerNorm(config.encoder_args.emb_dim)

        if config.encoder_args.pool_across_time_mode == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif config.encoder_args.pool_across_time_mode == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """Args: x (batch, time, channels). Returns pooled (batch, emb_dim), unpooled (batch, time, emb_dim)."""
        # input: (batch, time, channels)
        embed = self.ts_encoder(x)  # (batch, time, emb_dim)

        if self.norm_bool:
            embed = self.context_norm(embed)

        embed_pool = self.pool(rearrange(embed, "b t z -> b z t")).squeeze()
        # embed_pool: (batch, emb_dim); embed: (batch, time, emb_dim)
        return embed_pool, embed


class TSEncoder_(torch.nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.input_fc = torch.nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims, [hidden_dims] * depth + [output_dims], kernel_size=3
        )

    def forward(self, x, mask=None):
        """Args: x (batch, time, input_dims), mask optional binomial dropout mask.

        Returns (batch, time, output_dims).
        """
        # input: (batch, time, input_dims)
        nan_mask = ~x.isnan().any(axis=-1)  # TS2Vec may inject NaN timesteps
        x[~nan_mask] = 0

        x = self.input_fc(x)  # (batch, time, hidden)

        if mask == "binomial":
            mask = torch.from_numpy(
                np.random.binomial(1, 0.5, size=(x.size(0), x.size(1)))
            ).to(x.device)
            mask &= nan_mask
            x[~mask] = 0

        x = x.transpose(1, 2)  # (batch, hidden, time)
        x = self.feature_extractor(x)  # (batch, emb_dim, time)
        x = x.transpose(1, 2)  # (batch, time, emb_dim)
        return x


class DilatedConvEncoder(torch.nn.Module):
    """Stack of dilated conv blocks; I/O layout (batch, channels, time)."""

    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            *[
                ConvBlock(
                    channels[i - 1] if i > 0 else in_channels,
                    channels[i],
                    kernel_size=kernel_size,
                    dilation=2**i,
                    final=(i == len(channels) - 1),
                )
                for i in range(len(channels))
            ]
        )

    def forward(self, x):
        # input/output: (batch, channels, time)
        return self.net(x)


class ConvBlock(torch.nn.Module):
    """Residual conv block with GELU activations."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.conv2 = SamePadConv(
            out_channels, out_channels, kernel_size, dilation=dilation
        )
        self.projector = (
            torch.nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels or final
            else None
        )

    def forward(self, x):
        # input/output: (batch, channels, time)
        residual = x if self.projector is None else self.projector(x)
        x = torch.nn.functional.gelu(x)
        x = self.conv1(x)
        x = torch.nn.functional.gelu(x)
        x = self.conv2(x)
        return x + residual


class SamePadConv(torch.nn.Module):
    """Conv1d with same-length output via symmetric padding."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
