import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pulse.conv import Conv1D
from utils.common import get_true_rolled


class TimeVaryingModule(nn.Module):
    """Extract a time-varying latent from encoder output for nonstationary reconstruction."""

    def __init__(self, config):
        super(TimeVaryingModule, self).__init__()

        self.conv = Conv1D(
            config.encoder_args.emb_dim,
            config.model_args.time_vary_args.tv_dim,
            config.model_args.time_vary_args.tv_dim,
            kernel_size=3,
            dilation=1,
            groups=config.model_args.time_vary_args.tv_dim,
        )
        self.config = config

    def forward(self, x):
        """Extract time-varying latent from encoder output.

        Args:
            x: (batch, time, emb_dim) — encoder output (unpooled)
        Returns:
            tv: (batch, time, tv_dim) — full-resolution time-varying signal
            dtv: (batch, time, tv_dim) — pooled then upsampled (for recon input)
        """
        x = self.conv(x)  # (batch, time, tv_dim)
        pool_denom = self.config.model_args.time_vary_args.pool_denom
        t = x.shape[1]
        pooled_len = max(1, t // pool_denom)

        o = rearrange(x, "b t c -> b c t")
        o = F.adaptive_max_pool1d(o, pooled_len)
        if pooled_len != t:
            o = F.interpolate(o, size=t, mode="nearest")
        o = rearrange(o, "b c t -> b t c")
        return x, o

    def shift_start(self, x, start_ix):
        """Args: x (batch, time, channels), start_ix (batch,). Returns rolled x."""
        return get_true_rolled(x, start_ix)
