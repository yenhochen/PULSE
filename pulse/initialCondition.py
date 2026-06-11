"""Initial-condition encoders for PULSE reconstruction.

InitConditionEncoder maps an input window to the GRU hidden state (h0) that
seeds the reconstruction network. When shared_f_init is enabled in config,
SharedInitConditionEncoder reads encoder output instead of raw input.
"""

import math
import random
import torch
import torch.nn as nn
import numpy as np

from einops import rearrange
from pulse.conv import Conv1D

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


class InitConditionEncoder(nn.Module):
    """Conv encoder producing GRU initial hidden state from input window."""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.in_dim = config.data_args.input_dims
        self.context_dim = config.encoder_args.emb_dim
        self.recon_hidden_dim = config.model_args.recon_args.hidden_dim
        self.recon_gru_num_layers = config.model_args.recon_args.num_layers
        self.in_proj_kernel_size = config.model_args.init_args.in_proj_kernel_size
        self.in_proj_dilation = config.model_args.init_args.in_proj_dilation
        self.init_hidden_dim = config.model_args.init_args.hidden_dim

        self.init_proj = Conv1D(
            self.in_dim,
            self.init_hidden_dim,
            self.recon_hidden_dim,
            kernel_size=self.in_proj_kernel_size,
            dilation=self.in_proj_dilation,
            groups=math.gcd(self.recon_hidden_dim, self.init_hidden_dim),
            padding_mode="replicate",
        )

    def forward(self, x, sample_init=False, sample_right_boundary=20):
        """Args: x (batch, time, channels). Returns h0 (num_layers, batch, hidden_dim), start_ix, n_steps."""
        batch, time, _ = x.shape  # input: (batch, time, channels)

        if sample_init:
            start_ix = torch.randint(time - sample_right_boundary, (batch,))
        else:
            start_ix = torch.zeros(batch).long()

        n_steps = time - start_ix - 1

        x0 = self.init_proj(x)  # (batch, time, hidden)
        x0 = x0[torch.arange(batch), start_ix].squeeze()  # (batch, hidden)

        x0 = rearrange(x0, "b (i j) -> i b j", i=1).contiguous()
        x0 = torch.cat(
            [x0]
            + [
                torch.zeros_like(x0)
                for _ in range(self.config.model_args.recon_args.num_layers - 1)
            ]
        )  # (num_layers, batch, hidden_dim)

        return x0, start_ix, n_steps


class SharedInitConditionEncoder(InitConditionEncoder):
    """Same as InitConditionEncoder but input is encoder output (batch, time, emb_dim)."""

    def __init__(self, config):
        super().__init__(config)
        self.init_proj = Conv1D(
            self.context_dim,
            self.init_hidden_dim,
            self.recon_hidden_dim,
            kernel_size=self.in_proj_kernel_size,
            dilation=self.in_proj_dilation,
            groups=math.gcd(self.recon_hidden_dim, self.init_hidden_dim),
            padding_mode="replicate",
        )
