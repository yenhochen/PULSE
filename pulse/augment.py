import torch
import torch.nn as nn

from einops import rearrange

import random
import numpy as np

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


class DynamicAugmentations(nn.Module):
    """Recon-input utilities for PULSE training."""

    def __init__(self, config):
        super(DynamicAugmentations, self).__init__()

        self.config = config
        self.hidden_dim = self.config.model_args.recon_args.hidden_dim
        self.projector = nn.Identity()

    def get_recon_inputs(self, context, n_steps):
        """Args: context (batch, emb_dim), n_steps int. Returns (batch, n_steps, emb_dim)."""
        context = rearrange(context, "b c -> b 1 c")
        context = self.projector(context)

        batch, _, emb_dim = context.shape
        recon_inputs = context.expand(batch, n_steps, emb_dim).contiguous()
        return recon_inputs
