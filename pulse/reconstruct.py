import torch.nn as nn
from pulse.mlp import MLP


class ReconstructionNet(nn.Module):
    """GRU decoder: predicts future signal from recon inputs and initial hidden state."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.in_dim = config.data_args.input_dims
        self.context_dim = config.encoder_args.emb_dim
        self.hidden_dim = config.model_args.recon_args.hidden_dim
        self.gru_num_layers = config.model_args.recon_args.num_layers

        self.recon_gru = nn.GRU(
            (
                self.context_dim + config.model_args.time_vary_args.tv_dim
                if config.model_args.time_vary_args.include
                else self.context_dim
            ),
            self.hidden_dim,
            num_layers=self.gru_num_layers,
            batch_first=True,
        )

        self.out_proj = MLP(self.hidden_dim, self.hidden_dim, self.in_dim)

    def forward(self, inputs, h0):
        """Args: inputs (batch, time, context[+tv_dim]), h0 (num_layers, batch, hidden_dim).

        Returns out (batch, time, channels), hidden states (batch, time, hidden_dim).
        """
        out_, _ = self.recon_gru(inputs, h0)  # (batch, time, hidden_dim)
        out = self.out_proj(out_)  # (batch, time, channels)
        return out, out_
