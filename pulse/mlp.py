import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    """Two-layer MLP: input_dim -> hidden_dim -> output_dim."""

    def __init__(
        self, input_dim, hidden_dim, output_dim, activation=torch.nn.ReLU(), bias=True
    ):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim, bias=bias)
        self.activation = activation
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, x):
        """Args: x (batch, input_dim). Returns (batch, output_dim)."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
