import torch
import torch.nn as nn
from . import get_sigmas


class MLP1D(nn.Module):
    """
    Simple MLP for score estimation on 1D distribution.
    Noise conditioning via output rescaling (Improved NCSN).
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        
        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.output_dim = config.model.output_dim
        self.num_layers = config.model.num_layers
        self.num_classes = config.model.num_classes

        # Noise levels
        self.register_buffer("sigmas", get_sigmas(config))

        # Input layer
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim)
             for _ in range(self.num_layers - 2)]
        )

        # Output layer
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

        # Activation
        self.act = nn.SiLU()

    def forward(self, x, y):
        """
        Args:
            x: (B, input_dim)
            y: (B,) noise level indices
        Returns:
            score: (B, output_dim)
        """
        # Standard MLP forward (unconditional)
        h = self.input_layer(x)
        h = self.act(h)

        for layer in self.hidden_layers:
            h = layer(h)
            h = self.act(h)

        score = self.output_layer(h)

        # Noise conditioning via analytic rescaling
        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * (score.dim() - 1)))
        score = score / used_sigmas

        return score
