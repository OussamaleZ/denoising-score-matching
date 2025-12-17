import torch
import torch.nn as nn
import torch.nn.functional as F
from . import get_sigmas


class MLP1D(nn.Module):
    """
    Simple MLP for score estimation on 1D uniform distribution.
    Conditioned on noise level via label embedding.
    """
    def __init__(self, config):
        super(MLP1D, self).__init__()

        self.config = config
        
        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.output_dim = config.model.output_dim
        self.num_layers = config.model.num_layers
        self.num_classes = config.model.num_classes

        # Register sigmas buffer (noise levels)
        self.register_buffer('sigmas', get_sigmas(config))

        # Embedding for noise level conditioning
        self.sigma_embed = nn.Embedding(self.num_classes, self.hidden_dim)

        # Input layer
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(self.num_layers - 2):
            self.hidden_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        # Output layer
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

        # Activation (Swish/SiLU)
        self.act = nn.SiLU()

    def forward(self, x, y):
        """
        Args:
            x: input data of shape (batch_size, input_dim)
            y: noise level labels of shape (batch_size,) - indices into sigmas
        Returns:
            score: estimated score of shape (batch_size, output_dim)
        """
        # Get sigma embedding
        sigma_emb = self.sigma_embed(y)  # (batch_size, hidden_dim)

        # Input projection
        h = self.input_layer(x)
        h = h + sigma_emb  # Condition on noise level
        h = self.act(h)

        # Hidden layers
        for layer in self.hidden_layers:
            h = layer(h)
            h = h + sigma_emb  # Condition on noise level at each layer
            h = self.act(h)

        # Output
        score = self.output_layer(h)

        # Normalize by sigma (like NCSNv2)
        #used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))
        #score = score / used_sigmas

        return score