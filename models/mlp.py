import torch
import torch.nn as nn
import torch.nn.functional as F
from . import get_sigmas

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        
        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.output_dim = config.model.output_dim
        self.num_layers = config.model.num_layers

        self.register_buffer('sigmas', get_sigmas(config))

        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.network = nn.Sequential(*layers)
    
    def forward(self, x, y):
        """
        Forward pass of the MLP model.

        Input:
        param x: of shape (B, input_dim)
        param y: of shape (B,) representing the noise levels we are conditioning on
        
        output: of shape (B, output_dim)
        """
        output = self.network(x)
        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))
        output = output / used_sigmas
    
        return output