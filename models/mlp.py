import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()

        self.config = config
        
        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.output_dim = config.model.output_dim
        self.num_layers = config.model.num_layers

        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.network = nn.Sequential(*layers)
    
    def forward(self, x, y=None):
        return self.network(x)