from torch.utils.data import Dataset
import torch
import math

class Uniform2DGrid(Dataset):
    """
    Dataset enumerating points on a regular [0,1]^2 grid.
    """

    def __init__(self, grid_size=32, transform=None):
        self.grid_size = grid_size
        self.transform = transform
        self.data = self._build_grid()

    def _build_grid(self):
        lin = torch.linspace(0, 1, self.grid_size)
        x, y = torch.meshgrid(lin, lin, indexing='xy')
        coords = torch.stack([x.flatten(), y.flatten()], dim=-1)
        return coords

    def __len__(self):
        return self.grid_size ** 2

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, 0
