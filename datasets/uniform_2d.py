from torch.utils.data import Dataset
import torch
import math

class Uniform2D_Finite(Dataset):
    """
    Fixed finite dataset: x_1,...,x_N i.i.d. ~ Uniform([0,1]^2).
    Represents the empirical measure over these N samples.
    """
    def __init__(self, n_samples=10000, transform=None, seed=None):
        self.n_samples = n_samples
        self.transform = transform

        g = None
        if seed is not None:
            g = torch.Generator().manual_seed(seed)

        self.data = torch.rand(n_samples, 2, generator=g) # of shape (n_samples, 2)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, 0
    

class Uniform2D_Online(Dataset):
    """
    Dataset of i.i.d. samples drawn from the uniform distribution on [0,1]^2.
    Samples are generated online in __getitem__.
    """

    def __init__(self, n_samples=10000, transform=None):
        self.n_samples = n_samples
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = torch.rand(2)  # ~ Uniform([0,1]^2)
        if self.transform:
            sample = self.transform(sample)
        return sample, 0