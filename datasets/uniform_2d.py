import torch
from torch.utils.data import Dataset
import numpy as np

class Uniform2D(Dataset):
    def __init__(self, xmin=0, xmax=1, ymin=0, ymax=1, n_samples=100000, grid=False, grid_res=200, seed=None):
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.seed = seed
        rng = np.random.RandomState(seed)
        if grid:
            xs = np.linspace(xmin, xmax, grid_res)
            ys = np.linspace(ymin, ymax, grid_res)
            X, Y = np.meshgrid(xs, ys, indexing='xy')
            pts = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
            self.data = torch.from_numpy(pts)[:n_samples]
        else:
            samples = rng.rand(n_samples, 2).astype(np.float32)
            samples[:,0] = samples[:,0] * (xmax-xmin) + xmin
            samples[:,1] = samples[:,1] * (ymax-ymin) + ymin
            self.data = torch.from_numpy(samples)
    def __len__(self): 
        return len(self.data)
    def __getitem__(self, idx): 
        # Return a dummy label so DataLoader yields (X, y) tuples like other datasets
        return self.data[idx], 0