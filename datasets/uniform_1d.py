from torch.utils.data import Dataset
import torch

class Uniform1D(Dataset):
    """
    Synthetic dataset that supplies samples drawn uniformly from [0, 1].
    """

    def __init__(self, num_samples=1000, transform=None, seed=None):
        self.num_samples = num_samples
        self.transform = transform

        if seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)
            self.data = torch.rand(num_samples, 1)
            torch.set_rng_state(rng_state)
        else:
            self.data = torch.rand(num_samples, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, 0
