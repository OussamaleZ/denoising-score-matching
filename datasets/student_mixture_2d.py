from torch.utils.data import Dataset
import torch
from torch.distributions import StudentT


class StudentMixture2D_Finite(Dataset):
    """
    Fixed finite dataset: x_1,...,x_N i.i.d. ~ Mixture of 2 Student-t distributions in 2D.
    Component 1: mean [0,0], probability 1/5
    Component 2: mean [10,10], probability 4/5
    """
    def __init__(self, n_samples=10000, transform=None, seed=None, df=3.0, scale=1.0):
        self.n_samples = n_samples
        self.transform = transform
        self.df = df
        self.scale = scale
        
        # Probabilités des composantes
        self.probs = torch.tensor([1/5, 4/5])
        # Moyennes des composantes
        self.means = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
        
        g = None
        if seed is not None:
            g = torch.Generator().manual_seed(seed)
        
        # Générer les échantillons
        self.data = self._generate_samples(n_samples, generator=g)
    
    def _generate_samples(self, n_samples, generator=None):
        """Génère n_samples selon le mélange de Student-t."""
        # Fixer le seed globalement si un generator est fourni
        if generator is not None:
            torch.manual_seed(generator.initial_seed())
        
        samples = []
        
        for _ in range(n_samples):
            # Choisir la composante selon les probabilités
            if generator is not None:
                comp_idx = torch.multinomial(self.probs, 1, generator=generator).item()
            else:
                comp_idx = torch.multinomial(self.probs, 1).item()
            
            mean = self.means[comp_idx]
            
            # Générer un échantillon Student-t pour chaque dimension
            student_dist = StudentT(df=self.df, loc=0.0, scale=self.scale)
            sample = mean + torch.stack([
                student_dist.sample((1,))[0],
                student_dist.sample((1,))[0]
            ])
            
            samples.append(sample)
        
        return torch.stack(samples)  # Shape: (n_samples, 2)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, 0


class StudentMixture2D_Online(Dataset):
    """
    Dataset of i.i.d. samples drawn from a mixture of 2 Student-t distributions in 2D.
    Samples are generated online in __getitem__.
    Component 1: mean [0,0], probability 1/5
    Component 2: mean [10,10], probability 4/5
    """
    def __init__(self, n_samples=10000, transform=None, df=3.0, scale=1.0):
        self.n_samples = n_samples
        self.transform = transform
        self.df = df
        self.scale = scale
        
        # Probabilités des composantes
        self.probs = torch.tensor([1/5, 4/5])
        # Moyennes des composantes
        self.means = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        # Choisir la composante selon les probabilités
        comp_idx = torch.multinomial(self.probs, 1).item()
        mean = self.means[comp_idx]
        
        # Générer un échantillon Student-t pour chaque dimension
        student_dist = StudentT(df=self.df, loc=0.0, scale=self.scale)
        sample = mean + torch.stack([
            student_dist.sample((1,))[0],
            student_dist.sample((1,))[0]
        ])
        
        if self.transform:
            sample = self.transform(sample)
        return sample, 0

