import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, LSUN
from datasets.celeba import CelebA
from datasets.ffhq import FFHQ
from torch.utils.data import Subset
from datasets.uniform_2d import Uniform2D
from datasets.uniform_1d import Uniform1D_Finite, Uniform1D_Online
from datasets.student_mixture_2d import StudentMixture2D_Finite, StudentMixture2D_Online
import numpy as np

def get_dataset(args, config):
    if not hasattr(config.data, 'is_not_image'):
        if config.data.random_flip is False:
            tran_transform = test_transform = transforms.Compose([
                transforms.Resize(config.data.image_size),
                transforms.ToTensor()
            ])
        else:
            tran_transform = transforms.Compose([
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize(config.data.image_size),
                transforms.ToTensor()
            ])

    if config.data.dataset == 'CIFAR10':
        dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10'), train=True, download=True,
                          transform=tran_transform)
        test_dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10_test'), train=False, download=True,
                               transform=test_transform)

    elif config.data.dataset == 'CELEBA':
        if config.data.random_flip:
            dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ]), download=True)
        else:
            dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

        test_dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba_test'), split='test',
                              transform=transforms.Compose([
                                  transforms.CenterCrop(140),
                                  transforms.Resize(config.data.image_size),
                                  transforms.ToTensor(),
                              ]), download=True)


    elif config.data.dataset == 'LSUN':
        train_folder = '{}_train'.format(config.data.category)
        val_folder = '{}_val'.format(config.data.category)
        if config.data.random_flip:
            dataset = LSUN(root=os.path.join(args.exp, 'datasets', 'lsun'), classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.ToTensor(),
                             ]))
        else:
            dataset = LSUN(root=os.path.join(args.exp, 'datasets', 'lsun'), classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

        test_dataset = LSUN(root=os.path.join(args.exp, 'datasets', 'lsun'), classes=[val_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

    elif config.data.dataset == "FFHQ":
        if config.data.random_flip:
            dataset = FFHQ(path=os.path.join(args.exp, 'datasets', 'FFHQ'), transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ]), resolution=config.data.image_size)
        else:
            dataset = FFHQ(path=os.path.join(args.exp, 'datasets', 'FFHQ'), transform=transforms.ToTensor(),
                           resolution=config.data.image_size)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)

    elif config.data.dataset == 'Uniform2D':
        dataset = Uniform2D(xmin=config.data.xmin, xmax=config.data.xmax, ymin= config.data.ymin, ymax= config.data.ymax, n_samples=config.data.n_samples, grid= config.data.grid, grid_res=config.data.grid_res)
        test_dataset = Uniform2D(xmin=config.data.xmin, xmax=config.data.xmax, ymin= config.data.ymin, ymax= config.data.ymax, n_samples=config.data.n_test_samples, grid= config.data.test_grid, grid_res=config.data.test_grid_res)

    elif config.data.dataset == 'Uniform1D':
        n_samples = getattr(config.data, 'n_samples', 10000)
        n_test_samples = getattr(config.data, 'n_test_samples', 2000)
        test_seed = getattr(config.data, 'test_seed', 42)

        dataset = Uniform1D_Online(n_samples=n_samples, transform=None)
        test_dataset = Uniform1D_Finite(n_samples=n_test_samples, transform=None, seed=test_seed)

    elif config.data.dataset == 'StudentMixture2D':
        n_samples = getattr(config.data, 'n_samples', 10000)
        n_test_samples = getattr(config.data, 'n_test_samples', 2000)
        test_seed = getattr(config.data, 'test_seed', 42)
        df = getattr(config.data, 'df', 3.0)
        scale = getattr(config.data, 'scale', 1.0)

        dataset = StudentMixture2D_Online(n_samples=n_samples, transform=None, df=df, scale=scale)
        test_dataset = StudentMixture2D_Finite(n_samples=n_test_samples, transform=None, seed=test_seed, df=df, scale=scale)

    return dataset, test_dataset

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)

def data_transform(config, X):
    if not hasattr(config.data, 'is_not_image'):
        if config.data.uniform_dequantization:
            X = X / 256. * 255. + torch.rand_like(X) / 256.
        if config.data.gaussian_dequantization:
            X = X + torch.randn_like(X) * 0.01

        if config.data.rescaled:
            X = 2 * X - 1.
        elif config.data.logit_transform:
            X = logit_transform(X)

        if hasattr(config, 'image_mean'):
            return X - config.image_mean.to(X.device)[None, ...]

    return X

def inverse_data_transform(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)
