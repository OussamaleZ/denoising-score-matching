"""
Échantillonne des points avec le modèle appris et visualise la distribution 2D.
Usage: python sample_and_visualize_2d.py --ckpt-id 2000 --n-samples 5000
"""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import re
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models.mlp_2d import MLP2D
from models import get_sigmas, anneal_Langevin_dynamics
from models.ema import EMAHelper
from datasets.student_mixture_2d import StudentMixture2D_Finite


def dict2namespace(d):
    ns = argparse.Namespace()
    for k, v in d.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns


def load_config_safe(config_path):
    with open(config_path, 'r') as f:
        content = f.read()
    content = re.sub(r'!!python/object:argparse\.Namespace\n', '', content)
    content = re.sub(r'!!python/object/apply:torch\.device\n', '', content)
    config = yaml.safe_load(content)
    return dict2namespace(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--doc', type=str, default='student_mixture_2d')
    parser.add_argument('--ckpt-id', type=int, default=None)
    parser.add_argument('--n-samples', type=int, default=5000, help='Nombre de points à échantillonner')
    parser.add_argument('--batch-size', type=int, default=100, help='Taille du batch')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    
    # Trouver checkpoint
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        log_dir = Path('experiments') / 'logs' / args.doc
        ckpt_file = f'checkpoint_{args.ckpt_id}.pth' if args.ckpt_id else 'checkpoint.pth'
        ckpt_path = log_dir / ckpt_file
    
    # Charger config
    config_path = Path('configs/student_mixture_2d.yml')
    if not config_path.exists():
        config_path = Path('experiments') / 'logs' / args.doc / 'config.yml'
    
    try:
        config = load_config_safe(config_path)
    except:
        with open(config_path, 'r') as f:
            config = dict2namespace(yaml.safe_load(f))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    
    print(f" Checkpoint: {ckpt_path}")
    print(f"  Device: {device}")
    print(f" Nombre d'échantillons: {args.n_samples}")
    
    # Charger modèle
    states = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MLP2D(config)
    state_dict = states[0] if isinstance(states, (list, tuple)) else states
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    
    if config.model.ema and len(states) > 4:
        ema = EMAHelper(mu=config.model.ema_rate)
        ema.register(model)
        ema.load_state_dict(states[-1])
        ema.ema(model)
    
    model = model.to(device)
    model.eval()
    
    # Échantillonner
    sigmas = get_sigmas(config).to(device)
    all_samples = []
    batch_size = min(args.batch_size, args.n_samples)
    n_batches = (args.n_samples + batch_size - 1) // batch_size
    
    print(f" Batch size: {batch_size}, Batches: {n_batches}")
    
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Échantillonnage"):
            samples_collected = sum(s.shape[0] for s in all_samples) if all_samples else 0
            remaining = args.n_samples - samples_collected
            current_batch_size = min(batch_size, remaining)
            
            xmin, xmax = -10.0, 15.0   # large domaine
            init_samples = torch.rand(current_batch_size, 2, device=device)
            init_samples = init_samples * (xmax - xmin) + xmin

            
            # Sampling avec Langevin dynamics
            samples = anneal_Langevin_dynamics(
                init_samples, model, sigmas.cpu().numpy(),
                n_steps_each=config.sampling.n_steps_each,
                step_lr=config.sampling.step_lr,
                final_only=True,
                verbose=False,
                denoise=config.sampling.denoise
            )
            
            final_samples_2d = samples[-1]
            all_samples.append(final_samples_2d.cpu().numpy())
    
    # Concaténer
    all_samples = np.concatenate(all_samples, axis=0)
    
    print(f"\n✓ {len(all_samples)} points échantillonnés")
    print(f"  X: [{all_samples[:, 0].min():.2f}, {all_samples[:, 0].max():.2f}]")
    print(f"  Y: [{all_samples[:, 1].min():.2f}, {all_samples[:, 1].max():.2f}]")
    
    # Générer des échantillons de la distribution réelle
    print(" Génération de la distribution réelle...")
    df = getattr(config.data, 'df', 3.0)
    scale = getattr(config.data, 'scale', 1.0)
    test_seed = getattr(config.data, 'test_seed', 42)
    real_dataset = StudentMixture2D_Finite(n_samples=args.n_samples, transform=None, seed=test_seed, df=df, scale=scale)
    real_samples = torch.stack([real_dataset[i][0] for i in range(len(real_dataset))]).numpy()
    
    # Générer nom de fichier
    if args.out is None:
        n_steps = config.sampling.n_steps_each
        num_classes = len(sigmas)
        args.out = f'samples_2d_n{args.n_samples}_steps{n_steps}_sigmas{num_classes}_bs{batch_size}.png'
    
            # Visualiser - 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Limites fixes pour éviter les outliers de la distribution Student-t
    x_min, x_max = -7, 17
    y_min, y_max = -7, 17
    
    # Plot 1: Distribution réelle
    ax1 = axes[0]
    ax1.scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.5, s=1, c='green')
    ax1.scatter([0, 10], [0, 10], c='red', s=100, marker='x', linewidths=3, label='Moyennes')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title(f'Distribution réelle ({len(real_samples)} points)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    
    # Plot 2: Distribution échantillonnée
    ax2 = axes[1]
    ax2.scatter(all_samples[:, 0], all_samples[:, 1], alpha=0.5, s=1, c='blue')
    ax2.scatter([0, 10], [0, 10], c='red', s=100, marker='x', linewidths=3, label='Moyennes')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title(f'Distribution échantillonnée ({len(all_samples)} points)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    
    # Plot 3: Histogramme 2D
    ax3 = axes[2]
    h = ax3.hist2d(all_samples[:, 0], all_samples[:, 1], bins=50, cmap='Blues', density=True)
    ax3.scatter([0, 10], [0, 10], c='red', s=100, marker='x', linewidths=3, label='Moyennes')
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_title('Densité estimée')
    ax3.legend()
    plt.colorbar(h[3], ax=ax3)
    ax3.set_aspect('equal')
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f" Figure sauvegardée: {args.out}")

if __name__ == '__main__':
    main()