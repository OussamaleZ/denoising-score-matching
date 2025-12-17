"""
Visualise les scores 1D après entraînement.
Usage: 
    python visualize_scores_1d.py
    python visualize_scores_1d.py --doc autre_experience
    python visualize_scores_1d.py --ckpt experiments/logs/uniform_1d/checkpoint_2000.pth
"""

import argparse
import yaml
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import re

sys.path.insert(0, str(Path(__file__).parent))
from models.mlp_1d import MLP1D
from models import get_sigmas
from models.ema import EMAHelper


def dict2namespace(d):
    ns = argparse.Namespace()
    for k, v in d.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns


def load_config_safe(config_path):
    """Charge config en gérant les tags Python dans le YAML sauvegardé."""
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Supprimer les tags Python pour permettre le chargement
    content = re.sub(r'!!python/object:argparse\.Namespace\n', '', content)
    content = re.sub(r'!!python/object/apply:torch\.device\n', '', content)
    
    config = yaml.safe_load(content)
    return dict2namespace(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None, 
                       help='Chemin direct vers checkpoint')
    parser.add_argument('--doc', type=str, default='uniform_1d',
                       help='Nom de l\'expérience')
    parser.add_argument('--ckpt-id', type=int, default=None,
                       help='ID du checkpoint (ex: 2000)')
    parser.add_argument('--config', type=str, default=None,
                       help='Config originale (si différente de configs/uniform_1d.yml)')
    parser.add_argument('--out', default='scores_1d.png')
    args = parser.parse_args()
    
    # Trouver le checkpoint
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        log_dir = ckpt_path.parent
    else:
        log_dir = Path('experiments') / 'logs' / args.doc
        ckpt_file = f'checkpoint_{args.ckpt_id}.pth' if args.ckpt_id else 'checkpoint.pth'
        ckpt_path = log_dir / ckpt_file
    
    # Charger config (essayer d'abord la config originale, puis celle sauvegardée)
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path('configs/uniform_1d.yml')
        if not config_path.exists():
            config_path = log_dir / 'config.yml'
    
    try:
        config = load_config_safe(config_path)
    except:
        # Fallback: utiliser config originale
        config_path = Path('configs/uniform_1d.yml')
        with open(config_path, 'r') as f:
            config = dict2namespace(yaml.safe_load(f))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    
    print(f" Checkpoint: {ckpt_path}")
    print(f" Config: {config_path}")
    print(f"  Device: {device}")
    
    # Charger checkpoint
    states = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Créer et charger modèle
    model = MLP1D(config)
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
    
    # Visualiser
        # Visualiser
    x = torch.linspace(-2, 3, 500).unsqueeze(1).to(device)  # Changé de [0, 1] à [-2, 3]
    sigmas = get_sigmas(config).to(device)
    sigma_indices = [0, len(sigmas)//4, len(sigmas)//2, 3*len(sigmas)//4, len(sigmas)-1]
    
    plt.figure(figsize=(10, 6))
    with torch.no_grad():
        for idx in sigma_indices:
            y = torch.full((len(x),), idx, dtype=torch.long).to(device)
            scores = model(x, y).squeeze().cpu().numpy()
            plt.plot(x.squeeze().cpu().numpy(), scores, label=f'σ={sigmas[idx].item():.3f}', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('Score s(x, σ)')
    plt.title('Scores appris pour différents niveaux de bruit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-2, 3)  # Changé de [0, 1] à [-2, 3]
    plt.axvspan(0, 1, alpha=0.2, color='green', label='Support uniforme [0,1]')  # Marquer la zone [0,1]
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f" Sauvegardé: {args.out}")


if __name__ == '__main__':
    main()