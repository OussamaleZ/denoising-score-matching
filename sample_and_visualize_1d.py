"""
Sample 1D points from a trained MLP score model using annealed Langevin dynamics.

Usage:
  python sampling_1d.py --ckpt-id 20000 --n-samples 4096
"""

import argparse
import sys
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mlp_1d import MLP1D
from models import anneal_Langevin_dynamics, get_sigmas


def dict2namespace(d):
    ns = argparse.Namespace()
    for k, v in d.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns


def load_config_safe(config_path):
    """Charge config en gérant les tags Python dans le YAML sauvegardé."""
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Supprimer les tags Python pour permettre le chargement
    content = re.sub(r'!!python/object:argparse\.Namespace\n', '', content)
    content = re.sub(r'!!python/object/apply:torch\.device\n', '', content)
    
    config = yaml.safe_load(content)
    return dict2namespace(config)


def main():
    parser = argparse.ArgumentParser(description="1D Langevin sampling")
    parser.add_argument("--doc", default="uniform_1d")
    parser.add_argument("--ckpt-id", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--out", default="figures/samples_1d.png")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Déterminer le device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = Path("experiments/logs") / args.doc

    if args.ckpt_id is None:
        ckpt_path = log_dir / "checkpoint.pth"
        print("Using final checkpoint: checkpoint.pth")
    else:
        ckpt_path = log_dir / f"checkpoint_{args.ckpt_id}.pth"
        print(f"Using checkpoint: checkpoint_{args.ckpt_id}.pth")

    # Charger config depuis config.yml (pas depuis le checkpoint)
    config_path = Path('configs/uniform_1d.yml')
    if not config_path.exists():
        config_path = log_dir / 'config.yml'
    
    try:
        cfg = load_config_safe(config_path)
    except Exception as e:
        print(f"Erreur lors du chargement de {config_path}: {e}")
        # Fallback: utiliser config originale
        config_path = Path('configs/uniform_1d.yml')
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = dict2namespace(yaml.safe_load(f))
    
    cfg.device = device
    
    print(f"Config: {config_path}")
    print(f"Device: {device}")

    # Charger modèle
    model = MLP1D(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = state[0] if isinstance(state, (list, tuple)) else state
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    
    # Gérer EMA si présent
    if cfg.model.ema and len(state) > 4:
        from models.ema import EMAHelper
        ema = EMAHelper(mu=cfg.model.ema_rate)
        ema.register(model)
        ema.load_state_dict(state[-1])
        ema.ema(model)
    
    model.eval()

    # Noise levels
    sigmas = get_sigmas(cfg).cpu().numpy()

    # Sampling parameters
    n_steps_each = cfg.sampling.n_steps_each
    step_lr = cfg.sampling.step_lr
    denoise = cfg.sampling.denoise

    # Initial distribution (large interval, )
    xmin, xmax = -3.0, 4.0
    init = torch.rand(args.n_samples, 1, device=device)
    init = init * (xmax - xmin) + xmin

    # Langevin sampling
    with torch.no_grad():
        samples = anneal_Langevin_dynamics(
            init,
            model,
            sigmas,
            n_steps_each=n_steps_each,
            step_lr=step_lr,
            final_only=True,
            denoise=denoise,
        )[-1]

    samples = samples.cpu().numpy().squeeze()

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(samples, bins=100, density=True, alpha=0.7)
    ax.set_title("1D Langevin samples")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
