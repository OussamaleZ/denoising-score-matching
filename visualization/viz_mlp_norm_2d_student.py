import argparse
import sys
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mlp_2d import MLP2D


def namespaceify(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: namespaceify(v) for k, v in value.items()})
    return value


def resolve_config_path(cfg: str) -> Path:
    path = Path(cfg)
    if path.is_file():
        return path
    candidate = PROJECT_ROOT / "configs" / cfg
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Config not found: {cfg}")


def load_config_safe(config_path: Path) -> SimpleNamespace:
    """Charge config en gérant les tags Python dans le YAML sauvegardé."""
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Supprimer les tags Python pour permettre le chargement
    content = re.sub(r'!!python/object:argparse\.Namespace\n', '', content)
    content = re.sub(r'!!python/object/apply:torch\.device\n', '', content)
    
    config = yaml.safe_load(content)
    return namespaceify(config)


def load_config(cfg_path: Path) -> SimpleNamespace:
    try:
        return load_config_safe(cfg_path)
    except:
        with open(cfg_path, "r", encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return namespaceify(cfg)


def load_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, (list, tuple)):
        ckpt = ckpt[0]
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state_dict", "net", "ema"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint: {ckpt_path}")
    if any(key.startswith("module.") for key in ckpt):
        ckpt = {key[len("module.") :]: value for key, value in ckpt.items()}
    return ckpt


def build_model(cfg: SimpleNamespace) -> MLP2D:
    return MLP2D(cfg)


def resolve_checkpoint(args: argparse.Namespace, config: SimpleNamespace) -> Path:
    if args.ckpt:
        path = Path(args.ckpt)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path

    doc = args.doc or Path(args.config).stem
    
    # Utiliser checkpoint.pth par défaut, sauf si --ckpt-id est explicitement spécifié
    if args.ckpt_id is not None:
        filename = f"checkpoint_{args.ckpt_id}.pth"
    else:
        filename = "checkpoint.pth"
    
    path = Path(args.exp) / "logs" / doc / filename
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ||MLP2D(x)|| over [-7,17]×[-7,17].")
    parser.add_argument("--config", default="student_mixture_2d.yml", help="Config file name or path.")
    parser.add_argument("--exp", default="experiments", help="Experiment root.")
    parser.add_argument("--doc", default=None, help="Run name under logs/. Defaults to config stem.")
    parser.add_argument("--ckpt-id", type=int, default=None, help="Checkpoint suffix after checkpoint_.")
    parser.add_argument("--ckpt", default=None, help="Direct path to a checkpoint.")
    parser.add_argument("--n", type=int, default=300, help="Grid resolution per axis.")
    parser.add_argument("--out", default="mlp_norm.png", help="Output image path.")
    parser.add_argument("--sigma-idx", type=int, default=0, help="Noise level index (0 = highest noise).")
    return parser.parse_args()


def main():
    minx, maxx = -10.0, 20.0
    miny, maxy = -20.0, 20.0

    args = parse_args()
    cfg = load_config(resolve_config_path(args.config))
    ckpt_path = resolve_checkpoint(args, cfg)
    state_dict = load_state_dict(ckpt_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.device = device

    model = MLP2D(cfg).to(device)
    model.load_state_dict(state_dict, strict=False)
    
    # Gérer EMA si présent
    states = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if cfg.model.ema and len(states) > 4:
        from models.ema import EMAHelper
        ema = EMAHelper(mu=cfg.model.ema_rate)
        ema.register(model)
        ema.load_state_dict(states[-1])
        ema.ema(model)
    
    model.eval()

    xs = torch.linspace(minx, maxx, args.n)
    ys = torch.linspace(miny, maxy, args.n)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), dim=-1).reshape(-1, 2).to(device)
    
    # Labels pour le niveau de bruit (sigma index)
    labels = torch.full((grid.shape[0],), args.sigma_idx, dtype=torch.long, device=device)

    with torch.no_grad():
<<<<<<< HEAD
        outputs = model(grid)
=======
        outputs = model(grid, labels)
>>>>>>> f2e4032 (Add toy experiments (1D uniform, 2D student) and figures)
        
    outputs = outputs if outputs.ndim > 1 else outputs[:, None]
    norms = torch.linalg.norm(outputs, dim=1).reshape(args.n, args.n).cpu()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        norms.numpy(),
        origin="lower",
        extent=[minx, maxx, miny, maxy],
        cmap="viridis",
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label=r"$\|\mathrm{MLP2D}(x)\|$")
    ax.set(xlabel="x", ylabel="y", title="MLP2D output norm on [{minx},{maxx}]×[{miny},{maxy}] (σ_idx={sigma_idx})".format(minx=minx, maxx=maxx, miny=miny, maxy=maxy, sigma_idx=args.sigma_idx))
    fig.tight_layout()
    Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
