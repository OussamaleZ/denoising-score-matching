"""
Sample 2D points from a trained MLP score model using annealed Langevin dynamics.

Usage:
  python -m ncsnv2.visualization.sampling_2d --config uniform_2d.yml --doc <run> --ckpt-id 8000
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncsnv2.models import anneal_Langevin_dynamics, get_sigmas
from ncsnv2.visualization.viz_mlp_norm_2d import (
    build_model,
    load_config,
    load_state_dict,
    resolve_checkpoint,
    resolve_config_path,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Langevin sampling for 2D MLP score model.")
    parser.add_argument("--config", default="uniform_2d.yml", help="Config file name or path.")
    parser.add_argument("--exp", default="/Data/ncsnv2_checkpoints/exp/exp", help="Experiment root.")
    parser.add_argument("--doc", default=None, help="Run name under logs/. Defaults to config stem.")
    parser.add_argument("--ckpt-id", type=int, default=None, help="Checkpoint suffix after checkpoint_.")
    parser.add_argument("--ckpt", default=None, help="Direct path to a checkpoint.")
    parser.add_argument("--device", default="cpu", help="Device for model and sampling (cpu|cuda).")

    parser.add_argument("--batch-size", type=int, default=None, help="Number of 2D points to sample.")
    parser.add_argument("--n-steps-each", type=int, default=None, help="Langevin steps per noise level.")
    parser.add_argument("--step-lr", type=float, default=None, help="Base step size for Langevin updates.")
    parser.add_argument("--denoise", action="store_true", help="Apply final denoising step.")
    parser.add_argument("--no-denoise", action="store_true", help="Disable final denoising step.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument("--out", default="figures/samples_2d.png", help="Output image path.")
    parser.add_argument("--out-samples", default=None, help="Optional path to save samples as a .pt tensor.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    cfg_path = resolve_config_path(args.config)
    cfg = load_config(cfg_path)
    cfg.device = device

    ckpt_path = resolve_checkpoint(args, cfg)
    state_dict = load_state_dict(ckpt_path)

    model = build_model(cfg).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    sigmas_th = get_sigmas(cfg)
    sigmas = sigmas_th.cpu().numpy()

    batch_size = int(args.batch_size or getattr(cfg.sampling, "batch_size", 4096))
    n_steps_each = int(args.n_steps_each or getattr(cfg.sampling, "n_steps_each", 10))
    step_lr = float(args.step_lr or getattr(cfg.sampling, "step_lr", 5e-5))

    if args.denoise and args.no_denoise:
        raise ValueError("Use only one of --denoise or --no-denoise.")
    denoise = getattr(cfg.sampling, "denoise", True)
    if args.denoise:
        denoise = True
    if args.no_denoise:
        denoise = False

    xmin = float(getattr(cfg.data, "xmin", -1.0)) - 1.0 
    xmax = float(getattr(cfg.data, "xmax", 1.0)) + 1.0
    ymin = float(getattr(cfg.data, "ymin", -1.0)) - 1.0
    ymax = float(getattr(cfg.data, "ymax", 1.0)) + 1.0

    init = torch.rand(batch_size, 2, device=device)
    init[:, 0] = init[:, 0] * (xmax - xmin) + xmin
    init[:, 1] = init[:, 1] * (ymax - ymin) + ymin

    # Mimic runner's "data_init": start from data and add the largest noise.
    data_init = bool(getattr(cfg.sampling, "data_init", False))
    if data_init:
        init = init + sigmas_th[0] * torch.randn_like(init)

    all_samples = anneal_Langevin_dynamics(
        init,
        model,
        sigmas,
        n_steps_each=n_steps_each,
        step_lr=step_lr,
        final_only=True,
        verbose=False,
        denoise=denoise,
    )

    samples = all_samples[-1]
    if args.out_samples:
        out_samples = Path(args.out_samples).expanduser()
        out_samples.parent.mkdir(parents=True, exist_ok=True)
        torch.save(samples, out_samples)

    pts = samples.numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.5, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Langevin samples")
    fig.tight_layout()

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"Saved samples plot to {out}")


if __name__ == "__main__":
    main()
