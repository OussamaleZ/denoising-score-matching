import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_DIR = PROJECT_ROOT / "ncsnv2" / "configs"
from ncsnv2.models.mlp import MLP


def namespaceify(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: namespaceify(v) for k, v in value.items()})
    return value


def resolve_config_path(cfg: str) -> Path:
    path = Path(cfg)
    if path.is_file():
        return path
    candidate = CONFIG_DIR / cfg
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Config not found: {cfg}")


def load_config(cfg_path: Path) -> SimpleNamespace:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return namespaceify(cfg)


def load_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
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


def build_model(cfg: SimpleNamespace) -> MLP:
    return MLP(cfg)


def resolve_checkpoint(args: argparse.Namespace, config: SimpleNamespace) -> Path:
    if args.ckpt:
        path = Path(args.ckpt)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path

    doc = args.doc or Path(args.config).stem
    ckpt_id = args.ckpt_id
    sampling = getattr(config, "sampling", None)
    if ckpt_id is None and sampling:
        ckpt_id = getattr(sampling, "ckpt_id", None)

    filename = f"checkpoint_{ckpt_id}.pth" if ckpt_id is not None else "checkpoint.pth"
    path = Path(args.exp) / "logs" / doc / filename
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ||MLP(x)|| over [-1,2]×[-1,2].")
    parser.add_argument("--config", default="uniform_2d.yml", help="Config file name or path.")
    parser.add_argument("--exp", default="/Data/ncsnv2_checkpoints/exp/exp", help="Experiment root.")
    parser.add_argument("--doc", default=None, help="Run name under logs/. Defaults to config stem.")
    parser.add_argument("--ckpt-id", type=int, default=None, help="Checkpoint suffix after checkpoint_.")
    parser.add_argument("--ckpt", default=None, help="Direct path to a checkpoint.")
    parser.add_argument("--n", type=int, default=300, help="Grid resolution per axis.")
    parser.add_argument("--out", default="mlp_norm.png", help="Output image path.")
    return parser.parse_args()


def main():
    minx, maxx = -5.0, 6.0
    miny, maxy = -5.0, 6.0

    args = parse_args()
    cfg = load_config(resolve_config_path(args.config))
    ckpt_path = resolve_checkpoint(args, cfg)
    state_dict = load_state_dict(ckpt_path)

    model = MLP(cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    xs = torch.linspace(minx, maxx, args.n)
    ys = torch.linspace(miny, maxy, args.n)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), dim=-1).reshape(-1, 2)

    with torch.no_grad():
        outputs = model(grid)
        
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
    fig.colorbar(im, ax=ax, label=r"$\|\mathrm{MLP}(x)\|$")
    ax.set(xlabel="x", ylabel="y", title="MLP output norm on [{minx},{maxx}]×[{miny},{maxy}]".format(minx=minx, maxx=maxx, miny=miny, maxy=maxy))
    fig.tight_layout()
    Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
