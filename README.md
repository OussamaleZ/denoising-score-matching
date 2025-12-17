# PGMs Project — Denoising Score Matching (NCSNv2)

Course project for **Probabilistic Graphical Models**. The code is adapted from the official NCSNv2 implementation from the paper *Improved Techniques for Training Score-Based Generative Models* (Song & Ermon, 2020).

## What’s inside
- `ncsnv2/`: training/sampling code (`main.py`), configs, models, datasets, losses.
- `ncsnv2/notebooks/`: project notebook(s) and figures.

## Setup
From the repo root:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r ncsnv2/requirements.txt
```

Install PyTorch separately (CPU/GPU depending on your machine): https://pytorch.org/get-started/locally/

## Run (train / sample)
Run from `ncsnv2/` (configs are loaded via a relative `configs/` path):
```bash
cd ncsnv2

# Train (example: 2D uniform toy dataset)
python main.py --config uniform_2d.yml --doc uniform2d --exp ./exp --ni

# Sample (set `sampling.ckpt_id` in the config first)
python main.py --sample --config uniform_2d.yml --doc uniform2d --exp ./exp -i uniform2d --ni
```

## Notes
- Outputs go under `--exp` (logs, checkpoints, tensorboard, samples).
- See `ncsnv2/README.md` for project-specific details and credits.

