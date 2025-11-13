#!/usr/bin/env bash
set -e
python src/arz_pinn/eval_main.py \
  --config configs/config_ngsim.yaml \
  --ckpt checkpoints/ngsim/final_model.pt \
  --cache data/ngsim/processed/grid_cache.npz \
  --out outputs/ngsim/metrics.csv
