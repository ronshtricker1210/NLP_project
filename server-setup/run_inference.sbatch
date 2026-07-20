#!/bin/bash
#SBATCH --job-name=typo-infer
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Submit with:  sbatch run_inference.sbatch
# studentkillable: up to 1 day, low priority (can be preempted).
# Quick interactive test:
#   srun --account=gpu-students --partition=studentkillable --gres=gpu:1 --pty bash

set -euo pipefail
mkdir -p logs

# --- same scratch storage used during download; compute nodes are offline ----
export STORAGE="${STORAGE:-/vol/scratch/$USER}"
export HF_HOME="$STORAGE/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source "$STORAGE/envs/typo/bin/activate"

echo "Node: $(hostname)"
nvidia-smi || true

python run_inference.py
