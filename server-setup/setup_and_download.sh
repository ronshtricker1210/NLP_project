#!/bin/bash
# Run on the TAU CS login node (slurm-client.cs.tau.ac.il), NOT inside a Slurm job.
# Login nodes reach the internet; compute nodes generally do not.
#
# Usage:
#   ssh shiranhamami@slurm-client.cs.tau.ac.il
#   bash setup_and_download.sh
#
set -euo pipefail

# --- Everything lives on /vol/scratch (home quota is only 6 GB). --------------
export STORAGE="${STORAGE:-/vol/scratch/$USER}"
export HF_HOME="$STORAGE/hf_cache"
export PIP_CACHE_DIR="$STORAGE/pip_cache"
export TMPDIR="$STORAGE/tmp"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
echo ">> STORAGE=$STORAGE"

# --- Python venv on scratch (no conda on this cluster) -----------------------
ENV="$STORAGE/envs/typo"
if [ ! -d "$ENV" ]; then
  echo ">> Creating venv at $ENV"
  python3 -m venv "$ENV"
fi
source "$ENV/bin/activate"
pip install -U pip
pip install "huggingface_hub[cli]" torch transformers accelerate

# --- Download the model (open weights, resumable, no token) ------------------
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
echo ">> Downloading $MODEL into $HF_HOME"
hf download "$MODEL"

echo ""
echo ">> DONE. In job scripts set:"
echo "     export STORAGE=$STORAGE"
echo "     export HF_HOME=$HF_HOME"
echo "     export HF_HUB_OFFLINE=1"
echo "     source $ENV/bin/activate"
