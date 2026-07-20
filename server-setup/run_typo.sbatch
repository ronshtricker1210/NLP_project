#!/bin/bash
#SBATCH --job-name=typo
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --cpus-per-task=4
#SBATCH --mem=50000
#SBATCH --time=1440
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Usage (positional args): DATASET  CONFIGS  LIMIT  [MAXTOK]
#   sbatch run_typo.sbatch gsm8k real10,real40,real70 5
#   sbatch run_typo.sbatch math500 real10,real40,real70 0     # 0 = full dataset

set -e
DATASET="${1:-gsm8k}"
CONFIGS="${2:-real10,real40,real70}"
LIMIT="${3:-0}"
MAXTOK="${4:-4096}"

source ~/nlp_project/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd ~/nlp_project/scripts

echo "node: $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# --- load the model. On this cluster the scratch NFS read (~8-12 min) is the bottleneck
#     whether we copy to /tmp or load directly, so we load DIRECTLY from scratch (simpler).
#     Set COPY_LOCAL=1 to instead copy to node-local /tmp first (only helps if reloading).
SRC=$(ls -d /vol/scratch/$USER/hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/*/)
if [ "${COPY_LOCAL:-0}" = "1" ]; then
  export MODEL_PATH=/tmp/$USER/dsr1
  mkdir -p "$MODEL_PATH"; cp -rL "$SRC"/. "$MODEL_PATH"/
else
  export MODEL_PATH="$SRC"
fi

python run_typo.py --dataset "$DATASET" --configs "$CONFIGS" \
       --limit "$LIMIT" --max-new-tokens "$MAXTOK"

rm -rf /tmp/$USER 2>/dev/null || true
