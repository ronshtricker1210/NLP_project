#!/bin/bash
#SBATCH --job-name=typovllm
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

# Usage: DATASET  CONFIGS  LIMIT  [MAXTOK]  [VARIANT]
#   typo runs:  sbatch run_typo_vllm.sbatch gsm8k real10,real40,real70 75
#   clean base: sbatch run_typo_vllm.sbatch gsm8k real10 75 4096 clean

set -e
DATASET="${1:-gsm8k}"; CONFIGS="${2:-real10,real40,real70}"; LIMIT="${3:-0}"
MAXTOK="${4:-4096}"; VARIANT="${5:-typo}"

# uses the dedicated vllm env (kept separate so it can't break the transformers env)
source /vol/scratch/$USER/miniconda/etc/profile.d/conda.sh
conda activate /vol/scratch/$USER/envs/vllm
# RTX 2080 Ti has no nvcc: disable flashinfer's JIT-compiled sampler
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 HF_HOME=/vol/scratch/$USER/hf_cache
cd ~/nlp_project/scripts

echo "node: $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader | head -1
export MODEL_PATH=$(ls -d /vol/scratch/$USER/hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/*/)

python run_typo_vllm.py --dataset "$DATASET" --configs "$CONFIGS" \
       --limit "$LIMIT" --max-new-tokens "$MAXTOK" --variant "$VARIANT" --tp 2
