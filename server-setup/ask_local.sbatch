#!/bin/bash
#SBATCH --job-name=asklocal
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --cpus-per-task=4
#SBATCH --mem=50000
#SBATCH --time=25
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
source ~/nlp_project/env.sh   # persistent (home); activates the scratch conda env
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
echo "node: $(hostname)"; df -h /tmp | tail -1

SRC=$(ls -d /vol/scratch/$USER/hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/*/)
DST=/tmp/$USER/dsr1
mkdir -p "$DST"
echo "copying model to node-local $DST ..."
t0=$(date +%s)
cp -rL "$SRC"/. "$DST"/
echo "copy took $(( $(date +%s) - t0 )) s"
export MODEL_PATH="$DST"
python ask_local.py
rm -rf /tmp/$USER
