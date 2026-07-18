#!/bin/bash
#SBATCH --job-name=typo_smoke
#SBATCH --output=typo_smoke_%j.out
#SBATCH --error=typo_smoke_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$PWD/python_pkgs:${PYTHONPATH:-}"
export NLTK_DATA="$PWD/nltk_data"
export HF_HOME="$PWD/hf_home"

mkdir -p "$NLTK_DATA" "$HF_HOME"

python3 - <<'PY'
from pathlib import Path

import nltk

from typo_pipeline import Config, main

nltk.download("words", download_dir=str(Path("nltk_data").resolve()), quiet=True)

main(Config(
    subset_size=5,
    num_proc=1,
    allow_offline_fallback=False,
    out_dir=Path("typo_dataset_smoke").resolve(),
))
PY

echo "Smoke run done. Output written to ./typo_dataset_smoke/"