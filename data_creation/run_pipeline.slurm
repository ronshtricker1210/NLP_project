#!/bin/bash
#SBATCH --job-name=typo_pipeline
#SBATCH --output=typo_pipeline_%j.out
#SBATCH --error=typo_pipeline_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
# --- EDIT THESE for your TAU cluster account/partition ---
# #SBATCH --partition=<your_partition>
# #SBATCH --account=<your_account>

# Template Slurm batch script for building the typo dataset.
# Submit with:  sbatch run_pipeline.slurm

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# 1) Environment ------------------------------------------------------------
# Use whatever your cluster provides (module or conda/venv). Example:
# module load python/3.10
# source ~/venvs/typo/bin/activate

# 2) Dependencies (first run only; needs network on a login node) -----------
pip install -r requirements.txt
python -c "import nltk; nltk.download('words')"

# 3) Run --------------------------------------------------------------------
# Match num_proc to --cpus-per-task above. For the full run, set
# subset_size=500 and allow_offline_fallback=False.
python - <<'PY'
from typo_pipeline import Config, main
main(Config(
    subset_size=500,
    num_proc=8,
    allow_offline_fallback=False,
))
PY

echo "Done. Output written to ./typo_dataset/"
