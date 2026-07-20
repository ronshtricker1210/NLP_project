#!/bin/bash
#SBATCH --job-name=vllminstall
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --time=40
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --output=/a/home/cc/students/cs/%u/install_vllm.out
set -e
export STORAGE=/vol/scratch/$USER
export PIP_CACHE_DIR=$STORAGE/pip_cache TMPDIR=$STORAGE/tmp
source $STORAGE/miniconda/etc/profile.d/conda.sh
conda activate $STORAGE/envs/vllm
echo "node: $(hostname); installing vllm..."
pip install vllm
python -c "import vllm, torch; print('OK vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"
echo "VLLM_INSTALL_DONE"
