export STORAGE=/vol/scratch/$USER
export HF_HOME=$STORAGE/hf_cache
export TMPDIR=$STORAGE/tmp
source $STORAGE/miniconda/etc/profile.d/conda.sh
conda activate $STORAGE/envs/typo
