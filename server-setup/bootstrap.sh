#!/bin/bash
# Rebuild the disposable scratch environment (conda + torch + model + datasets) after a
# /vol/scratch purge. Run on the LOGIN node (needs internet). Idempotent: skips done steps.
#   ssh <user>@slurm-client.cs.tau.ac.il ; bash ~/nlp_project/bootstrap.sh
set -e
export STORAGE=/vol/scratch/$USER
export HF_HOME=$STORAGE/hf_cache
export PIP_CACHE_DIR=$STORAGE/pip_cache
export TMPDIR=$STORAGE/tmp
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$STORAGE/conda_pkgs"

echo ">>> [1/5] miniconda"
if [ ! -f $STORAGE/miniconda/etc/profile.d/conda.sh ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $STORAGE/miniconda.sh
  bash $STORAGE/miniconda.sh -b -p $STORAGE/miniconda
fi
source $STORAGE/miniconda/etc/profile.d/conda.sh
conda config --add pkgs_dirs $STORAGE/conda_pkgs 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo ">>> [2/5] conda env"
[ -d $STORAGE/envs/typo ] || conda create -y -p $STORAGE/envs/typo python=3.11
conda activate $STORAGE/envs/typo

echo ">>> [3/5] python deps (torch cu128 + libs)"
python -c "import torch; assert 'cu128' in torch.__version__" 2>/dev/null || \
  pip install --force-reinstall "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
pip install -q -U "huggingface_hub[cli]" transformers accelerate datasets
pip uninstall -y hf_xet 2>/dev/null || true

echo ">>> [4/5] model download"
D=$HF_HOME/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B
SZ=$(du -sm "$D/blobs" 2>/dev/null | cut -f1 || echo 0)
if [ "${SZ:-0}" -lt 14000 ]; then hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B; fi
find "$D" -name '*.incomplete' -delete 2>/dev/null || true

echo ">>> [5/6] cache typo datasets (compute nodes may run offline, so pre-cache here)"
python - <<'PY'
from datasets import load_dataset
repos = ["idoazou/gsm8k-typos", "idoazou/math500-typos", "idoazou/gpqa-typos"]
cfgs  = ["real0", "real10", "real40", "real70"]
for r in repos:
    for c in cfgs:
        load_dataset(r, c, split="test")
print("datasets cached")
PY

echo ">>> [6/6] separate vLLM env (fast inference). Pinned to versions that work on RTX 2080 Ti."
# vLLM 0.11.0 -> torch 2.8.0+cu128 (matches driver); needs transformers 4.x (5.x breaks its tokenizer).
# Runtime flags fp16 + enforce_eager + VLLM_USE_FLASHINFER_SAMPLER=0 live in run_typo_vllm.sbatch.
[ -d $STORAGE/envs/vllm ] || conda create -y -p $STORAGE/envs/vllm python=3.11
conda activate $STORAGE/envs/vllm
python -c "import vllm" 2>/dev/null || \
  pip install "vllm==0.11.0" --extra-index-url https://download.pytorch.org/whl/cu128
pip install -q "transformers>=4.55,<5" datasets      # force transformers 4.x for vLLM + add datasets
python -c "import vllm, torch, transformers, datasets; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"

echo "BOOTSTRAP DONE"
