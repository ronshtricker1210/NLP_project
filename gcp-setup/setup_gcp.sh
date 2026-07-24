#!/bin/bash
# One-time setup on the GCP L4 instance (idonlpinstance1).
# Installs the Python deps into ~/.local (no conda, no venv, no root needed) and verifies the
# model + typo datasets are cached under ~/.cache/huggingface. Idempotent — safe to re-run.
#
#   bash gcp-setup/setup_gcp.sh
set -e
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

echo ">>> [0/5] system build toolchain (this base image has CUDA + gcc but no g++/make/python3-dev)"
# vLLM's flashinfer + torch.compile JIT-compile CUDA/C++ at runtime and need the full toolchain.
# Needs sudo; skip cleanly if unavailable (then export VLLM_ENFORCE_EAGER=1 to use the no-JIT path).
if sudo -n true 2>/dev/null; then
  sudo apt-get update -qq || true
  sudo apt-get install -y --no-install-recommends g++ make python3-dev libpython3.10-dev || \
    echo "  [warn] toolchain install failed — set VLLM_ENFORCE_EAGER=1 VLLM_USE_FLASHINFER_SAMPLER=0"
else
  echo "  [warn] no sudo — skipping. If vLLM JIT fails, set VLLM_ENFORCE_EAGER=1 VLLM_USE_FLASHINFER_SAMPLER=0"
fi

echo ">>> [1/5] pip"
python3 -m pip --version >/dev/null 2>&1 || curl -sSL https://bootstrap.pypa.io/get-pip.py | python3 - --user

echo ">>> [2/5] vLLM (pulls a CUDA-12.x torch that runs on this driver) + libs"
# vLLM brings its own torch + a compatible transformers; install it first, then the rest.
python3 -m pip install --user -q -U vllm
python3 -m pip install --user -q -U "huggingface_hub[hf_transfer]" datasets math_verify accelerate

echo ">>> [3/5] sanity import"
python3 - <<'PY'
import torch, vllm, transformers, datasets
print("  torch", torch.__version__, "| cuda", torch.version.cuda, "| gpu?", torch.cuda.is_available())
print("  vllm", vllm.__version__, "| transformers", transformers.__version__, "| datasets", datasets.__version__)
PY

echo ">>> [4/5] model present? (${MODEL})"
python3 - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("${MODEL}")   # instant if already cached; downloads (~15GB) otherwise
print("  model ->", p)
PY

echo ">>> [5/5] cache the default typo-dataset configs"
python3 - <<'PY'
from huggingface_hub import snapshot_download
from datasets import load_dataset
repos = ["idoazou/gsm8k-typos", "idoazou/math500-typos", "idoazou/gpqa-typos"]
for r in repos:
    snapshot_download(r, repo_type="dataset")            # all configs' parquet (enables --configs all)
    for c in ["clean", "real0", "real30", "real70"]:     # warm the default sweep's arrow cache
        try: load_dataset(r, c, split="test")
        except Exception as e: print("   warn", r, c, type(e).__name__)
print("  datasets cached")
PY

echo "SETUP DONE — next:  source gcp-setup/env.sh && bash gcp-setup/run.sh --dataset gsm8k --variant both --limit 20"
