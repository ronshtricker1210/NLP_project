# Source before running anything on the GCP L4 instance:  source gcp-setup/env.sh
# (Unlike the TAU cluster there is no conda / no /vol/scratch — deps live in ~/.local,
#  and the model + datasets live in the default HF cache ~/.cache/huggingface.)

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PATH="$HOME/.local/bin:$PATH"        # pip --user console scripts (hf, huggingface-cli, ...)
export PYTHONUNBUFFERED=1

# Everything (model + all dataset configs) is already cached, so run fully offline: no per-load
# metadata ping to the Hub, no "unauthenticated requests" warning, guaranteed no re-download.
# setup_gcp.sh warms the Arrow cache for every config so this is safe. If you ever ADD a new
# config on the Hub, `unset HF_HUB_OFFLINE` for one run (or re-run setup_gcp.sh) to fetch it.
export HF_HUB_OFFLINE=1

# Resolve the cached model snapshot into MODEL_PATH (optional: the runners also accept the
# repo id and find it in the cache themselves). Silently no-op if not downloaded yet.
_snap="$(ls -d "$HF_HOME"/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/*/ 2>/dev/null | head -1)"
[ -n "$_snap" ] && export MODEL_PATH="$_snap"
unset _snap

# NOTE: this bare VM shipped with CUDA 12.9 + gcc but no g++/make/python3-dev, so vLLM's
# flashinfer + torch.compile runtime JIT failed. setup_gcp.sh installs those via apt; once
# present, the default full-speed path (cudagraphs + flashinfer) works with no extra env.
# If you ever run on a box that lacks the C++ toolchain, set:
#   export VLLM_ENFORCE_EAGER=1 VLLM_USE_FLASHINFER_SAMPLER=0
# to fall back to the no-JIT path (the runners honor VLLM_ENFORCE_EAGER).
