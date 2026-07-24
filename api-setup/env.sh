# source api-setup/env.sh before running — validates the API token and sets defaults.
# (API twin of gcp-setup/env.sh: no HF_HOME / MODEL_PATH here, nothing is cached
#  locally — the model runs on Nscale's GPUs behind router.huggingface.co.)

if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: HF_TOKEN is not set."
  echo "  Create a fine-grained token with 'Make calls to Inference Providers' at"
  echo "  https://huggingface.co/settings/tokens then:  export HF_TOKEN=hf_..."
  return 1 2>/dev/null || exit 1
fi

export API_BASE="${API_BASE:-https://router.huggingface.co/v1}"
export MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:nscale}"
export OUTDIR="${OUTDIR:-$HOME/nlp_project/results}"

echo "env OK  model=$MODEL  outdir=$OUTDIR"
