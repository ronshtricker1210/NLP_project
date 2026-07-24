# gcp-setup

GCP single-GPU port of `server-setup`, for the **"Silent Tax" typo project**. This runs the
same inference + scoring pipeline on the `idonlpinstance1` GCP instance (one **NVIDIA L4**)
instead of the TAU Slurm cluster. `server-setup/` is left untouched for the teammates who run
on TAU.

**Start here:** [GCP_RUNBOOK.md](GCP_RUNBOOK.md) — full context (what we did + why), how to run,
data locations, timing, scoring, running when SSH drops, and troubleshooting. This README is the
short reference + flag table.

## What's different from `server-setup`

| | `server-setup` (TAU cluster) | `gcp-setup` (this instance) |
|---|---|---|
| GPU | 2× RTX 2080 Ti (sm_75, no bf16) | 1× **L4** (Ada sm_89, **bf16** native) |
| `tensor_parallel_size` | 2 | **1** |
| precision | fp16 + `enforce_eager` + flashinfer off | **bfloat16**, cudagraphs on |
| env | miniconda on `/vol/scratch` (purges) | **pip `--user`** in `~/.local` |
| model + data | `/vol/scratch/$USER/hf_cache` | default **`~/.cache/huggingface`** |
| scheduler | Slurm `sbatch` | run directly (no Slurm) |
| network | compute nodes offline | has internet |

## Files

| File | Purpose |
|------|---------|
| `setup_gcp.sh` | One-time install (vLLM + libs into `~/.local`) and verify model/datasets are cached. |
| `env.sh` | `source` it before running — sets `HF_HOME`, `PATH`, resolves `MODEL_PATH`. |
| `run.sh` | Wrapper: inference (passthrough flags) → scoring. |
| `run_typo_vllm.py` | **Primary** engine. vLLM, single-L4 bf16. Enhanced CLI (below). |
| `run_typo.py` | Transformers fallback (slower; use if vLLM ever misbehaves). |
| `score.py` | CPU scoring: accuracy per config + flip rates vs the clean baseline. Handles multi-sample. |
| `vllm_smoke.py` | Minimal load + one-prompt test to confirm the GPU/install work. |

## Quickstart

```bash
# 1. one-time install (~a few min; model + datasets are already downloaded)
bash gcp-setup/setup_gcp.sh

# 2. smoke test the engine
source gcp-setup/env.sh && python3 gcp-setup/vllm_smoke.py     # expect VLLM_SMOKE_OK

# 3. first real run: 20 questions of GSM8K, clean baseline + a typo sweep, then score
bash gcp-setup/run.sh --dataset gsm8k --variant both --limit 20
```

Results are written to `~/nlp_project/results/<dataset>_<config>.jsonl` (one clean baseline file
`<dataset>_clean.jsonl` plus one per typo config).

## CLI flags (`run_typo_vllm.py`)

| flag | default | meaning |
|------|---------|---------|
| `--dataset` | `gsm8k` | `gsm8k` \| `math500` \| `gpqa` \| a comma list \| **`all`** |
| `--configs` | `real0,real30,real70` | comma list of typo configs, or **`all`** (every config on the Hub) |
| `--variant` | `typo` | `typo` \| `clean` \| **`both`** (clean baseline once + typo per config) |
| `--limit` | `0` (all) | questions per config |
| `--n-samples` | `1` | generations per question (for averaging / majority-vote flips) |
| `--max-new-tokens` | `4096` | generation budget |
| `--temperature` / `--top-p` | `0.6` / `0.95` | sampling (DeepSeek-R1 recommended) |
| `--tp` | `1` | GPUs (L4 = 1) |
| `--dtype` | `bfloat16` | native on Ada |
| `--gpu-mem-util` | `0.90` | vLLM VRAM fraction |
| `--enforce-eager` | off | disable cudagraphs (only if VRAM is tight) |

Config naming: `real{N}` = N% of the typos are real-word typos (e.g. `real0` = all non-word,
`real70` = 70% real-word). `rate{R}_real{N}` variants also exist (typo density R%). `--configs all`
picks up every one.

## Notes

- The model (`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, ~15 GB, bf16) and all three typo datasets
  are already cached under `~/.cache/huggingface`, so runs work offline too (`HF_HUB_OFFLINE=1`).
- Deps are installed with `pip --user` (no root, no conda). `~/.local/bin` is added to `PATH` by
  `env.sh`.
- One L4 (23 GB) holds the 7B in bf16 with room for KV cache. If you ever hit OOM, lower
  `--gpu-mem-util`, reduce `--max-new-tokens`, or add `--enforce-eager`.
