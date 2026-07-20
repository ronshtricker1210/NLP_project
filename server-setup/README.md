# server-setup

Backup of the TAU Slurm cluster setup + inference pipeline for the "Silent Tax" typo project.

**Start here:** [CLUSTER_RUNBOOK.md](CLUSTER_RUNBOOK.md) — full step-by-step guide to run the model
on the TAU CS cluster from a fresh account (SSH, storage, conda, model download, GPUs, jobs,
troubleshooting). It captures every gotcha we hit.

## Files

### Setup / environment
| File | Purpose |
|------|---------|
| `CLUSTER_RUNBOOK.md` | The main guide. Read this first. |
| `bootstrap.sh` | One command to rebuild everything after a `/vol/scratch` purge: both conda envs (transformers + vLLM 0.11.0), the model, and the datasets. |
| `env.sh` | Sourced to activate the scratch conda env (kept in home, persistent). |
| `setup_and_download.sh` | Original one-shot setup + model download (superseded by `bootstrap.sh`). |

### Inference pipeline
| File | Purpose |
|------|---------|
| `run_typo.py` / `run_typo.sbatch` | Transformers pipeline: batched generation of reasoning traces over the typo datasets → JSONL. Fallback engine. |
| `run_typo_vllm.py` / `run_typo_vllm.sbatch` | vLLM pipeline: same JSONL output, much faster throughput. Primary engine. |
| `score.py` | Reads the JSONL traces, extracts predicted answers (`\boxed{}` + fallbacks), computes accuracy per config and flip rates vs the clean baseline. Runs on CPU. |
| `vllm_smoke.py` | Minimal vLLM load + one-prompt test to verify the GPU works. |
| `ask_local.py` / `ask_local.sbatch` | Single-question demo (full chain-of-thought), with the node-local model-copy trick. |
| `run_inference.py` / `run_inference.sbatch` | Early smoke test (superseded). |

## Key cluster facts (see runbook for details)
- `/vol/scratch` **auto-purges** — keep code/results in home, rebuild scratch with `bootstrap.sh`.
- Student GPUs are **RTX 2080 Ti only** (Titan Xp too old). Need **2 GPUs** for the 7B model.
- torch must be **cu128** (not cu130); vLLM needs **fp16 + enforce_eager** on the 2080 Ti.
- Datasets: `idoazou/{gsm8k,math500,gpqa}-typos`, configs `real0..real70` (real-word typo ratio).

## Run recipe
```bash
# one-time (or after a scratch purge)
bash ~/nlp_project/bootstrap.sh

# generate reasoning traces (vLLM), 75 questions per config
sbatch run_typo_vllm.sbatch gsm8k real10,real40,real70 75

# score
python score.py --dataset gsm8k
```
