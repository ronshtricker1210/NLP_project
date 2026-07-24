# api-setup — typo pipeline over the DeepSeek API (no GPU)

API twin of `gcp-setup/`: same datasets, same prompts (identical seeded GPQA
shuffling), same JSONL schema, same `score.py` — but inference happens on
**DeepSeek-R1-Distill-Qwen-7B served by Nscale** through the HuggingFace
Inference Providers router, so there is no GCP instance, no vLLM install, no
model download. A MacBook and a token are enough.

| | gcp-setup (L4) | api-setup |
|---|---|---|
| hardware | 1× NVIDIA L4 on GCP | none (Nscale's GPUs, serverless) |
| setup | `setup_gcp.sh`, ~15 GB model cache | `pip install -r requirements.txt` |
| cost | GPU-hours while the VM is up | per token: $0.01/M in, $0.03/M out |
| MATH-500 full pass | GPU-bound | ~15 min at concurrency 100, ≈ $0.05 |
| auth | — | `HF_TOKEN` (fine-grained, "Make calls to Inference Providers") |

Verified 2026-07-24: one full MATH-500 pass (500 questions, concurrency 500)
completed with zero failures, avg 3,256 reasoning tokens/question, $0.049 at
list prices, 94.6% accuracy under `score.py`/math_verify — matching the model's
published benchmark, so the API serves the same model quality as local vLLM.

## Quick start

```bash
pip install -r api-setup/requirements.txt
export HF_TOKEN=hf_...                    # huggingface.co/settings/tokens
source api-setup/env.sh && python api-setup/api_smoke.py   # one-request sanity check

bash api-setup/run.sh --dataset math500 --variant both --limit 20   # small sweep + scoring
bash api-setup/run.sh --detach --dataset all --configs all --n-samples 5   # the real thing
```

Results land in `$OUTDIR` (default `~/nlp_project/results`) as
`{dataset}_{config}.jsonl` / `{dataset}_clean.jsonl`, scored automatically by
`score.py` at the end of `run.sh`.

## Mirroring results to the Hub

Pass `--hub-repo user/typo-results` (or `export HUB_REPO=...`) and every
completed config file is *also* pushed to a private HF dataset repo, twice
over — the same dual convention as `data_creation/generate_variants.py`:

1. the raw JSONL bytes, laid out by typo rate and real-word ratio:

```
results/{dataset}/clean.jsonl                # clean baseline
results/{dataset}/typo30/real{Y}.jsonl       # plain realY configs (fixed ~30% rate)
results/{dataset}/typo{X}/real{Y}.jsonl      # rateX_realY configs (X = 25/50/75)
```

2. a named dataset config (`push_to_hub(config_name=..., split="test")`), so
   results load exactly like the typo datasets themselves:

```python
from datasets import load_dataset
res = load_dataset("user/typo-results", "math500_typo30_real40", split="test")
```

The repo is created (private) on first use. Uploading needs **write**
permission: the inference-only token can't push, so either extend it
("Write access to contents/settings of repos under your namespace") or set a
separate `HF_WRITE_TOKEN`. A config with failed samples is not pushed until a
rerun completes it — the Hub only ever sees whole files.

## Differences that matter when comparing to GPU runs

- The router applies the chat template server-side; records store the plain
  user prompt in `prompt`. `generation` is reconstructed as
  `reasoning + "</think>" + final` so `score.py` parses it identically.
- Records carry extra fields (`cost_usd`, `latency_s`); `score.py` ignores them.
- Interrupted runs **resume** (finished `(idx, sample_idx)` pairs are skipped) —
  rerun the same command after any failure; nothing is re-billed.
- Nscale's serving precision is unpublished (gcp-setup runs bf16); measured
  MATH-500 accuracy matches, so any difference is negligible.
- `--n-samples N` issues N independent requests per question (the router has
  no batched `n`), which parallelizes freely.

## Known billing caveat (2026-07-24)

HF Inference Providers had a live bug charging a flat $0.01/request instead of
per-token (forum: discuss.huggingface.co/t/178135). Check your first small run
against https://huggingface.co/settings/billing before launching big sweeps;
per-token list prices make a full MATH-500 pass ≈ $0.05, so a bill near
$5/pass means the bug is still active.
