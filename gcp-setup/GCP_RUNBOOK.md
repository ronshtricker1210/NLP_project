# GCP Runbook — "Silent Tax" typo experiments on the L4 instance

Full context + step-by-step guide for running the DeepSeek-R1 typo-robustness experiments on the
GCP instance **`idonlpinstance1`** (single NVIDIA L4). This is the GCP counterpart of the TAU
`server-setup/` (which targets the CS Slurm cluster). `server-setup/` is left untouched so the
teammates running on TAU are unaffected.

---

## TL;DR — run it

```bash
cd ~/NLP_project

# 0. one-time setup (installs toolchain + Python deps, verifies model + datasets cached)
bash gcp-setup/setup_gcp.sh

# 1. sanity check the engine (expect: VLLM_SMOKE_OK)
source gcp-setup/env.sh && python3 gcp-setup/vllm_smoke.py

# 2. a real run — GSM8K, clean baseline + a typo sweep, 50 questions each, then auto-score
bash gcp-setup/run.sh --dataset gsm8k --variant both \
     --configs rate25_real10,rate25_real40,rate25_real70 --limit 50

# 2b. SAME run but survive SSH disconnect (see "Running when SSH drops" below)
bash gcp-setup/run.sh --detach --dataset gsm8k --variant both \
     --configs rate25_real10,rate25_real40,rate25_real70 --limit 50
```

Results land in `~/nlp_project/results/<dataset>_<config>.jsonl`.

---

## The machine (and why it needed fixing)

| Property | Value |
|---|---|
| GPU | 1× **NVIDIA L4**, 23 GB, Ada **sm_89** (supports bfloat16, cudagraphs, flashinfer) |
| Driver / CUDA | 580.159.03 / CUDA 13.0 runtime; CUDA **12.9** toolkit at `/usr/local/cuda` |
| CPU / RAM / disk | 8 vCPU / 31 GB / ~79 GB free on `/` |
| Python | system `/usr/bin/python3` = **3.10** (no conda, no venv) |
| Deps | installed with **`pip --user`** into `~/.local` |

**This is NOT a GCP "Deep Learning VM" image.** It's a bare Ubuntu 22.04 with the CUDA toolkit and
`gcc`, but it shipped **without** `g++`, `make`, or `python3-dev`, and `/opt` is empty (no conda).
vLLM's fast path JIT-compiles CUDA/C++ at runtime (flashinfer kernels + `torch.compile`), which
needs that full toolchain. `setup_gcp.sh` installs it. If you ever land on a box where you *can't*
install the toolchain, there's a no-JIT fallback (see Troubleshooting).

---

## What we did (context / history)

1. **Ported `server-setup/` → `gcp-setup/`.** Converted the TAU-Slurm pipeline to a single L4:
   `tensor_parallel_size=2 → 1`, `float16 + enforce_eager + flashinfer-off` (RTX 2080 Ti hacks) →
   native **bfloat16 + cudagraphs + flashinfer**, miniconda on `/vol/scratch` → `pip --user`, and
   dropped all Slurm `*.sbatch` files (run directly instead).

2. **Downloaded the model** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (~15 GB, bf16) into the
   default HF cache `~/.cache/huggingface`, so `from_pretrained(repo_id)` finds it automatically.

3. **Downloaded the 3 typo datasets** `idoazou/{gsm8k,math500,gpqa}-typos` (34 configs each) into
   the same cache and warmed their Arrow build cache.

4. **Fixed the half-provisioned toolchain:**
   - `sudo apt-get update` first (the package index was stale → "no installation candidate").
   - installed `python3-dev`/`libpython3.10-dev` (provides `Python.h`), `g++-12`, `make`.
   - registered `g++`/`c++`/`cc` via `update-alternatives` (nvcc looks for those exact names).
   - result: the full **cudagraphs + flashinfer** path works (verified via `vllm_smoke.py`).

5. **Enhanced the runner CLI** (`--dataset ...|all`, `--configs ...|all`, `--variant both`,
   `--n-samples`) and made **`score.py`** aggregate multi-sample runs.

6. **Made it fully offline** (`HF_HUB_OFFLINE=1` in `env.sh`) so runs never ping the Hub — no
   re-downloads, no "unauthenticated request" warnings. Everything is already cached locally.

7. **Added `--detach`** to `run.sh` so a run survives SSH disconnect.

---

## Files in `gcp-setup/`

| File | Purpose |
|------|---------|
| `GCP_RUNBOOK.md` | This document. |
| `README.md` | Short reference + full CLI flag table. |
| `setup_gcp.sh` | One-time setup: system toolchain (apt) + Python deps (`pip --user`) + verify cache. |
| `env.sh` | `source` before running — sets `HF_HOME`, `PATH`, `MODEL_PATH`, `HF_HUB_OFFLINE`. |
| `run.sh` | Wrapper: inference (flags pass through) → scoring. Supports `--detach`. |
| `run_typo_vllm.py` | **Primary** engine. vLLM, single-L4 bf16. |
| `run_typo.py` | Transformers fallback (slower; if vLLM ever misbehaves). |
| `score.py` | CPU scoring: accuracy per config + flip rates vs the clean baseline; handles `--n-samples`. |
| `vllm_smoke.py` | Minimal load + one-prompt test. |

---

## Where the data lives

Nothing is inside the git repo `~/NLP_project` — it's all under your **home dir**:

- **Input datasets** (from HuggingFace):
  `~/.cache/huggingface/hub/datasets--idoazou--{gsm8k,math500,gpqa}-typos/snapshots/<hash>/`
  Each config is a subfolder (`clean/`, `rate25_real10/`, `real30/`, …). The `.parquet` files are
  **symlinks** into the sibling `blobs/` store (standard HF cache layout) — the bytes are there,
  just indirected. ~69 MB total for all three.
- **Model:** `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/` (~15 GB).
- **Output results:** `~/nlp_project/results/<dataset>_<config>.jsonl` (one clean baseline file
  `<dataset>_clean.jsonl` + one per typo config). Detach logs: `~/nlp_project/results/run_<ts>.log`.

Browse the inputs: `ls ~/.cache/huggingface/hub/datasets--idoazou--gsm8k-typos/snapshots/*/`

---

## Datasets & config naming

| dataset | repo | clean field | typo field | gold field | kind |
|---|---|---|---|---|---|
| gsm8k | `idoazou/gsm8k-typos` | `question` | `problem_typo` | `answer` | math |
| math500 | `idoazou/math500-typos` | `problem` | `problem_typo` | `answer` | math |
| gpqa | `idoazou/gpqa-typos` | `Question` | `problem_typo` | `Correct Answer` | multiple-choice |

Config names (34 per dataset):
- `clean` — untouched questions (the baseline).
- `real{N}` (N = 0,10,…,70) — typos on words, **N% of them real-word** typos (rest non-word).
- `rate{R}_real{N}` (R = 25,50,75) — **R% typo density** (fraction of words corrupted), N% real-word.

`--configs all` picks up every config present in the cached snapshot.

---

## Running

### Command shapes

```bash
# one dataset, default sweep (real0,real30,real70), typos only
bash gcp-setup/run.sh --dataset gsm8k

# clean baseline + typos (variant both), a chosen density/real sweep, 50 q each
bash gcp-setup/run.sh --dataset gsm8k --variant both \
     --configs rate25_real10,rate25_real40,rate25_real70 --limit 50

# everything: all 3 datasets, every config, 50 q each
bash gcp-setup/run.sh --dataset all --configs all --limit 50

# multi-sample (average over N runs/question; flips use majority vote)
bash gcp-setup/run.sh --dataset math500 --configs real0,real30,real70 --n-samples 5
```

Key flags (full table in `README.md`): `--dataset` (name | comma list | `all`), `--configs`
(comma list | `all`), `--variant` (`typo`|`clean`|`both`), `--limit` (0 = all), `--n-samples`,
`--max-new-tokens` (default 4096), `--outdir` (default `~/nlp_project/results`).

### Timing (single L4, 50 q/config, DeepSeek-R1 reasoning traces ~370 tok/s)

- Model load + cudagraph capture: **~1 min** (once per process).
- **~2.5–3 min per config** of 50 questions.
- Each config is **one batched `llm.generate()`** call — you see vLLM's `Processed prompts N/50`
  bar (updates via `\r`), then a `wrote …jsonl` line when the config completes. Progress is
  **per-config**, which is inherent to batched inference (far faster than one-at-a-time).

### Scoring

`run.sh` scores automatically at the end. To (re-)score anytime (CPU, seconds):

```bash
source gcp-setup/env.sh
python3 gcp-setup/score.py --results ~/nlp_project/results --dataset gsm8k
```

Output: per-config `acc(all)` / `acc(answered)`, and `right→wrong` / `wrong→right` flips vs the
clean baseline. You can improve the parser and re-score without touching the GPU.

---

## Running when SSH drops (important)

A normal `bash run.sh …` dies if the SSH connection is lost (the shell gets `SIGHUP`). To keep the
job running after disconnect/logout, add **`--detach`** (or `-d`) as the **first** argument:

```bash
bash gcp-setup/run.sh --detach --dataset gsm8k --variant both \
     --configs rate25_real10,rate25_real40,rate25_real70 --limit 50
```

It relaunches itself in a new session (`setsid` + `nohup`, stdin from `/dev/null`), detaches, and
prints everything you need, then returns immediately:

```
detached: PID 12345   (keeps running after SSH disconnect / logout)
log:      ~/nlp_project/results/run_20260724_131500.log
follow:   tail -f ~/nlp_project/results/run_...log | tr '\r' '\n'
stop:     kill 12345
```

Then you can safely close the terminal. To check on it after reconnecting:

```bash
# is it still running?
pgrep -af run_typo_vllm.py

# follow the log (tr makes vLLM's \r progress bars render as lines)
tail -f ~/nlp_project/results/run_*.log | tr '\r' '\n'

# which configs are done
ls -la ~/nlp_project/results/*.jsonl

# stop it
kill <PID>          # or: pkill -f run_typo_vllm.py
```

**Why it survives:** `setsid` gives the job a new session with no controlling terminal, so the
`SIGHUP` sent when SSH closes never reaches it; `nohup` ignores `SIGHUP` regardless; `</dev/null`
frees it from the terminal's stdin.

*(Alternatives: `tmux` or `screen` also survive disconnect and let you reattach interactively, but
they aren't installed here — `--detach` covers the "must keep running" need without them.)*

---

## Output record schema (`*.jsonl`, one line per question×sample)

```
dataset, config, variant, idx, sample_idx,
clean_question, typo_question, prompt, gold_answer,
generation,                 # full model output (includes </think> reasoning + answer)
reasoning, final_answer_text,  # split on </think>
n_prompt_tokens, n_gen_tokens,
+ carried-through typo metadata: real_ratio, num_total, num_real, num_nonword,
  target_real_ratio, typo_techniques, level, subject, unique_id, Record ID (when present)
```

---

## First-run sanity result (GSM8K, 50 q, for reference)

| config | accuracy | notes |
|---|---|---|
| clean | 98.0% | baseline |
| rate25_real10 | 82.0% | −16 pts; flips 9 right→wrong, 1 wrong→right |

Reasoning length +23.5% under typos; self-doubt markers (wait/hmm/actually/reconsider) +41%.
Matches the proposal's hypothesis direction (accuracy down, reasoning longer, more self-correction).

---

## Troubleshooting

- **`fatal error: Python.h`** — `python3-dev` missing. `sudo apt-get update && sudo apt-get install
  -y python3-dev libpython3.10-dev`. (`setup_gcp.sh` does this.)
- **`gcc: cannot execute 'cc1plus'` / `nvcc fatal: Failed to preprocess host compiler`** — `g++`
  missing. `sudo apt-get install -y g++-12 make` and register: `sudo update-alternatives --install
  /usr/bin/g++ g++ /usr/bin/g++-12 60`.
- **apt "has no installation candidate"** — stale index; run `sudo apt-get update` first.
- **Can't install the toolchain at all** — use the no-JIT fallback (slower, no compilation):
  `export VLLM_ENFORCE_EAGER=1 VLLM_USE_FLASHINFER_SAMPLER=0` before running. The runners honor
  `VLLM_ENFORCE_EAGER`.
- **CUDA OOM on the L4 (23 GB)** — lower `--gpu-mem-util` (e.g. 0.85), reduce `--max-new-tokens`,
  or add `--enforce-eager`.
- **"unauthenticated requests to the HF Hub" warning** — harmless metadata ping; suppressed by
  `HF_HUB_OFFLINE=1` (already set in `env.sh`). Data is never re-downloaded — it's all cached.
- **vLLM console scripts not found** — `~/.local/bin` isn't on `PATH` by default; `env.sh` adds it.
```
