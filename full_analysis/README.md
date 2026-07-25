# full_analysis — typo-robustness analysis (gsm8k)

Wider analysis of the DeepSeek-R1-Distill-Qwen-7B typo runs, covering the four
proposal dimensions: **accuracy & flips**, **reasoning length**, **self-doubt**,
and **repair behavior**. Correctness reuses [`../api-setup/score.py`](../api-setup/score.py);
these modules add the rest.

## Data

Inputs are the model result files on the Hub: **`Dolevabudi/typo-results`**
(gsm8k: `clean` + `typo{25,50,75}` × `real{10,40,70}`). The raw JSONL is **not
committed** (~66 MB, reproducible). Fetch it into `data/gsm8k/`:

```python
import os, shutil
from huggingface_hub import HfApi, hf_hub_download
repo = "Dolevabudi/typo-results"
files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
         if f.endswith(".jsonl") and f.startswith("results/gsm8k/")]
dest = "data/gsm8k"; os.makedirs(dest, exist_ok=True)
for f in files:
    tmp = hf_hub_download(repo, f, repo_type="dataset")
    name = "gsm8k_" + f[len("results/gsm8k/"):].replace(".jsonl", "").replace("/", "_") + ".jsonl"
    shutil.copyfile(tmp, os.path.join(dest, name))
```

## Key policy (applies everywhere)

- **Answered-only.** A trace that hit the 4096-token cap before writing a final
  answer is *unanswered/truncated* and is excluded from metric calculations
  (each table still reports the answered `n` and, in accuracy, the truncation rate).
- The predicted answer is read from the **final section only** (after `</think>`) —
  never a trailing number from mid-reasoning.
- **clean** is the paired baseline; flips and deltas are all relative to it.

## Modules

| file | dimension | output tables (in `tables/`) |
| --- | --- | --- |
| `common.py` | shared loaders, strict extraction, bootstrap, McNemar | — |
| `accuracy_flips.py` | accuracy (strict / answered / completion), truncation, answered-only flips + McNemar | `accuracy_per_config`, `flips_vs_clean`, `accuracy_decomposition` |
| `reasoning_length.py` | absolute tokens + typo/clean ratio (median/mean/p90), length by outcome | `reasoning_length`, `reasoning_length_absolute`, `length_by_outcome` |
| `self_doubt.py` | self-doubt = `second_guess` + `uncertainty` marker density | `self_doubt_per_config`, `self_doubt_by_marker`, `self_doubt_by_outcome` |
| `repair_wordlevel.py` | **primary repair measure** — grounded in the actual corrupted words (diff clean vs typo). Per corrupted word: silent_fix / flagged / **misread** / not_used | `repair_wordlevel_per_config`, `repair_wordlevel_by_real`, `repair_wordlevel_by_outcome` |
| `marker_banks.py` | shared repair-side word banks (typo-noticing, repair words) + `classify_trace` (imported, not run) | — |
| `lexical_grid.py` | all four marker families across the rate×real grid | `lexical_grid` |
| `real_word_effect.py` | isolates the real-word axis: controlled paired real10-vs-real70 (same question/positions) + McNemar, per-typo logit (`num_real` vs `num_nonword`), silent-failure test | `realword_paired`, `realword_pertypo_logit`, `realword_silent_failure` |
| `repair_behavior_old_version.py` | *retired* keyword-based repair (notice×outcome, repair-word density, LLM-judge scaffold). Superseded by `repair_wordlevel.py`; kept for reference, not part of the active set | `repair_notice_outcome`, `repair_words_per_config`, `repair_notice_vs_realratio` |

**Repair note:** `repair_wordlevel.py` is the primary repair analysis — it checks
the real corrupted words, so it separates a silent FIX from a silent MISREAD
(the proposal's "silently reading through" vs "misreading as a different word").
`repair_behavior_old_version.py` only detects typo-flagging *keywords* and is blind
to that distinction; it is retired.

**Real-word finding (from `real_word_effect.py`):** because each rate corrupts the
same positions across real variants, swapping non-word typos for real-word ones on
the *same* questions lowers accuracy significantly at rate50/75 (−5% / −9%,
McNemar p<0.05). Per-typo, a real-word typo does ~2× the damage of a non-word one.
But real-word failures are *noticed more*, not less — the harm is
"noticed-but-unrecoverable", not silent.

## Run

```bash
cd full_analysis
python accuracy_flips.py
python reasoning_length.py
python self_doubt.py                 # both categories
python self_doubt.py --no-2guess     # uncertainty only (second_guess is a flat baseline)
python repair_wordlevel.py           # primary repair measure (silent_fix / flagged / misread)
python lexical_grid.py
python real_word_effect.py           # real-word isolation (paired + logit)

# retired keyword-based repair (reference only, not part of the active set):
python repair_behavior_old_version.py
python repair_behavior_old_version.py --judge --limit 50   # + LLM judge (needs HF_TOKEN)
```

On Windows consoles set `PYTHONIOENCODING=utf-8` for the Δ / → glyphs.
