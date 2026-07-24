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
| `repair_behavior.py` | repair = `typo-noticing` + `repair words`; notice×outcome; LLM-judge scaffold | `repair_notice_outcome`, `repair_words_per_config`, `repair_notice_vs_realratio` |
| `lexical_grid.py` | all four marker families across the rate×real grid | `lexical_grid` |

## Run

```bash
cd full_analysis
python accuracy_flips.py
python reasoning_length.py
python self_doubt.py                 # both categories
python self_doubt.py --no-2guess     # uncertainty only (second_guess is a flat baseline)
python repair_behavior.py            # lexical measure
python repair_behavior.py --judge --limit 50   # + LLM judge (needs HF_TOKEN)
python lexical_grid.py
```

On Windows consoles set `PYTHONIOENCODING=utf-8` for the Δ / → glyphs.
