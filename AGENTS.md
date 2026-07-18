# AGENTS.md — Instructions for the coding agent running this repo

> This file tells an AI coding agent (e.g. GitHub Copilot) what this repo is,
> how to run it on the university Slurm cluster, and how to verify it worked.
> It was authored on a firewalled corporate machine where the Hugging Face Hub
> and NLTK downloads are blocked; the **real run is expected on the cluster**,
> which has network access.

## 1. What this project does

Research question: **how do typos affect the reasoning ability of LLMs?**

This repo builds the *typo dataset* used for that study. The pipeline:

1. Loads a subset of the `MATH-500` dataset (`HuggingFaceH4/MATH-500`, split `test`).
2. Injects **keyboard-aware typos** into the `problem` text using the local
   [`multypo`](./multypo) package. Four techniques are used, matching the
   proposal:
   - `replace`   = adjacent-key substitution
   - `transpose` = swapping two letters
   - `delete`    = deletion of a letter
   - `insert`    = insertion of a letter
3. **Protects numbers and math**: LaTeX (`$...$`, `$$...$$`, `\(...\)`,
   `\[...\]`, `\command`) and bare numbers are never corrupted. Only plain
   prose words are eligible.
4. Guarantees **at least one typo per problem** (so every problem is a valid
   counterpart to the zero-typo baseline model — no 0-typo rows are allowed).
5. For each corrupted word, checks `nltk.corpus.words` to classify the typo as
   a **real word** (exists in the dictionary) or a **non-word**. Per-problem
   counts are stored.
6. Scores each problem by the real-word ratio:

   ```
   P = num_real / num_total          # denominator = words actually changed
   ```

7. Uses `pandas.cut` to split problems into `real_token_groups` bins
   (default 4: `0-25`, `25-50`, `50-75`, `75-100` percent) and saves **one
   dataset per bin** via `datasets.save_to_disk`.

## 2. Repo layout

| Path | Purpose |
| --- | --- |
| `typo_pipeline.py`   | The whole pipeline (loading, typos, scoring, binning, saving). |
| `multypo/`           | Local keyboard-aware typo generator (do not `pip install`, used locally). |
| `requirements.txt`   | Python deps (`datasets`, `nltk`, `pandas`, `numpy`). |
| `DATASET_USAGE.md`   | How to consume the produced binned datasets downstream. |
| `run_pipeline.slurm` | Template Slurm batch script for the cluster. |
| `typo_dataset/`      | **Output** (created on run): `bin_<range>/` folders. |

## 3. How to run on the cluster (the real run)

```bash
# from the repo root
module load python            # or activate your conda/venv
pip install -r requirements.txt
python -c "import nltk; nltk.download('words')"   # one-time corpus download

python typo_pipeline.py
```

### Recommended config for the full run

Edit the `Config` dataclass at the top of `typo_pipeline.py`:

| Field | Local-dev value | Full cluster run |
| --- | --- | --- |
| `subset_size` | `50` | `500` (all of MATH-500 test) |
| `num_proc` | `1` | e.g. `8`–`16` (match `--cpus-per-task`) |
| `allow_offline_fallback` | `True` | **`False`** (fail loudly instead of silently using the mock) |
| `typo_rate` | `0.30` | keep or tune |
| `real_token_groups` | `4` | keep or change bin count |

You can also run without editing code by importing and overriding:

```python
from typo_pipeline import Config, main
main(Config(subset_size=500, num_proc=8, allow_offline_fallback=False))
```

## 4. OS-agnostic guarantees (Windows dev ↔ Linux cluster)

- All paths use `pathlib`/`os.path.join` — no hard-coded separators.
- Parallelism uses `datasets.Dataset.map(num_proc=...)`, which works with
  `fork` (Linux) and `spawn` (Windows). The entry point is guarded by
  `if __name__ == "__main__":`, and heavy objects (dictionary set, typo
  generator) are lazily built once per worker via module-level caches, so they
  pickle correctly under `spawn`.
- No shell-outs or platform-specific system calls.

## 5. Offline / firewall behaviour (why fallbacks exist)

On a network that blocks the Hugging Face Hub / NLTK downloads, the pipeline
falls back to:
- a small built-in **mock** MATH-style dataset, and
- a tiny built-in **fallback dictionary**.

These are **only for local smoke tests**. On the cluster set
`allow_offline_fallback=False` so a network failure raises instead of silently
producing mock data. To run offline with *real* data, download the files on a
networked machine and point the config at them:
- `Config.local_data_path` → a `save_to_disk` dir, `.json`, or `.parquet`.
- `Config.local_words_path` → a newline-separated word list (replaces NLTK).

## 6. How to verify a successful run

1. Console prints a `PREVIEW` (original vs. typo text) and a `SUMMARY`.
2. `min typos / problem` in the summary must be **>= 1**.
3. `typo_dataset/` contains `bin_<range>/` folders; each loads back with
   `datasets.load_from_disk`.
4. Spot-check that LaTeX/numbers are **identical** between `problem` and
   `problem_typo`.
5. Each row's `real_ratio` lies within its bin's percentage range.

## 7. Output columns

Every produced row keeps the original MATH-500 fields plus:

| Column | Type | Meaning |
| --- | --- | --- |
| `problem_typo` | str | The corrupted problem text (feed this to the LLM). |
| `num_total` | int | Words actually changed in this problem (>= 1). |
| `num_real` | int | Changes that are valid English words. |
| `num_nonword` | int | Changes that are non-words. |
| `real_ratio` | float | `P = num_real / num_total`, in `[0, 1]`. |
| `real_bin` | str | Percentage bin label, e.g. `"25-50"`. |

See `DATASET_USAGE.md` for downstream usage.
