# Dataset Usage Guide

This guide explains the dataset produced by [`typo_pipeline.py`](./typo_pipeline.py)
and how to use it in the "how typos affect reasoning LLMs" experiment.

## 1. What gets produced

Running the pipeline creates an output directory (default `./typo_dataset/`)
containing **one saved dataset per real-word-ratio bin**:

```
typo_dataset/
├── bin_0-25/      # problems where 0–25% of typos are real words
├── bin_25-50/
├── bin_50-75/
└── bin_75-100/
```

Each `bin_<range>/` is a standard Hugging Face dataset saved with
`save_to_disk`, so it is loaded with `load_from_disk`.

> The number of bins is controlled by `Config.real_token_groups` (default `4`).
> A bin folder is only created if at least one problem falls into it.

## 2. Columns

Each row keeps the original `MATH-500` fields (`problem`, `solution`,
`answer`, `subject`, `level`, `unique_id`) **plus**:

| Column | Type | Meaning |
| --- | --- | --- |
| `problem_typo` | str | The problem text **with typos** — feed this to the model. |
| `num_total` | int | Number of words actually changed in this problem (always `>= 1`). |
| `num_real` | int | How many changed words are valid English words (real-word typos). |
| `num_nonword` | int | How many changed words are non-words. |
| `real_ratio` | float | `P = num_real / num_total`, a value in `[0, 1]`. |
| `real_bin` | str | The percentage bin this row belongs to, e.g. `"50-75"`. |

The original `problem` column is preserved untouched, so each row carries both
the clean and the corrupted version.

## 3. What the bins mean

`real_ratio` (P) measures **how "sneaky" the typos are**:

- **Low P (e.g. `0-25`)** → most typos produce *non-words* (obvious garbage
  like `teh`, `naswer`). Easier for a model to notice something is wrong.
- **High P (e.g. `75-100`)** → most typos produce *other real words*
  (`from` → `form`, `their` → `there`). Harder to detect; may silently change
  meaning.

Comparing accuracy across bins shows whether reasoning LLMs are more affected
by non-word noise or by real-word (semantically confusing) noise.

## 4. Loading a bin

```python
from datasets import load_from_disk

ds = load_from_disk("typo_dataset/bin_75-100")
print(ds)
print(ds[0]["problem"])       # original, clean
print(ds[0]["problem_typo"])  # corrupted version to send to the model
print(ds[0]["real_ratio"])    # e.g. 0.83
```

Load every bin at once:

```python
from pathlib import Path
from datasets import load_from_disk

bins = {
    p.name.replace("bin_", ""): load_from_disk(str(p))
    for p in sorted(Path("typo_dataset").glob("bin_*"))
}
for label, ds in bins.items():
    print(label, len(ds))
```

## 5. Using the dataset in the experiment

The core comparison is **typo condition vs. the zero-typo baseline**. Because
every problem has `num_total >= 1`, every row in the typo dataset has a direct
clean counterpart (its own `problem` field, and/or the original MATH-500 row
with the same `unique_id`).

Typical evaluation loop:

```python
from datasets import load_from_disk

ds = load_from_disk("typo_dataset/bin_50-75")

for row in ds:
    # 1) Baseline: ask the model to solve the clean problem
    baseline_pred = run_llm(row["problem"])

    # 2) Typo condition: ask the model to solve the corrupted problem
    typo_pred = run_llm(row["problem_typo"])

    # 3) Score both against the gold answer
    gold = row["answer"]
    log(unique_id=row["unique_id"],
        real_bin=row["real_bin"],
        real_ratio=row["real_ratio"],
        baseline_correct=is_correct(baseline_pred, gold),
        typo_correct=is_correct(typo_pred, gold))
```

Then aggregate accuracy **per bin** to see how robustness degrades as the
real-word ratio changes:

```python
import pandas as pd

df = pd.DataFrame(all_logs)
summary = df.groupby("real_bin").agg(
    n=("typo_correct", "size"),
    baseline_acc=("baseline_correct", "mean"),
    typo_acc=("typo_correct", "mean"),
)
summary["accuracy_drop"] = summary["baseline_acc"] - summary["typo_acc"]
print(summary)
```

## 6. Reproducibility notes

- Typos are seeded deterministically per row (`Config.seed + row_index`), so
  re-running with the same config reproduces the same dataset.
- Numbers and LaTeX are guaranteed untouched, so mathematical correctness of
  each problem is preserved — only the surrounding prose is corrupted.
- To regenerate with different settings (typo rate, number of bins, subset
  size), edit the `Config` dataclass in `typo_pipeline.py` and re-run.
