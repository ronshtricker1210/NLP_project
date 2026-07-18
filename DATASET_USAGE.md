# Dataset Usage Guide

The typo datasets are built by [`data_creation/generate_variants.py`](./data_creation/generate_variants.py)
for the "how typos affect reasoning LLMs" experiment.

## 1. What exists

For each source dataset there are **8 variants** that differ only in the fraction
of typos that are **real English words** (`sum -> sun`) vs **non-words**
(`triangle -> trianlge`): `real0`, `real10`, ..., `real70` (0% to 70% in 10% steps).

| HF repo (private) | Source | Rows | Text field |
| --- | --- | --- | --- |
| `idoazou/math500-typos` | MATH-500 test | 500 | `problem` |
| `idoazou/gsm8k-typos` | GSM8K test | 1319 | `question` |
| `idoazou/gpqa-typos` | GPQA Diamond | 198 | `Question` |

Held constant across variants: ~30% of eligible prose words corrupted (>= 1 per
problem), numbers/LaTeX never touched, 1 edit per typo (90%) or 2 edits (10%),
keyboard-aware replace/delete/insert/transpose. The same seed is used everywhere,
so the same words are corrupted in every variant of a problem — only the kind of
typo differs.

> Why max 70%? Most words have no 1-edit typo that forms a real word, so higher
> targets are not reliably reachable without biasing which words get corrupted.
> Each row records its actually achieved ratio in `real_ratio`.

## 2. Loading

The repos are private — authenticate first (`hf auth login`, or `export HF_TOKEN=...`
on the cluster):

```python
from datasets import load_dataset

ds = load_dataset("idoazou/math500-typos", "real70", split="test")
print(ds[0]["problem"])       # clean original
print(ds[0]["problem_typo"])  # corrupted version to send to the model
```

Local copies (same content) live under `data_creation/typo_variants/<dataset>/<variant>/`
and load with `datasets.load_from_disk`.

## 3. Columns

Original source columns are preserved, plus:

| Column | Type | Meaning |
| --- | --- | --- |
| `problem_typo` | str | Corrupted text (same name in all 3 datasets). |
| `num_total` | int | Words corrupted (>= 1). |
| `num_real` / `num_nonword` | int | Real-word / non-word typo counts. |
| `real_ratio` | float | Achieved P = num_real / num_total. |
| `target_real_ratio` | float | The variant's target (0.0-0.7). |
| `typo_originals` | list[str] | Corrupted words, before. |
| `typo_replacements` | list[str] | Corrupted words, after. |
| `typo_techniques` | list[str] | e.g. `"replace"`, `"delete+insert"`. |
| `typo_edit_counts` | list[int] | 1 or 2 edits per typo. |
| `typo_is_real` | list[bool] | Per-typo real-word flag. |

## 4. Typical evaluation loop

```python
for variant in ["real0", "real10", "real30", "real50", "real70"]:
    ds = load_dataset("idoazou/gsm8k-typos", variant, split="test")
    for row in ds:
        baseline_pred = run_llm(row["question"])
        typo_pred = run_llm(row["problem_typo"])
        log(variant=variant, real_ratio=row["real_ratio"],
            baseline_correct=is_correct(baseline_pred, row["answer"]),
            typo_correct=is_correct(typo_pred, row["answer"]))
```

Aggregate accuracy per variant (or per `real_ratio` when it deviates from the
target) to get the dose-response curve of real-word vs non-word noise. The
metadata lists support per-technique and 1-vs-2-edit breakdowns.

## 5. Regenerating

Deterministic — same command, same data:

```bash
cd data_creation
python generate_variants.py                          # all 3 datasets x real0..real70
python generate_variants.py --push --namespace idoazou --private
python generate_variants.py --help                   # rates, ratios, subset, seed...
```
