# full_analysis — typo-robustness analysis (gsm8k)

Wider analysis of the DeepSeek-R1-Distill-Qwen-7B typo runs, covering the four
proposal dimensions: **accuracy & flips**, **reasoning length**, **self-doubt**,
and **repair behavior**. Correctness reuses [`../api-setup/score.py`](../api-setup/score.py);
these modules add the rest.

## Quick start (fresh clone)

```bash
cd full_analysis

# 1. install dependencies
pip install huggingface_hub datasets numpy scikit-learn math_verify

# 2. download the raw generations into raw_results/<dataset>/
python tools/download_data.py                # gsm8k (default)

# 3. run every analysis and build the LaTeX report
python run_all.py                            # -> report_gsm8k.tex
```

`run_all.py` runs all the analysis modules (regenerating the CSVs in
`analysis_tables/<dataset>/<variant>/`) and then writes **`reports/<dataset>/<variant>/report_<dataset>.tex`** — a self-contained,
Overleaf-ready document with every result table. Open it in Overleaf (or
`pdflatex report_gsm8k.tex`). To only rebuild the `.tex` from existing CSVs:
`python run_all.py --skip-run`.

## Layout

```
run_all.py          entry point: runs every analysis module, then builds the report .tex
common.py           shared loading / scoring / dataset-to-folder mapping
marker_banks.py     shared lexical marker banks

analysis/           the core measures, one CSV set per dataset
                      accuracy_flips.py     accuracy, completion, paired flips (McNemar)
                      reasoning_length.py   generated-token lengths, answered-only
                      self_doubt.py         doubt-marker densities
                      repair_wordlevel.py   per-corrupted-word repair classification
                      real_word_effect.py   real-word vs non-word logistic models
                      lexical_grid.py       marker densities across the rate x real grid
                      baseline_compare.py   R1 vs Qwen2.5-7B (run separately, needs both runs)
fixes/              spellcheck_recovery.py  typo-recovery rates for the spellcheck arms
judge/              llm_judge.py            the LLM-judge pass (costs API credits; NOT in run_all)
                    llm_as_a_judges_tables.py  judge traces -> mean_by_* aggregate tables
                    build_judge_reports.py     those tables -> per-dimension report .tex/.pdf
                    plot_judge_scalar.py       judge score plots
tools/              download_data.py        fetch raw generations from the HF Hub
                    repair_dropped.py       detect/remove dropped-stream rows (audit tool)

raw_results/        raw generations (gitignored, ~500MB; see its README for the HF repos)
analysis_tables/    the CSVs every report and paper asset is built from
reports/            report PDFs + the LaTeX they compile from
legacy/             superseded artifacts, kept for the audit trail only
```

Scripts in subfolders are still run from this directory, e.g.
`python judge/llm_judge.py --all`, `python tools/download_data.py --dataset arc`.


## Other datasets (math500, gpqa, …)

The pipeline is dataset-agnostic. Any dataset with the **same config grid**
(`clean` + `typo{25,50,75}` × `real{10,40,70}`, one sample/question) works — just
point everything at it with `--dataset` (scoring auto-switches: gsm8k = numeric,
math500 = `math_verify` on `\boxed`, gpqa = multiple choice):

```bash
python tools/download_data.py --dataset math500
python run_all.py             --dataset math500
```

Under the hood every module reads the `NLP_DATASET` env var (set by `run_all.py`),
so a single module can also be run standalone, e.g. `NLP_DATASET=math500 python self_doubt.py`.

## Data

The inputs are the model result files on the Hub: **`Dolevabudi/typo-results`**
(under `results/<dataset>/`). The raw JSONL is **not committed** (reproducible) —
`download_data.py` fetches it into `data/<dataset>/`. Windows note: set
`PYTHONIOENCODING=utf-8` if a console chokes on the Δ / → glyphs the modules print.

## Key policy (applies everywhere)

- **Answered-only.** A trace that hit the 4096-token cap before writing a final
  answer is *unanswered/truncated* and is excluded from metric calculations
  (each table still reports the answered `n` and, in accuracy, the truncation rate).
- The predicted answer is read from the **final section only** (after `</think>`) —
  never a trailing number from mid-reasoning.
- **clean** is the paired baseline; flips and deltas are all relative to it.

## Modules

| file | dimension | output tables (in `tables/<dataset>/`) |
| --- | --- | --- |
| `run_all.py` | **entry point** — runs every module below, then writes `report.tex` | writes `report.tex` |
| `download_data.py` | fetch the gsm8k result JSONL from the Hub into `data/gsm8k/` | — |
| `common.py` | shared loaders, strict extraction, bootstrap, McNemar | — |
| `accuracy_flips.py` | accuracy (strict / answered), truncation-aware, answered-only flips + McNemar | `accuracy_per_config`, `flips_vs_clean` |
| `reasoning_length.py` | absolute tokens (mean/median/p90), length by outcome | `reasoning_length_absolute`, `length_by_outcome` |
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

Normally just `python run_all.py` (see Quick start). To run a single dimension
and read its printed tables:

```bash
python accuracy_flips.py
python reasoning_length.py
python self_doubt.py                 # both marker categories
python self_doubt.py --no-2guess     # uncertainty only (second_guess is a flat baseline)
python repair_wordlevel.py           # primary repair measure (silent_fix / flagged / misread)
python real_word_effect.py           # real-word isolation (paired + logit)
python lexical_grid.py               # overview of all four marker families

# retired keyword-based repair (reference only, not part of the active set):
python repair_behavior_old_version.py
python repair_behavior_old_version.py --judge --limit 50   # + LLM judge (needs HF_TOKEN)
```
