# Data used in the final paper

Everything in this folder is what `main.tex` is built from. The raw generation
logs are **not** here (~370 MB); they live on the Hugging Face Hub — see
[Raw JSONL](#raw-jsonl-hugging-face) below.

```
paper/data/
  tables/    derived CSVs, one folder per run  -> every number in the paper
  reports/   the per-dataset analysis reports we read while writing
```

---

## Raw JSONL (Hugging Face)

| What | Repo | Path inside repo |
|---|---|---|
| Model generations (all runs) | **`ronshtricker/typo-reasoning-results`** (dataset repo) | `results/<dataset>/<config>.jsonl` |
| Judge outputs | same repo | `judge_cache/*.jsonl` |
| Typo-perturbed question sets | `idoazou/gsm8k-typos`, `idoazou/math500-typos`, `idoazou/arc-typos`, `idoazou/gpqa-typos` | — |

`Dolevabudi/typo-results` is the older repo and holds **GSM8K only**; the
repo above is the complete one (108 files) and is what `download_data.py`
now defaults to.

| `--dataset` | Contents |
|---|---|
| `gsm8k` | 20 files: the main 20,000-cap run (`*_20000.jsonl`) **and** an earlier 4,096-cap run |
| `math500` | 20 files: R1 + `_Qwen2.5-7B-Instruct` control |
| `arc` | 20 files: R1 + `_Qwen2.5-7B-Instruct` control |
| `gsm8k_fix-spellcheck` | 10 files, the spell-check mitigation arm |
| `gsm8k_fix-spellcheck_grid` | 33 files, wider spell-check grid (rates 25/30/50/75 × real 0–70) |
| `gpqa` | 1 file, `clean` only — the grid was never completed |

The raw JSONL for the **warn** and **rewrite** mitigation arms no longer exists
anywhere; only their derived CSVs in `tables/gsm8k_fix-warn` and
`tables/gsm8k_fix-rewrite` survive.

Fetch with the script already in the repo:

```powershell
cd full_analysis
python download_data.py --dataset gsm8k     # then math500, arc
python run_all.py --dataset gsm8k --skip-run
```

`download_data.py` renames `results/<ds>/typo25/real10.jsonl` to
`data/<ds>/<ds>_typo25_real10.jsonl`, which is what the analysis scripts expect.

### JSONL schema (one row = one question in one configuration)

`idx`, `dataset`, `config`, `variant`, `clean_question`, `typo_question`,
`prompt`, `reasoning`, `generation`, `final_answer_text`, `gold_answer`,
`n_prompt_tokens`, `n_gen_tokens`, `latency_s`, `cost_usd`, `sample_idx`.

Extras: `level`/`subject`/`unique_id` (MATH-500), `Record ID` (ARC),
`finish_reason` (ARC, spellcheck arm), and on the spellcheck arm
`fix`, `spellchecked_question`, `spellcheck_changes`,
`spellcheck_total_typo_words`, `spellcheck_restored_exact_n`,
`spellcheck_changed_other_n`.

`idx` is the pairing key: row `idx=i` is the **same** source question in all ten
configurations (the generator is seeded at 42), which is what makes the McNemar
tests in the paper valid.

---

## `tables/` — which run folder is which

| Folder | Model | Prompt arm | Token cap | Used for |
|---|---|---|---|---|
| `gsm8k_20000` | R1-Distill-Qwen-7B | plain | 20,000 | **all GSM8K numbers in the paper** |
| `math500` | R1-Distill-Qwen-7B | plain | 17,000 | all MATH-500 numbers |
| `arc` | R1-Distill-Qwen-7B | plain | 17,000 | all ARC numbers |
| `gsm8k_fix-warn` | R1-Distill-Qwen-7B | *warn* mitigation | 20,000 | §5.6 warn row |
| `gsm8k_fix-rewrite` | R1-Distill-Qwen-7B | *rewrite* mitigation | 20,000 | §5.6 rewrite row |
| `gsm8k` | R1-Distill-Qwen-7B | *spellcheck* mitigation | 4,096 | §5.6 spellcheck restoration rates only |

> `gsm8k` (no suffix) is **not** the main GSM8K run. Its rows carry
> `fix: "spellcheck"` and a 4,096-token cap, so its clean accuracy is 83.7% vs
> 91.4% for the main run. Never compare it across folders. The main GSM8K run is
> `gsm8k_20000`.

`math500/` and `arc/` also contain files ending in `_Qwen2.5-7B-Instruct`
upstream — the non-reasoning control. There is **no** GSM8K control run.

### CSV → paper mapping

| CSV | Feeds |
|---|---|
| `accuracy_per_config.csv` | Fig. 2, Table 3 (appendix), all headline accuracies |
| `flips_vs_clean.csv` | §5.1 McNemar tests, Table 4 (appendix) |
| `accuracy_decomposition.csv` | §5.1 "93% quality loss, not truncation" |
| `reasoning_length_absolute.csv` | §5.1 token counts, §5.6 rewrite shortening |
| `length_by_outcome.csv` | §5.1 wrong-vs-correct trace length ratio |
| `realword_paired.csv` | Fig. 3 left, §5.2 controlled paired test |
| `realword_pertypo_logit.csv` | Fig. 3 right, §5.2 per-typo odds ratios |
| `realword_silent_failure.csv` | §5.3 explicit-notice fraction among wrong answers |
| `repair_wordlevel_by_real.csv` | Fig. 4, §5.3 silent / flagged / misread / unused |
| `repair_wordlevel_by_outcome.csv`, `repair_wordlevel_per_config.csv` | §5.3 supporting |
| `self_doubt_per_config.csv` | Fig. 5 left, §5.4 doubt density per 1k tokens |
| `self_doubt_by_marker.csv` | Fig. 5 right, §5.4 hedging-up / verification-down |
| `self_doubt_by_outcome.csv` | §5.4 supporting |
| `lexical_grid.csv` | §3.2 Table 1 corruption statistics |
| `spellcheck_recovery_*.csv` | §5.6 — **see warning below** |
| `baseline_model_compare.csv` | Table 2, R1 vs Qwen2.5-7B-Instruct |
| `judge_*.csv` (ARC only) | §5.3 judge corroboration |

### Two caveats baked into these files

1. **`spellcheck_recovery_*.csv` is wrong.** `spellcheck_recovery_stats()` in
   `api-setup/run_typo_api.py` reads per-row metadata that is constant across a
   whole file, so every value comes out an exact multiple of 500. The numbers in
   the paper (54.9% / 39.9% / 25.8% restored at ρ=10/40/70) were recomputed from
   the raw `spellcheck_changes` records. Fix the script before reusing this CSV.
2. **`math_verify` silently mis-scores on Windows.** It enforces its parsing
   timeout with a worker process that cannot spawn here, so `parse()` returns
   nothing and every MATH-500 answer scores wrong. `api-setup/score.py` now
   passes `parsing_timeout=None` / `timeout_seconds=None` when `os.name == "nt"`.
   Any MATH-500 CSV regenerated on Windows before that patch is invalid.

---

## `reports/` — analysis reports

| File | Run |
|---|---|
| `report_gsm8k_20000.tex`, `nlp_project_analysis_gsm8k_20kmax.pdf` | main GSM8K |
| `report_math500.tex`, `nlp_project_analysis_math500.pdf` | main MATH-500 |
| `report_arc.tex`, `nlp_project_analysis_arc.pdf` | main ARC |
| `report_gsm8k_fix-warn.tex` | warn mitigation |
| `report_gsm8k_fix-rewrite.tex` | rewrite mitigation |
| `report_gsm8k_fix-spellcheck_500.{tex,pdf}` | spellcheck mitigation |
| `report_gsm8k_fix-spellcheck_500_judge.{tex,pdf}` | LLM-judge pass (spellcheck arm) |
| `report_supplementary_baseline_fixes.{tex,pdf}` | R1 vs Qwen — **laxer scorer, 1–2 pp off; Table 2 uses `baseline_compare.py` instead** |

---

## Judge coverage

The LLM judge (`meta-llama/Llama-3.3-70B-Instruct`, prompt `v1`, temperature 0)
did not complete everywhere:

| Dataset | Status |
|---|---|
| ARC | 3,740 / 9,738 rows usable (74.8%) — the only judge evidence in the paper |
| GSM8K | ran against the **spellcheck** arm, not the main arm |
| MATH-500 | all 4,962 rows returned `repair_label = "error"` — unusable |

The deterministic word-level alignment (`repair_wordlevel.py`) covers all three
datasets and is the paper's primary evidence; the judge is corroboration only.
