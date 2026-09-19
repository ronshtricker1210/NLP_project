# Legacy

Pre-repair artifacts kept for the audit trail; nothing here feeds the current
reports or the paper.

- `report_arc_judge_prefix.{pdf,tex}`: ARC report snapshot built BEFORE the
  dropped-stream repair - its accuracy/completion numbers are the contaminated
  pre-fix values. The current report is `../reports/arc/results/report_arc.pdf`.
- `rerun_manifest_arc.json`, `rerun_manifest_gpqa.json`: which question idx per
  config were lost to mid-generation connection drops and re-generated with
  identical prompts (57 on ARC, 9 on GPQA).

- `report_gsm8k_fix-spellcheck_500.{pdf,tex}`: superseded by
  `../reports/gsm8k/fix_spellcheck/report_gsm8k_fix-spellcheck.pdf` (same
  numbers, adds n_answered; its judge tables live on in
  report_gsm8k_fix-spellcheck_500_judge.pdf, which stays current).
- `report_gsm8k_fix-spellcheck_grid-pilot20.{pdf,tex}`: 20-question pilot of
  the spellcheck grid; the published HF data is the full run, now reported in
  `../reports/gsm8k/fix_spellcheck_grid/report_gsm8k_fix-spellcheck_grid.pdf`.

- `report_gsm8k_budget4096.{pdf,tex}` and `raw_results_gsm8k_budget4096/`
  (raw data untracked; same files as results/gsm8k/*.jsonl on the HF repo):
  the early GSM8K grid at a 4,096-token generation budget. Superseded by the
  20,000-token run the paper reports (`../reports/gsm8k/results_20000/`); at
  4k, truncation confounds accuracy (19-108 capped traces per config).
  Its analysis tables remain in `../analysis_tables/gsm8k/results/` (which
  also holds the gsm8k judge tables - those are budget-independent).

- `create_judge_reports_gsm8k_only.py` + `judge_scalar_reports_gsm8k_only/`:
  an earlier, GSM8K-only judge report builder and its output. Superseded by
  `../judge/build_judge_reports.py`, which produces the same two dimensions for
  all three datasets (`../reports/<ds>/llm_as_a_judge/`). Both were added in the
  same commit (8c32b31); only the cross-dataset one was run for every dataset.
- `repair_behavior_old_version.py`: keyword-based repair measure, self-declared
  retired in its own docstring; superseded by `../analysis/repair_wordlevel.py`.

- `analysis_tables_gsm8k_budget4096/`: the analysis CSVs of the 4,096-budget
  GSM8K grid. Superseded by `../analysis_tables/gsm8k/results_20000/` (the run
  the paper reports). NOTE its `judge_*.csv` files are NOT from this grid at
  all: that judge pass ran on the spell-check arm (see the paper's judge-
  coverage note). The GSM8K judge tables computed on the main 20k run now live
  in `../analysis_tables/gsm8k/results_20000/`.
