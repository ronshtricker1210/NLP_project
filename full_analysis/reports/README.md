# Reports

Human-readable reports (PDF + the LaTeX they compile from), one folder per
dataset, one subfolder per analysis:

```
<dataset>/results/          main typo-grid report (report_<dataset>.pdf) - "the final"
<dataset>/results_20000/    same grid at a 20,000-token generation budget (gsm8k only;
                            this is the run the paper reports for GSM8K)
<dataset>/fix_warn/         "input may contain typos" warning mitigation
<dataset>/fix_rewrite/      rewrite-question-first mitigation
<dataset>/fix_spellcheck/   external spellchecker mitigation (incl. its judge report)
<dataset>/llm_as_a_judge/   LLM-judge reports for that dataset
supplementary/              cross-dataset baseline + fixes report
```

Each report_<name>.pdf is compiled from report_<name>.tex, which
`run_all.py --dataset <name>` regenerates from the CSVs in
`../results_data/<dataset>/<variant>/` (identical structure). fix_warn and
fix_rewrite raw generations were never published to the HF results repo, so
their CSVs cannot be re-derived; their reports are kept as built.

gpqa holds no reports yet: only a clean-baseline probe was run (see the paper's
dataset-substitution note).
