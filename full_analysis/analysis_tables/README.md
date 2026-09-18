# Analysis tables

The analysis CSVs every report and paper asset is built from. The structure
mirrors `../reports/`: one folder per dataset, one subfolder per analysis
variant (`results`, `results_20000`, `fix_warn`, `fix_rewrite`, ...).

Regenerate any folder from the raw generations (in `../raw_results/`, downloaded by
`download_data.py`) with:

    NLP_MAX_NEW_TOKENS=<cap> python run_all.py --dataset <name>

where `<name>` is `gsm8k`, `gsm8k_20000`, `math500`, `arc`, ... and the caps
are 4096 (gsm8k), 20000 (gsm8k_20000), 17000 (math500, arc). The mapping from
dataset name to folder lives in `common.dataset_layout`.

Note: the paper's GSM8K numbers come from `gsm8k/results_20000` (the 20k-token
budget run); `gsm8k/results` is the earlier 4096-budget grid. `judge_*.csv`
files are written by the LLM-judge scripts and sit alongside the lexical
tables of their dataset.
