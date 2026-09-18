# Raw results (experiment generations)

The raw model outputs of every experiment: one JSONL per (dataset, config),
one row per question, holding the prompt, the full reasoning trace, the final
answer text, and token/cost metadata. Everything in `../analysis_tables/` and
`../reports/` is derived from these files.

This folder is ~500 MB and NOT committed to git. The data lives on the
Hugging Face Hub, split across two repos with different roles:

| repo | role | contents |
|---|---|---|
| [ronshtricker/typo-reasoning-results](https://huggingface.co/datasets/ronshtricker/typo-reasoning-results) | raw files of record: what `download_data.py` fetches and the analysis reads | main grids (gsm8k 4k + 20k budget, math500, arc, gpqa), both fix-spellcheck grids, Qwen baselines. Raw JSONL only, no viewer subsets |
| [idoazou/typo-results](https://huggingface.co/datasets/idoazou/typo-results) | original mirror of the July runs, with 67 browsable subsets (`load_dataset("idoazou/typo-results", "arc_clean")`) | arc, math500, gsm8k fix runs; the ONLY copy of the fix-warn / fix-rewrite raw files (`results/gsm8k/*_fix-{warn,rewrite}.jsonl`). Its arc/gpqa rows predate the dropped-stream repair |

Shared files were verified row-identical between the repos (same rows,
different on-disk order). Note the analysis pipeline consumes the raw JSONLs
only - the per-config subsets (`typoX_realY`) are a browsing/`load_dataset`
convenience, not an input to `run_all.py`.

The corrupted QUESTION datasets (inputs, not results) are separate:
[idoazou/gsm8k-typos](https://huggingface.co/datasets/idoazou/gsm8k-typos),
[idoazou/math500-typos](https://huggingface.co/datasets/idoazou/math500-typos),
[idoazou/arc-typos](https://huggingface.co/datasets/idoazou/arc-typos),
[idoazou/gpqa-typos](https://huggingface.co/datasets/idoazou/gpqa-typos) -
these ARE loaded by subset name (e.g. `rate50_real40`) by run_typo_api.py at
generation time.

Download into this folder with:

    python download_data.py --dataset gsm8k     # also: math500, arc, gpqa, ...

`dropped_rows.json` files here are audit manifests from the dropped-stream
repair: per config, the question idx whose original rows were lost to a
mid-generation connection drop and re-generated with identical prompts
(tracked copies in ../legacy/).
