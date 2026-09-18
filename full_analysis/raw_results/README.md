# Raw results (experiment generations)

The raw model outputs of every experiment: one JSONL per (dataset, config),
one row per question, holding the prompt, the full reasoning trace, the final
answer text, and token/cost metadata. Everything in `../analysis_tables/` and
`../reports/` is derived from these files.

This folder is ~500 MB and NOT committed to git. The data of record lives on
the Hugging Face Hub:

| what | HF dataset repo | path |
|---|---|---|
| main grids: gsm8k (4k + 20k budget), math500, arc, gpqa, fix-spellcheck | [ronshtricker/typo-reasoning-results](https://huggingface.co/datasets/ronshtricker/typo-reasoning-results) | `results/<dataset>/...` |
| fix-warn / fix-rewrite grids | [idoazou/typo-results](https://huggingface.co/datasets/idoazou/typo-results) | `results/gsm8k/*_fix-{warn,rewrite}.jsonl` |
| corrupted question datasets (inputs, not results) | [idoazou/gsm8k-typos](https://huggingface.co/datasets/idoazou/gsm8k-typos), [idoazou/math500-typos](https://huggingface.co/datasets/idoazou/math500-typos), [idoazou/arc-typos](https://huggingface.co/datasets/idoazou/arc-typos), [idoazou/gpqa-typos](https://huggingface.co/datasets/idoazou/gpqa-typos) | per-config splits |

Download into this folder with:

    python download_data.py --dataset gsm8k     # also: math500, arc, gpqa, ...

`dropped_rows.json` files here are audit manifests from the dropped-stream
repair: per config, the question idx whose original rows were lost to a
mid-generation connection drop and re-generated with identical prompts
(tracked copies in ../legacy/).
