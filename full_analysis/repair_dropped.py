"""Remove dropped-stream rows from downloaded result files.

Runs made before api-setup commit 46cd701 ("reject dropped streams") could store
a partial generation when the router killed the SSE stream mid-reasoning: the
trace ends mid-sentence far below the token cap, the harness appends "</think>",
and there is no final-answer section. Those rows are transport failures, not
model behaviour, and they deflate completion / strict accuracy for the affected
configs (arc clean, arc typo50_real40, gpqa clean).

A row is treated as dropped iff ALL of:
  - no extractable final answer (it counts as unanswered), and
  - n_gen_tokens < the run's generation cap (so not budget exhaustion), and
  - final_answer_text is empty (the stream died inside the think block).

Truncated-in-final rows (final text present but unparseable) are NOT removed:
they cannot be told apart from genuine format misses, occur at similar rates in
every config, and so do not bias config comparisons.

Usage (after download_data.py):
    NLP_MAX_NEW_TOKENS=17000 python repair_dropped.py --dataset arc

Rewrites raw_results/<dataset>/*.jsonl in place (keeping <name>.jsonl.orig) and writes
raw_results/<dataset>/dropped_rows.json, the manifest of removed (config, idx). To
regenerate the removed traces for real instead, delete the dropped rows (this
script) and use run_typo_api.py's resume mode - it re-runs exactly the idx
missing from each output file.
"""
import os, json, shutil, argparse, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATASET, MAX_NEW_TOKENS, config_files, parse_tag,  # noqa: E402
                    eval_row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report, change nothing")
    args = ap.parse_args()

    manifest = {}
    for path in config_files():
        tag = parse_tag(path)["tag"]
        kept, dropped = [], []
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            e = eval_row(r)
            final = (r.get("final_answer_text") or "").strip()
            if not e["answered"] and not e["capped"] and not final:
                dropped.append(r["idx"])
            else:
                kept.append(line)
        print(f"{tag:20s} kept {len(kept):4d}  removed {len(dropped):3d}"
              f"  idx={dropped if len(dropped) <= 12 else dropped[:12] + ['...']}")
        manifest[tag] = dropped
        if dropped and not args.dry_run:
            orig = path + ".orig"
            if not os.path.exists(orig):
                shutil.copyfile(path, orig)
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(kept)

    total = sum(len(v) for v in manifest.values())
    print(f"\n{DATASET}: removed {total} dropped rows (cap={MAX_NEW_TOKENS})")
    if not args.dry_run:
        mpath = os.path.join(os.path.dirname(config_files()[0]), "dropped_rows.json")
        json.dump(manifest, open(mpath, "w"), indent=1)
        print(f"manifest -> {mpath}")


if __name__ == "__main__":
    main()
