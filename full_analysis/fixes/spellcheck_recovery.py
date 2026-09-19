"""Spellcheck typo-recovery summary tables.

Reads result JSONL files in data/<dataset>/ that include spellcheck metadata
(from api-setup/run_typo_api.py --fix spellcheck) and writes:
  - tables/<dataset>/spellcheck_recovery_per_config.csv
  - tables/<dataset>/spellcheck_recovery_overall.csv

If no spellcheck files are present, exits cleanly without writing tables.
"""
import os
import re
import csv
import glob
import json

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.environ.get("NLP_DATASET", "gsm8k")
DATA_DIR = os.path.join(HERE, "raw_results", DATASET)
import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from common import tables_dir
TABLES = tables_dir(DATASET)
os.makedirs(TABLES, exist_ok=True)


def _cfg_from_name(path):
    base = os.path.basename(path).replace(".jsonl", "")
    if base.startswith(DATASET + "_"):
        base = base[len(DATASET) + 1 :]
    return base


def _iter_spell_rows(path):
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        total = row.get("spellcheck_total_typo_words")
        restored = row.get("spellcheck_restored_exact_n")
        changed_other = row.get("spellcheck_changed_other_n")
        not_restored = row.get("spellcheck_not_restored_n")
        if isinstance(total, int) and isinstance(restored, int):
            yield {
                "total": total,
                "restored": restored,
                "changed_other": int(changed_other or 0),
                "not_restored": int(not_restored or (total - restored)),
            }


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"[spellcheck_recovery] no data dir: {DATA_DIR}")
        return

    files = sorted(glob.glob(os.path.join(glob.escape(DATA_DIR), f"{DATASET}_*.jsonl")))
    # Prefer explicit spellcheck suffix; fallback to rows with spellcheck metrics.
    cand = [f for f in files if "spellcheck" in os.path.basename(f)]
    if not cand:
        cand = files

    per_cfg = []
    overall_total = overall_restored = overall_changed_other = overall_not_restored = 0

    for fp in cand:
        cfg = _cfg_from_name(fp)
        rows = list(_iter_spell_rows(fp))
        if not rows:
            continue

        n_questions = len(rows)
        total = sum(r["total"] for r in rows)
        restored = sum(r["restored"] for r in rows)
        changed_other = sum(r["changed_other"] for r in rows)
        not_restored = sum(r["not_restored"] for r in rows)
        restored_frac = (restored / total) if total else 0.0

        per_cfg.append({
            "config": cfg,
            "n_questions": n_questions,
            "total_typo_words": total,
            "restored_exact_n": restored,
            "changed_other_n": changed_other,
            "not_restored_n": not_restored,
            "restored_exact_frac": round(restored_frac, 6),
            "restored_exact_pct": round(100.0 * restored_frac, 2),
        })

        overall_total += total
        overall_restored += restored
        overall_changed_other += changed_other
        overall_not_restored += not_restored

    if not per_cfg:
        print("[spellcheck_recovery] no rows with spellcheck metrics found")
        return

    per_cfg.sort(key=lambda r: r["config"])
    overall_frac = (overall_restored / overall_total) if overall_total else 0.0
    overall = [{
        "dataset": DATASET,
        "configs_count": len(per_cfg),
        "total_typo_words": overall_total,
        "restored_exact_n": overall_restored,
        "changed_other_n": overall_changed_other,
        "not_restored_n": overall_not_restored,
        "restored_exact_frac": round(overall_frac, 6),
        "restored_exact_pct": round(100.0 * overall_frac, 2),
    }]

    per_path = os.path.join(TABLES, "spellcheck_recovery_per_config.csv")
    all_path = os.path.join(TABLES, "spellcheck_recovery_overall.csv")
    _write_csv(per_path, per_cfg)
    _write_csv(all_path, overall)
    print(f"[spellcheck_recovery] wrote {per_path}")
    print(f"[spellcheck_recovery] wrote {all_path}")


if __name__ == "__main__":
    main()
