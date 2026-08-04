"""R1 vs Qwen2.5-7B-Instruct comparison on the configs that have both runs.

Writes tables/<dataset>/baseline_model_compare.csv and prints the table.

    NLP_DATASET=arc python baseline_compare.py
"""
import os
import csv
import glob

import common
from common import DATA_DIR, DATASET, TABLES, load_evals, mcnemar

SUFFIX = "_Qwen2.5-7B-Instruct"
ORDER = ["clean"] + [f"typo{r}_real{p}" for r in (25, 50, 75) for p in (10, 40, 70)]


def acc(evals):
    ans = [e for e in evals.values() if e["answered"]]
    return (sum(e["correct"] for e in ans) / len(ans) if ans else 0.0), len(ans)


def main():
    rows = []
    print(f"{'config':16s}{'R1':>8}{'n_a':>6}{'Qwen':>9}{'n_a':>6}{'R1-Qwen':>10}")
    r1_clean = qw_clean = None
    for cfg in ORDER:
        p1 = os.path.join(DATA_DIR, f"{DATASET}_{cfg}.jsonl")
        p2 = os.path.join(DATA_DIR, f"{DATASET}_{cfg}{SUFFIX}.jsonl")
        if not (os.path.exists(p1) and os.path.exists(p2)):
            continue
        e1, e2 = load_evals(p1), load_evals(p2)
        a1, n1 = acc(e1)
        a2, n2 = acc(e2)
        if cfg == "clean":
            r1_clean, qw_clean = a1, a2
        print(f"{cfg:16s}{a1:>8.4f}{n1:>6}{a2:>9.4f}{n2:>6}{a1-a2:>10.4f}")
        rows.append(dict(config=cfg, r1_acc=round(a1, 4), r1_n_answered=n1,
                         qwen_acc=round(a2, 4), qwen_n_answered=n2,
                         r1_minus_qwen=round(a1 - a2, 4),
                         r1_drop_vs_clean=round(a1 - r1_clean, 4),
                         qwen_drop_vs_clean=round(a2 - qw_clean, 4)))
    if not rows:
        raise SystemExit(f"no paired {SUFFIX} files in {DATA_DIR}")
    out = os.path.join(TABLES, "baseline_model_compare.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
