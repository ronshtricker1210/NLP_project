#!/usr/bin/env python3
"""Full analysis of a typo sweep: accuracy AND token usage per rate/real config.

Correctness comes from score.py (same extract/verify logic, flip analysis via
majority_correct); this adds per-config token/cost aggregation and lays both
out as rate x real matrices against the clean baseline.

Usage:
    python analyze.py --results ../results-typo --dataset gsm8k
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

from score import majority_correct, score_file


def analyze(results_dir, dataset, strip_suffix=""):
    files = sorted(glob.glob(os.path.join(results_dir, f"{dataset}_*.jsonl")))
    if not files:
        raise SystemExit(f"no {dataset}_*.jsonl in {results_dir}")

    stats = {}  # tag -> dict
    per_cfg = {}  # tag -> {idx: [(correct, pred, gold), ...]}
    for fp in files:
        tag = os.path.basename(fp).removesuffix(".jsonl").split("_", 1)[1]
        if strip_suffix:
            tag = tag.removesuffix(f"_{strip_suffix}")
        _, per = score_file(fp)
        per_cfg[tag] = per
        recs = [json.loads(l) for l in open(fp)]
        n = len(per)
        samples = [s for lst in per.values() for s in lst]
        answered = sum(1 for _, p, _ in samples if p is not None)
        correct = sum(c for c, _, _ in samples)
        gen_toks = [r["n_gen_tokens"] for r in recs]
        stats[tag] = {
            "n": n, "answered": answered, "acc": correct / max(len(samples), 1),
            "avg_gen": sum(gen_toks) / max(len(recs), 1),
            "p90_gen": sorted(gen_toks)[int(len(gen_toks) * 0.9)] if gen_toks else 0,
            "total_tokens": sum(r["n_prompt_tokens"] + r["n_gen_tokens"] for r in recs),
            "cost": sum(r.get("cost_usd", 0) for r in recs),
        }

    base = per_cfg.get("clean")
    for tag, per in per_cfg.items():
        r2w = w2r = 0
        if base and tag != "clean":
            for idx, samples in per.items():
                if idx in base:
                    b, c = majority_correct(base[idx]), majority_correct(samples)
                    r2w += b and not c
                    w2r += c and not b
        stats[tag]["r2w"], stats[tag]["w2r"] = r2w, w2r

    def sort_key(tag):
        m = re.fullmatch(r"rate(\d+)_real(\d+)", tag)
        return (0, 0, 0) if tag == "clean" else (1, int(m.group(1)), int(m.group(2))) if m else (2, 0, 0)

    clean_acc = stats.get("clean", {}).get("acc")
    print(f"=== {dataset}: per-config ===")
    print(f"{'config':<16}{'n':>5}{'answered':>10}{'acc':>8}{'Δclean':>8}"
          f"{'r→w':>6}{'w→r':>6}{'avg gen tok':>12}{'p90':>7}{'total tok':>11}{'cost $':>9}")
    for tag in sorted(stats, key=sort_key):
        s = stats[tag]
        delta = f"{s['acc']-clean_acc:+.1%}" if clean_acc is not None and tag != "clean" else "-"
        print(f"{tag:<16}{s['n']:>5}{s['answered']:>10}{s['acc']:>8.1%}{delta:>8}"
              f"{s['r2w']:>6}{s['w2r']:>6}{s['avg_gen']:>12.0f}{s['p90_gen']:>7}"
              f"{s['total_tokens']:>11,}{s['cost']:>9.4f}")

    rates = sorted({int(m.group(1)) for t in stats if (m := re.fullmatch(r"rate(\d+)_real(\d+)", t))})
    reals = sorted({int(m.group(2)) for t in stats if (m := re.fullmatch(r"rate(\d+)_real(\d+)", t))})
    for title, field, fmt in [("accuracy", "acc", "{:.1%}"), ("avg gen tokens", "avg_gen", "{:.0f}")]:
        print(f"\n=== {dataset}: {title} matrix (rows=typo rate, cols=real ratio) ===")
        if clean_acc is not None:
            print(f"clean baseline: {fmt.format(stats['clean'][field])}")
        print("rate\\real" + "".join(f"{f'real{r}':>10}" for r in reals))
        for rate in rates:
            row = f"{rate:>8}%"
            for real in reals:
                s = stats.get(f"rate{rate}_real{real}")
                row += f"{fmt.format(s[field]) if s else '-':>10}"
            print(row)

    total_tok = sum(s["total_tokens"] for s in stats.values())
    total_cost = sum(s["cost"] for s in stats.values())
    print(f"\ntotal: {sum(s['n'] for s in stats.values()):,} samples, "
          f"{total_tok:,} tokens, ${total_cost:.4f} (per-token list prices)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.expanduser("~/nlp_project/results"))
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--strip-suffix", default="",
                    help="strip a trailing _<suffix> from file tags, e.g. 20000")
    args = ap.parse_args()
    analyze(args.results, args.dataset, args.strip_suffix)
