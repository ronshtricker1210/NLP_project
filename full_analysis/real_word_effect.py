"""Isolating the REAL-WORD effect (does the *kind* of typo matter, not just how many?).

Most signals in this study scale with typo RATE and look flat across the real-word
ratio. But rate and real are confounded when compared naively. Two facts let us do
better here:
  * within a fixed rate, real10/real40/real70 corrupt the SAME words in the SAME
    positions of the SAME question — only the kind (real-word vs non-word) differs
    (verified: identical num_total for all 500 questions per rate).
  * every row records num_real / num_nonword typos actually applied.

So we can isolate the real-word effect three ways:

1. CONTROLLED PAIRED comparison: same question + same rate, real10 (mostly non-word)
   vs real70 (mostly real-word). Answered-in-both only, McNemar. This is the clean
   causal contrast — swap non-word typos for real-word ones and watch accuracy.
2. PER-TYPO harm: logistic regression P(correct) ~ num_real + num_nonword over all
   typo rows. Compares the marginal damage of one real-word vs one non-word typo.
3. SILENT-FAILURE signature: among WRONG answers, how often did the model explicitly
   notice the corruption, by real ratio? The proposal predicts real-word failures
   are quieter (less noticing).

Answered-only. Outputs printed tables + CSVs.
    python real_word_effect.py
"""
import os, sys, csv, json
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from sklearn.linear_model import LogisticRegression

from common import config_files, parse_tag, load_evals, mcnemar, TABLES
from marker_banks import classify_trace   # (notice_type, n_notice, n_repair)


def load_records(path):
    """{idx: rec} for answered rows, with correctness + typo-count metadata."""
    ev = load_evals(path)
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        e = ev[r["idx"]]
        if not e["answered"]:
            continue
        reasoning = r.get("reasoning", "") or ""
        notice, _, _ = classify_trace(reasoning)
        out[r["idx"]] = dict(idx=r["idx"], correct=e["correct"],
                             num_real=r.get("num_real", 0), num_nonword=r.get("num_nonword", 0),
                             num_total=r.get("num_total", 0), notice=notice)
    return out


def _write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    files = config_files()
    D, meta = {}, {}
    for f in files:
        t = parse_tag(f)
        meta[t["tag"]] = t
        D[t["tag"]] = load_records(f)
    rates = sorted({meta[t]["rate"] for t in meta if not meta[t]["is_clean"]})

    # ---- 1. CONTROLLED PAIRED comparison within each rate -----------------
    print("=== 1. controlled paired comparison: same question+rate, low vs high real-word ===")
    print("(real10 = mostly NON-word typos, real70 = mostly REAL-word; same corrupted positions)\n")
    print(f"{'rate':>6}{'compare':>14}{'n_both':>8}{'acc_low':>9}{'acc_high':>10}"
          f"{'Δacc':>8}{'r→w':>6}{'w→r':>6}{'McNemar p':>12}")
    paired_csv = []
    for rate in rates:
        for lo, hi in [(10, 70), (10, 40), (40, 70)]:
            A = D.get(f"typo{rate}_real{lo}", {})
            B = D.get(f"typo{rate}_real{hi}", {})
            shared = [i for i in A if i in B]
            n = len(shared)
            if not n:
                continue
            accA = sum(A[i]["correct"] for i in shared) / n
            accB = sum(B[i]["correct"] for i in shared) / n
            r2w = sum(1 for i in shared if A[i]["correct"] and not B[i]["correct"])
            w2r = sum(1 for i in shared if not A[i]["correct"] and B[i]["correct"])
            _, _, _, p = mcnemar({i: A[i]["correct"] for i in shared},
                                 {i: B[i]["correct"] for i in shared})
            pstr = f"{p:.2e}" if p is not None else "n/a"
            print(f"{rate:>5}%{f'real{lo}→{hi}':>14}{n:>8}{accA:>9.1%}{accB:>10.1%}"
                  f"{accB-accA:>+8.1%}{r2w:>6}{w2r:>6}{pstr:>12}")
            paired_csv.append(dict(rate=rate, compare=f"real{lo}_to_real{hi}", n_both=n,
                                   acc_low_real=round(accA, 4), acc_high_real=round(accB, 4),
                                   delta_acc=round(accB - accA, 4), r2w=r2w, w2r=w2r,
                                   mcnemar_p=(round(p, 8) if p is not None else None)))
    _write_csv(os.path.join(TABLES, "realword_paired.csv"), paired_csv)
    print("\n  Δacc<0 with r→w>w→r  => real-word typos are MORE harmful than the non-word\n"
          "  typos they replaced (same question, same positions).")

    # ---- 2. PER-TYPO harm: logistic regression ----------------------------
    print("\n=== 2. per-typo harm: logistic regression  correct ~ num_real + num_nonword ===")
    rows = [r for tag in D if not meta[tag]["is_clean"] for r in D[tag].values()]
    X = np.array([[r["num_real"], r["num_nonword"]] for r in rows], dtype=float)
    y = np.array([1 if r["correct"] else 0 for r in rows])
    clf = LogisticRegression(max_iter=1000, C=1e6)  # ~unregularised
    clf.fit(X, y)
    b_real, b_non = clf.coef_[0]
    print(f"  n={len(rows)} answered typo traces")
    print(f"  per REAL-word typo : log-odds {b_real:+.4f}  (odds x{np.exp(b_real):.3f} on correctness)")
    print(f"  per NON-word typo  : log-odds {b_non:+.4f}  (odds x{np.exp(b_non):.3f} on correctness)")
    worse = "REAL-word" if b_real < b_non else "NON-word"
    print(f"  => each {worse} typo hurts correctness more (more negative log-odds).")
    _write_csv(os.path.join(TABLES, "realword_pertypo_logit.csv"), [dict(
        n=len(rows), coef_num_real=round(b_real, 5), coef_num_nonword=round(b_non, 5),
        odds_real=round(float(np.exp(b_real)), 4), odds_nonword=round(float(np.exp(b_non)), 4),
        intercept=round(float(clf.intercept_[0]), 5))])

    # ---- 3. SILENT-FAILURE signature: noticing among WRONG, by real ratio -
    print("\n=== 3. do real-word failures stay silent? explicit-notice rate among WRONG answers ===")
    print(f"{'real ratio':>12}{'n_wrong':>9}{'explicit-notice%':>18}{'silent%':>10}")
    by_real = defaultdict(list)
    for tag in D:
        if meta[tag]["is_clean"]:
            continue
        for r in D[tag].values():
            by_real[meta[tag]["real"]].append(r)
    sig_csv = []
    for real in sorted(by_real):
        wrong = [r for r in by_real[real] if not r["correct"]]
        n = len(wrong)
        expl = sum(1 for r in wrong if r["notice"] == "explicit") / n if n else 0
        print(f"{f'real{real}':>12}{n:>9}{expl:>17.1%}{1-expl:>10.1%}")
        sig_csv.append(dict(real_ratio=real, n_wrong=n,
                            explicit_notice_frac=round(expl, 4),
                            silent_frac=round(1 - expl, 4)))
    _write_csv(os.path.join(TABLES, "realword_silent_failure.csv"), sig_csv)
    print("  proposal predicts real-word failures are QUIETER: explicit% should FALL as real rises.")

    print(f"\ntables written to {TABLES}")


if __name__ == "__main__":
    main()
