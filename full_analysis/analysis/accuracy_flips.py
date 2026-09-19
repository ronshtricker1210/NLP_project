"""Dimension 1 of the proposal: ACCURACY & FLIPS, truncation-aware.

Per config it reports three accuracy views so "didn't finish" is never confused
with "answered wrong":
  - completion   = fraction that produced a real final answer (not truncated)
  - acc|answered = accuracy among finished traces (reasoning quality)
  - acc(strict)  = correct / all, truncated counts as wrong  [headline]
                   (= completion * acc|answered)

It then, vs the clean baseline (paired by question idx):
  - counts flips across the 3 states correct/wrong/unanswered (3x3 transitions),
  - runs McNemar on strict correctness,
  - decomposes the accuracy drop into a "didn't finish" part and a
    "finished but wrong" part.

Outputs: printed tables + CSVs under analysis/tables/. Run:
    python accuracy_flips.py
"""
import os, sys, csv
from collections import defaultdict, Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Δ and → in a cp1252 console
except Exception:
    pass

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from common import (config_files, parse_tag, load_evals, bootstrap_ci,
                    mcnemar, TABLES, DATASET)


def summarize(evals):
    n = len(evals)
    answered = sum(e["answered"] for e in evals.values())
    correct = sum(e["correct"] for e in evals.values())
    capped = sum(e["capped"] for e in evals.values())
    unanswered = n - answered
    completion = answered / n if n else 0.0
    truncated = unanswered / n if n else 0.0          # no final answer produced
    acc_strict = correct / n if n else 0.0
    acc_answered = correct / answered if answered else 0.0
    lo, hi = bootstrap_ci([e["correct"] for e in evals.values()])
    return dict(n=n, answered=answered, unanswered=unanswered, correct=correct,
                capped=capped, completion=completion, truncated=truncated,
                acc_strict=acc_strict, acc_answered=acc_answered, ci=(lo, hi))


def main():
    files = config_files()
    tags, meta, S, EV = [], {}, {}, {}
    for f in files:
        t = parse_tag(f)
        ev = load_evals(f)
        tags.append(t["tag"]); meta[t["tag"]] = t
        EV[t["tag"]] = ev; S[t["tag"]] = summarize(ev)

    base = EV["clean"]
    cs = S["clean"]

    # ---- per-config table -------------------------------------------------
    print(f"=== {DATASET}: accuracy (truncation-aware) ===")
    print(f"{'config':16s}{'n':>5}{'trunc%':>8}{'compl%':>8}{'acc|ans':>9}"
          f"{'acc(strict)':>12}{'95% CI':>16}{'Δstrict':>9}{'capped%':>9}")
    rows_csv = []
    for tag in tags:
        s = S[tag]
        ci = f"[{s['ci'][0]:.1%},{s['ci'][1]:.1%}]"
        d = "-" if tag == "clean" else f"{s['acc_strict']-cs['acc_strict']:+.1%}"
        print(f"{tag:16s}{s['n']:>5}{s['truncated']:>8.1%}{s['completion']:>8.1%}"
              f"{s['acc_answered']:>9.1%}{s['acc_strict']:>12.1%}{ci:>16}{d:>9}"
              f"{s['capped']/s['n']:>9.1%}")
        rows_csv.append(dict(config=tag, n=s["n"], n_answered=s["answered"],
                             acc_answered=round(s["acc_answered"], 4),
                             acc_strict=round(s["acc_strict"], 4),
                             ci_lo=round(s["ci"][0], 4), ci_hi=round(s["ci"][1], 4)))
    _write_csv(os.path.join(TABLES, "accuracy_per_config.csv"), rows_csv)

    # ---- flips vs clean: ANSWERED-only (truncated pairs excluded) ---------
    # A question contributes to flips only if it was answered in BOTH clean and
    # the config. Ratios are over that answered base (n_both), reported per row.
    print("\n=== flips vs clean (answered-only; ratios over n_both) ===")
    print(f"{'config':16s}{'n_both':>7}{'dropped':>8}{'r→w':>6}{'w→r':>6}"
          f"{'r→w%':>7}{'w→r%':>7}{'net%':>7}{'McNemar p':>11}")
    flip_csv = []
    for tag in tags:
        if tag == "clean":
            continue
        ev = EV[tag]
        n_both = r2w = w2r = dropped = 0
        bc, cc = {}, {}     # answered-both correctness for McNemar
        for idx, b in base.items():
            if idx not in ev:
                continue
            if not (b["answered"] and ev[idx]["answered"]):
                dropped += 1          # unanswered in clean or config -> excluded
                continue
            n_both += 1
            bc[idx] = b["correct"]; cc[idx] = ev[idx]["correct"]
            if b["correct"] and not ev[idx]["correct"]:
                r2w += 1
            elif not b["correct"] and ev[idx]["correct"]:
                w2r += 1
        r2w_r = r2w / n_both if n_both else 0.0
        w2r_r = w2r / n_both if n_both else 0.0
        net_r = r2w_r - w2r_r
        b_, c_, stat, p = mcnemar(bc, cc)
        pstr = f"{p:.2e}" if p is not None else "n/a"
        print(f"{tag:16s}{n_both:>7}{dropped:>8}{r2w:>6}{w2r:>6}"
              f"{r2w_r:>7.1%}{w2r_r:>7.1%}{net_r:>7.1%}{pstr:>11}")
        flip_csv.append(dict(config=tag, n_both=n_both,
                             r2w=r2w, w2r=w2r,
                             r2w_ratio=round(r2w_r, 4), w2r_ratio=round(w2r_r, 4),
                             net_flip_ratio=round(net_r, 4),
                             mcnemar_b=b_, mcnemar_c=c_, mcnemar_stat=round(stat, 3),
                             mcnemar_p=(round(p, 8) if p is not None else None)))
    _write_csv(os.path.join(TABLES, "flips_vs_clean.csv"), flip_csv)

    # ---- decomposition: how much of the drop is "didn't finish" -----------
    print("\n=== accuracy-drop decomposition vs clean "
          "(Δ = completion-loss + quality-loss) ===")
    print(f"{'config':16s}{'Δstrict':>9}{'from not finishing':>20}{'from wrong answers':>20}")
    dec_csv = []
    aa_c, comp_c = cs["acc_answered"], cs["completion"]
    for tag in tags:
        if tag == "clean":
            continue
        s = S[tag]
        delta = s["acc_strict"] - cs["acc_strict"]                 # negative = drop
        completion_loss = s["acc_answered"] * (s["completion"] - comp_c)   # comp part
        quality_loss = comp_c * (s["acc_answered"] - aa_c)                 # quality part
        print(f"{tag:16s}{delta:>9.1%}{completion_loss:>20.1%}{quality_loss:>20.1%}")
        dec_csv.append(dict(config=tag, delta_strict=round(delta, 4),
                            completion_component=round(completion_loss, 4),
                            quality_component=round(quality_loss, 4)))
    _write_csv(os.path.join(TABLES, "accuracy_decomposition.csv"), dec_csv)

    # ---- rate x real matrices --------------------------------------------
    for title, field, fmt in [("acc(strict)", "acc_strict", "{:.1%}"),
                              ("acc|answered", "acc_answered", "{:.1%}"),
                              ("truncation rate", "truncated", "{:.1%}"),
                              ("completion", "completion", "{:.1%}")]:
        print(f"\n=== {title} matrix (rows=typo rate, cols=real ratio) "
              f"| clean={fmt.format(cs[field])} ===")
        rates = sorted({meta[t]["rate"] for t in tags if not meta[t]["is_clean"]})
        reals = sorted({meta[t]["real"] for t in tags if not meta[t]["is_clean"]})
        print("rate\\real" + "".join(f"{f'real{r}':>10}" for r in reals))
        for rate in rates:
            line = f"{rate:>7}% "
            for real in reals:
                tag = f"typo{rate}_real{real}"
                line += f"{fmt.format(S[tag][field]) if tag in S else '-':>10}"
            print(line)

    print(f"\ntables written to {TABLES}")


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
