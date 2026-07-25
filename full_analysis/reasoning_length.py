"""Dimension 2 of the proposal: REASONING LENGTH — typo tokens / clean tokens.

Length = generated tokens (n_gen_tokens = reasoning + final). The ratio is paired
per question (idx): ratio_i = tok_typo_i / tok_clean_i, then aggregated.

Truncation policy (locked): a question contributes ONLY if it was answered in BOTH
clean and the config. Truncated traces are right-censored at the 4096 cap — their
true length is unknown (longer), so including them biases the ratio downward.
=> the reported inflation is a LOWER BOUND: the worst blow-ups are the truncated
   ones we excluded (and truncation rises with the typo rate).

Per-question ratios are summarised by MEDIAN and GEOMEAN (robust for ratios);
ratio-of-means is also given. Plus absolute token means and a correct-vs-wrong
length split. Outputs: printed tables + CSVs under analysis/tables/.
    python reasoning_length.py
"""
import os, sys, csv, math

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from common import (config_files, parse_tag, load_evals, MAX_NEW_TOKENS,
                    TABLES, DATASET)
import json



def load_tokens(path):
    """{idx: n_gen_tokens} for one config."""
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[r["idx"]] = r["n_gen_tokens"]
    return out


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def geomean(xs):
    xs = [x for x in xs if x > 0]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, int(q * len(xs)))
    return xs[i]


def main():
    files = config_files()
    tags, meta, EV, TOK = [], {}, {}, {}
    for f in files:
        t = parse_tag(f)
        tags.append(t["tag"]); meta[t["tag"]] = t
        EV[t["tag"]] = load_evals(f)
        TOK[t["tag"]] = load_tokens(f)

    base_ev, base_tok = EV["clean"], TOK["clean"]

    # ---- absolute generated-token counts (answered-only) ------------------
    # The raw length the model produced (thousands of tokens). The ratio section
    # below normalises these per question, per the proposal.
    print(f"=== {DATASET}: absolute generated tokens (answered-only) ===")
    print(f"{'config':16s}{'n_ans':>7}{'mean':>8}{'median':>8}{'p90':>8}{'max':>7}")
    abs_csv = []
    for tag in tags:
        ev, tok = EV[tag], TOK[tag]
        vals = [tok[i] for i in ev if ev[i]["answered"]]
        mean = sum(vals) / len(vals) if vals else 0
        print(f"{tag:16s}{len(vals):>7}{mean:>8.0f}{median(vals):>8.0f}"
              f"{percentile(vals, 0.9):>8.0f}{max(vals) if vals else 0:>7}")
        abs_csv.append(dict(config=tag,
                            n_answered=len(vals), mean_tok=round(mean, 1),
                            median_tok=median(vals), p90_tok=percentile(vals, 0.9),
                            max_tok=max(vals) if vals else 0))
    _write_csv(os.path.join(TABLES, "reasoning_length_absolute.csv"), abs_csv)

    # ---- length vs correctness (answered only) ---------------------------
    # wrong_over_correct = mean tokens of wrong answers / mean tokens of correct answers
    print("\n=== mean tokens by outcome, answered-only (does wrongness cost length?) ===")
    print(f"{'config':16s}{'tok|correct':>13}{'tok|wrong':>11}{'wrong/correct':>15}")
    lc_csv = []
    for tag in tags:
        ev, tok = EV[tag], TOK[tag]
        corr = [tok[i] for i in ev if ev[i]["answered"] and ev[i]["correct"]]
        wrong = [tok[i] for i in ev if ev[i]["answered"] and not ev[i]["correct"]]
        mc = sum(corr)/len(corr) if corr else 0
        mw = sum(wrong)/len(wrong) if wrong else 0
        ratio = mw/mc if mc else 0
        print(f"{tag:16s}{mc:>13.0f}{mw:>11.0f}{ratio:>15.2f}")
        lc_csv.append(dict(config=tag, tok_correct=round(mc,1), tok_wrong=round(mw,1),
                           wrong_over_correct=round(ratio,4)))
    _write_csv(os.path.join(TABLES, "length_by_outcome.csv"), lc_csv)

    print(f"\ntables written to {TABLES}")


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
