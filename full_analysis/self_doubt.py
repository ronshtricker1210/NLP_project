"""Dimension 3 of the proposal: SELF-DOUBT via doubt-marker frequency.

(This is the lexical half only — the separate LLM-judge rating is intentionally
NOT implemented here.)

For each ANSWERED trace we count doubt markers in the reasoning text and report
two things, because markers and trace length are entangled (doubt makes traces
longer, so a pure per-token rate can hide the effect):
  - absolute: mean markers per trace
  - density : markers per 1,000 reasoning words

Markers are a curated, transparent set anchored on the proposal's own examples
(wait / hmm / actually / let me reconsider) plus tight variants, counted
individually so the mix is visible.

Key hypothesis test (proposal): real-word typos cause SILENT confident failures.
=> among WRONG answers, doubt should FALL as the real-word ratio rises.

Answered-only (truncated traces excluded). Outputs: printed tables + CSVs.
    python self_doubt.py
"""
import os, sys, re, csv, json, argparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from common import config_files, parse_tag, load_evals, _HERE

TABLES = os.path.join(_HERE, "tables")
os.makedirs(TABLES, exist_ok=True)

# Doubt markers in two transparent categories, chosen after checking each one's
# clean-vs-typo discrimination on this data (see the commit notes / word-bank test):
#   second_guess = the proposal's canonical set + direct reversals. Note that the
#                  proposal's own "let me reconsider" is ~0 here (kept for fidelity),
#                  and "wait"/"hmm" are near-baseline (in ~97% of CLEAN traces).
#   uncertainty  = hedging / "not certain" family; "maybe" carries the most signal.
# Typo-NOTICING words (typo, misspell, "probably means") are deliberately excluded —
# those are repair behaviour (dimension 4), not internal second-guessing.
MARKER_CATS = {
    "second_guess": {
        "wait":              r"\bwait\b",
        "hmm":               r"\bhm+\b",
        "actually":          r"\bactually\b",
        "reconsider":        r"\breconsider\b",          # incl. "let me reconsider" (~0 here)
        "but wait":          r"\bbut,? wait\b",
        "wait/actually no":  r"\b(?:wait,? no|actually,? no|no,? wait)\b",
        "hold on":           r"\bhold on\b",
        "on second thought": r"\bon second thought\b",
        "let me recheck":    r"\blet me (?:re-?check|recheck|re-?read|reread|verify|redo|double-?check)\b",
        "double-check":      r"\bdouble[- ]?check(?:ing|ed)?\b",
    },
    "uncertainty": {
        "not sure":            r"\b(?:not sure|unsure|not certain|not entirely sure|not totally sure)\b",
        "maybe":               r"\bmaybe\b",
        "perhaps":             r"\bperhaps\b",
        "possibly":            r"\bpossibly\b",
        "might be":            r"\bmight be\b",
        "could be":            r"\bcould be\b",
        "I guess":             r"\bi guess\b",
        "I assume":            r"\bi(?:'?ll)? assume\b",
        "presumably":          r"\bpresumably\b",
        "or maybe/alt":        r"\b(?:or maybe|alternatively|then again)\b",
        "unclear/confused":    r"\b(?:unclear|not clear|isn'?t clear|hard to tell|ambiguous|confus(?:ed|ing))\b",
        "not entirely":        r"\bnot entirely\b",
    },
}
# flat name -> (category, compiled regex)
MARKERS = {name: cat for cat, d in MARKER_CATS.items() for name in d}
COMPILED = {name: re.compile(rx, re.I) for cat, d in MARKER_CATS.items() for name, rx in d.items()}
CATS = list(MARKER_CATS)


def count_markers(text):
    """Return {marker: count}, {category: count}, and grand total for one string."""
    counts = {k: len(rx.findall(text)) for k, rx in COMPILED.items()}
    cat_counts = {c: sum(counts[k] for k in MARKER_CATS[c]) for c in CATS}
    return counts, cat_counts, sum(counts.values())


def analyze_file(path):
    """Per answered trace: marker counts, category counts, reasoning words, outcome."""
    ev = load_evals(path)
    recs = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        e = ev[r["idx"]]
        if not e["answered"]:
            continue
        reasoning = r.get("reasoning", "") or ""
        counts, cat_counts, total = count_markers(reasoning)
        words = max(len(reasoning.split()), 1)
        recs.append(dict(idx=r["idx"], counts=counts, cat=cat_counts, total=total,
                         words=words, correct=e["correct"], real=r.get("real_ratio")))
    return recs


# Which doubt categories are active (set from --no-2guess in main). "total" always
# means the sum over ACTIVE_CATS, so excluding a category also drops it from totals.
ACTIVE_CATS = list(MARKER_CATS)
SHORT = {"second_guess": "2guess", "uncertainty": "uncert"}
SHORT2 = {"second_guess": "2g", "uncertainty": "un", "total": "tot"}


def density(recs, key="total"):
    """markers per 1000 words over a group of records.
    key='total' = sum over ACTIVE_CATS; otherwise a single category name."""
    cats = ACTIVE_CATS if key == "total" else [key]
    m = sum(sum(x["cat"][c] for c in cats) for x in recs)
    w = sum(x["words"] for x in recs)
    return (m / w * 1000) if w else 0.0


def per_trace(recs, key="total"):
    """mean markers per trace; key='total' = sum over ACTIVE_CATS, else one category."""
    if not recs:
        return 0.0
    cats = ACTIVE_CATS if key == "total" else [key]
    return sum(sum(x["cat"][c] for c in cats) for x in recs) / len(recs)


def main():
    global ACTIVE_CATS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-2guess", action="store_true",
                    help="exclude the (flat, uninformative) second_guess category")
    args = ap.parse_args()
    if args.no_2guess:
        ACTIVE_CATS = ["uncertainty"]
    multi = len(ACTIVE_CATS) > 1   # only show a 'total' column when >1 category

    files = config_files()
    tags, meta, DATA = [], {}, {}
    for f in files:
        t = parse_tag(f)
        tags.append(t["tag"]); meta[t["tag"]] = t
        DATA[t["tag"]] = analyze_file(f)

    # ---- per-config summary (by category) --------------------------------
    active_str = " + ".join(ACTIVE_CATS)
    print(f"=== gsm8k: self-doubt marker frequency (answered-only) | categories: {active_str} ===")
    print("/1k = markers per 1,000 reasoning words (length-normalised);"
          " /tr = mean markers per answer\n")
    cols = [f"{SHORT[c]}/1k" for c in ACTIVE_CATS] + (["total/1k"] if multi else []) \
         + [f"{SHORT[c]}/tr" for c in ACTIVE_CATS] + (["total/tr"] if multi else [])
    print(f"{'config':16s}{'n_ans':>7}" + "".join(f"{c:>11}" for c in cols))
    sum_csv = []
    for tag in tags:
        recs = DATA[tag]
        d = {c: density(recs, c) for c in ACTIVE_CATS}
        t = {c: per_trace(recs, c) for c in ACTIVE_CATS}
        vals = [d[c] for c in ACTIVE_CATS] + ([density(recs, "total")] if multi else []) \
             + [t[c] for c in ACTIVE_CATS] + ([per_trace(recs, "total")] if multi else [])
        print(f"{tag:16s}{len(recs):>7}" + "".join(f"{v:>11.2f}" for v in vals))
        row = dict(config=tag, n_answered=len(recs))
        for c in ACTIVE_CATS:
            row[f"{c}_per_1k"] = round(d[c], 3)
            row[f"{c}_per_trace"] = round(t[c], 3)
        if multi:
            row["total_per_1k"] = round(density(recs, "total"), 3)
            row["total_per_trace"] = round(per_trace(recs, "total"), 3)
        sum_csv.append(row)
    _write_csv(os.path.join(TABLES, "self_doubt_per_config.csv"), sum_csv)

    # ---- per-marker: clean vs pooled-typo density + discrimination -------
    # Shows which individual markers actually respond to typos (delta vs clean).
    clean_recs = DATA["clean"]
    typo_recs = [x for tag in tags if not meta[tag]["is_clean"] for x in DATA[tag]]
    print("\n=== per-marker density (mk/1k words): clean vs pooled-typo, by category ===")
    print(f"{'marker':20s}{'cat':>14}{'clean':>9}{'typo':>9}{'Δ (signal)':>12}")
    mk_csv = []
    def mdensity(recs, name):
        m = sum(x["counts"][name] for x in recs); w = sum(x["words"] for x in recs)
        return (m / w * 1000) if w else 0.0
    for cat in ACTIVE_CATS:
        for name in MARKER_CATS[cat]:
            dc, dt = mdensity(clean_recs, name), mdensity(typo_recs, name)
            print(f"{name:20s}{cat:>14}{dc:>9.2f}{dt:>9.2f}{dt-dc:>+12.2f}")
            mk_csv.append(dict(marker=name, category=cat, clean_per_1k=round(dc, 3),
                               typo_per_1k=round(dt, 3), discrimination=round(dt-dc, 3)))
    _write_csv(os.path.join(TABLES, "self_doubt_by_marker.csv"), mk_csv)

    # ---- doubt by outcome (correct vs wrong), split by category ----------
    keys = list(ACTIVE_CATS) + (["total"] if multi else [])
    print("\n=== self-doubt density by outcome (markers per 1,000 words) ===")
    print("per category; w/c = wrong-over-correct ratio\n")
    hdr = f"{'config':16s}"
    for k in keys:
        s = SHORT2[k]
        hdr += f"{s+'_corr':>9}{s+'_wrong':>10}{s+'_w/c':>8}"
    print(hdr)
    oc_csv = []
    for tag in tags:
        recs = DATA[tag]
        corr = [x for x in recs if x["correct"]]
        wrong = [x for x in recs if not x["correct"]]
        row = {"config": tag}
        line = f"{tag:16s}"
        for k in keys:
            dc, dw = density(corr, k), density(wrong, k)
            wc = dw / dc if dc else 0
            line += f"{dc:>9.2f}{dw:>10.2f}{wc:>8.2f}"
            row[f"{SHORT2[k]}_correct"] = round(dc, 3)
            row[f"{SHORT2[k]}_wrong"] = round(dw, 3)
            row[f"{SHORT2[k]}_wrong_over_correct"] = round(wc, 3)
        print(line)
        oc_csv.append(row)
    _write_csv(os.path.join(TABLES, "self_doubt_by_outcome.csv"), oc_csv)

    # NOTE: real-word-ratio comparisons (doubt vs real10/40/70, silent-failure test)
    # live in real_word_effect.py, which isolates the real-word axis properly. They
    # were removed here to keep self_doubt focused on the doubt dimension per config.

    print(f"\ntables written to {TABLES}")


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
