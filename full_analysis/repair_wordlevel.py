"""Repair behaviour, grounded in the ACTUAL corrupted words (not just keywords).

The result files store both clean_question and typo_question but NOT the list of
corrupted words, so we recover every (original -> corrupted) pair by word-aligning
the two (difflib). Then, for each corrupted word, we look at the reasoning trace to
see which form the model actually used — turning the proposal's three categories
into measurable, per-word outcomes:

  RECOVERED : original word present, corrupted form absent  -> silently normalised
  NOTICED   : both forms present                            -> flagged & fixed ("X is probably Y")
  ECHOED    : corrupted form present, original absent        -> used the corrupted form
              (for a REAL-word typo like sum->sun this is the MISREAD signature;
               for a non-word typo it is usually a quoted garble)
  UNUSED    : neither form appears in the reasoning          -> word not referenced

Hypothesis (proposal): non-word typos trigger visible reactions and are recovered;
real-word typos slip through as misreads. So RECOVERED should dominate at low real
ratio, ECHOED should rise with real ratio.

Caveats: recovery is a lower bound (the model may paraphrase); short/common words
match trivially, so we require length >= 3 and report the real10-vs-real70 CONTRAST
(same length mix) rather than absolute rates. Answered-only.

    python repair_wordlevel.py
"""
import os, sys, re, csv, json, difflib
from collections import defaultdict, Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from common import config_files, parse_tag, load_evals, _HERE

TABLES = os.path.join(_HERE, "tables")
WORD = re.compile(r"[A-Za-z']+")
MIN_LEN = 3   # ignore very short words (the/of/a) — they match trivially


def corrupted_pairs(clean, typo):
    """Word-align clean vs typo; return [(original, corrupted), ...] that differ."""
    cw, tw = WORD.findall(clean), WORD.findall(typo)
    sm = difflib.SequenceMatcher(a=[w.lower() for w in cw], b=[w.lower() for w in tw])
    pairs = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace":
            a, b = cw[i1:i2], tw[j1:j2]
            for k in range(min(len(a), len(b))):   # position-align within the block
                if a[k].lower() != b[k].lower():
                    pairs.append((a[k], b[k]))
    return pairs


def present(word, reasoning_low):
    """Whole-word, case-insensitive membership in the (lowercased) reasoning."""
    if len(word) < MIN_LEN:
        return False
    return re.search(r"\b" + re.escape(word.lower()) + r"\b", reasoning_low) is not None


def classify_word(orig, typo, reasoning_low):
    o = present(orig, reasoning_low)
    t = present(typo, reasoning_low)
    if o and not t:
        return "recovered"
    if o and t:
        return "noticed"
    if t and not o:
        return "echoed"
    return "unused"


def analyze_file(path):
    ev = load_evals(path)
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        e = ev[r["idx"]]
        if not e["answered"]:
            continue
        reasoning_low = (r.get("reasoning", "") or "").lower()
        pairs = corrupted_pairs(r["clean_question"], r["typo_question"])
        cats = Counter()
        usable = 0
        for orig, typo in pairs:
            if len(orig) < MIN_LEN:      # skip words we can't reliably search
                continue
            usable += 1
            cats[classify_word(orig, typo, reasoning_low)] += 1
        rows.append(dict(idx=r["idx"], correct=e["correct"], cats=cats,
                         usable=usable, real=r.get("real_ratio")))
    return rows


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


CATS = ("recovered", "noticed", "echoed", "unused")


def dist(recs):
    """Pooled per-word category counts and fractions over problem records.
    Returns (fractions, counts, total)."""
    tot = Counter()
    for r in recs:
        tot.update(r["cats"])
    n = sum(tot.values()) or 1
    counts = {k: tot[k] for k in CATS}
    return {k: counts[k] / n for k in CATS}, counts, n


def main():
    files = config_files()
    tags, meta, DATA = [], {}, {}
    for f in files:
        t = parse_tag(f)
        if t["is_clean"]:
            continue      # clean has no corrupted words
        tags.append(t["tag"]); meta[t["tag"]] = t
        DATA[t["tag"]] = analyze_file(f)

    # ---- per-config word-handling distribution ---------------------------
    print("=== how the model handled each corrupted word, per config (answered-only) ===")
    print("Each corrupted word (from diffing clean vs typo) goes in ONE bucket by which form the")
    print("reasoning used. Columns are the % share of that config's corrupted words:")
    print("  n_corrupt_words = total corrupted words examined (the denominator)")
    print("  silent_fix%     = used ONLY the original word        (recovered silently)")
    print("  flagged%        = used BOTH the typo and the original (noticed & fixed)")
    print("  misread%        = used ONLY the corrupted word        (echoed / read as wrong word)")
    print("  not_used%       = neither form appears in the reasoning\n")
    print(f"{'config':16s}{'n_corrupt_words':>16}{'silent_fix%':>12}{'flagged%':>10}"
          f"{'misread%':>10}{'not_used%':>11}")
    cfg_csv = []
    for tag in tags:
        d, c, n = dist(DATA[tag])
        print(f"{tag:16s}{n:>16}{d['recovered']:>12.1%}{d['noticed']:>10.1%}"
              f"{d['echoed']:>10.1%}{d['unused']:>11.1%}")
        cfg_csv.append(dict(config=tag, n_corrupt_words=n,
                            silent_fix_n=c["recovered"], flagged_n=c["noticed"],
                            misread_n=c["echoed"], not_used_n=c["unused"],
                            silent_fix_frac=round(d["recovered"], 4),
                            flagged_frac=round(d["noticed"], 4),
                            misread_frac=round(d["echoed"], 4),
                            not_used_frac=round(d["unused"], 4)))
    _write_csv(os.path.join(TABLES, "repair_wordlevel_per_config.csv"), cfg_csv)

    # ---- the hypothesis: by real ratio (rates pooled) --------------------
    print("\n=== word handling BY REAL-WORD RATIO (rates pooled) — the key contrast ===")
    print("expect: non-word (real10) -> more silent_fix; real-word (real70) -> more misread\n")
    print(f"{'real ratio':>12}{'n_corrupt_words':>16}{'silent_fix%':>12}{'flagged%':>10}"
          f"{'misread%':>10}{'not_used%':>11}")
    by_real = defaultdict(list)
    for tag in tags:
        by_real[meta[tag]["real"]].extend(DATA[tag])
    real_csv = []
    for real in sorted(by_real):
        d, c, n = dist(by_real[real])
        print(f"{f'real{real}':>12}{n:>16}{d['recovered']:>12.1%}{d['noticed']:>10.1%}"
              f"{d['echoed']:>10.1%}{d['unused']:>11.1%}")
        real_csv.append(dict(real_ratio=real, n_corrupt_words=n,
                             silent_fix_frac=round(d["recovered"], 4),
                             flagged_frac=round(d["noticed"], 4),
                             misread_frac=round(d["echoed"], 4),
                             not_used_frac=round(d["unused"], 4)))
    _write_csv(os.path.join(TABLES, "repair_wordlevel_by_real.csv"), real_csv)

    # ---- does misreading predict a wrong answer? -------------------------
    # Per problem: misread rate among its corrupted words, split by final correctness.
    print("\n=== misread rate (used corrupted word) per problem, by final outcome ===")
    print(f"{'config':16s}{'misread%|correct':>18}{'misread%|wrong':>16}{'wrong/correct':>15}")
    oc_csv = []
    for tag in tags:
        def misread_rate(recs):
            e = sum(r["cats"]["echoed"] for r in recs)
            u = sum(r["usable"] for r in recs)
            return e / u if u else 0
        corr = [r for r in DATA[tag] if r["correct"]]
        wrong = [r for r in DATA[tag] if not r["correct"]]
        ec, ew = misread_rate(corr), misread_rate(wrong)
        print(f"{tag:16s}{ec:>18.1%}{ew:>16.1%}{(ew/ec if ec else 0):>15.2f}")
        oc_csv.append(dict(config=tag, misread_correct=round(ec, 4), misread_wrong=round(ew, 4),
                           wrong_over_correct=round(ew/ec if ec else 0, 3)))
    _write_csv(os.path.join(TABLES, "repair_wordlevel_by_outcome.csv"), oc_csv)

    print(f"\ntables written to {TABLES}")


if __name__ == "__main__":
    main()
