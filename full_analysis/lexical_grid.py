"""Lexical marker density across the full rate x real grid, for all four families:
    self-doubt : second_guess, uncertainty
    repair     : typo_noticing, repair_words

For each family it reports density = markers per 1,000 reasoning words, per config
(so the typo RATE is visible, not pooled away), plus a rate x real matrix per family.
Answered-only. Outputs printed tables + one CSV.
    python lexical_grid.py
"""
import os, sys, json, csv

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from common import config_files, parse_tag, load_evals, _HERE
from self_doubt import MARKER_CATS, COMPILED
from marker_banks import RM_COMPILED, TN_COMPILED

TABLES = os.path.join(_HERE, "tables")

FAMILIES = ["second_guess", "uncertainty", "typo_noticing", "repair_words"]


def family_counts(text):
    sg = sum(len(COMPILED[k].findall(text)) for k in MARKER_CATS["second_guess"])
    un = sum(len(COMPILED[k].findall(text)) for k in MARKER_CATS["uncertainty"])
    tn = sum(len(rx.findall(text)) for rx in TN_COMPILED.values())
    rw = sum(len(rx.findall(text)) for rx in RM_COMPILED.values())
    return {"second_guess": sg, "uncertainty": un, "typo_noticing": tn, "repair_words": rw}


def config_density(path):
    """Return {family: per-1k-word density} over answered traces of one config."""
    ev = load_evals(path)
    tot = {f: 0 for f in FAMILIES}
    words = 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if not ev[r["idx"]]["answered"]:
            continue
        reasoning = r.get("reasoning", "") or ""
        c = family_counts(reasoning)
        for f in FAMILIES:
            tot[f] += c[f]
        words += max(len(reasoning.split()), 1)
    return {f: (tot[f] / words * 1000 if words else 0.0) for f in FAMILIES}, words


def main():
    files = config_files()
    tags, meta, D = [], {}, {}
    for f in files:
        t = parse_tag(f)
        tags.append(t["tag"]); meta[t["tag"]] = t
        D[t["tag"]], _ = config_density(f)

    # ---- one combined per-config table ------------------------------------
    print("=== marker density per 1,000 reasoning words, per config (answered-only) ===")
    print(f"{'config':16s}"
          f"{'2nd_guess':>11}{'uncert':>9}{'typo_notice':>13}{'repair':>9}")
    rows = []
    for tag in tags:
        d = D[tag]
        print(f"{tag:16s}"
              f"{d['second_guess']:>11.2f}{d['uncertainty']:>9.2f}"
              f"{d['typo_noticing']:>13.2f}{d['repair_words']:>9.2f}")
        rows.append(dict(config=tag,
                         second_guess=round(d["second_guess"], 3),
                         uncertainty=round(d["uncertainty"], 3),
                         typo_noticing=round(d["typo_noticing"], 3),
                         repair_words=round(d["repair_words"], 3)))
    with open(os.path.join(TABLES, "lexical_grid.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- one rate x real matrix per family --------------------------------
    rates = sorted({meta[t]["rate"] for t in tags if not meta[t]["is_clean"]})
    reals = sorted({meta[t]["real"] for t in tags if not meta[t]["is_clean"]})
    clean = D["clean"]
    for fam in FAMILIES:
        print(f"\n=== {fam} density (rows=typo rate, cols=real ratio) | clean={clean[fam]:.2f} ===")
        print("rate\\real" + "".join(f"{f'real{r}':>10}" for r in reals))
        for rate in rates:
            line = f"{rate:>7}% "
            for real in reals:
                tag = f"typo{rate}_real{real}"
                line += f"{D[tag][fam]:>10.2f}" if tag in D else f"{'-':>10}"
            print(line)

    print(f"\ntable written to {os.path.join(TABLES, 'lexical_grid.csv')}")


if __name__ == "__main__":
    main()
