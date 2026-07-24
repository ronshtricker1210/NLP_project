"""Dimension 4 of the proposal: REPAIR BEHAVIOR — how the model handled the typo.

Two measures, per the proposal ("classify each trace ... using an LLM judge"):

1. LEXICAL measure (offline, runs now):
   - notice type: EXPLICIT (the trace flags a typo/error and/or restates the
     intended word) vs SILENT (no such language) — from the typo-noticing bank.
   - repair words: active corrective language ("read it as", "should be",
     "assume they mean", "correcting the typo", ...).
   Crossed with correctness this gives the proposal's behavioural cells:
     silent+correct  = silent read-through that worked
     silent+wrong    = SILENT FAILURE (the "cheap but dangerous" case)
     explicit+correct= noticed & fixed
     explicit+wrong  = noticed but failed to fix

2. LLM-JUDGE measure (the proposal's method): a strict fixed prompt classifies
   each trace into one category. Implemented and ready, but requires an API token
   and is only run when --judge is passed (costs tokens); otherwise skipped.

Answered-only. Clean is included for reference (its "typos" are none, so ~all
silent). Outputs: printed tables + CSVs under analysis/tables/.
    python repair_behavior.py                 # lexical only
    python repair_behavior.py --judge --limit 50   # + LLM judge on a subset
"""
import os, sys, re, csv, json, argparse
from collections import defaultdict, Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from common import config_files, parse_tag, load_evals, _HERE

TABLES = os.path.join(_HERE, "tables")
os.makedirs(TABLES, exist_ok=True)

# Typo-NOTICING bank — the model explicitly flags/interprets the corrupted text.
# This is repair behaviour (not self-doubt), so it lives here. Words chosen from the
# clean-vs-typo discrimination test ("typo" is the star: 1.2% of clean traces vs
# 44% of heavy-typo traces).
TYPO_NOTICING = {
    "typo":                 r"\btypos?\b",
    "misspell":             r"\bmis-?spell(?:ed|ing|s|ings)?\b",
    "probably/might mean":  r"\b(?:probably|might|could|must|likely|maybe)\s+(?:be\s+)?(?:a\s+typo|means?|meant)\b",
    "they/user mean":       r"\b(?:they|the user|it|author|question)\s+(?:mean|means|meant)\b",
    "assume/interpret":     r"\b(?:assum(?:e|es|ed|ing)|interpret(?:s|ed|ing)?)\b",
    "should be/say/read":   r"\bshould\s+(?:be|say|read|probably\s+be)\b",
    "doesn't make sense":   r"\b(?:does(?:n'?t| not)|didn'?t)\s+make\s+sense\b",
    "seems like typo/error":r"\b(?:seems|looks)\s+like\s+(?:a\s+)?(?:typo|misspelling|error|mistake)\b",
    "strange/weird/odd":    r"\b(?:strange|weird|odd|garbled)\b",
    "meant to be/say":      r"\bmeant\s+to\s+(?:be|say|read|write)\b",
    "error/mistake in":     r"\b(?:error|mistake|typos?)\s+in\s+the\b",
}
TN_COMPILED = {k: re.compile(v, re.I) for k, v in TYPO_NOTICING.items()}

# Active repair / correction language (distinct from merely noticing a problem).
REPAIR_MARKERS = {
    "read/treat it as":  r"\b(?:read|take|treat|interpret)\s+(?:it|this|that|the word|the problem)\s+as\b",
    "should be/say":     r"\bshould\s+(?:be|say|read|probably\s+be)\b",
    "meant/supposed to": r"\b(?:meant|supposed|intended)\s+to\s+(?:be|say|read|write|mean)\b",
    "assume/interpret":  r"\b(?:i(?:'?ll| will)?\s+)?(?:assum(?:e|es|ed|ing)|interpret(?:s|ed|ing)?)\b",
    "typo for / means":  r"\b(?:typo for|(?:probably|likely|must)\s+means?|means?\s+to\s+say)\b",
    "correct/fix typo":  r"\b(?:correct(?:ing|ed)?|fix(?:ing|ed)?)\s+(?:the\s+)?(?:typo|spelling|word|error|mistake)\b",
    "rewrite/rephrase":  r"\b(?:rewrit\w+|rephras\w+|reinterpret\w+)\b",
}
RM_COMPILED = {k: re.compile(v, re.I) for k, v in REPAIR_MARKERS.items()}

# A trace "explicitly notices" when it uses strong typo-flagging language. We
# require the high-precision markers (saying "typo"/"misspell"/"error in the"/
# "seems like a typo"/"doesn't make sense") rather than any interpretive verb,
# so ordinary "assume"/"means" in clean math doesn't count as noticing.
STRONG_NOTICE = {"typo", "misspell", "seems like typo/error", "doesn't make sense",
                 "error/mistake in", "meant to be/say", "strange/weird/odd"}


def classify_trace(reasoning):
    """Return (notice_type, n_notice, n_repair) for one reasoning string."""
    strong = sum(len(TN_COMPILED[k].findall(reasoning)) for k in STRONG_NOTICE)
    n_notice = sum(len(rx.findall(reasoning)) for rx in TN_COMPILED.values())
    n_repair = sum(len(rx.findall(reasoning)) for rx in RM_COMPILED.values())
    notice_type = "explicit" if strong > 0 else "silent"
    return notice_type, n_notice, n_repair


def load_traces(path):
    ev = load_evals(path)
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        e = ev[r["idx"]]
        if not e["answered"]:
            continue
        reasoning = r.get("reasoning", "") or ""
        nt, n_notice, n_repair = classify_trace(reasoning)
        out.append(dict(idx=r["idx"], notice=nt, n_notice=n_notice, n_repair=n_repair,
                        words=max(len(reasoning.split()), 1),
                        correct=e["correct"], real=r.get("real_ratio"),
                        rate=None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="also run the LLM-judge classifier (needs HF_TOKEN, costs tokens)")
    ap.add_argument("--limit", type=int, default=50, help="traces/config for the judge")
    args = ap.parse_args()

    files = config_files()
    tags, meta, DATA = [], {}, {}
    for f in files:
        t = parse_tag(f)
        tags.append(t["tag"]); meta[t["tag"]] = t
        DATA[t["tag"]] = load_traces(f)

    # ---- notice type x outcome (the 4 behavioural cells) ------------------
    print("=== gsm8k: repair behaviour — notice type x outcome (answered-only, %) ===")
    print(f"{'config':16s}{'n_ans':>7}{'explicit%':>10}"
          f"{'silent+OK':>11}{'silent+WRONG':>13}{'expl+OK':>9}{'expl+WRONG':>11}")
    cell_csv = []
    for tag in tags:
        recs = DATA[tag]; n = max(len(recs), 1)
        cell = Counter()
        for x in recs:
            cell[(x["notice"], x["correct"])] += 1
        expl = sum(1 for x in recs if x["notice"] == "explicit") / n
        s_ok = cell[("silent", True)] / n
        s_wr = cell[("silent", False)] / n
        e_ok = cell[("explicit", True)] / n
        e_wr = cell[("explicit", False)] / n
        print(f"{tag:16s}{len(recs):>7}{expl:>10.1%}{s_ok:>11.1%}{s_wr:>13.1%}"
              f"{e_ok:>9.1%}{e_wr:>11.1%}")
        cell_csv.append(dict(config=tag,
                             n_answered=len(recs), explicit_frac=round(expl, 4),
                             silent_correct=round(s_ok, 4), silent_wrong=round(s_wr, 4),
                             explicit_correct=round(e_ok, 4), explicit_wrong=round(e_wr, 4)))
    _write_csv(os.path.join(TABLES, "repair_notice_outcome.csv"), cell_csv)

    # ---- does explicit noticing help? accuracy within notice type --------
    print("\n=== accuracy within notice type (does noticing the typo help?) ===")
    print(f"{'config':16s}{'acc|silent':>12}{'acc|explicit':>14}{'n_silent':>10}{'n_explicit':>12}")
    for tag in tags:
        recs = DATA[tag]
        sil = [x for x in recs if x["notice"] == "silent"]
        exp = [x for x in recs if x["notice"] == "explicit"]
        a_s = sum(x["correct"] for x in sil) / len(sil) if sil else 0
        a_e = sum(x["correct"] for x in exp) / len(exp) if exp else 0
        print(f"{tag:16s}{a_s:>12.1%}{a_e:>14.1%}{len(sil):>10}{len(exp):>12}")

    # ---- repair-word density per config ----------------------------------
    print("\n=== repair-word usage (active correction language) ===")
    print(f"{'config':16s}{'repair/1k':>11}{'%>=1 repair':>13}")
    rw_csv = []
    for tag in tags:
        recs = DATA[tag]; n = max(len(recs), 1)
        m = sum(x["n_repair"] for x in recs); w = sum(x["words"] for x in recs)
        dns = m / w * 1000 if w else 0
        anyr = sum(1 for x in recs if x["n_repair"] > 0) / n
        print(f"{tag:16s}{dns:>11.2f}{anyr:>13.1%}")
        rw_csv.append(dict(config=tag, repair_per_1k=round(dns, 3), frac_any_repair=round(anyr, 4)))
    _write_csv(os.path.join(TABLES, "repair_words_per_config.csv"), rw_csv)

    # ---- proposal hypothesis: real-word typos pass UNNOTICED --------------
    # At each real ratio (pooled over rate), what fraction of traces explicitly
    # notice? If real-word typos are silent, explicit% should FALL as real rises.
    print("\n=== hypothesis: explicit-notice rate vs real-word ratio (pooled typo configs) ===")
    print(f"{'real ratio':>12}{'explicit%':>11}{'silent+wrong%':>15}{'n':>7}")
    by_real = defaultdict(list)
    for tag in tags:
        if meta[tag]["is_clean"]:
            continue
        for x in DATA[tag]:
            by_real[meta[tag]["real"]].append(x)
    hyp_csv = []
    for real in sorted(by_real):
        recs = by_real[real]; n = len(recs)
        expl = sum(1 for x in recs if x["notice"] == "explicit") / n
        sw = sum(1 for x in recs if x["notice"] == "silent" and not x["correct"]) / n
        print(f"{f'real{real}':>12}{expl:>11.1%}{sw:>15.1%}{n:>7}")
        hyp_csv.append(dict(real_ratio=real, explicit_frac=round(expl, 4),
                            silent_wrong_frac=round(sw, 4), n=n))
    _write_csv(os.path.join(TABLES, "repair_notice_vs_realratio.csv"), hyp_csv)

    if args.judge:
        run_judge(files, meta, args.limit)
    else:
        print("\n[LLM judge skipped — pass --judge (needs HF_TOKEN) to run the "
              "strict-prompt classifier as the proposal's primary measure]")

    print(f"\ntables written to {TABLES}")


# ---------------------------------------------------------------------------
# LLM-judge classifier (proposal's method). Strict fixed prompt, one category.
# ---------------------------------------------------------------------------
JUDGE_CATEGORIES = [
    "silent_readthrough",   # solved without ever flagging the typo
    "explicit_notice_fix",  # flagged the typo and recovered the intended word
    "misread_wrong_word",   # read the typo as a different real word and used it
    "confused_gaveup",      # got stuck / derailed by the corruption
    "no_typo",              # (clean) nothing to repair
]
JUDGE_PROMPT = """You are a strict annotator. You are given the ORIGINAL (clean) question, the
TYPO question actually shown to a model, and the model's REASONING trace. Classify
how the reasoning handled the typos into exactly ONE label:

- silent_readthrough: solved the intended problem without ever mentioning a typo/error.
- explicit_notice_fix: explicitly flagged a typo/misspelling/error and recovered the intended word.
- misread_wrong_word: interpreted a corrupted word as a DIFFERENT word than intended and reasoned from that.
- confused_gaveup: got derailed/stuck by the corruption without recovering.
- no_typo: there was effectively nothing to repair.

Answer with ONLY the label, nothing else.

ORIGINAL: {clean}
TYPO: {typo}
REASONING: {reasoning}
LABEL:"""


def run_judge(files, meta, limit):
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("\n[--judge] HF_TOKEN not set; cannot run the LLM judge. "
              "export HF_TOKEN=hf_... and retry.")
        return
    from openai import OpenAI
    client = OpenAI(api_key=token, base_url=os.environ.get("API_BASE", "https://router.huggingface.co/v1"))
    model = os.environ.get("JUDGE_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:nscale")
    print(f"\n=== LLM-judge repair classification (model={model}, limit={limit}/config) ===")
    rows = []
    for f in files:
        tag = parse_tag(f)["tag"]
        dist = Counter()
        for i, line in enumerate(open(f, encoding="utf-8")):
            if i >= limit:
                break
            r = json.loads(line)
            prompt = JUDGE_PROMPT.format(
                clean=r.get("clean_question", "")[:1500],
                typo=(r.get("typo_question") or r.get("clean_question", ""))[:1500],
                reasoning=(r.get("reasoning", "") or "")[:6000])
            try:
                resp = client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=8)
                label = (resp.choices[0].message.content or "").strip().split()[0].lower()
            except Exception as e:
                label = "error"
                print(f"  {tag} idx={r['idx']} judge failed: {str(e)[:80]}")
            dist[label] += 1
        total = sum(dist.values()) or 1
        summary = "  ".join(f"{k}={v}" for k, v in dist.most_common())
        print(f"{tag:16s} n={total:<4} {summary}")
        rows.append(dict(config=tag, n=total, **dict(dist)))
    _write_csv(os.path.join(TABLES, "repair_llm_judge.csv"), rows)


def _write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
