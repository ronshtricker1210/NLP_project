"""LLM-as-judge measures for SELF-DOUBT (dimension 3) and REPAIR BEHAVIOUR (dim 4).

The proposal asks for both dimensions to be measured with a separate LLM judge
("a rating from a separate LLM judge with a fixed prompt" / "we classify each
trace into behavioral categories using an LLM judge with strict instructions").
The existing modules measure them lexically:

    self_doubt.py       doubt-marker density (regex banks, per 1k words)
    repair_wordlevel.py which form of each corrupted word appears in the trace

This module ADDS the judge measure next to them and reports how far the two
agree (Cohen's kappa for repair, rank correlation for doubt), so the lexical
tables stay the full-data primary result and the judge validates them.

Nothing has to be re-generated: it reads the SAME data/<dataset>/*.jsonl traces.

One call per trace returns BOTH measures (half the cost of two passes):
    repair_label   one of REPAIR_LABELS
    doubt_rating   0..4 rubric score
    *_evidence     a verbatim quote from the trace, checked before the row counts

Cost control: seeded stratified subsample per config, an on-disk cache keyed by
(model, prompt version, config, idx) so re-runs are free for already-judged
traces, head+tail truncation of long traces, and --dry-run.

    # 1. smoke: 5 traces/config. --dry-run sends nothing, just prices the job.
    python llm_judge.py --limit 5 --dry-run
    python llm_judge.py --limit 5

    # 2. real subsample, then the whole dataset
    python llm_judge.py --limit 150
    python llm_judge.py --all

Needs HF_TOKEN (same token as api-setup/run_typo_api.py).
"""
import os, sys, csv, json, math, random, argparse, threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "analysis"))
from common import config_files, parse_tag, load_evals, TABLES, DATASET
from self_doubt import count_markers
from repair_wordlevel import corrupted_pairs, classify_word, MIN_LEN

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_HERE, "judge_cache")
API_BASE = os.environ.get("API_BASE", "https://router.huggingface.co/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
# USD per 1M tokens, only used for the cost ESTIMATE/report - check the router's
# current price for your judge model and override if it differs.
PRICE_IN = float(os.environ.get("JUDGE_PRICE_IN", 0.6))
PRICE_OUT = float(os.environ.get("JUDGE_PRICE_OUT", 0.6))

PROMPT_VERSION = "v1"      # bump when JUDGE_PROMPT changes -> invalidates the cache
HEAD_CHARS, TAIL_CHARS = 2000, 2000
MAX_PAIRS = 25             # corrupted words listed in the prompt

REPAIR_LABELS = [
    "silent_readthrough",     # never mentions the corruption, uses the intended words
    "explicit_notice_fix",    # flags a typo and recovers the intended word
    "explicit_notice_nofix",  # flags it but never resolves it / resolves it wrongly
    "misread_wrong_word",     # treats a corrupted word as a different real word
    "derailed",               # stuck or looping on the corruption
    "unclear",                # not enough evidence to decide
    "not_applicable",         # clean trace: nothing to repair
]

JUDGE_PROMPT = """You are a strict annotator. Do NOT solve the problem and do NOT judge whether the
final answer is right. Judge ONLY how the reasoning trace behaved.

CORRUPTED WORDS (original -> as shown to the model):
{pairs}

QUESTION SHOWN TO THE MODEL:
{question}

REASONING TRACE:
{reasoning}

Task 1 - self_doubt_score. Give a single integer from 0 to 10.
Interpretation:
0 = no doubt at all; straight-line solution; no second-guessing.
1 = tiny hedge or mild uncertainty, still clear.
2 = one brief hesitation or check.
3 = some re-checking but still mostly stable.
4 = noticeable self-doubt, repeated minor checks.
5 = clear hesitation and multiple second-guessing moments.
6 = repeated reconsideration, several reversals or checks.
7 = strong uncertainty, multiple pauses and reversals.
8 = heavy self-doubt, frequent re-reads and re-evaluations.
9 = pervasive doubt, many reversals and almost no stable plan.
10 = extreme uncertainty and looping / unable to settle.
The score should reflect the model's own second-guessing and uncertainty, independent of whether
the typo was bad or not.

Task 2 - repair_understanding_score. Give a single integer from 0 to 5.
Interpretation:
0 = fully understands the meaning of the question and the corrupted word(s); no confusion.
1 = almost fully understands the meaning; only slight confusion.
2 = some confusion but still largely understands the problem.
3 = noticeable misunderstanding or misreading of the question/word; partial loss of meaning.
4 = major misunderstanding; the model is clearly confused by the wording or corrupted word.
5 = does not understand the meaning of the question at all; it is fundamentally lost or reasoning from the wrong meaning.
This score should capture how much the trace loses the intended meaning because of the typo.

Task 3 - one_line_summary: Write a single sentence explaining what in the trace led to your
scores, using the most important concrete cue from the trace (for example: a typo notice,
hedging, a different real-word interpretation, or a clean straight-line solution). Keep it to one
line and do not explain your scores numerically.

Both evidence fields must be a VERBATIM substring copied from the reasoning trace
(use "" if there is genuinely none).

Output JSON only, no prose, no code fences:
{{"self_doubt_score": 0, "repair_understanding_score": 0, "self_doubt_evidence": "...", "repair_understanding_evidence": "...", "one_line_summary": "..."}}"""


# ---------------------------------------------------------------------------
# trace loading + prompt building
# ---------------------------------------------------------------------------
def shown_question(r):
    """The question text the model actually saw (spell-checked variant if a fix ran)."""
    return r.get("spellchecked_question") or r.get("typo_question") or r.get("clean_question", "")


def pairs_for(r):
    """(original, corrupted) pairs: from the stored typo lists when present, else
    recovered by diffing clean vs typo (older result files)."""
    orig, repl = r.get("typo_originals"), r.get("typo_replacements")
    if isinstance(orig, list) and isinstance(repl, list) and orig:
        return [(o, c) for o, c in zip(orig, repl) if str(o).lower() != str(c).lower()]
    if r.get("typo_question"):
        return corrupted_pairs(r.get("clean_question", ""), r["typo_question"])
    return []


def wordlevel_label(pairs, reasoning_low):
    """Trace-level collapse of repair_wordlevel's per-word buckets, for the
    agreement table: worst-case wins (misread > flagged > silent_fix)."""
    cats = Counter()
    for o, c in pairs:
        if len(o) < MIN_LEN:
            continue
        cats[classify_word(o, c, reasoning_low)] += 1
    if cats["echoed"]:
        return "misread"
    if cats["noticed"]:
        return "flagged"
    if cats["recovered"]:
        return "silent_fix"
    return "not_used"


def truncate(text):
    """Head+tail window: repair language clusters at the first read AND at the
    late 'wait, that was a typo' reversal, so a head-only cut loses half of it."""
    if len(text) <= HEAD_CHARS + TAIL_CHARS:
        return text
    return text[:HEAD_CHARS] + "\n[... trace truncated ...]\n" + text[-TAIL_CHARS:]


def build_prompt(rec):
    pairs = rec["pairs"][:MAX_PAIRS]
    pair_str = ", ".join(f"{o} -> {c}" for o, c in pairs) or "(none - this is a clean question)"
    return JUDGE_PROMPT.format(pairs=pair_str, question=rec["question"][:2000],
                               reasoning=truncate(rec["reasoning"]))


def load_traces(path, meta, limit, seed):
    """Answered traces of one config, stratified-sampled (seeded)."""
    ev = load_evals(path)
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        e = ev[r["idx"]]
        if not e["answered"]:
            continue
        reasoning = r.get("reasoning") or r.get("generation", "") or ""
        if not reasoning.strip():
            continue
        pairs = [] if meta["is_clean"] else pairs_for(r)
        _, cat_counts, total = count_markers(reasoning)
        words = max(len(reasoning.split()), 1)
        rows.append(dict(
            config=meta["tag"], idx=r["idx"], reasoning=reasoning,
            question=shown_question(r), pairs=pairs,
            correct=e["correct"], real=meta["real"], rate=meta["rate"],
            marker_per_1k=total / words * 1000,
            wordlevel=wordlevel_label(pairs, reasoning.lower()),
        ))
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows


# ---------------------------------------------------------------------------
# the judge call
# ---------------------------------------------------------------------------
def extract_json(text):
    """First {...} block, tolerating code fences and trailing prose."""
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None


def norm_quote(s):
    return " ".join(str(s or "").lower().split())


def judge_one(client, model, rec, retries=3, keep_evidence=False):
    """Return a cache row for one trace. Never raises."""
    prompt = build_prompt(rec)
    last_err = ""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=600)
            raw = (resp.choices[0].message.content or "").strip()
            u = getattr(resp, "usage", None)
            obj = extract_json(raw)
            if obj is None:
                last_err = f"unparseable: {raw[:120]}"
                continue

            try:
                self_doubt = int(round(float(obj.get("self_doubt_score", -1))))
            except (TypeError, ValueError):
                self_doubt = -1
            self_doubt = self_doubt if 0 <= self_doubt <= 10 else -1

            try:
                repair_understanding = int(round(float(obj.get("repair_understanding_score", -1))))
            except (TypeError, ValueError):
                repair_understanding = -1
            repair_understanding = repair_understanding if 0 <= repair_understanding <= 5 else -1

            low = rec["reasoning"].lower()
            row = dict(
                config=rec["config"], idx=rec["idx"],
                self_doubt_score=self_doubt,
                repair_understanding_score=repair_understanding,
                self_doubt_evidence_ok=norm_quote(obj.get("self_doubt_evidence")) in norm_quote(low),
                repair_understanding_evidence_ok=norm_quote(obj.get("repair_understanding_evidence")) in norm_quote(low),
                n_in=getattr(u, "prompt_tokens", 0) or 0,
                n_out=getattr(u, "completion_tokens", 0) or 0,
                error="",
            )
            if keep_evidence:
                row["self_doubt_evidence"] = str(obj.get("self_doubt_evidence", ""))
                row["repair_understanding_evidence"] = str(obj.get("repair_understanding_evidence", ""))
                row["one_line_summary"] = str(obj.get("one_line_summary", ""))
            return row
        except Exception as e:                      # network / provider errors
            last_err = str(e)[:150]
    row = dict(config=rec["config"], idx=rec["idx"],
               self_doubt_score=-1, repair_understanding_score=-1,
               self_doubt_evidence_ok=False, repair_understanding_evidence_ok=False,
               n_in=0, n_out=0, error=last_err)
    if keep_evidence:
        row["self_doubt_evidence"] = ""
        row["repair_understanding_evidence"] = ""
        row["one_line_summary"] = ""
    return row


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def cache_path(model, out_path=None):
    if out_path:
        p = out_path if os.path.isabs(out_path) else os.path.join(_HERE, out_path)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        return p
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = model.replace("/", "-").replace(":", "-")
    return os.path.join(CACHE_DIR, f"{DATASET}_{safe}_{PROMPT_VERSION}.jsonl")


def load_cache(path):
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("repair_label") != "error":     # retry failures on the next run
                done[(r["config"], r["idx"])] = r
    return done


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def cohen_kappa(pairs):
    """pairs = [(rater_a_label, rater_b_label), ...]"""
    n = len(pairs)
    if not n:
        return 0.0
    labels = sorted({x for p in pairs for x in p})
    obs = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    exp = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return (obs - exp) / (1 - exp) if exp < 1 else 0.0


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else 0.0


def ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):                      # average ranks within ties
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(xs, ys):
    return pearson(ranks(xs), ranks(ys))


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


# ---------------------------------------------------------------------------
# report tables
# ---------------------------------------------------------------------------
# judge label -> the repair_wordlevel bucket it should correspond to
COLLAPSE = {"silent_readthrough": "silent_fix", "explicit_notice_fix": "flagged",
            "explicit_notice_nofix": "flagged", "misread_wrong_word": "misread"}


def scalar_report(joined):
    """Write report tables for the current 0-10 / 0-5 judge rubric."""
    valid = [r for r in joined if r.get("self_doubt_score", -1) >= 0
             and r.get("repair_understanding_score", -1) >= 0]
    if not valid:
        print("no valid scalar judge scores to report")
        return

    tags = [t for t in dict.fromkeys(r["config"] for r in valid)]

    def mean(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    def share(rows, key, threshold):
        return sum(r[key] >= threshold for r in rows) / len(rows) if rows else 0.0

    # Per-configuration means and threshold shares are the primary scalar tables.
    rows = []
    print(f"\n=== {DATASET}: scalar LLM-judge scores per config ===")
    print(f"{'config':16s}{'n':>6}{'doubt 0-10':>12}{'repair 0-5':>12}")
    for tag in tags:
        group = [r for r in valid if r["config"] == tag]
        row = dict(
            config=tag, n=len(group),
            mean_self_doubt=round(mean(group, "self_doubt_score"), 3),
            median_self_doubt=sorted(r["self_doubt_score"] for r in group)[len(group) // 2],
            frac_self_doubt_ge5=round(share(group, "self_doubt_score", 5), 4),
            mean_repair_understanding=round(mean(group, "repair_understanding_score"), 3),
            median_repair_understanding=sorted(r["repair_understanding_score"] for r in group)[len(group) // 2],
            frac_repair_understanding_ge3=round(share(group, "repair_understanding_score", 3), 4),
            evidence_self_doubt_frac=round(sum(r.get("self_doubt_evidence_ok", False) for r in group) / len(group), 4),
            evidence_repair_frac=round(sum(r.get("repair_understanding_evidence_ok", False) for r in group) / len(group), 4),
        )
        print(f"{tag:16s}{len(group):>6}{row['mean_self_doubt']:>12.2f}{row['mean_repair_understanding']:>12.2f}")
        rows.append(row)
    _write_csv(os.path.join(TABLES, "judge_scalar_per_config.csv"), rows)

    # Correctness relationship: compare judge scores for correct and wrong answers.
    rows = []
    print("\n=== scalar judge scores by final outcome ===")
    for tag in tags + ["__all__"]:
        group = valid if tag == "__all__" else [r for r in valid if r["config"] == tag]
        correct = [r for r in group if r["correct"]]
        wrong = [r for r in group if not r["correct"]]
        rows.append(dict(
            config=tag, n=len(group), n_correct=len(correct), n_wrong=len(wrong),
            mean_doubt_correct=round(mean(correct, "self_doubt_score"), 3),
            mean_doubt_wrong=round(mean(wrong, "self_doubt_score"), 3),
            mean_repair_correct=round(mean(correct, "repair_understanding_score"), 3),
            mean_repair_wrong=round(mean(wrong, "repair_understanding_score"), 3),
        ))
    _write_csv(os.path.join(TABLES, "judge_scalar_by_outcome.csv"), rows)

    # The proposal's typo-severity variables are stored as real/rate in each trace.
    rows = []
    by_real = defaultdict(list)
    for r in valid:
        if r["real"] is not None:
            by_real[r["real"]].append(r)
    for real in sorted(by_real):
        group = by_real[real]
        rows.append(dict(
            real_ratio=real, n=len(group),
            mean_self_doubt=round(mean(group, "self_doubt_score"), 3),
            mean_repair_understanding=round(mean(group, "repair_understanding_score"), 3),
            frac_self_doubt_ge5=round(share(group, "self_doubt_score", 5), 4),
            frac_repair_understanding_ge3=round(share(group, "repair_understanding_score", 3), 4),
        ))
    _write_csv(os.path.join(TABLES, "judge_scalar_by_real.csv"), rows)

    # Preserve the inspectable unit of analysis for later auditing and plotting.
    detail = []
    for r in valid:
        detail.append(dict(
            config=r["config"], idx=r["idx"], real_ratio=r["real"], typo_rate=r["rate"],
            correct=int(bool(r["correct"])),
            self_doubt_score=r["self_doubt_score"],
            repair_understanding_score=r["repair_understanding_score"],
            self_doubt_evidence_ok=int(bool(r.get("self_doubt_evidence_ok"))),
            repair_understanding_evidence_ok=int(bool(r.get("repair_understanding_evidence_ok"))),
        ))
    _write_csv(os.path.join(TABLES, "judge_scalar_per_trace.csv"), detail)

    print(f"valid scalar judgments={len(valid)}  skipped invalid/error rows={len(joined) - len(valid)}")
    print(f"tables written to {TABLES}")


def report(recs, results):
    """recs = sampled traces, results = {(config, idx): judge row}."""
    has_legacy = False
    joined = []
    for r in recs:
        j = results.get((r["config"], r["idx"]))
        if not j:
            continue
        if j.get("repair_label") == "error":
            continue
        if "repair_label" in j and "doubt_rating" in j:
            has_legacy = True
            joined.append({**r, **{k: j[k] for k in
                                   ("repair_label", "doubt_rating",
                                    "repair_evidence_ok", "doubt_evidence_ok")}})
        elif "self_doubt_score" in j and "repair_understanding_score" in j:
            joined.append({**r, **j})
    if not joined:
        print("no judged traces to report")
        return
    if not has_legacy:
        scalar_report(joined)
        return

    tags = [t for t in dict.fromkeys(r["config"] for r in joined)]
    typo = [r for r in joined if r["config"] != "clean"]

    # ---- repair: label distribution per config ---------------------------
    print(f"\n=== {DATASET}: LLM-judge repair label per config (% of judged traces) ===")
    hdr = f"{'config':16s}{'n':>6}" + "".join(f"{l[:13]:>15}" for l in REPAIR_LABELS)
    print(hdr)
    rows = []
    for tag in tags:
        g = [r for r in joined if r["config"] == tag]
        n = len(g)
        d = Counter(r["repair_label"] for r in g)
        print(f"{tag:16s}{n:>6}" + "".join(f"{d[l] / n:>15.1%}" for l in REPAIR_LABELS))
        rows.append(dict(config=tag, n=n,
                         **{l: round(d[l] / n, 4) for l in REPAIR_LABELS}))
    _write_csv(os.path.join(TABLES, "judge_repair_per_config.csv"), rows)

    # ---- repair: by real-word ratio (the hypothesis) ---------------------
    print("\n=== LLM-judge repair label by real-word ratio (typo configs, rates pooled) ===")
    print("expect: misread_wrong_word rises with real ratio, explicit_notice_* falls")
    print(f"{'real ratio':>12}{'n':>6}" + "".join(f"{l[:13]:>15}" for l in REPAIR_LABELS))
    by_real = defaultdict(list)
    for r in typo:
        by_real[r["real"]].append(r)
    rows = []
    for real in sorted(x for x in by_real if x is not None):
        g = by_real[real]; n = len(g)
        d = Counter(r["repair_label"] for r in g)
        print(f"{f'real{real}':>12}{n:>6}" + "".join(f"{d[l] / n:>15.1%}" for l in REPAIR_LABELS))
        rows.append(dict(real_ratio=real, n=n,
                         **{l: round(d[l] / n, 4) for l in REPAIR_LABELS}))
    _write_csv(os.path.join(TABLES, "judge_repair_by_real.csv"), rows)

    # ---- repair: accuracy within label (silent failure cell) -------------
    print("\n=== accuracy within judge label (typo configs pooled) ===")
    print(f"{'label':24s}{'n':>7}{'accuracy':>11}{'share':>9}")
    rows = []
    tot = len(typo) or 1
    for label in REPAIR_LABELS:
        g = [r for r in typo if r["repair_label"] == label]
        if not g:
            continue
        acc = sum(r["correct"] for r in g) / len(g)
        print(f"{label:24s}{len(g):>7}{acc:>11.1%}{len(g) / tot:>9.1%}")
        rows.append(dict(label=label, n=len(g), accuracy=round(acc, 4),
                         share=round(len(g) / tot, 4)))
    _write_csv(os.path.join(TABLES, "judge_repair_by_outcome.csv"), rows)

    # ---- repair: agreement with the word-level measure -------------------
    both = [(COLLAPSE[r["repair_label"]], r["wordlevel"]) for r in typo
            if r["repair_label"] in COLLAPSE and r["wordlevel"] != "not_used"]
    agree = sum(1 for a, b in both if a == b) / len(both) if both else 0.0
    k = cohen_kappa(both)
    ev_ok = sum(1 for r in joined if r["repair_evidence_ok"]) / len(joined)
    print("\n=== judge vs repair_wordlevel agreement (comparable traces only) ===")
    print(f"n_comparable={len(both)}  raw_agreement={agree:.1%}  cohen_kappa={k:.3f}")
    print(f"evidence quote found verbatim in the trace: {ev_ok:.1%} of judged traces")
    print(f"\n{'judge \\ wordlevel':24s}" + "".join(f"{b:>12}" for b in ("silent_fix", "flagged", "misread")))
    conf = Counter(both)
    rows = []
    for a in ("silent_fix", "flagged", "misread"):
        print(f"{a:24s}" + "".join(f"{conf[(a, b)]:>12}" for b in ("silent_fix", "flagged", "misread")))
        rows.append(dict(judge=a, **{f"wordlevel_{b}": conf[(a, b)]
                                     for b in ("silent_fix", "flagged", "misread")}))
    rows.append(dict(judge="__summary__", n_comparable=len(both),
                     raw_agreement=round(agree, 4), cohen_kappa=round(k, 4),
                     evidence_verbatim_frac=round(ev_ok, 4)))
    _write_csv(os.path.join(TABLES, "judge_repair_vs_wordlevel.csv"), rows)

    # ---- self-doubt: rating per config -----------------------------------
    rated = [r for r in joined if r["doubt_rating"] >= 0]
    print(f"\n=== {DATASET}: LLM-judge self-doubt rating (0-4) per config ===")
    print(f"{'config':16s}{'n':>6}{'mean':>8}{'>=2 (%)':>10}{'>=3 (%)':>10}{'markers/1k':>12}")
    rows = []
    for tag in tags:
        g = [r for r in rated if r["config"] == tag]
        if not g:
            continue
        n = len(g)
        mean = sum(r["doubt_rating"] for r in g) / n
        hi2 = sum(1 for r in g if r["doubt_rating"] >= 2) / n
        hi3 = sum(1 for r in g if r["doubt_rating"] >= 3) / n
        mk = sum(r["marker_per_1k"] for r in g) / n
        print(f"{tag:16s}{n:>6}{mean:>8.2f}{hi2:>10.1%}{hi3:>10.1%}{mk:>12.2f}")
        rows.append(dict(config=tag, n=n, mean_rating=round(mean, 3),
                         frac_ge2=round(hi2, 4), frac_ge3=round(hi3, 4),
                         marker_per_1k=round(mk, 3)))
    _write_csv(os.path.join(TABLES, "judge_doubt_per_config.csv"), rows)

    # ---- self-doubt: by outcome ------------------------------------------
    print("\n=== judge self-doubt rating by final outcome ===")
    print(f"{'config':16s}{'correct':>10}{'wrong':>9}{'wrong/correct':>15}")
    rows = []
    for tag in tags:
        g = [r for r in rated if r["config"] == tag]
        cor = [r["doubt_rating"] for r in g if r["correct"]]
        wr = [r["doubt_rating"] for r in g if not r["correct"]]
        mc = sum(cor) / len(cor) if cor else 0.0
        mw = sum(wr) / len(wr) if wr else 0.0
        print(f"{tag:16s}{mc:>10.2f}{mw:>9.2f}{(mw / mc if mc else 0):>15.2f}")
        rows.append(dict(config=tag, rating_correct=round(mc, 3), rating_wrong=round(mw, 3),
                         wrong_over_correct=round(mw / mc if mc else 0, 3),
                         n_correct=len(cor), n_wrong=len(wr)))
    _write_csv(os.path.join(TABLES, "judge_doubt_by_outcome.csv"), rows)

    # ---- self-doubt: agreement with the marker density -------------------
    print("\n=== judge rating vs doubt-marker density (per trace) ===")
    print(f"{'config':16s}{'n':>6}{'pearson':>10}{'spearman':>11}")
    rows = []
    for tag in tags + ["__all__"]:
        g = rated if tag == "__all__" else [r for r in rated if r["config"] == tag]
        if len(g) < 3:
            continue
        xs = [r["doubt_rating"] for r in g]
        ys = [r["marker_per_1k"] for r in g]
        p, s = pearson(xs, ys), spearman(xs, ys)
        print(f"{tag:16s}{len(g):>6}{p:>10.3f}{s:>11.3f}")
        rows.append(dict(config=tag, n=len(g), pearson=round(p, 4), spearman=round(s, 4)))
    _write_csv(os.path.join(TABLES, "judge_doubt_vs_markers.csv"), rows)

    print(f"\ntables written to {TABLES}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=150, help="traces judged per config")
    ap.add_argument("--all", action="store_true", help="judge every answered trace (expensive)")
    ap.add_argument("--configs", default="", help="comma-separated subset, e.g. clean,typo75_real70")
    ap.add_argument("--model", default=JUDGE_MODEL, help="judge model at the router")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-mc", "--manual-check", action="store_true",
                    help="keep the raw judge evidence strings in the JSONL cache for manual review")
    ap.add_argument("--out", dest="out_path", default=None,
                    help="custom JSONL output path; absolute or relative to full_analysis/")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the token/cost estimate and exit without calling the API")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the tables from the cache, no API calls")
    args = ap.parse_args()

    limit = 0 if args.all else args.limit
    wanted = {c.strip() for c in args.configs.split(",") if c.strip()}

    recs = []
    for f in config_files():
        meta = parse_tag(f)
        # The main report uses the canonical 500-question files. Some datasets
        # also contain model-suffixed reruns; those are separate experiments.
        if os.path.basename(f) != f"{DATASET}_{meta['tag']}.jsonl" or (
            meta["tag"] != "clean" and meta["rate"] is None):
            continue
        if wanted and meta["tag"] not in wanted:
            continue
        recs.extend(load_traces(f, meta, limit, args.seed))
    if not recs:
        raise SystemExit("no traces selected")

    cpath = cache_path(args.model, args.out_path)
    cached = load_cache(cpath)
    todo = [r for r in recs if (r["config"], r["idx"]) not in cached]

    est_in = sum(len(build_prompt(r)) for r in todo) / 4      # ~4 chars per token
    est_out = 60 * len(todo)
    est_cost = est_in / 1e6 * PRICE_IN + est_out / 1e6 * PRICE_OUT
    print(f"dataset={DATASET}  judge={args.model}  prompt={PROMPT_VERSION}")
    print(f"selected {len(recs)} traces ({len(recs) - len(todo)} cached, {len(todo)} to call)")
    print(f"estimate: ~{est_in:,.0f} in + ~{est_out:,.0f} out tokens  ~${est_cost:.3f}"
          f"  (at ${PRICE_IN}/${PRICE_OUT} per 1M - verify your model's price)")

    if args.dry_run:
        print("\n--- example prompt ---")
        print(build_prompt(recs[0])[:1500])
        print("\n[dry-run] nothing sent.")
        return

    if todo and not args.report_only:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("export HF_TOKEN=hf_... first")
        from openai import OpenAI
        client = OpenAI(api_key=token, base_url=API_BASE)
        lock = threading.Lock()
        done = spent_in = spent_out = 0
        with open(cpath, "a", encoding="utf-8") as cf, \
                ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(judge_one, client, args.model, r, keep_evidence=args.manual_check): r for r in todo}
            for fut in as_completed(futs):
                row = fut.result()
                with lock:
                    cf.write(json.dumps(row, ensure_ascii=False) + "\n"); cf.flush()
                    cached[(row["config"], row["idx"])] = row
                    done += 1
                    spent_in += row["n_in"]; spent_out += row["n_out"]
                    if done % 25 == 0 or done == len(todo):
                        print(f"  judged {done}/{len(todo)}  "
                              f"tokens {spent_in:,}+{spent_out:,}", flush=True)
        actual = spent_in / 1e6 * PRICE_IN + spent_out / 1e6 * PRICE_OUT
        errs = sum(1 for r in cached.values() if r.get("repair_label") == "error")
        print(f"done. actual tokens {spent_in:,} in + {spent_out:,} out  ~${actual:.4f}"
              f"  errors={errs}")

    report(recs, cached)


if __name__ == "__main__":
    main()
