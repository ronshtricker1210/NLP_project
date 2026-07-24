"""Shared loading / scoring helpers for the wider typo analysis (gsm8k).

Correctness reuses api-setup/score.py (same normalisation and gsm8k gold parsing),
but answer EXTRACTION is deliberately stricter here: we only read the model's
answer from the final section (after </think>), never from mid-reasoning. A trace
that hit the 4096-token cap before writing a final answer is UNANSWERED, not a
lucky trailing-number guess. This keeps "didn't finish" separate from "got it wrong".
"""
import os, sys, json, glob, re, random

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "api-setup"))
from score import gold_gsm8k, norm_num, last_boxed, last_number, is_correct  # noqa: E402

DATA_DIR = os.path.join(_HERE, "data", "gsm8k")
MAX_NEW_TOKENS = 4096  # the cap used at generation time (run_typo_api.py default)


# ---- config identity -------------------------------------------------------
def parse_tag(path):
    """gsm8k_clean.jsonl -> {'tag':'clean','rate':0,'real':None}
       gsm8k_typo50_real40.jsonl -> {'tag':'typo50_real40','rate':50,'real':40}"""
    base = os.path.basename(path).replace(".jsonl", "").replace("gsm8k_", "")
    if base == "clean":
        return {"tag": "clean", "rate": 0, "real": None, "is_clean": True}
    m = re.fullmatch(r"typo(\d+)_real(\d+)", base)
    if not m:
        return {"tag": base, "rate": None, "real": None, "is_clean": False}
    return {"tag": base, "rate": int(m.group(1)), "real": int(m.group(2)), "is_clean": False}


def config_files():
    """Clean first, then typo configs sorted by (rate, real)."""
    files = glob.glob(os.path.join(DATA_DIR, "gsm8k_*.jsonl"))
    def key(f):
        t = parse_tag(f)
        return (0, 0, 0) if t["is_clean"] else (1, t["rate"] or 0, t["real"] or 0)
    return sorted(files, key=key)


# ---- answer extraction (strict: final section only) ------------------------
def extract_answer(final_text):
    """Model's gsm8k answer from the FINAL section only. None => unanswered."""
    b = last_boxed(final_text or "")
    if b is not None:
        return b
    return last_number(final_text or "")   # None if no number present


# ---- per-row evaluation ----------------------------------------------------
# state is one of: "correct", "wrong", "unanswered"
def eval_row(row):
    gold = gold_gsm8k(row["gold_answer"])
    final = row.get("final_answer_text", "") or ""
    pred = extract_answer(final)
    answered = pred is not None
    correct = bool(answered and is_correct("gsm8k_num", pred, gold))
    state = "correct" if correct else ("unanswered" if not answered else "wrong")
    return {
        "idx": row["idx"],
        "pred": pred,
        "gold": gold,
        "answered": answered,
        "correct": correct,
        "state": state,
        "n_gen_tokens": row.get("n_gen_tokens"),
        "capped": row.get("n_gen_tokens", 0) >= MAX_NEW_TOKENS,
    }


def load_evals(path):
    """{idx: eval_dict} for one config file (gsm8k has 1 sample/question)."""
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[r["idx"]] = eval_row(r)
    return out


# ---- statistics ------------------------------------------------------------
def bootstrap_ci(bools, n_boot=2000, alpha=0.05, seed=0):
    """95% CI for the mean of a 0/1 list via bootstrap."""
    xs = [1 if b else 0 for b in bools]
    if not xs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(xs)
    means = []
    for _ in range(n_boot):
        s = sum(xs[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return (lo, hi)


def mcnemar(base_correct, cfg_correct):
    """Paired test on correctness. base/cfg are dicts idx->bool over shared idx.
    Returns (b, c, stat, p) where b = base-right/cfg-wrong, c = base-wrong/cfg-right."""
    b = c = 0
    for idx in base_correct:
        if idx not in cfg_correct:
            continue
        bo, co = base_correct[idx], cfg_correct[idx]
        if bo and not co:
            b += 1
        elif not bo and co:
            c += 1
    n = b + c
    if n == 0:
        return b, c, 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / n   # continuity-corrected McNemar chi2 (df=1)
    try:
        from math import erf, sqrt
        # survival of chi2 df=1 = erfc(sqrt(stat/2))
        p = 1.0 - erf(sqrt(stat / 2.0)) if stat > 0 else 1.0
    except Exception:
        p = None
    return b, c, stat, p
