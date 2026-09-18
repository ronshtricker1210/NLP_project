"""Shared loading / scoring helpers for the wider typo analysis.

Dataset-agnostic: the target dataset comes from the NLP_DATASET environment
variable (default "gsm8k"). Raw generations are read from raw_results/<dataset>/
and tables are written to analysis_tables/<base>/<variant>/ (see dataset_layout),
so several datasets coexist without clobbering.

Correctness reuses api-setup/score.py per dataset kind (gsm8k = numeric,
math500 = math_verify on \\boxed, gpqa = multiple choice). Answer EXTRACTION is
deliberately stricter here: we only read the answer from the final section (after
</think>), never from mid-reasoning. A trace that hit the token cap before writing
a final answer is UNANSWERED, not a lucky trailing guess -- this keeps "didn't
finish" separate from "got it wrong".
"""
import os, sys, json, glob, re, random

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "api-setup"))
from score import (gold_gsm8k, norm_num, last_boxed, last_number,  # noqa: E402
                   extract_pred, is_correct)

# ---- dataset selection -----------------------------------------------------
DATASET = os.environ.get("NLP_DATASET", "gsm8k")

# kind = how score.py extracts/compares answers; gold = how to read the gold field.
DATASETS = {
    "gsm8k":   {"kind": "gsm8k_num", "gold": lambda r: gold_gsm8k(r["gold_answer"])},
    "math500": {"kind": "math",      "gold": lambda r: r["gold_answer"]},
    "gpqa":    {"kind": "mc",        "gold": lambda r: r["gold_answer"]},
    "arc":     {"kind": "mc",        "gold": lambda r: r["gold_answer"]},
}
# Variants like gsm8k_20000 / gsm8k_fix-warn reuse their base dataset's spec.
BASE_DATASET = next((k for k in DATASETS
                     if DATASET == k or DATASET.startswith(k + "_")), None)
if BASE_DATASET is None:
    raise SystemExit(f"unknown NLP_DATASET={DATASET!r}; choices: {list(DATASETS)}")

def dataset_layout(ds):
    """Folder pair for a dataset tag: gsm8k -> gsm8k/results,
    gsm8k_20000 -> gsm8k/results_20000, gsm8k_fix-warn -> gsm8k/fix_warn."""
    base = next((k for k in DATASETS if ds == k or ds.startswith(k + "_")), ds)
    variant = ds[len(base):].lstrip("_")
    if not variant:
        sub = "results"
    elif variant.startswith("fix-"):
        sub = "fix_" + variant[4:].replace("-", "_")
    else:
        sub = f"results_{variant}"
    return base, sub


def tables_dir(ds):
    return os.path.join(_HERE, "analysis_tables", *dataset_layout(ds))


def reports_dir(ds):
    return os.path.join(_HERE, "reports", *dataset_layout(ds))


DATA_DIR = os.path.join(_HERE, "raw_results", DATASET)
TABLES = tables_dir(DATASET)
os.makedirs(TABLES, exist_ok=True)
# Cap used at generation time; math500/arc API runs used 17000 (NLP_MAX_NEW_TOKENS=17000).
MAX_NEW_TOKENS = int(os.environ.get("NLP_MAX_NEW_TOKENS", 4096))


# ---- config identity -------------------------------------------------------
def parse_tag(path):
    """<ds>_clean.jsonl -> {'tag':'clean',...}; <ds>_typo50_real40.jsonl ->
       {'tag':'typo50_real40','rate':50,'real':40}."""
    base = os.path.basename(path).replace(".jsonl", "")
    if base.startswith(DATASET + "_"):
        base = base[len(DATASET) + 1:]
    if base == "clean":
        return {"tag": "clean", "rate": 0, "real": None, "is_clean": True}
    m = re.fullmatch(r"typo(\d+)_real(\d+)", base)
    if not m:
        return {"tag": base, "rate": None, "real": None, "is_clean": False}
    return {"tag": base, "rate": int(m.group(1)), "real": int(m.group(2)), "is_clean": False}


def config_files():
    """Clean first, then typo configs sorted by (rate, real), for DATASET."""
    files = glob.glob(os.path.join(glob.escape(DATA_DIR), f"{DATASET}_*.jsonl"))
    if not files:
        raise SystemExit(
            f"no data for dataset '{DATASET}' in {DATA_DIR}\n"
            f"  run:  python download_data.py --dataset {DATASET}")
    def key(f):
        t = parse_tag(f)
        return (0, 0, 0) if t["is_clean"] else (1, t["rate"] or 0, t["real"] or 0)
    return sorted(files, key=key)


# ---- answer extraction (strict: final section only) ------------------------
def _kind_gold(row):
    ds = row.get("dataset", BASE_DATASET)
    spec = DATASETS.get(ds, DATASETS[BASE_DATASET])
    return spec["kind"], spec["gold"](row)


def extract_answer(final_text, kind="gsm8k_num"):
    """Predicted answer from the FINAL section only (score.py logic, final text as
    the whole 'generation' so nothing mid-reasoning leaks in). None => unanswered."""
    return extract_pred(final_text or "", kind, final_text or "")


# ---- per-row evaluation ----------------------------------------------------
# state is one of: "correct", "wrong", "unanswered"
def eval_row(row):
    kind, gold = _kind_gold(row)
    final = row.get("final_answer_text", "") or ""
    pred = extract_answer(final, kind)
    answered = pred is not None
    correct = bool(answered and is_correct(kind, pred, gold))
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
    """{idx: eval_dict} for one config file (one sample/question)."""
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
