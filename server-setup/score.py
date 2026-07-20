"""
Score the reasoning traces produced by run_typo.py.

Reads results/*.jsonl, extracts the model's predicted answer from the generated text,
compares to the gold answer, and reports accuracy per config plus flip rates vs the clean
baseline (real0). Runs on CPU in seconds, so you can improve the parser and re-score without
touching the GPU.

Usage:
    python score.py --results ~/nlp_project/results --dataset gsm8k
"""
import os, re, json, glob, argparse
from collections import defaultdict

# math_verify gives proper symbolic equivalence for MATH500 (1/2 == 0.5 == \frac{1}{2}).
try:
    from math_verify import parse as mv_parse, verify as mv_verify
    HAVE_MATH_VERIFY = True
except Exception:
    HAVE_MATH_VERIFY = False


def last_boxed(text):
    """Return the content of the last \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth, out = 0, []
    for ch in text[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out).strip()


def norm_num(s):
    """Normalize a numeric string for GSM8K comparison."""
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    m = re.search(r"-?\d+\.?\d*", s)
    if not m:
        return None
    v = m.group(0)
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def gold_gsm8k(g):       # gold field is the full solution ending in "#### N"
    return norm_num(g.split("####")[-1])


def last_number(text):
    """Last standalone number in the text (GSM8K fallback when no \\boxed)."""
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1] if nums else None


def extract_pred(gen, kind, final_text=""):
    b = last_boxed(gen)
    if kind == "mc":
        src = b if b else (final_text or gen)
        m = re.findall(r"\b([ABCD])\b", src)
        return m[-1] if m else None
    if kind == "gsm8k_num":
        # prefer boxed, else the last number in the answer section (then whole gen)
        if b is not None:
            return b
        return last_number(final_text) or last_number(gen)
    return b  # generic math (MATH500): rely on boxed


def is_correct(kind, pred, gold):
    if pred is None:
        return False
    if kind == "mc":
        return pred.strip().upper() == str(gold).strip().upper()
    if kind == "gsm8k_num":
        return norm_num(pred) is not None and norm_num(pred) == gold
    # generic math (MATH500)
    if HAVE_MATH_VERIFY:
        try:
            return bool(mv_verify(mv_parse(str(gold)), mv_parse(str(pred))))
        except Exception:
            pass
    return norm_num(pred) is not None and norm_num(pred) == norm_num(gold)


def score_file(fp):
    """Return dict idx -> (is_correct, pred, gold) and the dataset name."""
    per = {}
    dataset = None
    for line in open(fp, encoding="utf-8"):
        r = json.loads(line)
        dataset = r["dataset"]
        if dataset == "gsm8k":
            kind, gold = "gsm8k_num", gold_gsm8k(r["gold_answer"])
        elif dataset == "gpqa":
            kind, gold = "mc", r["gold_answer"]
        else:  # math500
            kind, gold = "math", r["gold_answer"]
        pred = extract_pred(r["generation"], kind, r.get("final_answer_text", ""))
        per[r["idx"]] = (is_correct(kind, pred, gold), pred, gold)
    return dataset, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.expanduser("~/nlp_project/results"))
    ap.add_argument("--dataset", default=None, help="filter, e.g. gsm8k")
    args = ap.parse_args()

    if not HAVE_MATH_VERIFY:
        print("[warn] math_verify not installed - MATH500 uses numeric fallback (less accurate).\n"
              "       pip install math_verify\n")

    pattern = f"{args.dataset}_*.jsonl" if args.dataset else "*.jsonl"
    files = sorted(glob.glob(os.path.join(args.results, pattern)))
    by_config = {}   # config -> per-idx correctness
    for fp in files:
        cfg = os.path.basename(fp).replace(".jsonl", "").split("_", 1)[1]
        dataset, per = score_file(fp)
        by_config[cfg] = per
        acc = sum(c for c, _, _ in per.values()) / max(len(per), 1)
        print(f"{os.path.basename(fp):32s}  n={len(per):4d}  accuracy={acc:6.1%}")

    # flip analysis vs the clean baseline ("clean" file, or real0 if that is what exists)
    base_key = "clean" if "clean" in by_config else ("real0" if "real0" in by_config else None)
    if base_key:
        base = by_config[base_key]
        print(f"\nFlips vs {base_key} (baseline):")
        for cfg, per in sorted(by_config.items()):
            if cfg == base_key:
                continue
            r2w = w2r = 0
            for idx, (corr, _, _) in per.items():
                if idx in base:
                    if base[idx][0] and not corr:
                        r2w += 1
                    elif not base[idx][0] and corr:
                        w2r += 1
            print(f"  {cfg:8s}  right->wrong={r2w:4d}  wrong->right={w2r:4d}")
    else:
        print("\n(no baseline yet - run with --variant clean to enable flip analysis)")


if __name__ == "__main__":
    main()
