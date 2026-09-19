"""
API inference over the typo datasets — HuggingFace Inference Providers build.

Same JSONL output as the gcp-setup / server-setup vLLM versions, but no GPU:
questions go concurrently to DeepSeek-R1-Distill-Qwen-7B served by Nscale via
the HF router (router.huggingface.co). Differences from run_typo_vllm.py:
  - the server applies the chat template, so `prompt` stores the plain user
    prompt (not the templated chat string);
  - `generation` is reconstructed as reasoning + "</think>" + final answer when
    the API returns reasoning separately, so score.py works unchanged;
  - records gain additive fields (cost_usd, latency_s) that score.py ignores;
  - interrupted runs resume: finished (idx, sample_idx) pairs are skipped.

Examples (mirror run_typo_vllm.py):
    python run_typo_api.py --dataset gsm8k --variant both
    python run_typo_api.py --dataset all --configs all --limit 50
    python run_typo_api.py --dataset math500 --configs real0,real30,real70 --n-samples 5
"""
import os, json, re, time, argparse, random, sys, threading, types
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset, get_dataset_config_names
from openai import OpenAI

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:nscale"
API_BASE = os.environ.get("API_BASE", "https://router.huggingface.co/v1")
# Nscale via HF router, USD per 1M tokens (source: router.huggingface.co/v1/models).
PRICE_IN, PRICE_OUT = 0.15, 0.15

DATASETS = {
    "gsm8k":   {"repo": "idoazou/gsm8k-typos",   "clean": "question", "typo": "problem_typo", "gold": "answer",         "kind": "math"},
    "math500": {"repo": "idoazou/math500-typos", "clean": "problem",  "typo": "problem_typo", "gold": "answer",         "kind": "math"},
    "gpqa":    {"repo": "idoazou/gpqa-typos",    "clean": "Question", "typo": "problem_typo", "gold": "Correct Answer", "kind": "mc"},
    "arc":     {"repo": "idoazou/arc-typos",     "clean": "Question", "typo": "problem_typo", "gold": "Correct Answer", "kind": "mc"},
}
META_COLS = ["target_real_ratio", "real_ratio", "num_total", "num_real", "num_nonword",
             "typo_techniques", "typo_originals", "typo_replacements",
             "level", "subject", "unique_id", "Record ID"]

# A small, representative default sweep so `--configs` can be omitted for a first run.
DEFAULT_CONFIGS = ["real0", "real30", "real70"]

write_lock = threading.Lock()

# Protect math/LaTeX/numbers from spell-check edits.
_PROTECTED_PATTERN = re.compile(
    r"\$\$.*?\$\$"          # display math  $$ ... $$
    r"|\$.*?\$"               # inline math   $ ... $
    r"|\\\[.*?\\\]"       # display math  \[ ... \]
    r"|\\\(.*?\\\)"       # inline math   \( ... \)
    r"|\\[a-zA-Z]+"           # LaTeX command names
    r"|\d+(?:[.,]\d+)?",      # bare numbers
    re.DOTALL,
)
_WORD_PATTERN = re.compile(r"[A-Za-z]+")


def find_protected_spans(text):
    return [(m.start(), m.end()) for m in _PROTECTED_PATTERN.finditer(text)]


def overlaps_any(start, end, spans):
    for s, e in spans:
        if start < e and end > s:
            return True
    return False


def edit_distance_leq(a, b, max_dist=2):
    """Bounded Levenshtein check used to keep spell-corrections conservative."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > max_dist:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = cur[0]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            v = min(ins, delete, sub)
            cur.append(v)
            if v < row_min:
                row_min = v
        if row_min > max_dist:
            return False
        prev = cur
    return prev[-1] <= max_dist


def maybe_build_spellchecker(fix):
    if fix != "spellcheck":
        return None
    try:
        from spellchecker import SpellChecker
    except ImportError:
        raise SystemExit(
            "--fix spellcheck requires pyspellchecker. Install with: pip install pyspellchecker"
        )
    return SpellChecker(distance=2)


def apply_spellcheck(text, checker):
    """Conservative external spell-check over prose words only."""
    if checker is None:
        return text, []

    spans = find_protected_spans(text)
    out = []
    last = 0
    changes = []

    for m in _WORD_PATTERN.finditer(text):
        s, e = m.start(), m.end()
        tok = m.group(0)

        out.append(text[last:s])
        last = e

        # Skip protected regions, short words, and acronyms.
        if overlaps_any(s, e, spans) or len(tok) < 4 or tok.isupper():
            out.append(tok)
            continue

        lower = tok.lower()
        if not checker.unknown([lower]):
            out.append(tok)
            continue

        cand = checker.correction(lower)
        if not cand:
            out.append(tok)
            continue

        cand = cand.lower()
        if cand == lower or not edit_distance_leq(lower, cand, max_dist=2):
            out.append(tok)
            continue

        # Preserve leading capitalization style.
        repl = cand.capitalize() if tok[0].isupper() else cand
        out.append(repl)
        changes.append({"from": tok, "to": repl})

    out.append(text[last:])
    return "".join(out), changes


def spellcheck_recovery_stats(row, spellcheck_changes):
    """Compare spell-check edits against known typo pairs for this row.

    A typo is "exactly restored" iff a correction maps typo_replacement -> typo_original.
    """
    originals = row.get("typo_originals") or []
    replacements = row.get("typo_replacements") or []
    total = min(len(originals), len(replacements))
    if total == 0:
        return {
            "spellcheck_total_typo_words": 0,
            "spellcheck_restored_exact_n": 0,
            "spellcheck_changed_other_n": 0,
            "spellcheck_not_restored_n": 0,
            "spellcheck_restored_exact_frac": None,
            "spellcheck_restored_exact_pct": None,
        }

    # Multiset of observed corrections (from -> to), lower-cased for robust matching.
    corr_counts = {}
    changed_from = {}
    for c in spellcheck_changes:
        src = str(c.get("from", "")).lower()
        dst = str(c.get("to", "")).lower()
        key = (src, dst)
        corr_counts[key] = corr_counts.get(key, 0) + 1
        changed_from[src] = changed_from.get(src, 0) + 1

    restored = 0
    changed_other = 0
    for orig, rep in zip(originals[:total], replacements[:total]):
        o = str(orig).lower()
        r = str(rep).lower()
        exact_key = (r, o)
        if corr_counts.get(exact_key, 0) > 0:
            restored += 1
            corr_counts[exact_key] -= 1
            if corr_counts[exact_key] == 0:
                del corr_counts[exact_key]
            # Keep changed_from roughly balanced for repeated tokens.
            if changed_from.get(r, 0) > 0:
                changed_from[r] -= 1
                if changed_from[r] == 0:
                    del changed_from[r]
        elif changed_from.get(r, 0) > 0:
            changed_other += 1
            changed_from[r] -= 1
            if changed_from[r] == 0:
                del changed_from[r]

    not_restored = total - restored
    frac = restored / total if total else None
    return {
        "spellcheck_total_typo_words": total,
        "spellcheck_restored_exact_n": restored,
        "spellcheck_changed_other_n": changed_other,
        "spellcheck_not_restored_n": not_restored,
        "spellcheck_restored_exact_frac": round(frac, 6) if frac is not None else None,
        "spellcheck_restored_exact_pct": round(100.0 * frac, 2) if frac is not None else None,
    }


def resolve_datasets(arg):
    """--dataset accepts 'all', or a comma list like 'gsm8k,math500'."""
    if arg.strip().lower() == "all":
        return list(DATASETS)
    names = [d.strip() for d in arg.split(",") if d.strip()]
    bad = [d for d in names if d not in DATASETS]
    if bad:
        raise SystemExit(f"unknown dataset(s): {bad}. choices: {list(DATASETS)} or 'all'")
    return names


def resolve_configs(arg, repo):
    """--configs accepts 'all' (every config on the Hub) or a comma list."""
    if arg.strip().lower() == "all":
        cfgs = sorted(get_dataset_config_names(repo))
        if not cfgs:
            raise SystemExit(f"--configs all: no configs found on the Hub for {repo}")
        return cfgs
    return [c.strip() for c in arg.split(",") if c.strip()]


def build_prompt(kind, qtext, row, seed, fix=None):
    # Identical to run_typo_vllm.py (same seeded shuffle => same GPQA letters).
    # fix='rewrite' asks the model to correct typos before solving; fix='warn'
    # only mentions typos may exist (proposal section "Test simple fixes").
    if kind == "math":
        instr = "Please reason step by step, and put your final answer within \\boxed{}."
        if fix == "rewrite":
            instr = ("First rewrite the question with all typos corrected, then solve it "
                     "step by step, and put your final answer within \\boxed{}.")
        prompt = qtext + "\n\n" + instr
        if fix == "warn":
            prompt = "Note: the text may contain typos.\n\n" + prompt
        return prompt, None
    opts = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    order = list(range(4))
    random.Random(seed).shuffle(order)
    letters = "ABCD"
    lines, correct = [], None
    for i, o in enumerate(order):
        lines.append(f"{letters[i]}) {opts[o]}")
        if o == 0:
            correct = letters[i]
    instr = "Please reason step by step, and put the letter of the correct answer within \\boxed{}."
    if fix == "rewrite":
        instr = ("First rewrite the question with all typos corrected, then reason step by "
                 "step, and put the letter of the correct answer within \\boxed{}.")
    prompt = qtext + "\n\n" + "\n".join(lines) + "\n\n" + instr
    if fix == "warn":
        prompt = "Note: the text may contain typos.\n\n" + prompt
    return prompt, correct


def hub_path(dataset, config, variant, suffix=""):
    """Repo path for a result file: results/{dataset}/typo{rate}/real{ratio}.jsonl.
    Plain realY configs use the fixed ~30% corruption rate (see DATASET_USAGE.md),
    so they file under typo30; rateX_realY configs under typoX; clean baselines
    under results/{dataset}/clean.jsonl. A suffix (e.g. "_20000" for a different
    max-token budget) lands before .jsonl so variants coexist."""
    if variant == "clean":
        return f"results/{dataset}/clean{suffix}.jsonl"
    m = re.fullmatch(r"rate(\d+)_real(\d+)", config)
    if m:
        return f"results/{dataset}/typo{m.group(1)}/real{m.group(2)}{suffix}.jsonl"
    m = re.fullmatch(r"real(\d+)", config)
    if m:
        return f"results/{dataset}/typo30/real{m.group(1)}{suffix}.jsonl"
    return f"results/{dataset}/{config}{suffix}.jsonl"


def hub_config_name(dataset, config, variant, suffix=""):
    """Config name for push_to_hub, unique across datasets sharing one repo:
    math500_clean, math500_typo30_real40, gsm8k_typo25_real0_20000, ..."""
    leaf = hub_path(dataset, config, variant, suffix)      # results/ds/typoX/realY.jsonl
    parts = leaf.removeprefix("results/").removesuffix(".jsonl").split("/")
    return "_".join(parts)


def hub_upload(outpath, dataset, config, variant, args):
    """Mirror a finished result file to the HF dataset repo, the same way
    data_creation/generate_variants.py publishes the typo variants:
      1. the raw JSONL at results/{dataset}/typo{rate}/real{ratio}.jsonl
      2. push_to_hub(config_name=..., split="test") so it loads with
         load_dataset(repo, "<dataset>_typo<rate>_real<ratio>", split="test")."""
    from datasets import Dataset
    from huggingface_hub import HfApi
    token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
    suffix = f"_{args.file_suffix}" if args.file_suffix else ""
    dest = hub_path(dataset, config, variant, suffix)
    HfApi(token=token).upload_file(path_or_fileobj=outpath, path_in_repo=dest,
                                   repo_id=args.hub_repo, repo_type="dataset",
                                   commit_message=f"results: {dataset}/{config} ({variant}{suffix})")
    cfg_name = hub_config_name(dataset, config, variant, suffix)
    Dataset.from_json(outpath).push_to_hub(args.hub_repo, config_name=cfg_name,
                                           split="test", private=True, token=token)
    print(f"  pushed hf://datasets/{args.hub_repo}/{dest}  (config={cfg_name})", flush=True)


def split_generation(msg):
    """Rebuild the vLLM-style `generation` string (reasoning</think>final) from an
    API message, whether reasoning arrives as reasoning_content or inline tags."""
    reasoning = getattr(msg, "reasoning_content", None)
    content = msg.content or ""
    if reasoning:
        return reasoning.strip() + "\n</think>\n" + content.strip(), reasoning.strip(), content.strip()
    if "<think>" not in content and "</think>" not in content:
        # non-reasoning model: the whole output IS the final answer
        return content.strip(), "", content.strip()
    text = content.removeprefix("<think>").lstrip("\n")
    reasoning, sep, final = text.partition("</think>")
    if not sep:
        reasoning, final = text, ""
    return text, reasoning.strip(), final.strip()


def ask_one(client, args, prompt, retries):
    last_err = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            # stream=True keeps bytes flowing so the router's gateway timeout
            # (~10 min) can't kill long generations mid-chain.
            stream = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                timeout=1200,
                stream=True,
                stream_options={"include_usage": True},
            )
            reasoning_parts, content_parts, usage, finish = [], [], None, None
            for chunk in stream:
                if chunk.usage:
                    usage = chunk.usage
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    rc = getattr(delta, "reasoning_content", None)
                    if rc:
                        reasoning_parts.append(rc)
                    if delta.content:
                        content_parts.append(delta.content)
                    if choice.finish_reason:
                        finish = choice.finish_reason
            if finish is None:
                # server dropped the stream mid-generation; treat as a failed
                # attempt, never as a (silently truncated) result
                raise RuntimeError("stream ended without finish_reason")
            msg = types.SimpleNamespace(
                reasoning_content="".join(reasoning_parts) or None,
                content="".join(content_parts))
            generation, reasoning, final = split_generation(msg)
            if usage is None:  # provider sent no usage chunk; estimate ~3.5 chars/token
                est = lambda s: max(1, int(len(s) / 3.5))
                usage = types.SimpleNamespace(prompt_tokens=est(prompt),
                                              completion_tokens=est(generation))
            return {
                "generation": generation, "reasoning": reasoning, "final_answer_text": final,
                "finish_reason": finish,
                "n_prompt_tokens": usage.prompt_tokens, "n_gen_tokens": usage.completion_tokens,
                "cost_usd": round((usage.prompt_tokens * PRICE_IN + usage.completion_tokens * PRICE_OUT) / 1e6, 8),
                "latency_s": round(time.time() - t0, 1),
            }
        except Exception as e:
            last_err = e
            wait = 2 ** (attempt + 1)
            print(f"  attempt {attempt+1} failed: {str(e)[:120]} — retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"gave up after {retries} attempts: {last_err}")


def run_one(client, dataset, spec, config, variant, args, totals, checker=None):
    """Generate for a single (dataset, config, variant) and write one JSONL file."""
    ds = load_dataset(spec["repo"], config, split="test")
    use_typo = (variant == "typo")
    if use_typo and spec["typo"] not in ds.column_names:
        print(f"  [skip] {dataset}/{config}: no '{spec['typo']}' column (clean-only config)", flush=True)
        return
    n = ds.num_rows if args.limit == 0 else min(args.limit, ds.num_rows)
    tag = config if use_typo else "clean"
    if args.file_suffix:
        tag = f"{tag}_{args.file_suffix}"
    outpath = os.path.join(args.outdir, f"{dataset}_{tag}.jsonl")
    print(f"\n=== {dataset} / {config} ({variant}): {n} questions"
          f"{f' x{args.n_samples} samples' if args.n_samples > 1 else ''} ===", flush=True)

    done = set()
    if os.path.exists(outpath):
        for line in open(outpath, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["idx"], r["sample_idx"]))
        if done:
            print(f"  resuming: {len(done)} samples already in {outpath}", flush=True)

    tasks = []  # (idx, s_idx, row, typo_q, prompt, gold)
    for idx in range(n):
        row = ds[idx]
        typo_q = row[spec["typo"]] if spec["typo"] in ds.column_names else None
        q_text = typo_q if use_typo else row[spec["clean"]]
        spellchecked_q = q_text
        spellcheck_changes = []
        spellcheck_stats = {
            "spellcheck_total_typo_words": None,
            "spellcheck_restored_exact_n": None,
            "spellcheck_changed_other_n": None,
            "spellcheck_not_restored_n": None,
            "spellcheck_restored_exact_frac": None,
            "spellcheck_restored_exact_pct": None,
        }
        if use_typo and args.fix == "spellcheck":
            spellchecked_q, spellcheck_changes = apply_spellcheck(q_text, checker)
            spellcheck_stats = spellcheck_recovery_stats(row, spellcheck_changes)

        prompt, gold_letter = build_prompt(
            spec["kind"], spellchecked_q, row, seed=idx, fix=args.fix
        )
        gold = gold_letter if spec["kind"] == "mc" else row[spec["gold"]]
        for s_idx in range(args.n_samples):
            if (idx, s_idx) not in done:
                tasks.append((idx, s_idx, row, typo_q, spellchecked_q, spellcheck_changes,
                              spellcheck_stats, prompt, gold))
    if not tasks:
        print("  nothing to do", flush=True)
        return

    t0, ok, errs = time.time(), 0, 0
    with open(outpath, "a", encoding="utf-8") as f, \
         ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        # t = (idx, s_idx, row, typo_q, spellchecked_q, spellcheck_changes,
        #      spellcheck_stats, prompt, gold) -- send the built PROMPT (t[7]),
        #  not the bare question; index drifted when the spellcheck fields were added.
        futs = {pool.submit(ask_one, client, args, t[7], args.retries): t for t in tasks}
        for fut in as_completed(futs):
            (idx, s_idx, row, typo_q, spellchecked_q, spellcheck_changes,
             spellcheck_stats, prompt, gold) = futs[fut]
            try:
                gen = fut.result()
            except Exception as e:
                errs += 1
                print(f"  [{dataset}/{tag} idx={idx} s={s_idx}] FAILED: {str(e)[:120]}", flush=True)
                continue
            rec = {
                "dataset": dataset, "config": config, "variant": variant,
                "fix": args.fix, "idx": idx, "sample_idx": s_idx,
                "clean_question": row[spec["clean"]], "typo_question": typo_q, "prompt": prompt,
                "spellcheck_applied": bool(spellcheck_changes),
                "spellcheck_num_changes": len(spellcheck_changes),
                "spellchecked_question": spellchecked_q if args.fix == "spellcheck" else None,
                "spellcheck_changes": spellcheck_changes if args.fix == "spellcheck" else None,
                **spellcheck_stats,
                "gold_answer": gold, **gen,
            }
            for m in META_COLS:
                if m in row:
                    rec[m] = row[m]
            with write_lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            ok += 1
            totals["tokens"] += gen["n_prompt_tokens"] + gen["n_gen_tokens"]
            totals["cost"] += gen["cost_usd"]
    print(f"  {ok} samples in {time.time()-t0:.0f}s ({errs} failed — rerun to retry), "
          f"running cost ${totals['cost']:.4f}", flush=True)

    # completion order is arbitrary under concurrency — rewrite in dataset order
    with open(outpath, encoding="utf-8") as fin:
        recs = sorted((json.loads(l) for l in fin), key=lambda r: (r["idx"], r["sample_idx"]))
    with open(outpath + ".tmp", "w", encoding="utf-8") as fout:
        for r in recs:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(outpath + ".tmp", outpath)
    print(f"  wrote {outpath}", flush=True)
    if args.hub_repo and errs == 0:
        hub_upload(outpath, dataset, config, variant, args)
    elif args.hub_repo:
        print(f"  [hub] skipped push ({errs} failed samples) — rerun to complete, then it uploads", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="gsm8k",
                    help="gsm8k | math500 | gpqa | arc | a comma list | 'all'")
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                    help="comma list of typo configs, or 'all' for every config on the Hub")
    ap.add_argument("--variant", choices=["typo", "clean", "both"], default="typo",
                    help="'both' = clean baseline once + typo for each config")
    ap.add_argument("--limit", type=int, default=0, help="questions per config (0 = all)")
    ap.add_argument("--n-samples", type=int, default=1, help="generations per question (for averaging)")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--outdir", default=os.path.expanduser("~/nlp_project/results"))
    # --- API instead of GPU flags ---
    ap.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL),
                    help="model id at the router; ':nscale' pins the Nscale backend")
    ap.add_argument("--concurrency", type=int, default=50,
                    help="parallel requests (verified: 500 works; 50-100 avoids 504 retries)")
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--hub-repo", default=os.environ.get("HUB_REPO"),
                    help="HF dataset repo to mirror results to, e.g. user/typo-results "
                         "(layout: results/{dataset}/typo{rate}/real{ratio}.jsonl; "
                         "needs a write token in HF_WRITE_TOKEN or HF_TOKEN)")
    ap.add_argument("--fix", choices=["rewrite", "warn", "spellcheck"], default=None,
                    help="typo mitigation to test: 'rewrite' = ask the model to correct "
                         "typos before solving; 'warn' = only note typos may exist; "
                         "'spellcheck' = external spelling correction before prompting")
    ap.add_argument("--file-suffix", default="",
                    help="appended to local filenames and Hub paths/configs, e.g. "
                         "'20000' -> gsm8k_rate25_real10_20000.jsonl, so runs with "
                         "different budgets coexist")
    args = ap.parse_args()
    if args.fix and not args.file_suffix:  # keep fix results apart from no-fix runs
        args.file_suffix = f"fix-{args.fix}"

    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        sys.exit("export HF_TOKEN=hf_... first (fine-grained, 'Make calls to Inference Providers')")
    client = OpenAI(api_key=api_key, base_url=API_BASE)
    checker = maybe_build_spellchecker(args.fix)

    if args.hub_repo:  # make sure the target repo exists before spending tokens
        from huggingface_hub import HfApi
        token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
        HfApi(token=token).create_repo(args.hub_repo, repo_type="dataset",
                                       private=True, exist_ok=True)

    os.makedirs(args.outdir, exist_ok=True)
    datasets = resolve_datasets(args.dataset)
    want_clean = args.variant in ("clean", "both")
    want_typo = args.variant in ("typo", "both")

    totals = {"tokens": 0, "cost": 0.0}
    for dname in datasets:
        spec = DATASETS[dname]
        configs = resolve_configs(args.configs, spec["repo"])
        if want_clean:
            # one clean baseline: prefer the dedicated 'clean' config, else clean column of the first config
            all_cfgs = sorted(get_dataset_config_names(spec["repo"]))
            base_cfg = "clean" if "clean" in all_cfgs else configs[0]
            run_one(client, dname, spec, base_cfg, "clean", args, totals, checker=checker)
        if want_typo:
            for config in configs:
                if config == "clean":
                    continue
                run_one(client, dname, spec, config, "typo", args, totals, checker=checker)

    print(f"\nDONE  total tokens={totals['tokens']:,}  total cost=${totals['cost']:.4f}", flush=True)


if __name__ == "__main__":
    main()
