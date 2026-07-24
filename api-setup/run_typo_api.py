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
import os, json, re, time, argparse, random, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset, get_dataset_config_names
from openai import OpenAI

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:nscale"
API_BASE = os.environ.get("API_BASE", "https://router.huggingface.co/v1")
# Nscale list prices, USD per 1M tokens (reasoning tokens bill as output).
PRICE_IN, PRICE_OUT = 0.01, 0.03

DATASETS = {
    "gsm8k":   {"repo": "idoazou/gsm8k-typos",   "clean": "question", "typo": "problem_typo", "gold": "answer",         "kind": "math"},
    "math500": {"repo": "idoazou/math500-typos", "clean": "problem",  "typo": "problem_typo", "gold": "answer",         "kind": "math"},
    "gpqa":    {"repo": "idoazou/gpqa-typos",    "clean": "Question", "typo": "problem_typo", "gold": "Correct Answer", "kind": "mc"},
}
META_COLS = ["target_real_ratio", "real_ratio", "num_total", "num_real", "num_nonword",
             "typo_techniques", "level", "subject", "unique_id", "Record ID"]

# A small, representative default sweep so `--configs` can be omitted for a first run.
DEFAULT_CONFIGS = ["real0", "real30", "real70"]

write_lock = threading.Lock()


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


def build_prompt(kind, qtext, row, seed):
    # Identical to run_typo_vllm.py (same seeded shuffle => same GPQA letters).
    if kind == "math":
        return qtext + "\n\nPlease reason step by step, and put your final answer within \\boxed{}.", None
    opts = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    order = list(range(4))
    random.Random(seed).shuffle(order)
    letters = "ABCD"
    lines, correct = [], None
    for i, o in enumerate(order):
        lines.append(f"{letters[i]}) {opts[o]}")
        if o == 0:
            correct = letters[i]
    prompt = qtext + "\n\n" + "\n".join(lines) + \
        "\n\nPlease reason step by step, and put the letter of the correct answer within \\boxed{}."
    return prompt, correct


def hub_path(dataset, config, variant):
    """Repo path for a result file: results/{dataset}/typo{rate}/real{ratio}.jsonl.
    Plain realY configs use the fixed ~30% corruption rate (see DATASET_USAGE.md),
    so they file under typo30; rateX_realY configs under typoX; clean baselines
    under results/{dataset}/clean.jsonl."""
    if variant == "clean":
        return f"results/{dataset}/clean.jsonl"
    m = re.fullmatch(r"rate(\d+)_real(\d+)", config)
    if m:
        return f"results/{dataset}/typo{m.group(1)}/real{m.group(2)}.jsonl"
    m = re.fullmatch(r"real(\d+)", config)
    if m:
        return f"results/{dataset}/typo30/real{m.group(1)}.jsonl"
    return f"results/{dataset}/{config}.jsonl"


def hub_config_name(dataset, config, variant):
    """Config name for push_to_hub, unique across datasets sharing one repo:
    math500_clean, math500_typo30_real40, gsm8k_typo25_real0, ..."""
    leaf = hub_path(dataset, config, variant)              # results/ds/typoX/realY.jsonl
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
    dest = hub_path(dataset, config, variant)
    HfApi(token=token).upload_file(path_or_fileobj=outpath, path_in_repo=dest,
                                   repo_id=args.hub_repo, repo_type="dataset",
                                   commit_message=f"results: {dataset}/{config} ({variant})")
    cfg_name = hub_config_name(dataset, config, variant)
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
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                timeout=1200,
            )
            msg = resp.choices[0].message
            generation, reasoning, final = split_generation(msg)
            usage = resp.usage
            return {
                "generation": generation, "reasoning": reasoning, "final_answer_text": final,
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


def run_one(client, dataset, spec, config, variant, args, totals):
    """Generate for a single (dataset, config, variant) and write one JSONL file."""
    ds = load_dataset(spec["repo"], config, split="test")
    use_typo = (variant == "typo")
    if use_typo and spec["typo"] not in ds.column_names:
        print(f"  [skip] {dataset}/{config}: no '{spec['typo']}' column (clean-only config)", flush=True)
        return
    n = ds.num_rows if args.limit == 0 else min(args.limit, ds.num_rows)
    tag = config if use_typo else "clean"
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
        prompt, gold_letter = build_prompt(spec["kind"], q_text, row, seed=idx)
        gold = gold_letter if spec["kind"] == "mc" else row[spec["gold"]]
        for s_idx in range(args.n_samples):
            if (idx, s_idx) not in done:
                tasks.append((idx, s_idx, row, typo_q, prompt, gold))
    if not tasks:
        print("  nothing to do", flush=True)
        return

    t0, ok, errs = time.time(), 0, 0
    with open(outpath, "a", encoding="utf-8") as f, \
         ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(ask_one, client, args, t[4], args.retries): t for t in tasks}
        for fut in as_completed(futs):
            idx, s_idx, row, typo_q, prompt, gold = futs[fut]
            try:
                gen = fut.result()
            except Exception as e:
                errs += 1
                print(f"  [{dataset}/{tag} idx={idx} s={s_idx}] FAILED: {str(e)[:120]}", flush=True)
                continue
            rec = {
                "dataset": dataset, "config": config, "variant": variant,
                "idx": idx, "sample_idx": s_idx,
                "clean_question": row[spec["clean"]], "typo_question": typo_q, "prompt": prompt,
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
                    help="gsm8k | math500 | gpqa | a comma list | 'all'")
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
    args = ap.parse_args()

    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        sys.exit("export HF_TOKEN=hf_... first (fine-grained, 'Make calls to Inference Providers')")
    client = OpenAI(api_key=api_key, base_url=API_BASE)

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
            run_one(client, dname, spec, base_cfg, "clean", args, totals)
        if want_typo:
            for config in configs:
                if config == "clean":
                    continue
                run_one(client, dname, spec, config, "typo", args, totals)

    print(f"\nDONE  total tokens={totals['tokens']:,}  total cost=${totals['cost']:.4f}", flush=True)


if __name__ == "__main__":
    main()
