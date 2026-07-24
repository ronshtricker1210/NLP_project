"""
vLLM inference over the typo datasets — GCP / single-L4 build.

Same JSONL output as the TAU server-setup version, but tuned for one NVIDIA L4
(Ada, sm_89): native bfloat16, tensor_parallel_size=1, cudagraphs on. Batches every
question in a config through the GPU at once (continuous batching).

Examples:
    # one dataset, default config sweep, clean baseline + typos
    python run_typo_vllm.py --dataset gsm8k --variant both

    # everything: all three datasets, every config on the Hub, 50 q each (smoke-ish)
    python run_typo_vllm.py --dataset all --configs all --limit 50

    # a specific density sweep, 5 samples/question for averaging
    python run_typo_vllm.py --dataset math500 --configs real0,real30,real70 --n-samples 5
"""
import os, json, time, argparse, random, glob
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

DATASETS = {
    "gsm8k":   {"repo": "idoazou/gsm8k-typos",   "clean": "question", "typo": "problem_typo", "gold": "answer",         "kind": "math"},
    "math500": {"repo": "idoazou/math500-typos", "clean": "problem",  "typo": "problem_typo", "gold": "answer",         "kind": "math"},
    "gpqa":    {"repo": "idoazou/gpqa-typos",    "clean": "Question", "typo": "problem_typo", "gold": "Correct Answer", "kind": "mc"},
}
META_COLS = ["target_real_ratio", "real_ratio", "num_total", "num_real", "num_nonword",
             "typo_techniques", "level", "subject", "unique_id", "Record ID"]

# A small, representative default sweep so `--configs` can be omitted for a first run.
DEFAULT_CONFIGS = ["real0", "real30", "real70"]


def resolve_datasets(arg):
    """--dataset accepts 'all', or a comma list like 'gsm8k,math500'."""
    if arg.strip().lower() == "all":
        return list(DATASETS)
    names = [d.strip() for d in arg.split(",") if d.strip()]
    bad = [d for d in names if d not in DATASETS]
    if bad:
        raise SystemExit(f"unknown dataset(s): {bad}. choices: {list(DATASETS)} or 'all'")
    return names


def hub_configs(repo):
    """List config names available in the locally-cached snapshot of a dataset repo.
    Each config is a subdir holding a parquet file. Returns None if not cached."""
    cache = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                         "hub", "datasets--" + repo.replace("/", "--"), "snapshots")
    snaps = sorted(glob.glob(os.path.join(cache, "*")))
    if not snaps:
        return None
    snap = snaps[-1]
    cfgs = sorted(d for d in os.listdir(snap)
                  if os.path.isdir(os.path.join(snap, d)) and glob.glob(os.path.join(snap, d, "*.parquet")))
    return cfgs or None


def resolve_configs(arg, repo):
    """--configs accepts 'all' (every config in the repo snapshot) or a comma list."""
    if arg.strip().lower() == "all":
        cfgs = hub_configs(repo)
        if not cfgs:
            raise SystemExit(f"--configs all: no cached configs for {repo}; run setup_gcp.sh first")
        return cfgs
    return [c.strip() for c in arg.split(",") if c.strip()]


def build_prompt(kind, qtext, row, seed):
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


def run_one(llm, tok, sampling, dataset, spec, config, variant, args):
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

    rows, prompts, golds = [], [], []
    for idx in range(n):
        row = ds[idx]
        typo_q = row[spec["typo"]] if spec["typo"] in ds.column_names else None
        q_text = typo_q if use_typo else row[spec["clean"]]
        prompt, gold_letter = build_prompt(spec["kind"], q_text, row, seed=idx)
        gold = gold_letter if spec["kind"] == "mc" else row[spec["gold"]]
        chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False)
        rows.append((idx, row, typo_q, prompt)); prompts.append(chat); golds.append(gold)

    t0 = time.time()
    outs = llm.generate(prompts, sampling)
    print(f"  batched {n} questions in {time.time()-t0:.0f}s", flush=True)

    with open(outpath, "w", encoding="utf-8") as f:
        for (idx, row, typo_q, prompt), out, gold in zip(rows, outs, golds):
            for s_idx, comp in enumerate(out.outputs):
                text = comp.text
                reasoning, sep, final = text.partition("</think>")
                if not sep:
                    reasoning, final = text, ""
                rec = {
                    "dataset": dataset, "config": config, "variant": variant,
                    "idx": idx, "sample_idx": s_idx,
                    "clean_question": row[spec["clean"]], "typo_question": typo_q, "prompt": prompt,
                    "gold_answer": gold, "generation": text,
                    "reasoning": reasoning.strip(), "final_answer_text": final.strip(),
                    "n_prompt_tokens": len(out.prompt_token_ids),
                    "n_gen_tokens": len(comp.token_ids),
                }
                for m in META_COLS:
                    if m in row:
                        rec[m] = row[m]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  wrote {outpath}", flush=True)


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
    # --- model / engine (L4 defaults) ---
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL),
                    help="HF repo id or local path (defaults to the cached DeepSeek model)")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size = num GPUs (L4: 1)")
    ap.add_argument("--dtype", default="bfloat16", help="bfloat16 on L4/Ada (native)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="context length; auto = max_new_tokens + 1536 (prompt headroom)")
    ap.add_argument("--enforce-eager", action="store_true",
                    default=os.environ.get("VLLM_ENFORCE_EAGER", "") == "1",
                    help="disable cudagraphs/torch.compile (default on if VLLM_ENFORCE_EAGER=1, "
                         "e.g. bare VMs with no g++/CUDA toolchain for runtime JIT)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    datasets = resolve_datasets(args.dataset)

    tok = AutoTokenizer.from_pretrained(args.model)
    mml = args.max_model_len or (args.max_new_tokens + 1536)
    llm = LLM(model=args.model, dtype=args.dtype, tensor_parallel_size=args.tp,
              max_model_len=mml, gpu_memory_utilization=args.gpu_mem_util,
              enforce_eager=args.enforce_eager, trust_remote_code=True)
    sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                              max_tokens=args.max_new_tokens, n=args.n_samples)

    want_clean = args.variant in ("clean", "both")
    want_typo = args.variant in ("typo", "both")

    for dname in datasets:
        spec = DATASETS[dname]
        configs = resolve_configs(args.configs, spec["repo"])
        if want_clean:
            # one clean baseline: prefer the dedicated 'clean' config, else clean column of the first config
            cached = hub_configs(spec["repo"]) or []
            base_cfg = "clean" if "clean" in cached else configs[0]
            run_one(llm, tok, sampling, dname, spec, base_cfg, "clean", args)
        if want_typo:
            for config in configs:
                if config == "clean":
                    continue
                run_one(llm, tok, sampling, dname, spec, config, "typo", args)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
