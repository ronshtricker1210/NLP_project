"""
Run DeepSeek-R1-Distill-Qwen-7B over the typo datasets and save full reasoning traces.

Transformers fallback engine (use run_typo_vllm.py for speed). Loads the model ONCE, then
loops over one or more typo-rate configs (e.g. real10,real40,real70), writing one JSONL line
per question with the chain-of-thought, final answer, gold answer, and all typo metadata.

GCP note: defaults to the cached model in ~/.cache/huggingface; override with --model or
the MODEL_PATH env var.

Example:
    python run_typo.py --dataset gsm8k --configs real10,real40,real70 --limit 5
"""
import os, json, time, argparse, random
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# dataset -> field mapping
DATASETS = {
    "gsm8k":   {"repo": "idoazou/gsm8k-typos",   "clean": "question", "typo": "problem_typo", "gold": "answer",         "kind": "math"},
    "math500": {"repo": "idoazou/math500-typos", "clean": "problem",  "typo": "problem_typo", "gold": "answer",         "kind": "math"},
    "gpqa":    {"repo": "idoazou/gpqa-typos",    "clean": "Question", "typo": "problem_typo", "gold": "Correct Answer", "kind": "mc"},
}
# extra metadata columns to carry through if present
META_COLS = ["target_real_ratio", "real_ratio", "num_total", "num_real", "num_nonword",
             "typo_techniques", "level", "subject", "unique_id", "Record ID"]


def build_prompt(kind, qtext, row, seed):
    """Return (prompt_text, gold_letter_or_None)."""
    if kind == "math":
        return qtext + "\n\nPlease reason step by step, and put your final answer within \\boxed{}.", None
    # multiple choice (gpqa): shuffle the 4 options deterministically per-row
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--configs", required=True, help="comma list, e.g. real10,real40,real70")
    ap.add_argument("--variant", choices=["typo", "clean"], default="typo",
                    help="'typo' uses problem_typo; 'clean' uses the untouched question (baseline)")
    ap.add_argument("--limit", type=int, default=0, help="max questions per config (0 = all)")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="questions generated together (3-5x faster; lower if OOM)")
    ap.add_argument("--outdir", default=os.path.expanduser("~/nlp_project/results"))
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL),
                    help="HF repo id or local path (defaults to the cached DeepSeek model)")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    model_path = args.model
    os.makedirs(args.outdir, exist_ok=True)

    print(f"loading model from {model_path}", flush=True)
    print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"                       # left-pad for correct batched decoder generation
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")

    for config in args.configs.split(","):
        config = config.strip()
        ds = load_dataset(spec["repo"], config, split="test")
        n = ds.num_rows if args.limit == 0 else min(args.limit, ds.num_rows)
        tag = config if args.variant == "typo" else "clean"
        outpath = os.path.join(args.outdir, f"{args.dataset}_{tag}.jsonl")
        print(f"\n=== {args.dataset} / {config} ({args.variant}): {n} questions -> {outpath} ===", flush=True)

        # pre-build prompts + gold for all questions in this config
        items = []
        for idx in range(n):
            row = ds[idx]
            typo_q = row[spec["typo"]]
            q_text = typo_q if args.variant == "typo" else row[spec["clean"]]
            prompt, gold_letter = build_prompt(spec["kind"], q_text, row, seed=idx)
            gold = gold_letter if spec["kind"] == "mc" else row[spec["gold"]]
            chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True, tokenize=False)
            items.append((idx, row, typo_q, prompt, gold, chat))

        with open(outpath, "w") as f:
            done = 0
            for b in range(0, n, args.batch_size):
                batch = items[b:b + args.batch_size]
                enc = tok([it[5] for it in batch], return_tensors="pt", padding=True).to(model.device)
                plen = enc["input_ids"].shape[1]
                t0 = time.time()
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                                     temperature=0.6, top_p=0.95, pad_token_id=tok.pad_token_id)
                dt = time.time() - t0
                for j, (idx, row, typo_q, prompt, gold, chat) in enumerate(batch):
                    gen_ids = out[j][plen:]
                    n_gen = int((gen_ids != tok.pad_token_id).sum())
                    text = tok.decode(gen_ids, skip_special_tokens=True)
                    reasoning, sep, final = text.partition("</think>")
                    if not sep:
                        reasoning, final = text, ""
                    rec = {
                        "dataset": args.dataset, "config": config, "variant": args.variant, "idx": idx,
                        "clean_question": row[spec["clean"]], "typo_question": typo_q, "prompt": prompt,
                        "gold_answer": gold, "generation": text,
                        "reasoning": reasoning.strip(), "final_answer_text": final.strip(),
                        "n_prompt_tokens": plen, "n_gen_tokens": n_gen,
                    }
                    for m in META_COLS:
                        if m in row:
                            rec[m] = row[m]
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done += len(batch)
                print(f"  [{done}/{n}] batch of {len(batch)} in {dt:.0f}s", flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
