"""
vLLM version of run_typo.py — same JSONL output, but batches every question in a config
through the GPU at once (continuous batching), giving ~10-20x throughput over transformers.

Example:
    python run_typo_vllm.py --dataset gsm8k --configs real10,real40,real70 --limit 0
"""
import os, json, time, argparse, random
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DATASETS = {
    "gsm8k":   {"repo": "idoazou/gsm8k-typos",   "clean": "question", "typo": "problem_typo", "gold": "answer",         "kind": "math"},
    "math500": {"repo": "idoazou/math500-typos", "clean": "problem",  "typo": "problem_typo", "gold": "answer",         "kind": "math"},
    "gpqa":    {"repo": "idoazou/gpqa-typos",    "clean": "Question", "typo": "problem_typo", "gold": "Correct Answer", "kind": "mc"},
}
META_COLS = ["target_real_ratio", "real_ratio", "num_total", "num_real", "num_nonword",
             "typo_techniques", "level", "subject", "unique_id", "Record ID"]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--configs", required=True)
    ap.add_argument("--variant", choices=["typo", "clean"], default="typo")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--outdir", default=os.path.expanduser("~/nlp_project/results"))
    ap.add_argument("--tp", type=int, default=2, help="tensor parallel size = num GPUs")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="context length; auto = max_new_tokens + 1536 (prompt headroom)")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    model_path = os.environ["MODEL_PATH"]
    os.makedirs(args.outdir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_path)
    # float16 (not bfloat16): the RTX 2080 Ti (sm_75) does not support bf16 in vLLM.
    # enforce_eager=True: skip torch.compile/cudagraph (compute nodes have no nvcc).
    mml = args.max_model_len or (args.max_new_tokens + 1536)   # room for the prompt
    llm = LLM(model=model_path, dtype="float16", tensor_parallel_size=args.tp,
              max_model_len=mml, gpu_memory_utilization=0.90,
              enforce_eager=True, trust_remote_code=True)
    sampling = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_new_tokens)

    for config in args.configs.split(","):
        config = config.strip()
        ds = load_dataset(spec["repo"], config, split="test")
        n = ds.num_rows if args.limit == 0 else min(args.limit, ds.num_rows)
        tag = config if args.variant == "typo" else "clean"
        outpath = os.path.join(args.outdir, f"{args.dataset}_{tag}.jsonl")
        print(f"\n=== {args.dataset} / {config} ({args.variant}): {n} questions ===", flush=True)

        # build all prompts for this config, then generate in one batched call
        rows, prompts, golds = [], [], []
        for idx in range(n):
            row = ds[idx]
            typo_q = row[spec["typo"]]
            q_text = typo_q if args.variant == "typo" else row[spec["clean"]]
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
                text = out.outputs[0].text
                reasoning, sep, final = text.partition("</think>")
                if not sep:
                    reasoning, final = text, ""
                rec = {
                    "dataset": args.dataset, "config": config, "variant": args.variant, "idx": idx,
                    "clean_question": row[spec["clean"]], "typo_question": typo_q, "prompt": prompt,
                    "gold_answer": gold, "generation": text,
                    "reasoning": reasoning.strip(), "final_answer_text": final.strip(),
                    "n_prompt_tokens": len(out.prompt_token_ids),
                    "n_gen_tokens": len(out.outputs[0].token_ids),
                }
                for m in META_COLS:
                    if m in row:
                        rec[m] = row[m]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  wrote {outpath}", flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
