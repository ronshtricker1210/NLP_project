"""Download a dataset's model-result files from the Hub into data/<dataset>/.

The raw JSONL (~66 MB for gsm8k) is not committed; it lives on the public dataset
repo Dolevabudi/typo-results under results/<dataset>/. Run once after cloning:

    pip install huggingface_hub
    python download_data.py                 # gsm8k (default)
    python download_data.py --dataset math500

Then run the analysis with:  python run_all.py --dataset <dataset>
"""
import os, shutil, argparse
from huggingface_hub import HfApi, hf_hub_download

REPO = "Dolevabudi/typo-results"
_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gsm8k", help="gsm8k | math500 | gpqa | ...")
    ap.add_argument("--repo", default=REPO, help="HF dataset repo holding the results")
    args = ap.parse_args()

    dest = os.path.join(_HERE, "data", args.dataset)
    os.makedirs(dest, exist_ok=True)
    prefix = f"results/{args.dataset}/"
    files = [f for f in HfApi().list_repo_files(args.repo, repo_type="dataset")
             if f.endswith(".jsonl") and f.startswith(prefix)]
    if not files:
        raise SystemExit(f"no {args.dataset} result files under {prefix} in {args.repo}")
    for f in sorted(files):
        tmp = hf_hub_download(args.repo, f, repo_type="dataset")
        # results/<ds>/typo25/real10.jsonl -> <ds>_typo25_real10.jsonl
        leaf = f[len(prefix):].replace(".jsonl", "").replace("/", "_")
        name = f"{args.dataset}_{leaf}.jsonl"
        shutil.copyfile(tmp, os.path.join(dest, name))
        print("downloaded", name)
    print(f"\n{len(files)} files -> {dest}\nnext: python run_all.py --dataset {args.dataset}")


if __name__ == "__main__":
    main()
