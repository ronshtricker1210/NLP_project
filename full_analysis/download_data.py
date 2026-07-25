"""Download the gsm8k model-result files from the Hub into data/gsm8k/.

The raw JSONL (~66 MB) is not committed; it lives on the public dataset repo
Dolevabudi/typo-results. Run this once after cloning:

    pip install huggingface_hub
    python download_data.py

Then run the analysis with:  python run_all.py
"""
import os, shutil
from huggingface_hub import HfApi, hf_hub_download

REPO = "Dolevabudi/typo-results"
_HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(_HERE, "data", "gsm8k")


def main():
    os.makedirs(DEST, exist_ok=True)
    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
             if f.endswith(".jsonl") and f.startswith("results/gsm8k/")]
    if not files:
        raise SystemExit(f"no gsm8k result files found in {REPO}")
    for f in sorted(files):
        tmp = hf_hub_download(REPO, f, repo_type="dataset")
        # results/gsm8k/typo25/real10.jsonl -> gsm8k_typo25_real10.jsonl
        leaf = f[len("results/gsm8k/"):].replace(".jsonl", "").replace("/", "_")
        name = "gsm8k_" + leaf + ".jsonl"
        shutil.copyfile(tmp, os.path.join(DEST, name))
        print("downloaded", name)
    print(f"\n{len(files)} files -> {DEST}\nnext: python run_all.py")


if __name__ == "__main__":
    main()
