#!/usr/bin/env python
"""Interactive Slurm + Hugging Face runner for the typo dataset pipeline."""

from __future__ import annotations

import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
SSH_KNOWN_HOSTS_PATH: Optional[Path] = None
SSH_CONTROL_PATH: Optional[Path] = None

DATASET_PRESETS: Dict[str, Dict[str, Any]] = {
    "1": {
        "label": "MATH-500",
        "dataset_name": "HuggingFaceH4/MATH-500",
        "dataset_config_name": None,
        "dataset_split": "test",
        "text_field": "problem",
        "default_limit": 500,
    },
    "2": {
        "label": "GSM8K test",
        "dataset_name": "openai/gsm8k",
        "dataset_config_name": "main",
        "dataset_split": "test",
        "text_field": "question",
        "default_limit": 1319,
    },
    "3": {
        "label": "GPQA Diamond",
        "dataset_name": "Idavidrein/gpqa",
        "dataset_config_name": "gpqa_diamond",
        "dataset_split": "train",
        "text_field": "Question",
        "default_limit": 198,
        "requires_hf_token": True,
    },
    "4": {
        "label": "Custom Hugging Face dataset",
        "dataset_name": "",
        "dataset_config_name": None,
        "dataset_split": "test",
        "text_field": "problem",
        "default_limit": 50,
    },
}


def prompt(text: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    value = input(f"{text}{suffix}: ").strip()
    if not value and default is not None:
        return default
    return value


def prompt_bool(text: str, default: bool = True) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        value = input(f"{text} [{default_text}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer y or n.")


def prompt_int(text: str, default: int, minimum: Optional[int] = None) -> int:
    while True:
        value = prompt(text, str(default))
        try:
            parsed = int(value)
        except ValueError:
            print("Please enter an integer.")
            continue
        if minimum is not None and parsed < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return parsed


def prompt_float(text: str, default: float, minimum: float = 0.0) -> float:
    while True:
        value = prompt(text, str(default))
        try:
            parsed = float(value)
        except ValueError:
            print("Please enter a number.")
            continue
        if parsed < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return parsed


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    value = value.strip(".-_")
    return value or datetime.now().strftime("typo-run-%Y%m%d-%H%M%S")


def run_checked(args: List[str], cwd: Optional[Path] = None) -> None:
    printable = " ".join(args)
    print(f"\n$ {printable}")
    subprocess.run(args, cwd=str(cwd) if cwd else None, check=True)


def run_capture(args: List[str], cwd: Optional[Path] = None) -> str:
    result = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return result.stdout


def configure_ssh_reuse(use_temp_known_hosts: bool) -> None:
    global SSH_KNOWN_HOSTS_PATH, SSH_CONTROL_PATH
    if use_temp_known_hosts and SSH_KNOWN_HOSTS_PATH is None:
        handle = tempfile.NamedTemporaryFile(prefix="slurm_known_hosts_", delete=False)
        handle.close()
        SSH_KNOWN_HOSTS_PATH = Path(handle.name)
    if os.name == "nt":
        SSH_CONTROL_PATH = None
        return
    if SSH_CONTROL_PATH is None:
        SSH_CONTROL_PATH = Path(tempfile.gettempdir()) / f"slurm_ssh_mux_{os.getpid()}"


def ssh_options(use_temp_known_hosts: bool) -> List[str]:
    options: List[str] = []
    if not use_temp_known_hosts:
        pass
    else:
        if SSH_KNOWN_HOSTS_PATH is None:
            configure_ssh_reuse(use_temp_known_hosts)
        options.extend(
            [
                "-o",
                f"UserKnownHostsFile={SSH_KNOWN_HOSTS_PATH}",
                "-o",
                "StrictHostKeyChecking=accept-new",
            ]
        )
    if SSH_CONTROL_PATH is not None:
        control_path = str(SSH_CONTROL_PATH).replace("\\", "/")
        options.extend(
            [
                "-o",
                "ControlMaster=auto",
                "-o",
                "ControlPersist=10m",
                "-o",
                f"ControlPath={control_path}",
            ]
        )
    return options


def ssh(user_host: str, command: str, use_temp_known_hosts: bool) -> None:
    args = ["ssh", *ssh_options(use_temp_known_hosts), user_host, "bash", "-s"]
    printable = " ".join(args)
    print(f"\n$ {printable}")
    script = f"set -e\n{command}\n".encode("utf-8")
    subprocess.run(args, input=script, check=True)


def ssh_capture(user_host: str, command: str, use_temp_known_hosts: bool) -> str:
    args = ["ssh", *ssh_options(use_temp_known_hosts), user_host, "bash", "-s"]
    result = subprocess.run(
        args,
        input=f"set -e\n{command}\n".encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return result.stdout.decode("utf-8", errors="replace")


def scp_to_remote(paths: List[Path], user_host: str, remote_dir: str, use_temp_known_hosts: bool) -> None:
    run_checked(
        [
            "scp",
            *ssh_options(use_temp_known_hosts),
            *[str(path) for path in paths],
            f"{user_host}:{remote_dir}/",
        ]
    )


def scp_from_remote(user_host: str, remote_path: str, local_path: Path, use_temp_known_hosts: bool) -> None:
    if local_path.exists():
        shutil.rmtree(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    run_checked(["scp", "-r", *ssh_options(use_temp_known_hosts), f"{user_host}:{remote_path}", str(local_path)])


def choose_dataset() -> Dict[str, Any]:
    print("\nDataset options from the proposal:")
    for key, preset in DATASET_PRESETS.items():
        print(f"  {key}. {preset['label']}")
    choice = prompt("Choose dataset", "1")
    preset = dict(DATASET_PRESETS.get(choice, DATASET_PRESETS["1"]))

    if choice == "4":
        preset["dataset_name"] = prompt("Hugging Face dataset id, e.g. org/name")
        config_name = prompt("Dataset config name, if any", "")
        preset["dataset_config_name"] = config_name or None
        preset["dataset_split"] = prompt("Dataset split", preset["dataset_split"])
        preset["text_field"] = prompt("Text/question field to typo", preset["text_field"])
    else:
        if prompt_bool("Edit dataset id/config/split/text field?", False):
            preset["dataset_name"] = prompt("Dataset id", preset["dataset_name"])
            config_default = preset["dataset_config_name"] or ""
            config_name = prompt("Dataset config name, if any", config_default)
            preset["dataset_config_name"] = config_name or None
            preset["dataset_split"] = prompt("Dataset split", preset["dataset_split"])
            preset["text_field"] = prompt("Text/question field to typo", preset["text_field"])
    return preset


def collect_config() -> Dict[str, Any]:
    dataset = choose_dataset()
    default_run_name = sanitize_name(
        f"{dataset['label'].lower()}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_name = sanitize_name(prompt("Run/config name", default_run_name))

    print("\nSize and typo settings:")
    limit_default = dataset.get("default_limit") or 50
    subset_size = prompt_int("How many examples to process?", int(limit_default), minimum=1)
    typo_rate = prompt_float("Typo rate, fraction of eligible words", 0.30, minimum=0.0)
    groups = prompt_int("Number of real-word-ratio groups/bins", 4, minimum=1)
    seed = prompt_int("Random seed", 42)
    min_word_len = prompt_int("Minimum word length to typo", 2, minimum=1)
    real_word_retry_attempts = prompt_int(
        "Max typo tries per selected word to find a real-word typo", 1, minimum=1
    )

    print("\nTypo weights. They do not need to sum to 1; the script will pass them as relative weights.")
    weights = {
        "delete": prompt_float("Weight for delete", 0.20),
        "insert": prompt_float("Weight for insert", 0.20),
        "replace": prompt_float("Weight for adjacent-key replace", 0.40),
        "transpose": prompt_float("Weight for transpose", 0.20),
    }
    if sum(weights.values()) <= 0:
        raise ValueError("At least one typo weight must be > 0.")

    print("\nSlurm settings:")
    slurm_user = prompt("TAU Slurm username", "ronshtricker")
    slurm_host = prompt("Slurm login host", "slurm-client.cs.tau.ac.il")
    partition = prompt("Slurm partition", "studentkillable")
    account = prompt("Slurm account, blank if not needed", "")
    cpus = prompt_int("CPUs per task", 8, minimum=1)
    memory = prompt("Memory", "8G")
    time_limit = prompt("Time limit", "00:30:00")
    scratch_base = prompt("Remote scratch base", "/specific/scratches/scratch")
    install_deps = prompt_bool("Install/update Python deps on remote scratch?", True)
    use_temp_known_hosts = prompt_bool(
        "Use temporary SSH known_hosts with accept-new for TAU routed Slurm clients?",
        True,
    )

    print("\nHugging Face settings:")
    repo_default = f"{slurm_user}/{run_name}"
    repo_id = prompt("Dataset repo id username/name", repo_default)
    private = prompt_bool("Create/upload as private dataset repo?", True)
    upload_config_name = sanitize_name(prompt("Hugging Face config name", run_name))
    forward_hf_token = prompt_bool(
        "Forward local Hugging Face token to Slurm for gated/private source datasets?",
        bool(dataset.get("requires_hf_token")),
    )

    return {
        **dataset,
        "run_name": run_name,
        "subset_size": subset_size,
        "typo_rate": typo_rate,
        "real_token_groups": groups,
        "seed": seed,
        "min_word_len": min_word_len,
        "real_word_retry_attempts": real_word_retry_attempts,
        "typo_weights": weights,
        "slurm_user": slurm_user,
        "slurm_host": slurm_host,
        "partition": partition,
        "account": account or None,
        "cpus": cpus,
        "memory": memory,
        "time_limit": time_limit,
        "scratch_base": scratch_base.rstrip("/"),
        "install_deps": install_deps,
        "use_temp_known_hosts": use_temp_known_hosts,
        "repo_id": repo_id,
        "private": private,
        "upload_config_name": upload_config_name,
        "forward_hf_token": forward_hf_token,
    }


def write_remote_files(work_dir: Path, config: Dict[str, Any]) -> Dict[str, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / "job_config.json"
    runner_path = work_dir / "run_remote_pipeline.py"
    slurm_path = work_dir / "run_interactive.slurm"

    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    runner_path.write_text(
        """
from pathlib import Path
import json

import nltk

from typo_pipeline import Config, main

with open("job_config.json", encoding="utf-8") as fh:
    cfg = json.load(fh)

nltk_data = Path("nltk_data").resolve()
nltk_data.mkdir(exist_ok=True)
nltk.download("words", download_dir=str(nltk_data), quiet=True)

config = Config(
    dataset_name=cfg["dataset_name"],
    dataset_config_name=cfg.get("dataset_config_name"),
    dataset_split=cfg["dataset_split"],
    subset_size=int(cfg["subset_size"]),
    text_field=cfg["text_field"],
    allow_offline_fallback=False,
    typo_rate=float(cfg["typo_rate"]),
    typo_weights=tuple((name, float(weight)) for name, weight in cfg["typo_weights"].items()),
    min_word_len=int(cfg["min_word_len"]),
    real_word_retry_attempts=int(cfg.get("real_word_retry_attempts", 1)),
    real_token_groups=int(cfg["real_token_groups"]),
    num_proc=int(cfg["cpus"]),
    seed=int(cfg["seed"]),
    out_dir=Path("typo_dataset").resolve(),
)

main(config)
""".lstrip(),
        encoding="utf-8",
    )

    account_line = f"#SBATCH --account={config['account']}\n" if config.get("account") else ""
    install_line = (
        "python3 -m pip install --upgrade --target ./python_pkgs -r requirements.txt\n"
        if config.get("install_deps")
        else ""
    )
    slurm_path.write_text(
        f"""#!/bin/bash
#SBATCH --job-name={config['run_name'][:50]}
#SBATCH --output={config['run_name']}_%j.out
#SBATCH --error={config['run_name']}_%j.err
#SBATCH --time={config['time_limit']}
#SBATCH --partition={config['partition']}
{account_line}#SBATCH --cpus-per-task={config['cpus']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem={config['memory']}

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$PWD/python_pkgs:${{PYTHONPATH:-}}"
export NLTK_DATA="$PWD/nltk_data"
export HF_HOME="$PWD/hf_home"
if [ -f "$PWD/.hf_token" ]; then
    export HF_TOKEN="$(cat "$PWD/.hf_token")"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

mkdir -p "$NLTK_DATA" "$HF_HOME"
{install_line}srun python3 run_remote_pipeline.py

echo "Done. Dataset written to $PWD/typo_dataset"
""",
        encoding="utf-8",
        newline="\n",
    )
    return {"config": config_path, "runner": runner_path, "slurm": slurm_path}


def local_hf_token() -> Optional[str]:
    from huggingface_hub import get_token

    token = get_token()
    if token:
        return token
    print("Paste a Hugging Face token. Input will not be visible.")
    token = getpass.getpass("HF token: ").strip()
    return token or None


def copy_hf_token_to_remote(user_host: str, remote_dir: str, use_temp_known_hosts: bool) -> None:
    token = local_hf_token()
    if not token:
        raise RuntimeError("No Hugging Face token available for the Slurm job.")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(token)
        token_path = Path(handle.name)
    try:
        scp_to_remote([token_path], user_host, remote_dir, use_temp_known_hosts)
        ssh(
            user_host,
            f"cd {remote_dir} && mv {token_path.name} .hf_token && chmod 600 .hf_token",
            use_temp_known_hosts,
        )
    finally:
        token_path.unlink(missing_ok=True)


def build_upload_metadata(config: Dict[str, Any], dataset_dict: "DatasetDict") -> Dict[str, Any]:
    bin_counts = {name: len(dataset) for name, dataset in dataset_dict.items()}
    return {
        "run_name": config["run_name"],
        "source_dataset": config["dataset_name"],
        "source_dataset_config": config.get("dataset_config_name"),
        "source_split": config["dataset_split"],
        "text_field": config["text_field"],
        "examples_requested": config["subset_size"],
        "rows_uploaded": sum(bin_counts.values()),
        "typo_rate": config["typo_rate"],
        "real_word_retry_attempts": config.get("real_word_retry_attempts", 1),
        "real_word_ratio_groups": config["real_token_groups"],
        "typo_weights": config["typo_weights"],
        "bin_counts": bin_counts,
    }


def upload_to_hugging_face(config: Dict[str, Any], dataset_dir: Path) -> str:
    from datasets import DatasetDict, load_dataset, load_from_disk
    from huggingface_hub import HfApi

    token = local_hf_token()
    if token:
        print("\nHugging Face token found locally.")
    if not token:
        print("Paste a Hugging Face write token. Input will not be visible.")
        token = getpass.getpass("HF token: ").strip()
    if not token:
        raise RuntimeError("No Hugging Face token provided.")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=config["repo_id"],
        repo_type="dataset",
        private=bool(config["private"]),
        exist_ok=True,
    )

    splits = {}
    for bin_dir in sorted(dataset_dir.glob("bin_*")):
        split_name = bin_dir.name.replace("-", "_")
        splits[split_name] = load_from_disk(str(bin_dir))
    if not splits:
        raise RuntimeError(f"No bin_* datasets found under {dataset_dir}")

    dataset_dict = DatasetDict(splits)
    metadata = build_upload_metadata(config, dataset_dict)
    print("\nUploading splits: " + ", ".join(f"{name}={len(ds)}" for name, ds in dataset_dict.items()))
    dataset_dict.push_to_hub(
        config["repo_id"],
        config_name=config["upload_config_name"],
        private=bool(config["private"]),
        token=token,
    )

    bin_count_lines = "\n".join(f"- `{name}`: `{count}` rows" for name, count in metadata["bin_counts"].items())
    card = f"""# {config['upload_config_name']}

Generated by `interactive_slurm_hf_run.py` on TAU Slurm.

- Source dataset: `{config['dataset_name']}`
- Dataset config: `{config.get('dataset_config_name') or ''}`
- Split: `{config['dataset_split']}`
- Text field corrupted: `{config['text_field']}`
- Examples requested: `{config['subset_size']}`
- Typo rate: `{config['typo_rate']}`
- Real-word retry attempts: `{config.get('real_word_retry_attempts', 1)}`
- Real-word-ratio groups: `{config['real_token_groups']}`
- Typo weights: `{config['typo_weights']}`
- Rows uploaded: `{metadata['rows_uploaded']}`

## Bin counts

{bin_count_lines}

Use `problem_typo` for the typo-corrupted question. The original clean question remains in `{config['text_field']}`.
"""
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo=f"{config['upload_config_name']}_RUN.md",
        repo_id=config["repo_id"],
        repo_type="dataset",
        commit_message=f"Add run notes for {config['upload_config_name']}",
    )
    api.upload_file(
        path_or_fileobj=json.dumps(metadata, indent=2).encode("utf-8"),
        path_in_repo=f"{config['upload_config_name']}_metadata.json",
        repo_id=config["repo_id"],
        repo_type="dataset",
        commit_message=f"Add metadata for {config['upload_config_name']}",
    )

    first_split = next(iter(splits))
    loaded = load_dataset(
        config["repo_id"],
        config["upload_config_name"],
        split=first_split,
        token=token,
    )
    print(f"Verified Hub load: {first_split} has {len(loaded)} rows.")
    if len(loaded):
        print("Sample typo question:")
        print(str(loaded[0].get("problem_typo", ""))[:700])

    return f"https://huggingface.co/datasets/{config['repo_id']}"


def main() -> int:
    print("Interactive typo dataset runner: local prompts -> Slurm job -> Hugging Face upload")
    config = collect_config()
    user_host = f"{config['slurm_user']}@{config['slurm_host']}"
    remote_dir = f"{config['scratch_base']}/{config['slurm_user']}_{config['run_name']}"
    local_work = PROJECT_ROOT / "interactive_runs" / config["run_name"]
    local_remote_files = local_work / "remote_files"
    local_download = local_work / "typo_dataset"

    print("\nSummary:")
    print(json.dumps({k: v for k, v in config.items() if k != "typo_weights"}, indent=2))
    print(f"typo_weights: {config['typo_weights']}")
    if not prompt_bool("Start Slurm run and upload?", True):
        print("Cancelled before running.")
        return 1

    files = write_remote_files(local_remote_files, config)
    use_temp_known_hosts = bool(config["use_temp_known_hosts"])
    configure_ssh_reuse(use_temp_known_hosts)

    ssh(user_host, f"mkdir -p {remote_dir}", use_temp_known_hosts)
    scp_to_remote(
        [
            PROJECT_ROOT / "typo_pipeline.py",
            PROJECT_ROOT / "typo_generator.py",
            PROJECT_ROOT / "requirements.txt",
            files["config"],
            files["runner"],
            files["slurm"],
        ],
        user_host,
        remote_dir,
        use_temp_known_hosts,
    )

    if config.get("forward_hf_token"):
        copy_hf_token_to_remote(user_host, remote_dir, use_temp_known_hosts)

    ssh(user_host, f"cd {remote_dir} && perl -pi -e 's/\\r$//' run_interactive.slurm && bash -n run_interactive.slurm", use_temp_known_hosts)
    ssh(user_host, f"cd {remote_dir} && sbatch --wait run_interactive.slurm", use_temp_known_hosts)

    print("\nSlurm output logs:")
    logs = ssh_capture(user_host, f"cd {remote_dir} && cat {config['run_name']}_*.out {config['run_name']}_*.err", use_temp_known_hosts)
    print(logs)

    scp_from_remote(user_host, f"{remote_dir}/typo_dataset", local_download, use_temp_known_hosts)
    url = upload_to_hugging_face(config, local_download)

    print("\nAll done.")
    print(f"Remote Slurm directory: {remote_dir}")
    print(f"Local downloaded dataset: {local_download}")
    print(f"Hugging Face dataset: {url}")
    print(f"Config name: {config['upload_config_name']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130)