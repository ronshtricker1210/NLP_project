"""Create the six minimal aggregate tables from an LLM-judge JSONL trace file."""
import argparse
import csv
import json
import os
from collections import defaultdict

from common import config_files, parse_tag, load_evals


SCORES = {
    "self-doubt": "self_doubt_score",
    "repair_behavioure": "repair_understanding_score",
}


def load_latest_judgments(path):
    latest = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            latest[(row["config"], row["idx"])] = row
    return latest


def load_correctness():
    evaluations = {}
    for path in config_files():
        meta = parse_tag(path)
        for idx, evaluation in load_evals(path).items():
            evaluations[(meta["tag"], idx)] = evaluation
    return evaluations


def write_csv(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def mean_rows(groups, key_name, score_key):
    rows = []
    for group_name in sorted(groups):
        values = groups[group_name]
        rows.append({
            key_name: group_name,
            "n": len(values),
            f"{score_key}_mean": round(sum(values) / len(values), 3),
        })
    return rows


def create_tables(judgments, evaluations, output_root, dimension, score_key):
    output_dir = os.path.join(output_root, f"llm_as_a_judges_{dimension}")
    os.makedirs(output_dir, exist_ok=True)

    valid = []
    for judgment in judgments.values():
        try:
            score = float(judgment[score_key])
        except (KeyError, TypeError, ValueError):
            continue
        if score < 0:
            continue
        evaluation = evaluations.get((judgment["config"], judgment["idx"]))
        if not evaluation or not evaluation["answered"]:
            continue
        if judgment["config"] == "clean":
            typo_rate, real_ratio = 0, None
        else:
            parts = judgment["config"].split("_")
            typo_rate = int(parts[0].removeprefix("typo"))
            real_ratio = int(parts[1].removeprefix("real"))
        valid.append({
            "score": score,
            "config": judgment["config"],
            "typo_rate": typo_rate,
            "real_ratio": real_ratio,
            "correct": bool(evaluation["correct"]),
        })

    by_real = defaultdict(list)
    by_typo = defaultdict(list)
    by_outcome = defaultdict(list)
    for row in valid:
        if row["real_ratio"] is not None:
            by_real[row["real_ratio"]].append(row["score"])
        by_typo[row["typo_rate"]].append(row["score"])
        by_outcome["correct" if row["correct"] else "wrong"].append(row["score"])

    write_csv(
        os.path.join(output_dir, f"{dimension}_mean_by_real.csv"),
        ["real_ratio", "n", f"{score_key}_mean"],
        mean_rows(by_real, "real_ratio", score_key),
    )
    write_csv(
        os.path.join(output_dir, f"{dimension}_mean_by_typo.csv"),
        ["typo_percentage", "n", f"{score_key}_mean"],
        mean_rows(by_typo, "typo_percentage", score_key),
    )
    write_csv(
        os.path.join(output_dir, f"{dimension}_mean_by_correct_wrong.csv"),
        ["outcome", "n", f"{score_key}_mean"],
        mean_rows(by_outcome, "outcome", score_key),
    )
    print(f"{output_dir}: {len(valid)} valid traces")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=os.path.join("manual_check", "gsm8k_first2_5.jsonl"),
        help="judge JSONL trace file",
    )
    parser.add_argument(
        "--output-root",
        default=".",
        help="directory under which the two requested output directories are created",
    )
    parser.add_argument("--dataset", default=None,
                        help="dataset used for source correctness evaluation")
    args = parser.parse_args()

    if args.dataset:
        os.environ["NLP_DATASET"] = args.dataset
    judgments = load_latest_judgments(args.input)
    evaluations = load_correctness()
    for dimension, score_key in SCORES.items():
        create_tables(judgments, evaluations, args.output_root, dimension, score_key)


if __name__ == "__main__":
    main()
