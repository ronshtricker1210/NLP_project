"""Create aggregate plots for the scalar LLM-judge results."""
import argparse
import csv
import math
import os
from collections import defaultdict


METRICS = (
    ("repair_understanding_score", "Repair-understanding"),
    ("self_doubt_score", "Self-doubt"),
)


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean_ci(values):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    margin = 1.96 * math.sqrt(variance / n)
    return mean, margin


def parse_config(row):
    config = row["config"]
    if config == "clean":
        return 0, None
    parts = config.split("_")
    return int(parts[0].removeprefix("typo")), int(parts[1].removeprefix("real"))


def plot_metric(rows, metric, title, ylabel, axis_name, values, output_path):
    groups = defaultdict(list)
    for row in rows:
        typo_rate, real_ratio = parse_config(row)
        if axis_name == "typo_rate":
            x = typo_rate
        else:
            if real_ratio is None:
                continue
            x = real_ratio
        groups[x].append(float(row[metric]))

    points = []
    for x in values:
        if x in groups:
            mean, margin = mean_ci(groups[x])
            points.append((x, mean, margin, len(groups[x])))

    if not points:
        raise ValueError(f"No rows available for {title}")

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    x_values = [point[0] for point in points]
    means = [point[1] for point in points]
    axis.errorbar(
        x_values,
        means,
        fmt="o-",
        color="#176b87",
        markerfacecolor="#f4a261",
        markeredgecolor="#173f4f",
        markeredgewidth=1.0,
        linewidth=2.2,
        markersize=7,
        capsize=4,
    )
    axis.set_title(title, pad=12, fontweight="bold")
    axis.set_xlabel("Typo percentage" if axis_name == "typo_rate" else "Real-word ratio (%)")
    axis.set_ylabel(ylabel)
    axis.set_xticks(values)
    axis.set_ylim(0, 2)
    axis.grid(axis="y", color="#d9e2e6", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)

    print(f"{output_path}: " + ", ".join(
        f"x={x}, mean={mean:.3f}, n={n}" for x, mean, _, n in points
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=os.path.join("analysis_tables", "gsm8k", "results", "judge_scalar_per_trace.csv"),
        help="per-trace scalar judge CSV",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("reports", "judge_graphs"),
        help="directory for PNG plots",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_rows(args.input)
    for metric, label in METRICS:
        plot_metric(
            rows,
            metric,
            f"{label} by typo percentage",
            f"Mean {label.lower()} score",
            "typo_rate",
            [0, 25, 50, 75],
            os.path.join(args.out_dir, f"{metric}_by_typo_rate.png"),
        )
        plot_metric(
            rows,
            metric,
            f"{label} by real-word ratio",
            f"Mean {label.lower()} score",
            "real_ratio",
            [10, 40, 70],
            os.path.join(args.out_dir, f"{metric}_by_real_ratio.png"),
        )


if __name__ == "__main__":
    main()
