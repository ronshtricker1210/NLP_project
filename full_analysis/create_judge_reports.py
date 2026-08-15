"""Create separate self-doubt and repair-understanding judge reports."""
import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict


CONFIG_ORDER = ["clean"] + [
    f"typo{rate}_real{real}"
    for rate in (25, 50, 75)
    for real in (10, 40, 70)
]


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row, key):
    return float(row[key])


def i(row, key):
    return int(row[key])


def esc(value):
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"),
                     ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def aggregate_per_typo(rows, key):
    groups = defaultdict(list)
    for row in rows:
        config = row["config"]
        rate = 0 if config == "clean" else int(config.split("_")[0][4:])
        groups[rate].append(f(row, key))
    return [(rate, len(groups[rate]), sum(groups[rate]) / len(groups[rate]))
            for rate in (0, 25, 50, 75)]


def aggregate_per_real(rows, key):
    groups = defaultdict(list)
    for row in rows:
        if row["config"] == "clean":
            continue
        real = int(row["config"].split("_")[1][4:])
        groups[real].append(f(row, key))
    return [(real, len(groups[real]), sum(groups[real]) / len(groups[real]))
            for real in (10, 40, 70)]


def fmt(value):
    return f"{value:.3f}"


def make_graph(points, title, xlabel, ylabel, output):
    import matplotlib.pyplot as plt

    xs = [p[0] for p in points]
    ys = [p[2] for p in points]
    figure, axis = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    axis.plot(xs, ys, "o-", color="#176b87", linewidth=2.2, markersize=7,
              markerfacecolor="#f4a261", markeredgecolor="#173f4f",
              markeredgewidth=1.0)
    axis.set_title(title, pad=12, fontweight="bold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_xticks(xs)
    axis.set_ylim(0, 2)
    axis.grid(axis="y", color="#d9e2e6", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def table(title, headers, rows):
    columns = "l" + "r" * len(headers)
    lines = [r"\begin{table}[H]", r"\centering", rf"\caption{{{esc(title)}}}",
             r"\small", rf"\begin{{tabular}}{{{columns}}}", r"\toprule",
             " & ".join(rf"\textbf{{{esc(header)}}}" for header in headers) + r" \\",
             r"\midrule"]
    for row in rows:
        lines.append(" & ".join(esc(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def build_report(kind, metric, scale, title, definition, interpretation,
                 trace_rows, config_rows, graph_typo, graph_real, output_tex):
    typo_points = aggregate_per_typo(trace_rows, metric)
    real_points = aggregate_per_real(trace_rows, metric)
    config_lookup = {row["config"]: row for row in config_rows}

    typo_table = table(
        "Mean score pooled by typo percentage. Clean is the 0% baseline; typo points pool all three real-word ratios.",
        ["typo %", "n", "mean score"],
        [[str(x), str(n), fmt(mean)] for x, n, mean in typo_points],
    )
    real_table = table(
        "Mean score pooled by real-word ratio. Each point pools the 25%, 50%, and 75% typo configurations.",
        ["real-word ratio %", "n", "mean score"],
        [[str(x), str(n), fmt(mean)] for x, n, mean in real_points],
    )

    examples = [
        ["clean", config_lookup["clean"]],
        ["typo75_real10", config_lookup["typo75_real10"]],
        ["typo75_real70", config_lookup["typo75_real70"]],
    ]
    aggregate_key = {
        "self_doubt_score": "mean_self_doubt",
        "repair_understanding_score": "mean_repair_understanding",
    }[metric]
    example_rows = [[name, str(i(row, "n")), fmt(f(row, aggregate_key)),
                     fmt(f(row, aggregate_key)),
                     fmt(f(row, "mean_repair_understanding"))]
                    for name, row in examples]
    example_table = table(
        "Examples from the per-configuration aggregate table. The relevant score is shown alongside both judge dimensions for context.",
        ["config", "n", "selected score", "self-doubt", "repair-understanding"],
        example_rows,
    )

    doc = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}", r"\usepackage{booktabs}",
        r"\usepackage{graphicx}", r"\usepackage{float}",
        r"\usepackage{adjustbox}", r"\usepackage{caption}",
        r"\captionsetup{font=small,labelfont=bf,skip=4pt}",
        rf"\title{{LLM-as-a-Judge Report: {esc(title)}\\\large GSM8K}}",
        r"\date{}", r"\begin{document}", r"\maketitle",
        rf"\section*{{What is measured?}} {definition}",
        rf"\section*{{How to read the score}} The judge uses a {scale} scale. "
        r"The reported value in the tables and graphs is the arithmetic mean over valid judged traces in the group. "
        r"These are behavioral scores, not correctness scores: a model can be wrong without showing doubt, and it can be correct while expressing uncertainty.",
        rf"\section*{{Examples}} {interpretation}", example_table,
        rf"\section*{{Aggregate result by typo percentage}} {typo_table}",
        rf"\begin{{figure}}[H]\centering\includegraphics[width=0.82\textwidth]{{graphs/{esc(os.path.basename(graph_typo))}}}\caption{{{esc(title)} pooled by typo percentage.}}\end{{figure}}",
        rf"\section*{{Aggregate result by real-word ratio}} {real_table}",
        rf"\begin{{figure}}[H]\centering\includegraphics[width=0.82\textwidth]{{graphs/{esc(os.path.basename(graph_real))}}}\caption{{{esc(title)} pooled by real-word ratio.}}\end{{figure}}",
        r"\section*{Conclusion}", rf"{interpretation}",
        r"\end{document}",
    ]
    with open(output_tex, "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(doc))


def compile_pdf(tex_path):
    directory = os.path.dirname(tex_path)
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", os.path.basename(tex_path)],
        cwd=directory, capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stdout[-2000:] + result.stderr[-1000:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--compile-pdf", action="store_true")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    tables_dir = os.path.join(here, "tables", args.dataset)
    output_dir = os.path.join(here, "reports", "judge_scalar_reports")
    graph_dir = os.path.join(output_dir, "graphs")
    os.makedirs(graph_dir, exist_ok=True)

    per_trace = read_csv(os.path.join(tables_dir, "judge_scalar_per_trace.csv"))
    per_config = read_csv(os.path.join(tables_dir, "judge_scalar_per_config.csv"))

    reports = [
        ("self_doubt_score", "Self-doubt", "0--10",
         "Self-doubt",
         r"Self-doubt is the judge's assessment of how much the reasoning trace second-guesses itself, independently of whether the final answer is correct. A score of 0 means a straight-line trace with no hesitation; 10 means pervasive uncertainty, reversals, or looping. The judge also returns verbatim evidence and a one-line justification, which are retained in the manual-check JSONL.",
         r"The typo-percentage trend rises above the clean baseline (0.684 at 0\% versus 1.052 at 25\%), but it is not monotonic: it falls to 0.865 at 50\% and 0.799 at 75\%. For real-word ratio, the requested increasing claim is not supported in this run: the pooled means decrease from 1.001 at 10\% to 0.889 at 40\% and 0.826 at 70\%. The graph therefore reports the observed result rather than drawing an increasing line that the data does not show.",
         "self_doubt_report.tex"),
        ("repair_understanding_score", "Repair-understanding", "0--5",
         "Repair-understanding",
         r"Repair-understanding is the judge's assessment of how much the model loses the intended meaning of the question because of the corrupted wording. A score of 0 means the meaning is fully understood; 5 means the model is fundamentally lost. This is not a label for whether the arithmetic answer is correct.",
         r"The typo-percentage trend supports the requested direction: the pooled mean increases from 0.053 at 0\% to 0.580 at 25\%, 0.576 at 50\%, and 0.764 at 75\%. The real-word ratio comparison is less monotonic: it is 0.554 at 10\%, 0.697 at 40\%, and 0.669 at 70\%. Thus more typo corruption is associated with lower understanding in this sample, while the real-word-ratio effect peaks at 40\% rather than increasing through 70\%.",
         "repair_understanding_report.tex"),
    ]

    import matplotlib.pyplot  # noqa: F401
    for metric, label, scale, title, definition, interpretation, filename in reports:
        graph_typo = os.path.join(graph_dir, f"{metric}_by_typo_rate.png")
        graph_real = os.path.join(graph_dir, f"{metric}_by_real_ratio.png")
        make_graph(aggregate_per_typo(per_trace, metric),
                   f"{label} by typo percentage", "Typo percentage", f"Mean {label.lower()} score", graph_typo)
        make_graph(aggregate_per_real(per_trace, metric),
                   f"{label} by real-word ratio", "Real-word ratio (%)", f"Mean {label.lower()} score", graph_real)
        tex_path = os.path.join(output_dir, filename)
        build_report(metric, metric, scale, title, definition, interpretation,
                 per_trace, per_config, graph_typo, graph_real, tex_path)
        if args.compile_pdf:
            compile_pdf(tex_path)
        print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
