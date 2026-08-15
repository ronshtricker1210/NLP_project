"""Create separate self-doubt and repair-behaviour LaTeX/PDF reports."""
import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_analysis"))
from common import config_files, parse_tag, load_evals


REPORTS = {
    "self-doubt": {
        "score_key": "self_doubt_score",
        "label": "Self-doubt",
        "scale": "0--10",
        "directory": "llm_as_a_judges_self-doubt",
        "evidence_key": "self_doubt_evidence",
        "evidence_ok_key": "self_doubt_evidence_ok",
        "definition": (
            "The judge scores visible self-questioning in the reasoning trace, "
            "independently of whether the final answer is correct. A score of 0 "
            "means a straight-line trace with no hesitation; 10 means pervasive "
            "uncertainty, reversals, or looping."
        ),
        "conclusion": (
            "In this run, self-doubt decreases as the real-word ratio increases: "
            "the mean is 1.001 at 10 percent real-word typos, 0.889 at 40 percent, "
            "and 0.826 at 70 percent. This is consistent with the interpretation "
            "that non-real-word typos are more visibly strange and can trigger more "
            "doubt, while real-word typos can be read fluently and silently."
        ),
        "files": (
            "self-doubt_mean_by_real.csv",
            "self-doubt_mean_by_typo.csv",
            "self-doubt_mean_by_correct_wrong.csv",
        ),
    },
    "repair_behavioure": {
        "score_key": "repair_understanding_score",
        "label": "Repair-understanding",
        "scale": "0--5",
        "directory": "llm_as_a_judges_repair_behavioure",
        "evidence_key": "repair_understanding_evidence",
        "evidence_ok_key": "repair_understanding_evidence_ok",
        "definition": (
            "The judge scores how much the model loses the intended meaning of the "
            "question because of corrupted wording. A score of 0 means the meaning "
            "is fully understood; 5 means the model is fundamentally lost. This is "
            "a semantic-understanding score, not a direct correctness label."
        ),
        "conclusion": (
            "In this run, more typos are associated with less ability to recover and "
            "understand the question meaning: the mean rises from 0.053 for clean "
            "traces to 0.580 at 25 percent typos, 0.576 at 50 percent, and 0.764 "
            "at 75 percent. Higher scores mean more loss of the intended meaning."
        ),
        "files": (
            "repair_behavioure_mean_by_real.csv",
            "repair_behavioure_mean_by_typo.csv",
            "repair_behavioure_mean_by_correct_wrong.csv",
        ),
    },
}


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_latest(path):
    latest = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            latest[(row["config"], row["idx"])] = row
    return latest


def load_evaluation_map():
    result = {}
    for path in config_files():
        meta = parse_tag(path)
        for idx, evaluation in load_evals(path).items():
            result[(meta["tag"], idx)] = evaluation
    return result


def latex(value):
    text = str(value)
    replacements = [
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
        ("_", r"\_"), ("#", r"\#"), ("{", r"\{"), ("}", r"\}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def table(caption, headers, rows):
    spec = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[H]", r"\centering", rf"\caption{{{latex(caption)}}}",
        r"\small", rf"\begin{{adjustbox}}{{max width=\textwidth}}",
        rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
        " & ".join(rf"\textbf{{{latex(header)}}}" for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"])
    return "\n".join(lines)


def valid_rows(judgments, evaluations, score_key, evidence_key):
    rows = []
    for judgment in judgments.values():
        try:
            score = float(judgment[score_key])
        except (KeyError, TypeError, ValueError):
            continue
        evaluation = evaluations.get((judgment["config"], judgment["idx"]))
        if score < 0 or not evaluation or not evaluation["answered"]:
            continue
        config = judgment["config"]
        if config == "clean":
            typo_rate, real_ratio = 0, "clean"
        else:
            parts = config.split("_")
            typo_rate = int(parts[0].removeprefix("typo"))
            real_ratio = int(parts[1].removeprefix("real"))
        rows.append({
            "config": config,
            "idx": judgment["idx"],
            "score": score,
            "typo_rate": typo_rate,
            "real_ratio": real_ratio,
            "outcome": "correct" if evaluation["correct"] else "wrong",
            "evidence": judgment.get(evidence_key, ""),
            "summary": judgment.get("one_line_summary", ""),
            "evidence_ok": judgment.get(evidence_key + "_ok", False),
        })
    return rows


def choose_examples(rows):
    examples = []
    for target in (0, 2, 5, 8, 10):
        candidates = [row for row in rows if round(row["score"]) == target]
        if candidates:
            examples.append(min(candidates, key=lambda row: (not row["evidence"], row["idx"])))
    if len(examples) < 3:
        ordered = sorted(rows, key=lambda row: row["score"])
        examples = ordered[::max(1, len(ordered) // 3)][:3]
    unique = []
    seen = set()
    for row in examples:
        key = (row["config"], row["idx"])
        if key not in seen:
            unique.append(row)
            seen.add(key)
    return unique[:5]


def build_report(info, rows, tables_dir, output_dir, input_name, dataset):
    score_key = info["score_key"]
    label = info["label"]
    dimension_dir = os.path.join(output_dir, info["directory"])
    os.makedirs(dimension_dir, exist_ok=True)

    real_rows = read_csv(os.path.join(dimension_dir, info["files"][0]))
    typo_rows = read_csv(os.path.join(dimension_dir, info["files"][1]))
    outcome_rows = read_csv(os.path.join(dimension_dir, info["files"][2]))

    example_blocks = []
    for row in choose_examples(rows):
        example_blocks.extend([
            rf"\paragraph{{Trace {latex(row['config'])}, index {row['idx']}.}}",
            rf"\textbf{{Score:}} {row['score']:.0f} on the {info['scale']} scale. "
            rf"\textbf{{Final-answer outcome:}} {latex(row['outcome'])}.",
            rf"\textbf{{Judge evidence:}} \begin{{quote}}{latex(row['evidence'] or '(none)')}\end{{quote}}",
            rf"\textbf{{One-line summary:}} {latex(row['summary'] or '(none)')}\medskip",
        ])

    doc = [
        r"\documentclass[11pt]{article}", r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}", r"\usepackage{adjustbox}", r"\usepackage{float}",
        r"\usepackage{caption}", r"\captionsetup{font=small,labelfont=bf,skip=4pt}",
        rf"\title{{LLM-as-a-Judge Report: {latex(label)}\\\large {latex(dataset.upper())}}}",
        r"\date{}", r"\begin{document}", r"\maketitle",
        rf"\section*{{What was measured?}} {info['definition']} The judge returns the score, "
        r"a verbatim evidence quote when available, and a one-line summary. The tables "
        rf"below use {info['scale']} scores and include only valid answered traces from {latex(input_name)}.",
        rf"\section*{{Examples at different scores}} The examples below show the score, "
        r"final-answer outcome, the judge's evidence, and its one-line summary. An empty "
        r"evidence field means that the trace contained no suitable verbatim cue for that dimension.",
        "\n".join(example_blocks),
        r"\section*{Aggregate tables}",
        table("Mean score by real-word typo ratio", list(real_rows[0].keys()), [list(row.values()) for row in real_rows]),
        table("Mean score by typo percentage", list(typo_rows[0].keys()), [list(row.values()) for row in typo_rows]),
        table("Mean score by final-answer correctness", list(outcome_rows[0].keys()), [list(row.values()) for row in outcome_rows]),
        rf"\section*{{Consequences}} {info['conclusion']}",
        r"\end{document}",
    ]
    tex_path = os.path.join(dimension_dir, f"{info['directory']}.tex")
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(doc))
    return tex_path


def compile_pdf(tex_path):
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", os.path.basename(tex_path)],
        cwd=os.path.dirname(tex_path), capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stdout[-2000:] + result.stderr[-1000:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--compile-pdf", action="store_true")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    input_arg = args.input or os.path.join("full_analysis", "manual_check", f"{args.dataset}_llm_judge.jsonl")
    output_arg = args.output_root or os.path.join("full_analysis", f"{args.dataset}_llm_as_a_judge")
    input_path = input_arg if os.path.isabs(input_arg) else os.path.join(here, input_arg)
    output_root = output_arg if os.path.isabs(output_arg) else os.path.join(here, output_arg)
    tables_dir = os.path.join(here, "tables", args.dataset)
    judgments = load_latest(input_path)
    evaluations = load_evaluation_map()
    for info in REPORTS.values():
        rows = valid_rows(judgments, evaluations, info["score_key"], info["evidence_key"])
        tex_path = build_report(info, rows, tables_dir, output_root, os.path.basename(input_path), args.dataset)
        if args.compile_pdf:
            compile_pdf(tex_path)
        print(f"wrote {tex_path} ({len(rows)} valid traces)")


if __name__ == "__main__":
    main()
