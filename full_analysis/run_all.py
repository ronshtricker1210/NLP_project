"""Run every active analysis, then build one Overleaf-ready LaTeX report of all
result tables.

  python run_all.py                # run all modules + write report.tex
  python run_all.py --skip-run     # just rebuild report.tex from existing CSVs

Steps:
  1. runs each active module (regenerating its CSVs in tables/),
  2. reads those CSVs and emits report.tex — a self-contained document
     (\\documentclass ... \\end{document}) with a booktabs table per result,
     grouped by proposal dimension. Copy report.tex into Overleaf and compile.
"""
import os, sys, csv, subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(_HERE, "tables")
REPORT = os.path.join(_HERE, "report.tex")

# active modules to run (imported-only files are excluded, so is the retired one)
MODULES = ["accuracy_flips", "reasoning_length", "self_doubt",
           "repair_wordlevel", "real_word_effect", "lexical_grid"]

# report layout: (section title, [(csv file, caption), ...])
SECTIONS = [
    ("Accuracy and Flips", [
        ("accuracy_per_config.csv", "Accuracy per config: strict (truncated=wrong), answered-only, and completion rate, with truncation rate and bootstrap CI."),
        ("flips_vs_clean.csv", "Answered-only flips vs the clean baseline (right->wrong, wrong->right) with McNemar test."),
        ("accuracy_decomposition.csv", "Accuracy drop vs clean split into a not-finishing (completion) part and a wrong-answer (quality) part."),
    ]),
    ("Reasoning Length", [
        ("reasoning_length_absolute.csv", "Absolute generated tokens per config (answered-only)."),
        ("reasoning_length.csv", "Typo/clean token ratio, paired per question (median / geomean / mean / p90); a lower bound because truncated blow-ups are excluded."),
        ("length_by_outcome.csv", "Mean tokens for correct vs wrong answers."),
    ]),
    ("Self-doubt", [
        ("self_doubt_per_config.csv", "Doubt-marker density per config: second_guess and uncertainty, per 1k words and per trace."),
        ("self_doubt_by_marker.csv", "Per-marker density, clean vs pooled-typo, with the typo-vs-clean discrimination."),
        ("self_doubt_by_outcome.csv", "Doubt density split by correct vs wrong answer, per category."),
    ]),
    ("Repair Behaviour (word-level)", [
        ("repair_wordlevel_per_config.csv", "How each corrupted word was handled (silent fix / flagged / misread / not used), per config."),
        ("repair_wordlevel_by_real.csv", "Word handling pooled by real-word ratio: the non-word vs real-word contrast."),
        ("repair_wordlevel_by_outcome.csv", "Misread rate per problem, split by final correctness."),
    ]),
    ("Real-word Effect", [
        ("realword_paired.csv", "Controlled paired comparison within a fixed rate (same question and positions): low vs high real-word ratio, with McNemar."),
        ("realword_pertypo_logit.csv", "Per-typo logistic regression: marginal harm of one real-word vs one non-word typo."),
        ("realword_silent_failure.csv", "Explicit-notice rate among wrong answers, by real ratio (test of the silent-failure hypothesis)."),
    ]),
    ("Lexical Grid (overview)", [
        ("lexical_grid.csv", "All four marker families (second_guess, uncertainty, typo_noticing, repair_words) across configs."),
    ]),
]

TEX_SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
               "^": r"\textasciicircum{}"}


def esc(s):
    return "".join(TEX_SPECIAL.get(c, c) for c in str(s))


def csv_to_latex(path, caption):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return f"% (empty: {os.path.basename(path)})\n"
    header, body = rows[0], rows[1:]
    ncol = len(header)
    colspec = "l" + "r" * (ncol - 1)
    out = [r"\begin{table}[h!]", r"\centering", rf"\caption{{{esc(caption)}}}",
           r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{colspec}}}",
           r"\toprule",
           " & ".join(rf"\textbf{{{esc(h)}}}" for h in header) + r" \\",
           r"\midrule"]
    for row in body:
        row = (row + [""] * ncol)[:ncol]
        out.append(" & ".join(esc(c) for c in row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(out)


def build_report():
    doc = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{caption}",
        r"\captionsetup{font=small,labelfont=bf}",
        r"\title{The Silent Tax: Typo-Robustness Analysis (GSM8K)\\"
        r"\large DeepSeek-R1-Distill-Qwen-7B}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\noindent All tables are \emph{answered-only}: traces that hit the "
        r"4096-token cap before writing a final answer are excluded from the "
        r"metrics (truncation rate is reported separately). Configs are named "
        r"\texttt{typo\{rate\}\_real\{ratio\}}: \emph{rate} = \% of words "
        r"corrupted, \emph{real} = \% of typos that form real words. Baseline is "
        r"the clean input.",
        "",
    ]
    for title, items in SECTIONS:
        doc.append(rf"\section*{{{esc(title)}}}")
        for fname, caption in items:
            path = os.path.join(TABLES, fname)
            if os.path.exists(path):
                doc.append(csv_to_latex(path, caption))
            else:
                doc.append(f"% missing: {fname}\n")
        doc.append(r"\clearpage")
    doc.append(r"\end{document}")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(doc))
    print(f"wrote {REPORT}")


def run_modules():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for m in MODULES:
        print(f"running {m}.py ...", end=" ", flush=True)
        r = subprocess.run([sys.executable, os.path.join(_HERE, m + ".py")],
                           cwd=_HERE, env=env, capture_output=True, text=True)
        print("OK" if r.returncode == 0 else f"FAIL\n{r.stderr[-400:]}")


def main():
    if "--skip-run" not in sys.argv:
        run_modules()
    build_report()
    print("\nCopy report.tex into Overleaf (or run: pdflatex report.tex).")


if __name__ == "__main__":
    main()
