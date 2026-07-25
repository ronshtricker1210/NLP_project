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
        ("accuracy_per_config.csv", "Accuracy per config: answered-only and strict (truncated=wrong), with bootstrap CI and the token-cap rate."),
        ("flips_vs_clean.csv", "Answered-only flips vs the clean baseline (right->wrong, wrong->right) with McNemar test."),
    ]),
    ("Reasoning Length", [
        ("reasoning_length_absolute.csv", "Absolute generated tokens per config (answered-only)."),
        ("length_by_outcome.csv", "Mean generated tokens for correct vs wrong answers. wrong_over_correct = tok_wrong / tok_correct (mean tokens of wrong answers divided by mean tokens of correct answers)."),
    ]),
    ("Self-doubt", [
        ("self_doubt_per_config.csv", "Doubt-marker density per config: second_guess and uncertainty, per 1k words and per trace."),
        ("self_doubt_by_marker.csv", "Per-marker density (markers per 1000 reasoning words): clean vs pooled-typo. discrimination = typo minus clean (larger = more typo-responsive). All bank markers, ranked by discrimination."),
        ("self_doubt_by_outcome.csv", "Doubt density (markers per 1000 reasoning words) split by final outcome, per category (2g = second_guess, un = uncertainty, tot = both). For a group, density = total markers in that group / total reasoning words in that group x 1000. *_correct = over correct-answer traces; *_wrong = over wrong-answer traces; *_wrong_over_correct = _wrong / _correct."),
    ]),
    ("Repair Behaviour (word-level)", [
        ("repair_wordlevel_per_config.csv", "How each corrupted word was handled, per config. Each corrupted word (recovered by diffing the clean vs typo question) is placed in one bucket by which form appears in the reasoning: silent_fix = only the original word (quiet recovery); flagged = both the original and the corrupted form (noticed and fixed); misread = only the corrupted form (read as the wrong word); not_used = neither form referenced. Columns *_n are counts; *_frac are the share of n_corrupt_words."),
        ("repair_wordlevel_by_real.csv", "Word handling pooled by real-word ratio: the non-word vs real-word contrast."),
        ("repair_wordlevel_by_outcome.csv", "Misread rate split by final correctness. Within each outcome group, misread rate = (corrupted words used only in the corrupted form) / (all corrupted words examined), pooled over the group. misread_correct = over problems answered correctly; misread_wrong = over problems answered wrong; wrong_over_correct = misread_wrong / misread_correct."),
    ]),
    ("Real-word Effect", [
        ("realword_paired.csv", "Controlled paired comparison of two TYPO variants against EACH OTHER (baseline is the low-real variant, NOT clean -- so this is not the vs-clean accuracy table). Within a fixed rate the same questions are corrupted in the same positions, only the typo KIND differs. Restricted to questions answered in both variants. acc_low_real / acc_high_real = accuracy on the lower / higher real-word variant; delta_acc = acc_high_real - acc_low_real; r2w = questions correct in the low-real variant but wrong in the high-real one; w2r = the reverse; mcnemar_p = McNemar test on that paired change. Negative delta with r2w > w2r means real-word typos are more harmful than the non-word typos they replaced."),
        ("realword_pertypo_logit.csv", "Logistic regression correct ~ num_real + num_nonword over all answered typo traces (num_real / num_nonword = count of real-word / non-word typos in the problem). coef_* = change in log-odds of a correct answer per one added typo of that kind (more negative = more harmful); odds_* = exp(coef); intercept is the model constant. n = number of traces fit."),
        ("realword_silent_failure.csv", "Among WRONG answers only, does the model flag the corruption, by real ratio? explicit_notice_frac = fraction of wrong-answer traces whose reasoning uses strong typo-flagging language (typo / misspelled / doesn't make sense / ...); silent_frac = 1 - explicit_notice_frac. The proposal predicts real-word failures are quieter (explicit_notice_frac should fall as real rises)."),
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
    # Consistent base font (\small) for every table; adjustbox only SHRINKS a table
    # if it is wider than the text block, so narrow tables are never blown up.
    out = [r"\begin{table}[H]", r"\centering", rf"\caption{{{esc(caption)}}}",
           r"\small", r"\begin{adjustbox}{max width=\textwidth}",
           rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
           " & ".join(rf"\textbf{{{esc(h)}}}" for h in header) + r" \\",
           r"\midrule"]
    for row in body:
        row = (row + [""] * ncol)[:ncol]
        out.append(" & ".join(esc(c) for c in row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""]
    return "\n".join(out)


def build_report():
    doc = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{adjustbox}",
        r"\usepackage{float}",
        r"\usepackage{caption}",
        r"\captionsetup{font=small,labelfont=bf,skip=4pt}",
        r"\title{The Silent Tax: Typo-Robustness Analysis (GSM8K)\\"
        r"\large DeepSeek-R1-Distill-Qwen-7B}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\noindent All tables are \emph{answered-only}: traces that hit the "
        r"4096-token cap before writing a final answer are excluded from the "
        r"metrics (the cap rate is shown as \texttt{capped\_frac} in the accuracy "
        r"table). Configs are named "
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
