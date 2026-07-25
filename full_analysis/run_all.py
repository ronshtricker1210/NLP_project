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
import os, sys, csv, subprocess, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
# Set from --dataset in main(); these module-level defaults follow the env var.
DATASET = os.environ.get("NLP_DATASET", "gsm8k")
TABLES = os.path.join(_HERE, "tables", DATASET)
REPORT = os.path.join(_HERE, f"report_{DATASET}.tex")

# active modules to run (imported-only files are excluded, so is the retired one)
MODULES = ["accuracy_flips", "reasoning_length", "self_doubt",
           "repair_wordlevel", "real_word_effect", "lexical_grid"]

# Free-text explanatory notes inserted into the report as raw LaTeX (via "__note__").
SELF_DOUBT_NOTE = r"""\noindent The self-doubt markers are curated words grouped into two
\emph{banks}, each matched as a case-insensitive whole-word pattern in the reasoning
text. Density is matches per 1000 reasoning words.
\begin{itemize}\setlength{\itemsep}{1pt}
  \item \textbf{second\_guess} --- the proposal's canonical self-correction words plus
  direct reversals: \emph{wait, hmm, actually, reconsider, but wait, hold on, on second
  thought, let me recheck, double-check}.
  \item \textbf{uncertainty} --- hedging / ``not certain'' words: \emph{maybe, perhaps,
  possibly, might be, could be, not sure, I guess, presumably, or maybe / alternatively,
  unclear / confused}.
\end{itemize}
\noindent Example reasoning: ``\emph{Wait, maybe the total is 12\dots\ hmm, actually let
me recheck}'' contributes \emph{wait, hmm, actually} (second\_guess) and \emph{maybe}
(uncertainty). A separate \emph{typo-noticing} bank (\emph{typo, misspelled, doesn't make
sense, \dots}) is treated as repair behaviour, not self-doubt.
\medskip
"""

MCNEMAR_NOTE = r"""\medskip
\noindent\textbf{McNemar test (used in the flips table above).}
\begin{itemize}\setlength{\itemsep}{1pt}
  \item \textbf{What it is:} a paired significance test for before/after binary
  outcomes. It looks only at the questions that \emph{changed} (the discordant pairs:
  $b$ = right$\to$wrong, $c$ = wrong$\to$right) and ignores those that stayed the same.
  \item \textbf{Null hypothesis:} typos break correct answers as often as they fix
  wrong ones ($b = c$, i.e.\ no net effect).
  \item \textbf{Statistic:} $(|b - c| - 1)^2 / (b + c)$ (column mcnemar\_stat);
  mcnemar\_p is its chi-square $p$-value with 1 degree of freedom.
  \item \textbf{When we reject it:} when mcnemar\_p $< 0.05$ --- the
  right$\to$wrong vs wrong$\to$right imbalance is too large to be chance.
\end{itemize}
"""

REPAIR_CATEGORIES_NOTE = r"""\noindent A \emph{corrupted word} is a word that received a
typo --- it differs between the clean and typo question (e.g.\ \emph{sum} $\to$
\emph{sun}). We recover these by diffing the two questions, then place each into one
bucket by which form appears in the reasoning:
\begin{itemize}\setlength{\itemsep}{1pt}
  \item \textbf{silent\_fix} --- only the original word (\emph{sum}) appears: a quiet recovery.
  \item \textbf{flagged} --- both forms appear (\emph{sum} and \emph{sun}): noticed and fixed.
  \item \textbf{misread} --- only the corrupted form (\emph{sun}) appears: read as the wrong word.
  \item \textbf{not\_used} --- neither form appears in the reasoning.
\end{itemize}
\noindent Columns \texttt{*\_n} are counts; \texttt{*\_frac} are the share of
\texttt{n\_corrupt\_words}.
\medskip
"""

REALWORD_NOTE = r"""\noindent This section isolates the effect of typo \emph{kind}
(real-word vs non-word), holding the question, corrupted positions, and rate fixed ---
something the vs-clean accuracy section cannot do. Three views:
\begin{itemize}\setlength{\itemsep}{1pt}
  \item a \textbf{controlled paired} comparison of low-real vs high-real variants on
  the \emph{same} questions, with a McNemar test (paired table);
  \item a \textbf{per-typo logistic regression} comparing the marginal harm of one
  real-word vs one non-word typo (logit table);
  \item whether wrong answers \textbf{flag the corruption} more or less as the
  real-word share rises (silent-failure table).
\end{itemize}
\noindent Read the signs, deltas and $p$-values in the tables below for this
dataset's result.
\medskip
"""

# report layout: (section title, [(csv file, caption), ...])
SECTIONS = [
    ("Accuracy and Flips", [
        ("accuracy_per_config.csv", "Accuracy per config: answered-only and strict (truncated=wrong), with 95% bootstrap CI on acc_strict."),
        ("flips_vs_clean.csv", "Answered-only flips vs clean, paired by question over n_both (answered in both conditions). r2w / w2r = questions correct on clean but wrong on this config / wrong on clean but correct on this config; r2w_ratio = r2w / n_both, w2r_ratio = w2r / n_both; net_flip_ratio = r2w_ratio - w2r_ratio = (r2w - w2r) / n_both, the net rate at which this config worsens answers relative to clean."),
        ("__note__", MCNEMAR_NOTE),
    ]),
    ("Reasoning Length", [
        ("reasoning_length_absolute.csv", "Absolute generated tokens per config (answered-only)."),
        ("length_by_outcome.csv", "Mean generated tokens for correct vs wrong answers. wrong_over_correct = tok_wrong / tok_correct (mean tokens of wrong answers divided by mean tokens of correct answers)."),
    ]),
    ("Self-doubt", [
        ("__note__", SELF_DOUBT_NOTE),
        ("self_doubt_per_config.csv", "Doubt-marker density per config: second_guess and uncertainty, per 1k words and per trace."),
        ("self_doubt_by_marker.csv", "Per-marker density (markers per 1000 reasoning words): clean vs pooled-typo. discrimination = typo minus clean (larger = more typo-responsive). All bank markers, ranked by discrimination."),
        ("self_doubt_by_outcome.csv", "Doubt density (markers per 1000 reasoning words) split by final outcome, per category (2g = second_guess, un = uncertainty, tot = both). For a group, density = total markers in that group / total reasoning words in that group x 1000. *_correct = over correct-answer traces; *_wrong = over wrong-answer traces; *_wrong_over_correct = _wrong / _correct."),
    ]),
    ("Repair Behaviour", [
        ("repair_wordlevel_per_config.csv", "How each corrupted word was handled, per config (categories defined below the table)."),
        ("__note__", REPAIR_CATEGORIES_NOTE),
        ("repair_wordlevel_by_real.csv", "Word handling pooled by real-word ratio: the non-word vs real-word contrast."),
        ("repair_wordlevel_by_outcome.csv", "Misread rate split by final correctness. A typo'd word is 'misread' when the reasoning uses only its corrupted form (e.g. sun instead of sum). Within each outcome group, misread rate = (typo'd words that were misread) / (all typo'd words examined), pooled over the group. misread_correct = over problems answered correctly; misread_wrong = over problems answered wrong; wrong_over_correct = misread_wrong / misread_correct."),
    ]),
    ("Real-word Effect", [
        ("__note__", REALWORD_NOTE),
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
        rf"\title{{The Silent Tax: Typo-Robustness Analysis ({esc(DATASET.upper())})\\"
        r"\large DeepSeek-R1-Distill-Qwen-7B}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\noindent All tables are \emph{answered-only}: traces that hit the token "
        r"cap before writing a final answer are excluded from the metrics. Configs "
        r"are named \texttt{typo\{rate\}\_real\{ratio\}}: \emph{rate} = \% of words "
        r"corrupted, \emph{real} = \% of typos that form real words. Baseline is "
        r"the clean input.",
        "",
    ]
    for title, items in SECTIONS:
        doc.append(rf"\section*{{{esc(title)}}}")
        for fname, caption in items:
            if fname == "__note__":
                doc.append(caption)          # raw LaTeX, inserted verbatim
                continue
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
    global DATASET, TABLES, REPORT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.environ.get("NLP_DATASET", "gsm8k"),
                    help="which dataset under data/<dataset>/ to analyse (default gsm8k)")
    ap.add_argument("--skip-run", action="store_true",
                    help="rebuild the report from existing tables/<dataset>/ CSVs only")
    args = ap.parse_args()

    DATASET = args.dataset
    os.environ["NLP_DATASET"] = DATASET      # every module reads this
    TABLES = os.path.join(_HERE, "tables", DATASET)
    REPORT = os.path.join(_HERE, f"report_{DATASET}.tex")

    if not args.skip_run:
        run_modules()
    build_report()
    rep = os.path.basename(REPORT)
    print(f"\nOpen {rep} in Overleaf (or run: pdflatex {rep}).")


if __name__ == "__main__":
    main()
