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
from common import tables_dir, reports_dir  # noqa: E402
TABLES = tables_dir(DATASET)
REPORT = os.path.join(reports_dir(DATASET), f"report_{DATASET}.tex")

# active modules to run (imported-only files are excluded, so is the retired one).
# llm_judge.py is NOT here on purpose: it needs HF_TOKEN and spends API credits.
# Run it yourself (python llm_judge.py --limit 150); its tables are picked up below
# if they exist, and silently skipped if they don't.
MODULES = ["analysis/accuracy_flips", "analysis/reasoning_length",
           "analysis/self_doubt", "analysis/repair_wordlevel",
           "analysis/real_word_effect", "analysis/lexical_grid",
           "fixes/spellcheck_recovery"]

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

JUDGE_NOTE = r"""\noindent\textbf{LLM judge.} The proposal specifies a second,
non-lexical measure for both dimensions: a separate model with a fixed, strict prompt.
The judge is a different model from the one under test, called at temperature 0 on a
seeded random subsample of answered traces per config. It is shown the corrupted-word
list (\emph{original} $\to$ \emph{as shown}), the question, and the reasoning trace
(head+tail window). It returns two scalar ratings and evidence: \emph{self\_doubt\_score}
from 0--10, where 0 means no second-guessing and 10 means pervasive uncertainty or
looping; and \emph{repair\_understanding\_score} from 0--5, where 0 means the question's
meaning is fully understood and 5 means the model is fundamentally lost because of the
corruption. The judge also returns a one-line justification and verbatim evidence quotes;
quotes are checked against the reasoning trace. The judge is a subsample measure, while
the lexical tables remain the full-data result.
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
        ("accuracy_per_config.csv", "Accuracy per config. n = questions; n_answered = questions that produced a final answer (not truncated at the token cap). acc_answered = correct / n_answered; acc_strict = correct / n (truncated counted as wrong); ci_lo/ci_hi = 95% bootstrap CI on acc_strict."),
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
        ("__note__", JUDGE_NOTE),
        ("judge_scalar_per_config.csv", "New scalar LLM-judge analysis per config. mean/median self_doubt_score use the 0-10 scale; mean/median repair_understanding_score use the 0-5 scale; threshold columns show the share of traces with substantial doubt (>=5) or substantial loss of meaning (>=3)."),
        ("judge_scalar_by_outcome.csv", "New scalar LLM-judge scores split by final answer outcome. The correct/wrong columns are means over traces whose final answer was correct or wrong."),
        ("judge_scalar_by_real.csv", "New scalar LLM-judge scores pooled by real-word typo ratio, showing whether higher real-word corruption is associated with more doubt or loss of understanding."),
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
    ("Spellcheck Recovery", [
        ("spellcheck_recovery_per_config.csv", "External spellcheck recovery by config. restored_exact_pct = percentage of typo words that were corrected back to exactly their original clean word."),
        ("spellcheck_recovery_overall.csv", "Overall external spellcheck recovery for this dataset (pooled across spellcheck result files)."),
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
    TABLES = tables_dir(DATASET)
    REPORT = os.path.join(reports_dir(DATASET), f"report_{DATASET}.tex")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)

    if not args.skip_run:
        run_modules()
    build_report()
    rep = os.path.basename(REPORT)
    print(f"\nOpen {rep} in Overleaf (or run: pdflatex {rep}).")


if __name__ == "__main__":
    main()
