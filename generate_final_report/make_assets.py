"""
Build every figure and generated LaTeX table for the final report.

Reads the derived CSVs under full_analysis/analysis_tables and writes into
./figures and ./tables. Note tables/tab_datastats.tex is NOT generated here:
it is maintained in paper/tables/ and copied in (see README).

Written for this report. Reads only the analysis_tables CSVs on the
ido/final-reorg branch; writes into ./figures and ./tables next to this file.

Run source per MANIFEST.md:
    GSM8K     -> gsm8k/results_20000   (20,000-token cap, the main arm)
    MATH-500  -> math500/results       (17,000)
    ARC       -> arc/results           (17,000)
    warn      -> gsm8k/fix_warn
    rewrite   -> gsm8k/fix_rewrite
    judge     -> each dataset's main run (GSM8K judge re-run on 20k)
"""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
# Derived CSVs live in the repo; override with NLP_TABLES if they are elsewhere.
TABLES_IN = Path(os.environ.get(
    "NLP_TABLES", HERE.parent / "full_analysis" / "analysis_tables"))
# ASSET_FMT=png renders previews instead of the print-ready PDFs
FMT = os.environ.get("ASSET_FMT", "pdf")
FIGDIR = HERE / ("figures" if FMT == "pdf" else "_preview")
TABDIR = HERE / "tables"
FIGDIR.mkdir(exist_ok=True)
TABDIR.mkdir(exist_ok=True)

# --- palette -------------------------------------------------------------
# Reference categorical palette, slots 1-4 in fixed order. Slots 1-3 are the
# all-pairs-validated set; slot 4 joins them only in the stacked chart, which
# uses the adjacent pairlist.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

DATASETS = [("gsm8k", "GSM8K", BLUE),
            ("math500", "MATH-500", ORANGE),
            ("arc", "ARC-Challenge", AQUA)]

RUN = {"gsm8k": "results_20000", "math500": "results", "arc": "results"}
RATES, RHOS = [25, 50, 75], [10, 40, 70]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.6,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(ds, name, run=None):
    return pd.read_csv(TABLES_IN / ds / (run or RUN[ds]) / f"{name}.csv")


def pct(x, nd=1):
    """Percentage rounded half AWAY FROM ZERO, which is what the prose does.
    Python's default is banker's rounding, so 73.25 -> 73.2 rather than 73.3."""
    return f"{Decimal(str(x * 100)).quantize(Decimal('1.' + '0' * nd), ROUND_HALF_UP)}"


def style(ax, ygrid=True):
    """Recessive axes: no top/right spine, faint horizontal grid only."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ygrid:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.tick_params(length=2.5, width=0.6)


def by_rate(df, col):
    """Mean / min / max of `col` across the three rho values, per typo rate."""
    out = {}
    for r in RATES:
        v = [df.loc[df.config == f"typo{r}_real{p}", col].iloc[0] for p in RHOS]
        out[r] = (sum(v) / 3, min(v), max(v))
    return out


# =====================================================================
# Figure 1 - the accuracy tax, three panels (supports 5.1)
# =====================================================================
def fig_accuracy():
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.82))
    a, b, c = axes

    for ds, label, color in DATASETS:
        acc = load(ds, "accuracy_per_config")
        length = load(ds, "reasoning_length_absolute")
        clean = acc.loc[acc.config == "clean", "acc_answered"].iloc[0] * 100
        clean_tok = length.loc[length.config == "clean", "mean_tok"].iloc[0]

        m = by_rate(acc, "acc_answered")
        xs = [0] + RATES
        ys = [clean] + [m[r][0] * 100 for r in RATES]
        lo = [clean] + [m[r][1] * 100 for r in RATES]
        hi = [clean] + [m[r][2] * 100 for r in RATES]

        a.plot(xs, ys, color=color, marker="o", markersize=4, label=label,
               markeredgecolor="white", markeredgewidth=0.7)
        a.fill_between(xs, lo, hi, color=color, alpha=0.13, linewidth=0)

        b.plot(xs, [y - clean for y in ys], color=color, marker="o",
               markersize=4, label=label, markeredgecolor="white",
               markeredgewidth=0.7)

        t = by_rate(length, "mean_tok")
        b_tok = [1.0] + [t[r][0] / clean_tok for r in RATES]
        c.plot(xs, b_tok, color=color, marker="o", markersize=4, label=label,
               markeredgecolor="white", markeredgewidth=0.7)

    a.set_ylabel("accuracy, answered-only (%)")
    a.set_title("(a) accuracy", loc="left", color=INK2)
    b.axhline(0, color=GRID, linewidth=0.8)
    b.set_ylabel("change vs. clean (pp)")
    b.set_title("(b) accuracy tax", loc="left", color=INK2)
    c.axhline(1.0, color=GRID, linewidth=0.8)
    c.set_ylabel("reasoning tokens / clean")
    c.set_title("(c) reasoning length", loc="left", color=INK2)

    for ax in axes:
        ax.set_xlabel("corrupted words (%)")
        ax.set_xticks([0] + RATES)
        style(ax)
    a.legend(frameon=False, loc="lower left", handlelength=1.4)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(FIGDIR / f"fig_accuracy.{FMT}")
    plt.close(fig)


# =====================================================================
# Figure 2 - the real-word effect, two panels (supports 5.2)
# =====================================================================
def fig_realword():
    """Two stacked panels at COLUMN width, so the float can sit next to 5.2."""
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.30, 3.75))
    short = {"GSM8K": "GSM8K", "MATH-500": "MATH-500", "ARC-Challenge": "ARC"}

    # (a) per-typo loss in the odds of a correct answer
    xs = range(len(DATASETS))
    real, nonword = [], []
    for ds, _, _ in DATASETS:
        lg = load(ds, "realword_pertypo_logit").iloc[0]
        real.append((1 - lg.odds_real) * 100)
        nonword.append((1 - lg.odds_nonword) * 100)
    w = 0.34
    a.bar([x - w / 2 - 0.01 for x in xs], real, w, color=BLUE, label="real-word")
    a.bar([x + w / 2 + 0.01 for x in xs], nonword, w, color=ORANGE, label="non-word")
    for x, (rv, nv) in enumerate(zip(real, nonword)):
        a.text(x - w / 2 - 0.01, rv + 0.15, f"{rv:.1f}", ha="center",
               fontsize=6.5, color=INK2)
        a.text(x + w / 2 + 0.01, nv + 0.15, f"{nv:.1f}", ha="center",
               fontsize=6.5, color=INK2)
    a.set_xticks(list(xs))
    a.set_xticklabels([short[d[1]] for d in DATASETS])
    a.set_ylabel("odds lost per typo (%)", fontsize=7.5)
    a.set_title("(a) per-typo damage (logistic fit)", loc="left",
                color=INK2, fontsize=7.5)
    a.legend(frameon=False, handlelength=1.1, fontsize=6.8, ncol=2,
             loc="upper center", bbox_to_anchor=(0.5, 1.02))
    a.set_ylim(top=max(real + nonword) * 1.32)
    style(a)

    # (b) controlled paired test: same questions, same typo count, rho 10 -> 70
    w2 = 0.26
    for i, (ds, label, color) in enumerate(DATASETS):
        rp = load(ds, "realword_paired")
        rp = rp[rp["compare"] == "real10_to_real70"].set_index("rate")
        vals = [rp.loc[r, "delta_acc"] * 100 for r in RATES]
        ps = [rp.loc[r, "mcnemar_p"] for r in RATES]
        pos = [j + (i - 1) * (w2 + 0.02) for j in range(len(RATES))]
        b.bar(pos, vals, w2, color=color, label=short[label])
        for x, v, p in zip(pos, vals, ps):
            if p < 0.05:
                b.text(x, v - 0.9, "*", ha="center", fontsize=8, color=INK)
    b.axhline(0, color=GRID, linewidth=0.8)
    b.set_xticks(range(len(RATES)))
    b.set_xticklabels([f"{r}%" for r in RATES])
    b.set_xlabel("corrupted words", fontsize=7.5)
    b.set_ylabel(r"$\Delta$ accuracy (pp)", fontsize=7.5)
    b.set_title(r"(b) same words, $\rho$: 10% $\to$ 70%", loc="left",
                color=INK2, fontsize=7.5)
    b.legend(frameon=False, handlelength=1.1, fontsize=6.8, ncol=3,
             loc="lower center", bbox_to_anchor=(0.5, -0.02))
    b.set_ylim(bottom=min(b.get_ylim()[0], -17))
    style(b)

    for ax in (a, b):
        ax.tick_params(labelsize=7)
    fig.tight_layout(h_pad=1.1)
    fig.savefig(FIGDIR / f"fig_realword.{FMT}")
    plt.close(fig)


# =====================================================================
# Figure 3 - word-level repair behaviour (supports 5.3)
# =====================================================================
def fig_repair():
    """Two rows: handling against the typo RATE (pooled over rho) and against
    the real-word ratio (pooled over rates). Section 5.3 claims a trend on both
    axes, so both are drawn. Shares are recomputed from the raw word counts, not
    averaged from per-config fractions."""
    cats = [("silent_fix_n", "silent fix", BLUE),
            ("flagged_n", "flagged", ORANGE),
            ("misread_n", "misread", AQUA),
            ("not_used_n", "not used", YELLOW)]
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 2.92), sharey=True)

    def shares(df, configs):
        sub = df.loc[configs]
        total = sub["n_corrupt_words"].sum()
        return [sub[col].sum() / total * 100 for col, _, _ in cats]

    for col_i, (ds, label, _) in enumerate(DATASETS):
        per = load(ds, "repair_wordlevel_per_config").set_index("config")
        groups = [
            ("typo rate", [f"{r}\\%" for r in RATES],
             [[f"typo{r}_real{p}" for p in RHOS] for r in RATES]),
            ("real-word typos", [f"{p}\\%" for p in RHOS],
             [[f"typo{r}_real{p}" for r in RATES] for p in RHOS]),
        ]
        for row_i, (xlabel, ticks, grouping) in enumerate(groups):
            ax = axes[row_i][col_i]
            vals = [shares(per, g) for g in grouping]
            bottom = [0.0] * len(grouping)
            for k, (_, name, color) in enumerate(cats):
                heights = [v[k] for v in vals]
                ax.bar(range(len(grouping)), heights, 0.6, bottom=bottom,
                       color=color, label=name, edgecolor="white", linewidth=1.0)
                bottom = [b + h for b, h in zip(bottom, heights)]
            ax.set_xticks(range(len(grouping)))
            ax.set_xticklabels([t.replace("\\", "") for t in ticks])
            ax.set_xlabel(xlabel)
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_locator(MultipleLocator(25))
            style(ax, ygrid=False)
            if row_i == 0:
                ax.set_title(label, loc="center", color=INK2, pad=4)
    # One short y-label per row, and the row tag as a left-aligned title, so the
    # two rows' labels cannot collide in the left margin.
    for row_i, tag in enumerate(("(a) by typo rate", "(b) by real-word ratio")):
        axes[row_i][0].set_ylabel("share of words (%)", fontsize=7.5)
        axes[row_i][0].set_title(tag, loc="left", fontsize=7.5, color=INK2, pad=4)

    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, frameon=False, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.055), handlelength=1.1, columnspacing=1.6)
    fig.tight_layout(w_pad=1.0, h_pad=0.9, rect=(0, 0, 1, 0.955))
    fig.savefig(FIGDIR / f"fig_repair.{FMT}")
    plt.close(fig)


# =====================================================================
# Figure 4 - self-doubt: hedging vs verification (supports 5.4)
# =====================================================================
def fig_doubt():
    """Marker density across the grid. The wrong-vs-correct ratio that used to
    sit in a second panel is in Table A4 instead."""
    fig, a = plt.subplots(1, 1, figsize=(7.1, 2.05))
    order = ["clean"] + [f"typo{r}_real{p}" for r in RATES for p in RHOS]
    labels = ["clean"] + [f"{r}/{p}" for r in RATES for p in RHOS]

    for ds, label, color in DATASETS:
        sd = load(ds, "self_doubt_per_config").set_index("config")
        a.plot(range(len(order)), [sd.loc[c, "uncertainty_per_1k"] for c in order],
               color=color, marker="o", markersize=3.2, label=label,
               markeredgecolor="white", markeredgewidth=0.6)
        a.plot(range(len(order)), [sd.loc[c, "second_guess_per_1k"] for c in order],
               color=color, linestyle=(0, (3, 2)), linewidth=1.2, alpha=0.85)

    a.set_ylabel("markers per 1k reasoning words")
    a.set_title("uncertainty (solid) vs. second-guessing (dashed)",
                loc="left", color=INK2)
    a.set_xticks(range(len(order)))
    a.set_xticklabels(labels, rotation=45, ha="right")
    a.set_xlabel(r"configuration ($r$/$\rho$)")
    style(a)

    handles, lbls = a.get_legend_handles_labels()
    fig.legend(handles, lbls, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.045), fontsize=6.8,
               handlelength=1.3, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(FIGDIR / f"fig_doubt.{FMT}")
    plt.close(fig)


# =====================================================================
# Figure 5 - prompt-level mitigations on GSM8K (supports 5.5)
# =====================================================================
def fig_fixes():
    order = ["clean"] + [f"typo{r}_real{p}" for r in RATES for p in RHOS]
    labels = ["clean"] + [f"{r}/{p}" for r in RATES for p in RHOS]
    base = load("gsm8k", "accuracy_per_config").set_index("config")
    warn = load("gsm8k", "accuracy_per_config", run="fix_warn").set_index("config")
    rew = load("gsm8k", "accuracy_per_config", run="fix_rewrite").set_index("config")

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 1.88))
    for df, name, color, ls in ((base, "no fix", BLUE, "-"),
                                (warn, "warn", ORANGE, "-"),
                                (rew, "rewrite", AQUA, "-")):
        a.plot(range(len(order)), [df.loc[c, "acc_answered"] * 100 for c in order],
               color=color, marker="o", markersize=3.2, linestyle=ls, label=name,
               markeredgecolor="white", markeredgewidth=0.6)
    a.set_ylabel("accuracy, answered-only (%)")
    a.set_title("(a) GSM8K accuracy under each fix", loc="left", color=INK2)
    a.legend(frameon=False, handlelength=1.4)

    for df, name, color in ((warn, "warn", ORANGE), (rew, "rewrite", AQUA)):
        b.plot(range(len(order)),
               [(df.loc[c, "acc_answered"] - base.loc[c, "acc_answered"]) * 100
                for c in order],
               color=color, marker="o", markersize=3.2, label=name,
               markeredgecolor="white", markeredgewidth=0.6)
    b.axhline(0, color=GRID, linewidth=0.8)
    b.set_ylabel(r"$\Delta$ accuracy vs. no fix (pp)")
    b.set_title("(b) neither fix recovers the loss", loc="left", color=INK2)
    b.legend(frameon=False, handlelength=1.4)

    for ax in (a, b):
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel(r"configuration ($r$/$\rho$)")
        style(ax)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(FIGDIR / f"fig_fixes.{FMT}")
    plt.close(fig)


# =====================================================================
# LaTeX tables
# =====================================================================
def w(name, body):
    (TABDIR / name).write_text(body, encoding="utf-8")


def tab_accuracy_flips():
    """Full grid: accuracy, answered count, tokens, and paired flips."""
    rows = []
    order = ["clean"] + [f"typo{r}_real{p}" for r in RATES for p in RHOS]
    per = {ds: load(ds, "accuracy_per_config").set_index("config") for ds, _, _ in DATASETS}
    tok = {ds: load(ds, "reasoning_length_absolute").set_index("config") for ds, _, _ in DATASETS}
    flip = {ds: load(ds, "flips_vs_clean").set_index("config") for ds, _, _ in DATASETS}
    for c in order:
        cells = []
        for ds, _, _ in DATASETS:
            acc = per[ds].loc[c, "acc_answered"] * 100
            na = int(per[ds].loc[c, "n_answered"])
            tk = int(round(tok[ds].loc[c, "mean_tok"]))
            if c == "clean":
                cells += [f"{acc:.1f}", f"{na}", f"{tk}", "--", "--", "--"]
            else:
                f = flip[ds].loc[c]
                p = f.mcnemar_p
                ps = "$<$1e-6" if p < 1e-6 else (f"{p:.1e}" if p < 0.001 else f"{p:.3f}")
                cells += [f"{acc:.1f}", f"{na}", f"{tk}",
                          f"{int(f.r2w)}", f"{int(f.w2r)}", ps]
        rows.append(c.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    head = (r"\textbf{config} & " +
            " & ".join([r"acc & $n_a$ & tok & r$\to$w & w$\to$r & $p$"] * 3) + r" \\")
    w("tab_grid.tex", "\n".join([
        r"\begin{tabular}{l rrrrrr rrrrrr rrrrrr}", r"\toprule",
        r"& \multicolumn{6}{c}{\textbf{GSM8K}} & \multicolumn{6}{c}{\textbf{MATH-500}}"
        r" & \multicolumn{6}{c}{\textbf{ARC-Challenge}} \\",
        r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}\cmidrule(lr){14-19}",
        head, r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))


def tab_length():
    """Reasoning length, and length split by final outcome (supports 5.1)."""
    order = ["clean"] + [f"typo{r}_real{p}" for r in RATES for p in RHOS]
    abs_ = {ds: load(ds, "reasoning_length_absolute").set_index("config") for ds, _, _ in DATASETS}
    out = {ds: load(ds, "length_by_outcome").set_index("config") for ds, _, _ in DATASETS}
    rows = []
    for c in order:
        cells = []
        for ds, _, _ in DATASETS:
            cells += [f"{int(round(abs_[ds].loc[c, 'mean_tok']))}",
                      f"{int(round(abs_[ds].loc[c, 'median_tok']))}",
                      f"{out[ds].loc[c, 'wrong_over_correct']:.2f}"]
        rows.append(c.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    w("tab_length.tex", "\n".join([
        r"\begin{tabular}{l rrr rrr rrr}", r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{GSM8K}} & \multicolumn{3}{c}{\textbf{MATH-500}}"
        r" & \multicolumn{3}{c}{\textbf{ARC-Challenge}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"\textbf{config} & " + " & ".join([r"mean & med. & w/c"] * 3) + r" \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))


def tab_doubt():
    """Hedging / second-guessing density and the wrong-over-correct ratio (5.4)."""
    order = ["clean"] + [f"typo{r}_real{p}" for r in RATES for p in RHOS]
    per = {ds: load(ds, "self_doubt_per_config").set_index("config") for ds, _, _ in DATASETS}
    out = {ds: load(ds, "self_doubt_by_outcome").set_index("config") for ds, _, _ in DATASETS}
    rows = []
    for c in order:
        cells = []
        for ds, _, _ in DATASETS:
            cells += [f"{per[ds].loc[c, 'uncertainty_per_1k']:.1f}",
                      f"{per[ds].loc[c, 'second_guess_per_1k']:.1f}",
                      f"{out[ds].loc[c, 'un_wrong_over_correct']:.2f}"]
        rows.append(c.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    w("tab_doubt.tex", "\n".join([
        r"\begin{tabular}{l rrr rrr rrr}", r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{GSM8K}} & \multicolumn{3}{c}{\textbf{MATH-500}}"
        r" & \multicolumn{3}{c}{\textbf{ARC-Challenge}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"\textbf{config} & " + " & ".join([r"hedge & 2nd-g. & w/c"] * 3) + r" \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))


def tab_realword_paired():
    """The controlled paired contrast at every rate and every rho pair."""
    paired = {ds: load(ds, "realword_paired").set_index(["rate", "compare"])
              for ds, _, _ in DATASETS}

    def pfmt(p):
        if p < 1e-6:
            return r"$<$1e-6"
        return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"

    rows = []
    for rate in RATES:
        for lo, hi in ((10, 70), (10, 40), (40, 70)):
            key = (rate, f"real{lo}_to_real{hi}")
            cells = []
            for ds, _, _ in DATASETS:
                r = paired[ds].loc[key]
                star = r"$^{*}$" if r.mcnemar_p < 0.05 else ""
                cells += [f"{r.delta_acc*100:+.1f}{star}",
                          f"{int(r.r2w)}/{int(r.w2r)}", pfmt(r.mcnemar_p)]
            rows.append(f"{rate} & {lo}$\\to${hi} & " + " & ".join(cells) + r" \\")

    w("tab_realword_paired.tex", "\n".join([
        r"\begin{tabular}{ll rrr rrr rrr}", r"\toprule",
        r"& & \multicolumn{3}{c}{\textbf{GSM8K}} & \multicolumn{3}{c}{\textbf{MATH-500}}"
        r" & \multicolumn{3}{c}{\textbf{ARC-Challenge}} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"\textbf{$r$} & \textbf{$\rho$} & " +
        " & ".join([r"$\Delta$acc & r/w & $p$"] * 3) + r" \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))


def tab_realword_logit():
    """The per-typo logistic fit, one column per dataset."""
    lg = {ds: load(ds, "realword_pertypo_logit").iloc[0] for ds, _, _ in DATASETS}

    def row(label, fn):
        return label + " & " + " & ".join(fn(lg[ds]) for ds, _, _ in DATASETS) + r" \\"

    rows = [
        row(r"traces in the fit, $n$", lambda d: f"{int(d.n)}"),
        row(r"intercept", lambda d: f"{d.intercept:+.3f}"),
        row(r"coefficient, real-word", lambda d: f"{d.coef_num_real:+.4f}"),
        row(r"coefficient, non-word", lambda d: f"{d.coef_num_nonword:+.4f}"),
        row(r"odds multiplier, real-word", lambda d: f"{d.odds_real:.3f}"),
        row(r"odds multiplier, non-word", lambda d: f"{d.odds_nonword:.3f}"),
        row(r"odds lost per real-word typo",
            lambda d: f"{(1-d.odds_real)*100:.1f}\\%"),
        row(r"odds lost per non-word typo",
            lambda d: f"{(1-d.odds_nonword)*100:.1f}\\%"),
        r"\midrule",
        row(r"\textbf{relative loss, real / non-word}",
            lambda d: r"\textbf{%.1f$\times$}" % ((1 - d.odds_real) /
                                                  (1 - d.odds_nonword))),
    ]
    w("tab_realword_logit.tex", "\n".join([
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"& \textbf{GSM8K} & \textbf{MATH-500} & \textbf{ARC-C} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))


def tab_support():
    """Token length and hedging density in one table: the supporting numbers
    behind the 5.1 length claims and the 5.4 uncertainty claims."""
    order = ["clean"] + [f"typo{r}_real{p}" for r in RATES for p in RHOS]
    abs_ = {ds: load(ds, "reasoning_length_absolute").set_index("config") for ds, _, _ in DATASETS}
    lout = {ds: load(ds, "length_by_outcome").set_index("config") for ds, _, _ in DATASETS}
    sd = {ds: load(ds, "self_doubt_per_config").set_index("config") for ds, _, _ in DATASETS}
    sdo = {ds: load(ds, "self_doubt_by_outcome").set_index("config") for ds, _, _ in DATASETS}
    rows = []
    for c in order:
        cells = []
        for ds, _, _ in DATASETS:
            cells += [f"{int(round(abs_[ds].loc[c, 'mean_tok']))}",
                      f"{int(round(lout[ds].loc[c, 'tok_correct']))}",
                      f"{int(round(lout[ds].loc[c, 'tok_wrong']))}",
                      f"{sd[ds].loc[c, 'uncertainty_per_1k']:.1f}",
                      f"{sd[ds].loc[c, 'second_guess_per_1k']:.1f}",
                      f"{sdo[ds].loc[c, 'un_wrong_over_correct']:.2f}"]
        rows.append(c.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    w("tab_support.tex", "\n".join([
        r"\begin{tabular}{l rrrrrr rrrrrr rrrrrr}", r"\toprule",
        r"& \multicolumn{6}{c}{\textbf{GSM8K}} & \multicolumn{6}{c}{\textbf{MATH-500}}"
        r" & \multicolumn{6}{c}{\textbf{ARC-Challenge}} \\",
        r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}\cmidrule(lr){14-19}",
        r"\textbf{config} & " +
        " & ".join([r"tok & tok$_{c}$ & tok$_{w}$ & uncert. & 2nd-g. & u.w/c"] * 3) +
        r" \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))


def tab_judge():
    """LLM-judge scalar scores by rate, rho and outcome (5.3 and 5.4)."""
    rows = []

    def cells(fn):
        return " & ".join(fn(ds) for ds, _, _ in DATASETS)

    per = {ds: load(ds, "judge_scalar_per_config").set_index("config")
           for ds, _, _ in DATASETS}
    rea = {ds: load(ds, "judge_scalar_by_real").set_index("real_ratio")
           for ds, _, _ in DATASETS}
    outc = {ds: load(ds, "judge_scalar_by_outcome").set_index("config")
            for ds, _, _ in DATASETS}

    def mean_at_rate(ds, r, col):
        return sum(per[ds].loc[f"typo{r}_real{p}", col] for p in RHOS) / 3

    rows.append(r"\multicolumn{7}{l}{\emph{clean}} \\")
    rows.append("\\quad --- & " + cells(
        lambda ds: f"{per[ds].loc['clean','mean_repair_understanding']:.2f} & "
                   f"{per[ds].loc['clean','mean_self_doubt']:.2f}") + r" \\")
    rows.append(r"\midrule \multicolumn{7}{l}{\emph{by typo rate }$r$} \\")
    for r in RATES:
        rows.append(f"\\quad {r}\\% & " + cells(
            lambda ds, r=r: f"{mean_at_rate(ds, r, 'mean_repair_understanding'):.2f} & "
                            f"{mean_at_rate(ds, r, 'mean_self_doubt'):.2f}") + r" \\")
    rows.append(r"\midrule \multicolumn{7}{l}{\emph{by real-word ratio }$\rho$} \\")
    for p in RHOS:
        rows.append(f"\\quad {p}\\% & " + cells(
            lambda ds, p=p: f"{rea[ds].loc[p,'mean_repair_understanding']:.2f} & "
                            f"{rea[ds].loc[p,'mean_self_doubt']:.2f}") + r" \\")
    rows.append(r"\midrule \multicolumn{7}{l}{\emph{by final answer}} \\")
    for key, lab in (("correct", "correct"), ("wrong", "wrong")):
        rows.append(f"\\quad {lab} & " + cells(
            lambda ds, key=key: f"{outc[ds].loc['__all__', f'mean_repair_{key}']:.2f} & "
                                f"{outc[ds].loc['__all__', f'mean_doubt_{key}']:.2f}") + r" \\")
    w("tab_judge.tex", "\n".join([
        r"\begin{tabular}{l rr rr rr}", r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{GSM8K}} & \multicolumn{2}{c}{\textbf{MATH-500}}"
        r" & \multicolumn{2}{c}{\textbf{ARC-C}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"\textbf{condition} & " + " & ".join([r"repair & doubt"] * 3) + r" \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))


def tab_repair_words():
    """Word-level handling per configuration, plus misread split by final
    outcome. Backs the Section 5.3 claims that Figure 4 does not show: the
    per-RATE trend (Figure 4 pools rates and splits by rho) and the
    misread-vs-correctness link."""
    order = [f"typo{r}_real{p}" for r in RATES for p in RHOS]
    per = {ds: load(ds, "repair_wordlevel_per_config").set_index("config")
           for ds, _, _ in DATASETS}
    out = {ds: load(ds, "repair_wordlevel_by_outcome").set_index("config")
           for ds, _, _ in DATASETS}
    rows = []
    for i, c in enumerate(order):
        cells = []
        for ds, _, _ in DATASETS:
            r_, o_ = per[ds].loc[c], out[ds].loc[c]
            cells += [pct(r_.silent_fix_frac), pct(r_.flagged_frac),
                      pct(r_.misread_frac), pct(r_.not_used_frac),
                      pct(o_.misread_correct), pct(o_.misread_wrong)]
        rate, rho = c.replace("typo", "").split("_real")
        rows.append(f"{rate} & {rho} & " + " & ".join(cells) + r" \\")
    w("tab_repair.tex", "\n".join([
        r"\begin{tabular}{ll rrrrrr rrrrrr rrrrrr}", r"\toprule",
        r"& & \multicolumn{6}{c}{\textbf{GSM8K}} & \multicolumn{6}{c}{\textbf{MATH-500}}"
        r" & \multicolumn{6}{c}{\textbf{ARC-Challenge}} \\",
        r"\cmidrule(lr){3-8}\cmidrule(lr){9-14}\cmidrule(lr){15-20}",
        r"\textbf{$r$} & \textbf{$\rho$} & " +
        " & ".join([r"silent & flag & mis. & unused & mis$_{c}$ & mis$_{w}$"] * 3) +
        r" \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))


def tab_fixes():
    """Warn, rewrite and spell-check deltas against the unmodified prompt.
    All four arms are the 20,000-token GSM8K run, so they share a baseline."""
    order = [f"typo{r}_real{p}" for r in RATES for p in RHOS]
    base = load("gsm8k", "accuracy_per_config").set_index("config")
    arms = [("warn", load("gsm8k", "accuracy_per_config", run="fix_warn")),
            ("rewrite", load("gsm8k", "accuracy_per_config", run="fix_rewrite")),
            ("spell", load("gsm8k", "accuracy_per_config",
                           run="fix_spellcheck_20000"))]
    arms = [(n, d.set_index("config")) for n, d in arms]

    rows = []
    for c in order:
        r_, p_ = c.replace("typo", "").split("_real")
        cells = [f"{base.loc[c, 'acc_answered']*100:.1f}"]
        for _, d in arms:
            cells += [f"{d.loc[c, 'acc_answered']*100:.1f}",
                      f"{(d.loc[c, 'acc_answered']-base.loc[c, 'acc_answered'])*100:+.1f}"]
        rows.append(f"{r_}/{p_} & " + " & ".join(cells) + r" \\")

    means = [(d.loc[order, "acc_answered"] - base.loc[order, "acc_answered"]).mean() * 100
             for _, d in arms]
    rows.append(r"\midrule \textbf{mean} & & " +
                " & ".join(r"& \textbf{%+.1f}" % m for m in means) + r" \\")

    w("tab_fixes.tex", "\n".join([
        r"\begin{tabular}{lr rr rr rr}", r"\toprule",
        r"& \textbf{no fix} & \multicolumn{2}{c}{\textbf{warn}}"
        r" & \multicolumn{2}{c}{\textbf{rewrite}}"
        r" & \multicolumn{2}{c}{\textbf{spell check}} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"\textbf{$r$/$\rho$} & acc & acc & $\Delta$ & acc & $\Delta$"
        r" & acc & $\Delta$ \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))
    print("  mitigation means (pp): warn %+.1f  rewrite %+.1f  spell %+.1f" % tuple(means))


if __name__ == "__main__":
    print("figures:")
    for f in (fig_accuracy, fig_realword, fig_repair, fig_doubt, fig_fixes):
        f(); print("  ", f.__name__)
    print("tables:")
    for t in (tab_accuracy_flips, tab_support, tab_realword_paired, tab_realword_logit, tab_judge,
              tab_repair_words, tab_fixes):
        t(); print("  ", t.__name__)
    print("done ->", FIGDIR, "and", TABDIR)
