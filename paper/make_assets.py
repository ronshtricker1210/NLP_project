"""Build all figures (PDF) and the long appendix tables (LaTeX) for the paper.

Reads the analysis CSVs produced by ``full_analysis/run_all.py`` and writes
``paper/figures/*.pdf`` and ``paper/tables/*.tex``.

    python make_assets.py

The GSM8K numbers come from ``tables/gsm8k_20000`` (the 20k-token-cap run used
throughout the paper); MATH-500 and ARC-Challenge come from ``tables/math500``
and ``tables/arc``.
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES_SRC = os.path.join(HERE, "..", "full_analysis", "results_data")
FIG_DIR = os.path.join(HERE, "figures")
TAB_DIR = os.path.join(HERE, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# dataset key -> (directory under tables/, display name, colour)
DATASETS = [
    ("gsm8k", "gsm8k/results_20000", "GSM8K", "#1f77b4"),
    ("math500", "math500/results", "MATH-500", "#d62728"),
    ("arc", "arc/results", "ARC-Challenge", "#2ca02c"),
]
RATES = [25, 50, 75]
REALS = [10, 40, 70]
CONFIGS = [f"typo{r}_real{p}" for r in RATES for p in REALS]

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def load(dirname, name):
    """CSV -> list of dicts."""
    with open(os.path.join(TABLES_SRC, dirname, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def keyed(rows, key="config"):
    return {r[key]: r for r in rows}


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", os.path.relpath(path, HERE))


# ---------------------------------------------------------------------------
# Figure 1: the accuracy tax and the reasoning-length tax
# ---------------------------------------------------------------------------
def fig_accuracy():
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15))
    ax_acc, ax_delta, ax_len = axes

    for ds, d, label, colour in DATASETS:
        acc = keyed(load(d, "accuracy_per_config.csv"))
        ln = keyed(load(d, "reasoning_length_absolute.csv"))
        clean = float(acc["clean"]["acc_answered"])
        clean_tok = float(ln["clean"]["mean_tok"])

        xs = [0] + RATES
        means, lo, hi = [clean], [clean], [clean]
        tok = [1.0]
        for rate in RATES:
            vals = [float(acc[f"typo{rate}_real{p}"]["acc_answered"]) for p in REALS]
            means.append(float(np.mean(vals)))
            lo.append(min(vals))
            hi.append(max(vals))
            tok.append(float(np.mean(
                [float(ln[f"typo{rate}_real{p}"]["mean_tok"]) for p in REALS])) / clean_tok)

        ax_acc.plot(xs, means, "o-", color=colour, label=label, lw=1.4, ms=3.5)
        ax_acc.fill_between(xs, lo, hi, color=colour, alpha=0.15, lw=0)

        ax_delta.plot(xs, [100 * (m - clean) for m in means], "o-",
                      color=colour, label=label, lw=1.4, ms=3.5)
        ax_len.plot(xs, tok, "o-", color=colour, label=label, lw=1.4, ms=3.5)

    ax_acc.set_xlabel("corrupted words (%)")
    ax_acc.set_ylabel("accuracy (answered-only)")
    ax_acc.set_xticks([0] + RATES)
    ax_acc.set_title("(a) absolute accuracy")
    ax_acc.legend(loc="lower left", frameon=False)

    ax_delta.axhline(0, color="0.4", lw=0.7)
    ax_delta.set_xlabel("corrupted words (%)")
    ax_delta.set_ylabel("$\\Delta$ accuracy vs. clean (pp)")
    ax_delta.set_xticks([0] + RATES)
    ax_delta.set_title("(b) accuracy tax")

    ax_len.axhline(1.0, color="0.4", lw=0.7)
    ax_len.set_xlabel("corrupted words (%)")
    ax_len.set_ylabel("reasoning tokens / clean")
    ax_len.set_xticks([0] + RATES)
    ax_len.set_title("(c) reasoning-length tax")

    fig.tight_layout()
    save(fig, "fig1_accuracy.pdf")


# ---------------------------------------------------------------------------
# Figure 2: the real-word effect
# ---------------------------------------------------------------------------
def fig_realword():
    fig, (ax_odds, ax_paired) = plt.subplots(1, 2, figsize=(7.1, 2.2))

    width = 0.35
    xs = np.arange(len(DATASETS))
    real_pen, non_pen = [], []
    for ds, d, label, colour in DATASETS:
        row = load(d, "realword_pertypo_logit.csv")[0]
        real_pen.append(100 * (1 - float(row["odds_real"])))
        non_pen.append(100 * (1 - float(row["odds_nonword"])))
    ax_odds.bar(xs - width / 2, real_pen, width, label="real-word typo",
                color="#b2182b", edgecolor="black", lw=0.4)
    ax_odds.bar(xs + width / 2, non_pen, width, label="non-word typo",
                color="#92c5de", edgecolor="black", lw=0.4)
    for x, (a, b) in enumerate(zip(real_pen, non_pen)):
        ax_odds.text(x - width / 2, a + 0.15, f"{a:.1f}", ha="center", fontsize=6)
        ax_odds.text(x + width / 2, b + 0.15, f"{b:.1f}", ha="center", fontsize=6)
    ax_odds.set_xticks(xs)
    ax_odds.set_xticklabels([d[2] for d in DATASETS])
    ax_odds.set_ylabel("odds of a correct answer\nlost per typo (%)")
    ax_odds.set_title("(a) per-typo damage (logistic fit)")
    ax_odds.legend(frameon=False, loc="upper right")

    width = 0.25
    xs = np.arange(len(RATES))
    for i, (ds, d, label, colour) in enumerate(DATASETS):
        rows = {(int(r["rate"]), r["compare"]): r for r in load(d, "realword_paired.csv")}
        vals, sig = [], []
        for rate in RATES:
            r = rows[(rate, "real10_to_real70")]
            vals.append(100 * float(r["delta_acc"]))
            sig.append(float(r["mcnemar_p"]) < 0.05)
        bars = ax_paired.bar(xs + (i - 1) * width, vals, width, color=colour,
                             label=label, edgecolor="black", lw=0.4)
        for b, s in zip(bars, sig):
            if s:
                ax_paired.text(b.get_x() + b.get_width() / 2,
                               b.get_height() - 0.9, "*", ha="center", fontsize=9)
    ax_paired.axhline(0, color="0.3", lw=0.7)
    ax_paired.set_xticks(xs)
    ax_paired.set_xticklabels([f"{r}% corrupted" for r in RATES])
    ax_paired.set_ylabel("$\\Delta$ accuracy, real10 $\\to$ real70 (pp)")
    ax_paired.set_title("(b) same words, real-word vs. non-word")
    ax_paired.legend(frameon=False, loc="lower left")

    fig.tight_layout()
    save(fig, "fig2_realword.pdf")


# ---------------------------------------------------------------------------
# Figure 3: repair behaviour and the silent-failure test
# ---------------------------------------------------------------------------
def fig_repair():
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.1),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    cats = ["silent_fix_frac", "flagged_frac", "misread_frac", "not_used_frac"]
    names = ["silent fix", "flagged", "misread", "not used"]
    colours = ["#4393c3", "#92c5de", "#d6604d", "#dddddd"]

    for ax, (ds, d, label, colour) in zip(axes[:3], DATASETS):
        rows = {r["real_ratio"]: r for r in load(d, "repair_wordlevel_by_real.csv")}
        xs = np.arange(len(REALS))
        bottom = np.zeros(len(REALS))
        for cat, nm, col in zip(cats, names, colours):
            vals = np.array([100 * float(rows[str(p)][cat]) for p in REALS])
            ax.bar(xs, vals, 0.65, bottom=bottom, label=nm, color=col,
                   edgecolor="black", lw=0.3)
            bottom += vals
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{p}%" for p in REALS])
        ax.set_xlabel("real-word typos")
        ax.set_ylim(0, 138)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_title(label)
        ax.grid(False)
        if ax is axes[0]:
            ax.set_ylabel("corrupted words (%)")
        else:
            ax.set_yticklabels([])
    axes[0].legend(frameon=False, loc="upper left", ncol=2, fontsize=6,
                   handlelength=1.1, columnspacing=1.0)

    ax = axes[3]
    for ds, d, label, colour in DATASETS:
        rows = {r["real_ratio"]: r for r in load(d, "realword_silent_failure.csv")}
        ys = [100 * float(rows[str(p)]["explicit_notice_frac"]) for p in REALS]
        ax.plot(REALS, ys, "o-", color=colour, label=label, lw=1.4, ms=3.5)
    ax.set_xticks(REALS)
    ax.set_xlabel("real-word typos (%)")
    ax.set_ylabel("wrong answers that\nexplicitly flagged a typo (%)")
    ax.set_ylim(0, 100)
    ax.set_title("silent-failure test")
    ax.legend(frameon=False, loc="lower right", fontsize=6)

    fig.tight_layout()
    save(fig, "fig3_repair.pdf")


# ---------------------------------------------------------------------------
# Figure 4: what the doubt signal actually is
# ---------------------------------------------------------------------------
def fig_doubt():
    fig, (ax_marker, ax_notice) = plt.subplots(1, 2, figsize=(7.1, 2.2))
    xs = np.arange(len(CONFIGS) + 1)
    labels = ["clean"] + [c.replace("typo", "").replace("_real", "/") for c in CONFIGS]

    for ds, d, label, colour in DATASETS:
        grid = keyed(load(d, "lexical_grid.csv"))
        order = ["clean"] + CONFIGS
        ax_marker.plot(xs, [float(grid[c]["uncertainty"]) for c in order], "o-",
                       color=colour, lw=1.4, ms=3, label=f"{label}: hedging")
        ax_marker.plot(xs, [float(grid[c]["second_guess"]) for c in order], "s--",
                       color=colour, lw=1.0, ms=2.5, alpha=0.55,
                       label=f"{label}: self-correction")
        ax_notice.plot(xs, [float(grid[c]["typo_noticing"]) for c in order], "o-",
                       color=colour, lw=1.4, ms=3, label=label)

    for ax in (ax_marker, ax_notice):
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
    ax_marker.set_ylabel("markers per 1k words")
    ax_marker.set_ylim(5.0, 30.0)
    ax_marker.set_title("(a) hedging rises, self-correction does not")
    ax_marker.legend(frameon=False, ncol=2, fontsize=5.5, loc="upper left")
    ax_notice.set_ylabel("typo-noticing words per 1k")
    ax_notice.set_title("(b) explicit typo talk")
    ax_notice.legend(frameon=False, fontsize=6)

    fig.tight_layout()
    save(fig, "fig4_doubt.pdf")


# ---------------------------------------------------------------------------
# Figure 5: prompt-level mitigations (GSM8K)
# ---------------------------------------------------------------------------
def fig_fixes():
    base = keyed(load("gsm8k/results_20000", "accuracy_per_config.csv"))
    warn = keyed(load("gsm8k/fix_warn", "accuracy_per_config.csv"))
    rew = keyed(load("gsm8k/fix_rewrite", "accuracy_per_config.csv"))
    order = ["clean"] + CONFIGS
    labels = ["clean"] + [c.replace("typo", "").replace("_real", "/") for c in CONFIGS]
    xs = np.arange(len(order))

    fig, (ax_abs, ax_delta) = plt.subplots(1, 2, figsize=(7.1, 2.2))
    for tab, name, colour, marker in ((base, "no fix", "#333333", "o"),
                                      (warn, "warn", "#1f77b4", "s"),
                                      (rew, "rewrite", "#d62728", "^")):
        ys = [100 * float(tab[c]["acc_answered"]) for c in order]
        ax_abs.plot(xs, ys, marker=marker, ls="-", color=colour, label=name,
                    lw=1.3, ms=3)
    ax_abs.set_xticks(xs)
    ax_abs.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
    ax_abs.set_ylabel("accuracy (%)")
    ax_abs.set_title("(a) GSM8K accuracy under each fix")
    ax_abs.legend(frameon=False)

    width = 0.38
    for i, (tab, name, colour) in enumerate(((warn, "warn", "#1f77b4"),
                                             (rew, "rewrite", "#d62728"))):
        ys = [100 * (float(tab[c]["acc_answered"]) - float(base[c]["acc_answered"]))
              for c in order]
        ax_delta.bar(xs + (i - 0.5) * width, ys, width, color=colour, label=name,
                     edgecolor="black", lw=0.3)
    ax_delta.axhline(0, color="0.3", lw=0.7)
    ax_delta.set_xticks(xs)
    ax_delta.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
    ax_delta.set_ylabel("$\\Delta$ accuracy vs. no fix (pp)")
    ax_delta.set_title("(b) neither fix recovers the loss")
    ax_delta.legend(frameon=False)

    fig.tight_layout()
    save(fig, "fig5_fixes.pdf")


# ---------------------------------------------------------------------------
# Appendix tables
# ---------------------------------------------------------------------------
def _fmt_p(p):
    p = float(p)
    if p < 1e-6:
        return "$<$1e-6"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def tab_full_accuracy():
    lines = [r"\begin{tabular}{l rrr rrr rrr}", r"\toprule",
             r" & \multicolumn{3}{c}{GSM8K} & \multicolumn{3}{c}{MATH-500}"
             r" & \multicolumn{3}{c}{ARC-Challenge} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"config & acc & $n_a$ & tok & acc & $n_a$ & tok"
             r" & acc & $n_a$ & tok \\", r"\midrule"]
    accs = {ds: keyed(load(d, "accuracy_per_config.csv")) for ds, d, _, _ in DATASETS}
    lens = {ds: keyed(load(d, "reasoning_length_absolute.csv")) for ds, d, _, _ in DATASETS}
    for c in ["clean"] + CONFIGS:
        cells = []
        for ds, d, _, _ in DATASETS:
            cells += [f"{100*float(accs[ds][c]['acc_answered']):.1f}",
                      lens[ds][c]["n_answered"],
                      f"{float(lens[ds][c]['mean_tok']):.0f}"]
        name = c.replace("_", r"\_")
        lines.append(f"{name} & " + " & ".join(str(x) for x in cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("tab_full_accuracy.tex", lines)


def tab_full_flips():
    lines = [r"\begin{tabular}{l rrr rrr rrr}", r"\toprule",
             r" & \multicolumn{3}{c}{GSM8K} & \multicolumn{3}{c}{MATH-500}"
             r" & \multicolumn{3}{c}{ARC-Challenge} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"config & r$\to$w & w$\to$r & $p$ & r$\to$w & w$\to$r & $p$"
             r" & r$\to$w & w$\to$r & $p$ \\", r"\midrule"]
    flips = {ds: keyed(load(d, "flips_vs_clean.csv")) for ds, d, _, _ in DATASETS}
    for c in CONFIGS:
        cells = []
        for ds, d, _, _ in DATASETS:
            r = flips[ds][c]
            cells += [r["r2w"], r["w2r"], _fmt_p(r["mcnemar_p"])]
        name = c.replace("_", r"\_")
        lines.append(f"{name} & " + " & ".join(str(x) for x in cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("tab_full_flips.tex", lines)


def tab_full_doubt():
    lines = [r"\begin{tabular}{l rr rr rr}", r"\toprule",
             r" & \multicolumn{2}{c}{GSM8K} & \multicolumn{2}{c}{MATH-500}"
             r" & \multicolumn{2}{c}{ARC-Challenge} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"config & s.corr. & hedge & s.corr. & hedge & s.corr. & hedge \\",
             r"\midrule"]
    grids = {ds: keyed(load(d, "lexical_grid.csv")) for ds, d, _, _ in DATASETS}
    for c in ["clean"] + CONFIGS:
        cells = []
        for ds, d, _, _ in DATASETS:
            r = grids[ds][c]
            cells += [f"{float(r['second_guess']):.1f}", f"{float(r['uncertainty']):.1f}"]
        name = c.replace("_", r"\_")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("tab_full_doubt.tex", lines)


def tab_full_repair():
    lines = [r"\begin{tabular}{l r rrrr}", r"\toprule",
             r"dataset / config & \#words & silent fix & flagged & misread & not used \\",
             r"\midrule"]
    for ds, d, label, _ in DATASETS:
        lines.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{label}}}}} \\")
        for r in load(d, "repair_wordlevel_per_config.csv"):
            name = r["config"].replace("_", r"\_")
            lines.append(
                f"\\quad {name} & {r['n_corrupt_words']} & "
                f"{100*float(r['silent_fix_frac']):.1f} & {100*float(r['flagged_frac']):.1f} & "
                f"{100*float(r['misread_frac']):.1f} & {100*float(r['not_used_frac']):.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("tab_full_repair.tex", lines)


def _write(name, lines):
    path = os.path.join(TAB_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.relpath(path, HERE))


if __name__ == "__main__":
    fig_accuracy()
    fig_realword()
    fig_repair()
    fig_doubt()
    fig_fixes()
    tab_full_accuracy()
    tab_full_flips()
    tab_full_doubt()
    tab_full_repair()
