# generate_final_report

Everything needed to build the final report PDF.

The paper's prose is written outside LaTeX (in a plain-text draft) and pasted
into `report.tex`. Every number, figure and table is generated from the derived
CSVs in `full_analysis/analysis_tables/` — nothing is typed by hand.

## Build

```bash
cd generate_final_report
python make_assets.py                 # regenerate figures/ and tables/
pdflatex report && bibtex report && pdflatex report && pdflatex report
```

Four passes are needed: `pdflatex` once to collect labels, `bibtex` for the
bibliography, then twice more so cross-references and citation numbers settle.
The result is `report.pdf`.

## What is here

| File | Role |
|---|---|
| `report.tex` | the paper |
| `make_assets.py` | draws the five figures and writes the generated table fragments |
| `custom.bib` | bibliography, copied from `../paper/custom.bib` |
| `figures/` | generated — do not edit by hand |
| `tables/` | generated — **except `tab_datastats.tex`**, see below |
| `verify_text.py` | checks the paper still matches the plain-text draft |

### tab_datastats.tex is the one exception

`tables/tab_datastats.tex` is **not** produced by `make_assets.py`. It is
maintained in `../paper/tables/` and built from the published typo datasets'
own per-question bookkeeping. If it changes upstream, copy it in again:

```bash
cp ../paper/tables/tab_datastats.tex tables/
```

## Where the numbers come from

`make_assets.py` reads one folder per dataset. The choice matters — several
retired runs are still on disk and are **not** what the paper reports:

| Numbers | Folder | Token cap |
|---|---|---|
| GSM8K | `gsm8k/results_20000` | 20,000 |
| MATH-500 | `math500/results` | 17,000 |
| ARC-Challenge | `arc/results` | 17,000 |
| warn arm | `gsm8k/fix_warn` | 20,000 |
| rewrite arm | `gsm8k/fix_rewrite` | 20,000 |
| spell-check arm | `gsm8k/fix_spellcheck_20000` | 20,000 |

All six share the same 20k/17k generation budget, so the mitigation deltas in
Table 3 are a like-for-like comparison against the no-fix baseline.

**Do not use** `gsm8k/results` or `gsm8k/fix_spellcheck`: those are the retired
4,096-token runs. Their clean accuracy is 95.2% and 83.7% against the main run's
91.4%, so mixing them silently corrupts every comparison.

Set `NLP_TABLES` to read the CSVs from somewhere other than
`../full_analysis/analysis_tables`.

## Checking the prose

The draft is the source of truth for sections 1–9. After editing either side:

```bash
python verify_text.py /path/to/report_text.txt
```

It prints one line per section. Differences that are expected and fine:

- section and subsection headings, which LaTeX generates
- multi-key citations such as `\citep{gan2024,chai2024,zhao2026}`
- deliberate additions, e.g. the `(Figure 3a)` cross-references

Anything else means the paper and the draft have diverged and one of them needs
updating.

## Layout notes

Two things that are easy to undo by accident:

- **`\usepackage{stfloats}`** lets a full-width `figure*`/`table*` sit at the
  bottom of a page, not only the top. Without it wide floats queue up and drift
  several pages from the text that cites them. Install with
  `miktex packages install sttools` if it is missing.
- **Float declaration position.** A float can never be placed before the point
  it is declared. Figures 3 and 4 are declared immediately after their
  subsection heading rather than after the prose, which is what lets them land
  on the same page as their own section.

The appendix is single-column (`\onecolumn`) with `[H]` placement so the wide
tables get the full page width and stack in order.
