# Final report

`main.tex` — "The Silent Tax: How Typos Reshape Reasoning in Thinking LLMs".
Output: `main.pdf` (8 body pages + references + 2 appendix pages).

## Build

`latexmk` does not work on the machine this was written on (MiKTeX without a Perl
script engine). Build directly:

```powershell
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

On Overleaf it just compiles (pdfLaTeX + BibTeX).

## Files

- `acl.sty`, `acl_natbib.bst` — ACL style files, from the repo root of
  <https://github.com/acl-org/acl-style-files> (branch `master`). If they are
  missing, `main.tex` falls back to `geometry` + `natbib` so it still builds.
- `custom.bib` — bibliography.
- `figures/*.pdf`, `tables/tab_full_*.tex` — generated, do not edit by hand.
- `make_assets.py` — regenerates everything in `figures/` and `tables/` from
  `../full_analysis/tables/`. Run it after `../full_analysis/run_all.py`.

```powershell
python make_assets.py
```
