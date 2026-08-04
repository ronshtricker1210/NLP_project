# HANDOFF.md — what this project is and where everything lives

For an agent picking this repo up cold. `AGENTS.md` covers only the dataset
generation stage; this file covers the whole project end to end, including the
final paper.

## 1. The study in one paragraph

We measure how **typos** degrade a **reasoning** LLM. Two axes are varied
independently: the **typo rate** $r \in \{25,50,75\}\%$ of eligible words, and
the **real-word ratio** $\rho \in \{10,40,70\}\%$ — the share of typos that land
on another valid English word (`sum`→`sun`) rather than a non-word
(`triangle`→`trianlge`). That gives 9 typo configurations plus `clean`. The
generator is seeded at 42, so question `idx=i` is the *same* source question in
all ten conditions; every comparison in the paper is paired (McNemar), not a
difference of independent accuracies.

Subject model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` via the HF router
(OpenAI-compatible, `https://router.huggingface.co/v1`), temperature 0.6,
top_p 0.95, 1 sample per question. Datasets: GSM8K test, MATH-500,
ARC-Challenge, 500 questions each. Non-reasoning control:
`Qwen2.5-7B-Instruct`. Trace judge: `meta-llama/Llama-3.3-70B-Instruct`.

## 2. Findings (what the paper claims)

1. **Large monotone typo tax.** clean → `typo75_real70`: GSM8K −31.0 pp
   (91.4→60.4), ARC −10.4 pp, MATH-500 −8.2 pp. Decomposition says ~93% is a
   *reasoning-quality* loss, not truncation. Traces get **longer** under typos
   (GSM8K 1582→2593 mean tokens), not shorter.
2. **Real-word typos hurt more per typo.** Odds ratio 1.9× (GSM8K), 1.3×
   (MATH-500), 5.1× (ARC). Confirmed by a controlled paired test holding the
   typo *count* fixed: at r=75 on GSM8K, ρ=10→70 costs −14.1 pp
   ($p{=}1.3{\times}10^{-7}$).
3. **The proposal's "silent failure" hypothesis is REFUTED.** Real-word
   corruptions are noticed *equally or more* often. The judge's
   accuracy-by-label breakdown gives the real mechanism:
   silent readthrough 84.1% correct, explicit-notice-and-fix 77.2%,
   **noticed-but-not-fixed 59.1%**, misread 57.9%. The failure is
   *noticed-but-unrepairable*, not unnoticed.
4. **The doubt signature is hedging, not verification.** Per 1k tokens,
   clean→75/70: "maybe" +5.66, "wait" +2.43, "perhaps" +1.13 — while
   "double-check" −0.35, "let me recheck" −0.24, "actually" −0.07.
5. **Reasoning distillation does not confer robustness.** R1 degrades *more*
   than Qwen2.5-7B-Instruct (MATH-500 −8.2 vs −6.8; ARC −10.4 vs −7.7).
6. **All three mitigations fail.** Warning the model: −0.5 pp (null). Asking it
   to rewrite the question first: −4.7 pp (harmful; shortens the trace ~40%).
   External spell-check: restores 54.9% of typos at ρ=10 but only 25.8% at
   ρ=70 — structurally blind to the channel that does the damage.

## 3. Repo map

| Path | What it is |
|---|---|
| `data_creation/` | Typo generator + HF dataset builder. See `AGENTS.md`. |
| `api-setup/` | **The runner used for all final results.** `run_typo_api.py` calls the HF router; `score.py` is the answer extractor/grader. |
| `server-setup/`, `gcp-setup/` | Earlier Slurm / vLLM / GCP attempts. Not used for the final numbers. |
| `full_analysis/` | All analysis. `run_all.py` orchestrates; outputs land in `tables/<run>/` and `reports/`. |
| `paper/` | The final ACL paper. `main.tex` → `main.pdf`. |
| `paper/data/` | The exact derived CSVs and reports the paper is built from, plus `MANIFEST.md` mapping every CSV to a paper number. |
| `NLP_Project_Proposal.md` | The original proposal (useful for seeing what changed). |

## 4. Getting the data and reproducing

Raw JSONL generation logs (~370 MB) are **not** in git. They live on the Hub:

- generations + judge cache: **`ronshtricker/typo-reasoning-results`** (dataset
  repo, 108 files). `Dolevabudi/typo-results` is the older repo and has GSM8K
  only — no MATH-500, no ARC, no controls.
- typo question sets: `idoazou/{gsm8k,math500,arc,gpqa}-typos`

```powershell
cd full_analysis
python download_data.py --dataset gsm8k      # then math500, arc
python run_all.py --dataset gsm8k --skip-run # --skip-run = analyse only, no API calls
cd ../paper
python make_assets.py                        # regenerates figures/ and tables/
pdflatex -interaction=nonstopmode main.tex; bibtex main; pdflatex main; pdflatex main
```

`latexmk` does not work on this machine (MiKTeX without a Perl script engine) —
run `pdflatex`/`bibtex` directly as above. `acl.sty` and `acl_natbib.bst` came
from the **root of the `master` branch** of
<https://github.com/acl-org/acl-style-files> (the `/latex/` subpath and the
`main` branch both 404).

## 5. Traps — read before touching anything

1. **`tables/gsm8k` is NOT the main GSM8K run.** It is the *spellcheck*
   mitigation arm at a 4,096-token cap (clean accuracy 83.7%). The main GSM8K
   run is **`tables/gsm8k_20000`** (20,000-token cap, clean 91.4%). Same for
   `data/gsm8k/*.jsonl`, whose rows carry `fix: "spellcheck"`. Never compare
   accuracies across these folders.
2. **`math_verify` silently mis-scores on Windows.** It enforces its parsing
   timeout with a worker process that cannot spawn, so `parse()` returns nothing
   and *every* MATH-500 answer scores wrong — no exception, no warning.
   `api-setup/score.py` now passes `parsing_timeout=None` /
   `timeout_seconds=None` when `os.name == "nt"`. Any MATH-500 number computed
   on Windows before that patch is invalid.
3. **`spellcheck_recovery_*.csv` is produced by buggy code.**
   `spellcheck_recovery_stats()` in `api-setup/run_typo_api.py` reads per-row
   metadata that is constant across a whole file, so every value is an exact
   multiple of 500. The paper's numbers (54.9 / 39.9 / 25.8% restored) were
   recomputed from the raw `spellcheck_changes` records. Fix the script before
   reusing that CSV.
4. **`config_files()` in `full_analysis/common.py` globs the
   `_Qwen2.5-7B-Instruct` control files into the main pipeline.** A fresh
   `run_all.py` on ARC or MATH-500 will silently add junk config rows. Filter
   them out first.
5. **`report_supplementary_baseline_fixes.tex` uses a laxer scorer** (GSM8K
   clean 92.4 vs strict 91.4). Table 2 in the paper uses
   `full_analysis/baseline_compare.py` instead. Its Qwen columns do match.
6. **The warn and rewrite mitigation JSONL are gone.** Only the derived CSVs in
   `tables/gsm8k_fix-warn` and `tables/gsm8k_fix-rewrite` survive — those arms
   would have to be re-run from scratch to re-analyse.

## 6. Known gaps (ranked by value of fixing)

1. **GSM8K judge ran against the spellcheck arm, not the main arm.** Highest
   value re-run: `python download_data.py --dataset gsm8k` then
   `python llm_judge.py --all`.
2. **MATH-500 judge cache is 100% errors** — all 4,962 rows have
   `repair_label = "error"`. No judge evidence at all for MATH-500.
3. **ARC judge had 5,998/9,738 API errors** → 74.8% coverage (3,740 usable
   rows). Retrying just the failed rows would complete it cheaply.
4. **No decoding-variance estimate** — every run is `--n-samples 1`. Even 3
   seeds on one configuration would strengthen the paper.
5. **No GSM8K non-reasoning control on disk**, so Table 2 covers only MATH-500
   and ARC.
6. **Mitigations only tested on GSM8K**, never MATH-500 or ARC.
7. **GPQA-Diamond was promised in the proposal**; ARC-Challenge was substituted
   after the GPQA grid could not be completed (only `results/gpqa/clean.jsonl`
   exists).
8. **No human validation** that corrupted questions remain answerable by people
   (the proposal promised it).
9. ARC `clean` truncation: only 455/500 answered vs 461–489 for typo
   conditions, which *deflates* the measured clean-vs-typo gap. The paper leans
   on paired flips, which are immune to this.
10. Judge ↔ word-level alignment agreement is only fair (Cohen's κ = 0.24 on
    ARC, 0.13 on GSM8K) — the judge labels a whole trace, the alignment labels
    each typo.

## 7. Paper status

`paper/main.pdf` — 11 pages: **8 body pages** (the guideline limit) + 1 page of
references + 2 appendix pages. Compiles clean, no undefined references or
citations. Title: *The Silent Tax: How Typos Reshape Reasoning in Thinking
LLMs*. Figures are PDF vector, appendix is single-column so the wide grids fit.
