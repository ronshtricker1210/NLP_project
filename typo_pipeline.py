#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
typo_pipeline.py
================

Typo-generation and dataset-binning pipeline for the "how typos affect
reasoning LLMs" study.

Pipeline overview
-----------------
1. Load a (small, local-dev) subset of the ``MATH-500`` dataset.
2. For every *problem* (a "row") inject keyboard-aware typos into the
   ``problem`` text using the local ``multypo`` package. Numbers and
   mathematical expressions (LaTeX) are protected and never corrupted.
   Every problem receives **at least one** typo so it can be compared
   one-to-one against the zero-typo baseline model.
3. For each corrupted word, use ``nltk.corpus.words`` to decide whether the
   produced typo is a *real word* (exists in the English dictionary) or a
   *non-word*. Per-problem counts are stored.
4. Score each problem by the real-word ratio ::

       P = (real-word typos) / (total words actually changed)

   The denominator only counts words that were genuinely modified.
5. Use :func:`pandas.cut` to split the problems into ``REAL_TOKEN_GROUPS``
   bins (e.g. 0-25%, 25-50%, 50-75%, 75-100%) and emit one dataset per bin.

The script is fully OS-agnostic: all paths go through :mod:`pathlib`/``os``,
and parallelism uses :meth:`datasets.Dataset.map` with ``num_proc`` (which
works with both ``fork`` on Linux and ``spawn`` on Windows). The heavy shared
objects (the dictionary set and the typo generator) are lazily built once per
worker process via module-level caches.

Run locally::

    python typo_pipeline.py
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Central, tweakable configuration for the pipeline."""

    # --- data loading (local-dev mode) ---
    dataset_name: str = "HuggingFaceH4/MATH-500"
    dataset_split: str = "test"
    subset_size: int = 50          # keep small for local Windows testing
    text_field: str = "problem"    # only this field receives typos
    # Offline support (useful behind a corporate firewall that blocks the Hub).
    # If set, load the dataset from this local path instead of the Hub. It may
    # be a directory saved via ``save_to_disk`` or a ``.json``/``.parquet`` file.
    local_data_path: Optional[Path] = None
    # If the Hub/NLTK are unreachable, fall back to a tiny built-in mock so the
    # pipeline logic can still be smoke-tested locally.
    allow_offline_fallback: bool = True
    # Optional local newline-separated dictionary to replace nltk.corpus.words
    # when the corpus cannot be downloaded.
    local_words_path: Optional[Path] = None

    # --- typo generation ---
    language: str = "english"
    typo_rate: float = 0.30        # fraction of eligible words to corrupt
    # Probability weights for the 4 techniques required by the proposal:
    #   replace   = adjacent-key substitution
    #   transpose = swapping two letters
    #   delete    = deletion of a letter
    #   insert    = insertion of a letter
    typo_weights: Tuple[Tuple[str, float], ...] = (
        ("delete", 0.20),
        ("insert", 0.20),
        ("replace", 0.40),
        ("transpose", 0.20),
    )
    min_word_len: int = 2          # words shorter than this are never typo'd

    # --- binning ---
    real_token_groups: int = 4     # number of P-ratio bins

    # --- execution ---
    num_proc: int = 1              # 1 for local Windows dev; raise on Slurm
    seed: int = 42

    # --- output ---
    out_dir: Path = Path(__file__).resolve().parent / "typo_dataset"

    @property
    def typo_weights_dict(self) -> Dict[str, float]:
        return {name: w for name, w in self.typo_weights}


CONFIG = Config()


# ---------------------------------------------------------------------------
# Offline fallbacks (only used when the Hub / NLTK cannot be reached)
# ---------------------------------------------------------------------------

# A handful of MATH-500-style problems (with LaTeX) so the pipeline can be
# smoke-tested without network access. These mirror the real schema fields.
_MOCK_PROBLEMS: List[dict] = [
    {
        "problem": r"What is the value of $\frac{3}{4} + \frac{1}{8}$ when "
                   r"expressed as a common fraction in lowest terms?",
        "answer": r"\frac{7}{8}",
        "subject": "Prealgebra", "level": 2, "unique_id": "mock/0",
    },
    {
        "problem": r"A triangle has sides of length 3, 4 and 5 units. "
                   r"Compute its area in square units.",
        "answer": "6", "subject": "Geometry", "level": 1, "unique_id": "mock/1",
    },
    {
        "problem": r"If $x^2 - 5x + 6 = 0$, what is the sum of all possible "
                   r"values of $x$?",
        "answer": "5", "subject": "Algebra", "level": 3, "unique_id": "mock/2",
    },
    {
        "problem": r"Evaluate $\sqrt{144}$ and then multiply the result by "
                   r"seven to obtain the final integer answer.",
        "answer": "84", "subject": "Prealgebra", "level": 1, "unique_id": "mock/3",
    },
    {
        "problem": r"How many distinct positive divisors does the number "
                   r"$60$ have in total?",
        "answer": "12", "subject": "Number Theory", "level": 2, "unique_id": "mock/4",
    },
]

# Minimal fallback dictionary for real-word detection when the NLTK corpus is
# unavailable. This is NOT a substitute for the full corpus (the real run on
# Slurm uses nltk.corpus.words); it only lets local smoke tests classify a few
# common words as "real".
_FALLBACK_WORDS: frozenset = frozenset(
    w.lower()
    for w in (
        "what is the value of when expressed as a common fraction in lowest "
        "terms triangle has sides length and units compute its area square "
        "if sum all possible values evaluate then multiply result by seven to "
        "obtain final integer answer how many distinct positive divisors does "
        "number have total the and for with from into over under about above "
        "below between word real none same time part place work case point "
        "government company system program question during without before "
        "great small large little other another which their there these those "
        "where when while because through against around before behind beside"
    ).split()
)


# ---------------------------------------------------------------------------
# Math / number protection
# ---------------------------------------------------------------------------

# Spans matched here are considered "protected" and are never corrupted.
# Order matters: display math ($$...$$) must be tried before inline math.
_PROTECTED_PATTERN = re.compile(
    r"\$\$.*?\$\$"          # display math  $$ ... $$
    r"|\$.*?\$"             # inline math   $ ... $
    r"|\\\[.*?\\\]"         # display math  \[ ... \]
    r"|\\\(.*?\\\)"         # inline math   \( ... \)
    r"|\\[a-zA-Z]+"         # LaTeX command names, e.g. \frac, \sqrt, \times
    r"|\d+(?:[.,]\d+)?",    # bare numbers (integers / decimals)
    re.DOTALL,
)

# Candidate prose words: runs of ASCII letters only. Anything containing a
# digit or backslash therefore cannot match and stays untouched.
_WORD_PATTERN = re.compile(r"[A-Za-z]+")


def find_protected_spans(text: str) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` character spans that must not be corrupted."""
    return [(m.start(), m.end()) for m in _PROTECTED_PATTERN.finditer(text)]


def _overlaps_any(start: int, end: int, spans: List[Tuple[int, int]]) -> bool:
    """True if ``[start, end)`` overlaps any protected span."""
    for s, e in spans:
        if start < e and end > s:
            return True
    return False


# ---------------------------------------------------------------------------
# Lazily-initialised shared resources (built once per process)
# ---------------------------------------------------------------------------

_WORD_SET: Optional[frozenset] = None
_GENERATOR = None  # type: ignore[var-annotated]


def get_nltk_word_set() -> frozenset:
    """
    Return a lowercase set of valid English words from ``nltk.corpus.words``.

    The corpus is downloaded on first use if it is missing. The result is
    cached in a module-level global so each worker process builds it once.
    """
    global _WORD_SET
    if _WORD_SET is not None:
        return _WORD_SET

    # 1) User-provided local dictionary file takes priority.
    if CONFIG.local_words_path and Path(CONFIG.local_words_path).exists():
        with open(CONFIG.local_words_path, encoding="utf-8") as fh:
            _WORD_SET = frozenset(line.strip().lower() for line in fh if line.strip())
        return _WORD_SET

    # 2) Try the real NLTK corpus (downloading it if the network allows).
    try:
        import nltk
        from nltk.corpus import words as nltk_words

        try:
            nltk_words.words()  # force access; raises LookupError if missing
        except LookupError:
            nltk.download("words", quiet=True)
            from nltk.corpus import words as nltk_words

        _WORD_SET = frozenset(w.lower() for w in nltk_words.words())
        return _WORD_SET
    except Exception as exc:  # network blocked, corpus missing, etc.
        if not CONFIG.allow_offline_fallback:
            raise
        print(
            f"[warn] nltk.corpus.words unavailable ({exc.__class__.__name__}); "
            f"using a tiny fallback dictionary. Real-word detection will be "
            f"approximate. Provide Config.local_words_path or run on a machine "
            f"with network access for accurate results."
        )
        _WORD_SET = _FALLBACK_WORDS
        return _WORD_SET


def get_generator():
    """Return a cached :class:`multypo.MultiTypoGenerator` for this process."""
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR

    from multypo import MultiTypoGenerator

    _GENERATOR = MultiTypoGenerator(
        language=CONFIG.language,
        use_excluding_set=True,
        typo_distribution=CONFIG.typo_weights_dict.copy(),
    )
    return _GENERATOR


# ---------------------------------------------------------------------------
# Typo generation
# ---------------------------------------------------------------------------


@dataclass
class TypoResult:
    """Outcome of corrupting a single piece of text."""

    text: str          # the corrupted text
    total: int         # number of words actually changed
    real: int          # how many changes are valid English words
    nonword: int       # how many changes are non-words


def _apply_one_typo(
    generator,
    word: str,
    typo_types: List[str],
    type_weights: List[float],
) -> Tuple[str, bool]:
    """
    Corrupt a single word with one typo, guaranteeing a change when possible.

    A typo type is sampled according to ``type_weights``; if it happens to be
    inapplicable (e.g. ``transpose`` on a word with no swappable pair) the
    remaining types are tried as fallbacks. ``delete`` always changes a word
    of length >= 2, so a change is effectively always produced.
    """
    first = random.choices(typo_types, weights=type_weights, k=1)[0]
    order = [first] + [t for t in typo_types if t != first]
    for typo_type in order:
        new_word, changed = generator.apply_single_typo(word, typo_type)
        if changed:
            return new_word, True
    return word, False


def apply_typos_to_text(
    text: str,
    generator,
    word_set: frozenset,
    typo_rate: float,
    min_word_len: int,
) -> TypoResult:
    """
    Inject typos into ``text`` and classify each change as real/non-word.

    Only plain prose words (ASCII letters, not inside a protected math span,
    not a known number word) are eligible. At least one typo is always applied
    when at least one eligible word exists.
    """
    protected = find_protected_spans(text)
    ignore_set = getattr(generator, "ignore_set", set())

    eligible = [
        m
        for m in _WORD_PATTERN.finditer(text)
        if (m.end() - m.start()) >= min_word_len
        and not _overlaps_any(m.start(), m.end(), protected)
        and m.group().lower() not in ignore_set
    ]

    if not eligible:
        # No prose to corrupt (essentially never happens for MATH-500).
        return TypoResult(text=text, total=0, real=0, nonword=0)

    # Guarantee >= 1 typo; never round down to zero.
    n_target = max(1, round(typo_rate * len(eligible)))
    n_target = min(n_target, len(eligible))
    chosen = random.sample(eligible, n_target)

    typo_types = list(CONFIG.typo_weights_dict.keys())
    type_weights = list(CONFIG.typo_weights_dict.values())

    total = real = nonword = 0
    replacements: List[Tuple[int, int, str]] = []
    for match in chosen:
        original = match.group()
        new_word, changed = _apply_one_typo(
            generator, original, typo_types, type_weights
        )
        if not changed:
            continue
        total += 1
        if new_word.lower() in word_set:
            real += 1
        else:
            nonword += 1
        replacements.append((match.start(), match.end(), new_word))

    # Splice replacements back in, right-to-left so offsets stay valid.
    new_text = text
    for start, end, new_word in sorted(replacements, key=lambda r: r[0], reverse=True):
        new_text = new_text[:start] + new_word + new_text[end:]

    return TypoResult(text=new_text, total=total, real=real, nonword=nonword)


# ---------------------------------------------------------------------------
# Per-row processing (used by datasets.map, must be top-level / picklable)
# ---------------------------------------------------------------------------


def process_row(example: dict, idx: int) -> dict:
    """
    Corrupt one dataset row and attach typo statistics.

    New columns
    -----------
    problem_typo : str   -> the corrupted problem text
    num_total    : int   -> number of words actually changed
    num_real     : int   -> changes that are real English words
    num_nonword  : int   -> changes that are non-words
    real_ratio   : float -> P = num_real / num_total (in [0, 1])
    """
    # Deterministic, per-row seeding so runs are reproducible even in parallel.
    random.seed(CONFIG.seed + idx)

    generator = get_generator()
    word_set = get_nltk_word_set()

    result = apply_typos_to_text(
        text=example[CONFIG.text_field],
        generator=generator,
        word_set=word_set,
        typo_rate=CONFIG.typo_rate,
        min_word_len=CONFIG.min_word_len,
    )

    real_ratio = result.real / result.total if result.total > 0 else float("nan")

    return {
        "problem_typo": result.text,
        "num_total": result.total,
        "num_real": result.real,
        "num_nonword": result.nonword,
        "real_ratio": real_ratio,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_mock_dataset(config: Config):
    """Build an in-memory dataset from the built-in mock problems."""
    from datasets import Dataset

    rows = (_MOCK_PROBLEMS * ((config.subset_size // len(_MOCK_PROBLEMS)) + 1))
    rows = rows[: config.subset_size]
    return Dataset.from_list(rows)


def load_math_subset(config: Config):
    """
    Load the first ``subset_size`` rows of the configured dataset.

    Resolution order:
    1. ``config.local_data_path`` (offline: save_to_disk dir, .json or .parquet)
    2. the Hugging Face Hub
    3. a small built-in mock (only if ``allow_offline_fallback`` and the Hub is
       unreachable) so the pipeline can be smoke-tested with no network.
    """
    from datasets import Dataset, load_dataset, load_from_disk

    # 1) Explicit local path.
    if config.local_data_path:
        path = Path(config.local_data_path)
        if not path.exists():
            raise FileNotFoundError(f"local_data_path does not exist: {path}")
        if path.is_dir():
            dataset = load_from_disk(str(path))
        elif path.suffix == ".json":
            dataset = Dataset.from_json(str(path))
        elif path.suffix == ".parquet":
            dataset = Dataset.from_parquet(str(path))
        else:
            raise ValueError(f"Unsupported local_data_path type: {path.suffix}")
        n = min(config.subset_size, len(dataset))
        return dataset.select(range(n))

    # 2) Hugging Face Hub, with 3) offline fallback on failure.
    try:
        dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    except Exception as exc:
        if not config.allow_offline_fallback:
            raise
        print(
            f"[warn] could not reach the Hub for '{config.dataset_name}' "
            f"({exc.__class__.__name__}); falling back to the built-in mock "
            f"dataset. Use Config.local_data_path or run on a networked machine "
            f"(e.g. the Slurm cluster) for the real data."
        )
        return _load_mock_dataset(config)

    n = min(config.subset_size, len(dataset))
    return dataset.select(range(n))


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


def _bin_labels(groups: int) -> List[str]:
    """Human-readable percentage labels, e.g. ['0-25', '25-50', ...]."""
    edges = np.linspace(0, 100, groups + 1)
    return [f"{int(round(edges[i]))}-{int(round(edges[i + 1]))}" for i in range(groups)]


def bin_dataset(dataset, groups: int):
    """
    Split ``dataset`` into ``groups`` bins by ``real_ratio`` using pandas.cut.

    Returns a :class:`datasets.DatasetDict` keyed by percentage-range labels
    (e.g. ``"0-25"``). Rows with an undefined ratio (no typos) are dropped
    with a warning, since a 0-typo problem cannot belong to the typo dataset.
    """
    from datasets import Dataset, DatasetDict

    df = dataset.to_pandas()

    undefined = df["real_ratio"].isna().sum()
    if undefined:
        print(
            f"[warn] dropping {undefined} problem(s) with 0 typos "
            f"(cannot be binned / compared to the baseline)."
        )
        df = df[df["real_ratio"].notna()].reset_index(drop=True)

    edges = np.linspace(0.0, 1.0, groups + 1)
    labels = _bin_labels(groups)
    df["real_bin"] = pd.cut(
        df["real_ratio"],
        bins=edges,
        labels=labels,
        include_lowest=True,
    )

    bins = {}
    for label in labels:
        sub = df[df["real_bin"] == label].reset_index(drop=True)
        if len(sub) == 0:
            continue
        # ``real_bin`` is categorical; store as plain string for portability.
        sub = sub.copy()
        sub["real_bin"] = sub["real_bin"].astype(str)
        bins[label] = Dataset.from_pandas(sub, preserve_index=False)

    return DatasetDict(bins)


def save_bins(dataset_dict, out_dir: Path) -> None:
    """Persist each bin to its own sub-directory via ``save_to_disk``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, ds in dataset_dict.items():
        target = out_dir / f"bin_{label}"
        ds.save_to_disk(str(target))
        print(f"[save] bin '{label}': {len(ds):>4d} rows -> {target}")


# ---------------------------------------------------------------------------
# Reporting helpers (for local inspection)
# ---------------------------------------------------------------------------


def preview(dataset, config: Config, n: int = 3) -> None:
    """Print a few before/after examples and the typo statistics."""
    print("\n" + "=" * 78)
    print(f"PREVIEW: {min(n, len(dataset))} example problem(s)")
    print("=" * 78)
    for i in range(min(n, len(dataset))):
        row = dataset[i]
        print(f"\n--- Problem {i} "
              f"(total={row['num_total']}, real={row['num_real']}, "
              f"nonword={row['num_nonword']}, P={row['real_ratio']:.3f}) ---")
        print("ORIGINAL:", row[config.text_field][:300])
        print("TYPO    :", row["problem_typo"][:300])


def summarize(dataset) -> None:
    """Print aggregate statistics across the processed dataset."""
    totals = dataset["num_total"]
    reals = dataset["num_real"]
    nonwords = dataset["num_nonword"]
    print("\n" + "-" * 78)
    print("SUMMARY")
    print("-" * 78)
    print(f"problems processed : {len(dataset)}")
    print(f"typos (total)      : {sum(totals)}")
    print(f"  real-word typos  : {sum(reals)}")
    print(f"  non-word typos   : {sum(nonwords)}")
    print(f"min typos / problem: {min(totals)}  (must be >= 1)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(config: Config = CONFIG) -> None:
    print(f"[load] {config.dataset_name} [{config.dataset_split}] "
          f"(subset={config.subset_size})")
    dataset = load_math_subset(config)

    print(f"[typo] generating typos on '{config.text_field}' "
          f"(num_proc={config.num_proc}) ...")
    processed = dataset.map(
        process_row,
        with_indices=True,
        num_proc=config.num_proc,
        desc="Injecting typos",
    )

    preview(processed, config)
    summarize(processed)

    print(f"\n[bin ] splitting into {config.real_token_groups} bins by P ...")
    binned = bin_dataset(processed, config.real_token_groups)

    print(f"[save] writing bins to {config.out_dir}")
    save_bins(binned, config.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
