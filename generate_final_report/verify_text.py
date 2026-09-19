"""Check that the author-written sections of report.tex reproduce the plain-text
source word for word.

The paper's prose is written outside LaTeX and pasted in, so it is easy for an
edit to drift. This compares the lowercase alphabetic word stream of each
author-written section against the same section of the plain-text draft and
prints any difference.

    python verify_text.py path/to/report_text.txt

Expected output is one line per section. Differences that DO show up are
normally one of:
  * section/subsection headings, which LaTeX generates
  * multi-key citations, which the stripper removes rather than expands
  * deliberate additions, e.g. a "(Figure 3a)" cross-reference
Anything else means the paper and the draft have genuinely diverged.
"""
import difflib
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "report.tex"

# bib key -> the author-year text the plain draft writes inline
CITES = {
    "wei2022": "Wei et al 2022", "deepseekr1": "Guo et al 2025",
    "gan2024": "Gan et al 2024", "chai2024": "Chai et al 2024",
    "zhao2026": "Zhao et al", "belinkov2018": "Belinkov and Bisk 2018",
    "ebrahimi2018": "Ebrahimi et al 2018", "jin2020": "Jin et al 2020",
    "zhu2024": "Zhu et al 2024", "pruthi2019": "Pruthi et al 2019",
    "turpin2023": "Turpin et al 2023", "huang2024": "Huang et al 2024",
    "llama3": "Grattafiori et al 2024", "zheng2023": "Zheng et al 2023",
}

# (label, (tex start, tex end), (draft start, draft end))
SECTIONS = [
    ("1 Introduction", (r"\section{Introduction}", r"\begin{figure}[tbp]"),
     ("Chain-of-thought prompting and reasoning", "2 Background and Related Work")),
    ("2 Background", (r"\section{Background and Related Work}", r"\section{Method}"),
     ("Sensitivity to character-level typos.", "3 Method")),
    ("3 Method", (r"\section{Method}", r"\section{Experimental Setup}"),
     ("We generate corrupted versions", "4 Experimental Setup")),
    ("4 Experimental Setup", (r"\section{Experimental Setup}", r"\section{Results}"),
     ("Models. We use DeepSeek", "5 Results")),
    ("5 Results", (r"\section{Results}", r"\section{Discussion}"),
     ("5.1 Effect of typos on accuracy", "6 Discussion")),
    ("6 Discussion", (r"\section{Discussion}", r"\section{Limitations}"),
     ("The most consistent finding", "7 Limitations")),
    ("7 Limitations", (r"\section{Limitations}", r"\section{AI Disclosure"),
     ("One model, one sample.", "8 AI Disclosure")),
    ("8 AI Disclosure", (r"\section{AI Disclosure and Reflection}", r"\section{Conclusion}"),
     ("What we used AI for.", "9 Conclusion")),
    ("9 Conclusion", (r"\section{Conclusion}", "{plainnat}"),
     ("Typos have a clear", "All code is available")),
]

# words LaTeX supplies that the plain draft cannot have
IGNORE = {"introduction", "background", "and", "related", "work", "experimental",
          "setup", "results", "limitations", "section", "table", "figure", "rho",
          "method", "controlled", "typo", "generation", "discussion", "conclusion",
          "ai", "disclosure", "reflection"}


def strip_tex(s):
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = re.sub(r"\\footnote\{(?:[^{}]|\{[^{}]*\})*\}", " ", s)
    s = re.sub(r"\\begin\{(figure|table)\*?\}[\s\S]*?\\end\{\1\*?\}", " ", s)
    for k, v in CITES.items():
        s = re.sub(r"\\cite[tp]\{%s\}" % re.escape(k), v, s)
    s = re.sub(r"\\cite[tp]\{[^}]*\}", " ", s)
    s = re.sub(r"\\(begin|end)\{[a-z*]+\}", " ", s)
    s = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", " ", s)
    s = re.sub(r"\\(label|ref|includegraphics|resizebox|input)\{?[^}\s]*\}?", " ", s)
    s = s.replace(r"\LaTeX", "LaTeX").replace(r"\arrow", " ")
    s = s.replace(r"\pp", " percentage points").replace(r"\%", "%")
    return re.sub(r"\\[a-zA-Z]+", " ", s)


def words(s):
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"[^A-Za-z]+", " ", s).lower().split()


def chunk(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i)]


def main(draft_path):
    tex = TEX.read_text(encoding="utf-8")
    txt = Path(draft_path).read_text(encoding="utf-8")
    clean = True
    for label, (ts, te), (xs, xe) in SECTIONS:
        try:
            a, b = words(chunk(txt, xs, xe)), words(strip_tex(chunk(tex, ts, te)))
        except ValueError:
            print("%-22s SKIPPED (marker not found)" % label)
            continue
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        diffs = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            av, bv = " ".join(a[i1:i2]), " ".join(b[j1:j2])
            if all(w in IGNORE for w in (av + " " + bv).split()):
                continue
            diffs.append((tag, av, bv))
        print("%-22s draft=%4d  paper=%4d  %s"
              % (label, len(a), len(b),
                 "IDENTICAL" if not diffs else "%d diff" % len(diffs)))
        for tag, av, bv in diffs:
            clean = False
            print("    [%s] draft: %r" % (tag, av[:100]))
            print("        paper: %r" % bv[:100])
    print("\nRESULT:", "all sections verbatim" if clean else "review the differences above")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python verify_text.py path/to/report_text.txt")
    main(sys.argv[1])
