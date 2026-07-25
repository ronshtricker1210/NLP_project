"""Shared repair-side lexical banks (typo-noticing + repair words) and the
notice-type classifier. Kept in one place so the ACTIVE modules (lexical_grid,
real_word_effect) don't depend on the retired repair_behavior_old_version.py.

Self-doubt banks (second_guess / uncertainty) live in self_doubt.py.
"""
import re

# Typo-NOTICING bank — the model explicitly flags/interprets the corrupted text.
# Words chosen from the clean-vs-typo discrimination test ("typo" is the star:
# 1.2% of clean traces vs 44% of heavy-typo traces).
TYPO_NOTICING = {
    "typo":                 r"\btypos?\b",
    "misspell":             r"\bmis-?spell(?:ed|ing|s|ings)?\b",
    "probably/might mean":  r"\b(?:probably|might|could|must|likely|maybe)\s+(?:be\s+)?(?:a\s+typo|means?|meant)\b",
    "they/user mean":       r"\b(?:they|the user|it|author|question)\s+(?:mean|means|meant)\b",
    "assume/interpret":     r"\b(?:assum(?:e|es|ed|ing)|interpret(?:s|ed|ing)?)\b",
    "should be/say/read":   r"\bshould\s+(?:be|say|read|probably\s+be)\b",
    "doesn't make sense":   r"\b(?:does(?:n'?t| not)|didn'?t)\s+make\s+sense\b",
    "seems like typo/error":r"\b(?:seems|looks)\s+like\s+(?:a\s+)?(?:typo|misspelling|error|mistake)\b",
    "strange/weird/odd":    r"\b(?:strange|weird|odd|garbled)\b",
    "meant to be/say":      r"\bmeant\s+to\s+(?:be|say|read|write)\b",
    "error/mistake in":     r"\b(?:error|mistake|typos?)\s+in\s+the\b",
}
TN_COMPILED = {k: re.compile(v, re.I) for k, v in TYPO_NOTICING.items()}

# Active repair / correction language (distinct from merely noticing a problem).
REPAIR_MARKERS = {
    "read/treat it as":  r"\b(?:read|take|treat|interpret)\s+(?:it|this|that|the word|the problem)\s+as\b",
    "should be/say":     r"\bshould\s+(?:be|say|read|probably\s+be)\b",
    "meant/supposed to": r"\b(?:meant|supposed|intended)\s+to\s+(?:be|say|read|write|mean)\b",
    "assume/interpret":  r"\b(?:i(?:'?ll| will)?\s+)?(?:assum(?:e|es|ed|ing)|interpret(?:s|ed|ing)?)\b",
    "typo for / means":  r"\b(?:typo for|(?:probably|likely|must)\s+means?|means?\s+to\s+say)\b",
    "correct/fix typo":  r"\b(?:correct(?:ing|ed)?|fix(?:ing|ed)?)\s+(?:the\s+)?(?:typo|spelling|word|error|mistake)\b",
    "rewrite/rephrase":  r"\b(?:rewrit\w+|rephras\w+|reinterpret\w+)\b",
}
RM_COMPILED = {k: re.compile(v, re.I) for k, v in REPAIR_MARKERS.items()}

# A trace "explicitly notices" when it uses strong typo-flagging language (not just
# any interpretive verb), so ordinary "assume"/"means" in clean math doesn't count.
STRONG_NOTICE = {"typo", "misspell", "seems like typo/error", "doesn't make sense",
                 "error/mistake in", "meant to be/say", "strange/weird/odd"}


def classify_trace(reasoning):
    """Return (notice_type, n_notice, n_repair) for one reasoning string."""
    strong = sum(len(TN_COMPILED[k].findall(reasoning)) for k in STRONG_NOTICE)
    n_notice = sum(len(rx.findall(reasoning)) for rx in TN_COMPILED.values())
    n_repair = sum(len(rx.findall(reasoning)) for rx in RM_COMPILED.values())
    notice_type = "explicit" if strong > 0 else "silent"
    return notice_type, n_notice, n_repair
