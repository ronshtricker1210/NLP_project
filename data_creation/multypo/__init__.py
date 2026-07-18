"""Small local keyboard-aware typo generator used by typo_pipeline."""

from __future__ import annotations

import random
from typing import Dict, Iterable, Tuple


_KEY_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
_ADJACENT_KEYS = {}
for row_index, row in enumerate(_KEY_ROWS):
    for column_index, key in enumerate(row):
        neighbours = set()
        for delta in (-1, 1):
            other_column = column_index + delta
            if 0 <= other_column < len(row):
                neighbours.add(row[other_column])
        for other_row_index in (row_index - 1, row_index + 1):
            if 0 <= other_row_index < len(_KEY_ROWS):
                other_row = _KEY_ROWS[other_row_index]
                for other_column in (column_index - 1, column_index, column_index + 1):
                    if 0 <= other_column < len(other_row):
                        neighbours.add(other_row[other_column])
        _ADJACENT_KEYS[key] = tuple(sorted(neighbours))


class MultiTypoGenerator:
    """Generate one typo at a time using simple QWERTY-aware operations."""

    def __init__(
        self,
        language: str = "english",
        use_excluding_set: bool = True,
        typo_distribution: Dict[str, float] | None = None,
    ) -> None:
        self.language = language
        self.typo_distribution = typo_distribution or {}
        self.ignore_set = _english_ignore_set() if use_excluding_set else set()

    def apply_single_typo(self, word: str, typo_type: str) -> Tuple[str, bool]:
        if typo_type == "delete":
            return self._delete(word)
        if typo_type == "insert":
            return self._insert(word)
        if typo_type == "replace":
            return self._replace(word)
        if typo_type == "transpose":
            return self._transpose(word)
        raise ValueError(f"unsupported typo type: {typo_type}")

    def _delete(self, word: str) -> Tuple[str, bool]:
        if len(word) < 2:
            return word, False
        index = random.randrange(len(word))
        return word[:index] + word[index + 1 :], True

    def _insert(self, word: str) -> Tuple[str, bool]:
        if not word:
            return word, False
        index = random.randrange(len(word) + 1)
        base = word[index - 1] if index > 0 else word[0]
        inserted = self._nearby_or_random_letter(base)
        return word[:index] + inserted + word[index:], True

    def _replace(self, word: str) -> Tuple[str, bool]:
        candidates = [i for i, char in enumerate(word) if char.lower() in _ADJACENT_KEYS]
        if not candidates:
            return word, False
        index = random.choice(candidates)
        replacement = random.choice(_ADJACENT_KEYS[word[index].lower()])
        replacement = _match_case(replacement, word[index])
        return word[:index] + replacement + word[index + 1 :], True

    def _transpose(self, word: str) -> Tuple[str, bool]:
        if len(word) < 2:
            return word, False
        candidates = [i for i in range(len(word) - 1) if word[i] != word[i + 1]]
        if not candidates:
            return word, False
        index = random.choice(candidates)
        chars = list(word)
        chars[index], chars[index + 1] = chars[index + 1], chars[index]
        return "".join(chars), True

    def _nearby_or_random_letter(self, char: str) -> str:
        lower = char.lower()
        if lower in _ADJACENT_KEYS:
            letter = random.choice(_ADJACENT_KEYS[lower])
        else:
            letter = random.choice("abcdefghijklmnopqrstuvwxyz")
        return _match_case(letter, char)


def _match_case(new_char: str, original_char: str) -> str:
    return new_char.upper() if original_char.isupper() else new_char


def _english_ignore_set() -> set[str]:
    return {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "if",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
    }


__all__ = ["MultiTypoGenerator"]