"""Shared helpers for semantic pattern construction."""

from __future__ import annotations

from typing import Pattern

try:  # pragma: no cover - regex optional for runtime precision
    import regex as _regex
except ImportError:  # pragma: no cover
    _regex = None
    import re as _re
else:  # pragma: no cover
    _re = _regex

if _regex is not None:
    _LETTER_CLASS = r"\p{L}"
else:
    _LETTER_CLASS = r"[^\W\d_]"


def build_token_boundary_pattern(text: str) -> Pattern[str] | None:
    """Return a regex that respects token boundaries for ``text``.

    The pattern prevents partial matches against alphabetic neighbours while still
    allowing digits or punctuation to sit adjacent to the token. If ``text`` is empty
    after trimming, ``None`` is returned so callers can fall back to simpler checks.
    """

    trimmed = text.strip()
    if not trimmed:
        return None
    escaped = _re.escape(trimmed)
    leading_alnum = trimmed[0].isalnum()
    trailing_alnum = trimmed[-1].isalnum()
    if leading_alnum and trailing_alnum:
        return _re.compile(rf"(?<!{_LETTER_CLASS}){escaped}(?!{_LETTER_CLASS})")
    return _re.compile(escaped)


def extract_letter_tokens(text: str) -> list[str]:
    """Return contiguous alphabetic tokens using the active regex backend."""

    pattern = r"\p{L}+" if _regex is not None else r"[^\W\d_]+"
    return _re.findall(pattern, text)


__all__ = ["build_token_boundary_pattern", "extract_letter_tokens"]
