"""Shared helpers for semantic pattern construction."""

from __future__ import annotations

import re

_LETTER_CLASS = r"[^\W\d_]"


def build_token_boundary_pattern(text: str) -> re.Pattern[str] | None:
    """Return a regex that respects token boundaries for ``text``.

    The pattern prevents partial matches against alphabetic neighbours while still
    allowing digits or punctuation to sit adjacent to the token. If ``text`` is empty
    after trimming, ``None`` is returned so callers can fall back to simpler checks.
    """

    trimmed = text.strip()
    if not trimmed:
        return None
    escaped = re.escape(trimmed)
    leading_alnum = trimmed[0].isalnum()
    trailing_alnum = trimmed[-1].isalnum()
    if leading_alnum and trailing_alnum:
        return re.compile(rf"(?<!{_LETTER_CLASS}){escaped}(?!{_LETTER_CLASS})", re.UNICODE)
    return re.compile(escaped)


__all__ = ["build_token_boundary_pattern"]
