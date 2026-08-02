from __future__ import annotations

import re
import unicodedata

from ftfy import fix_text
from unidecode import unidecode

_SPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(text: str) -> str:
    fixed = fix_text(text).replace("\xa0", " ")
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFC", fixed)).strip()


def ascii_words(text: str) -> str:
    normalized = unidecode(normalize_text(text)).lower()
    return _SPACE_RE.sub(" ", normalized).strip()


def ascii_compact(text: str) -> str:
    return _NON_ALNUM_RE.sub("", ascii_words(text))
