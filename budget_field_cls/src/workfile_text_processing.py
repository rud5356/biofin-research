from __future__ import annotations

import re
from typing import Iterable

from document_text import clean_extracted_text


_SECTION_SEPARATOR = r"[\.\u00b7\u318d]?"
_SECTION_START_PATTERNS = [
    re.compile(r"(?im)^\s*1\)\s*\uc0ac\uc5c5\ubaa9\uc801\s*" + _SECTION_SEPARATOR + r"\s*\ub0b4\uc6a9"),
    re.compile(r"(?im)^\s*4\.\s*\uc0ac\uc5c5\ubaa9\uc801\s*" + _SECTION_SEPARATOR + r"\s*\ub0b4\uc6a9"),
    re.compile(r"(?im)^\s*4\.\s*\uc0ac\uc5c5\ubaa9\uc801"),
    re.compile(r"(?im)^\s*\uc0ac\uc5c5\ubaa9\uc801\s*" + _SECTION_SEPARATOR + r"\s*\ub0b4\uc6a9"),
    re.compile(r"(?im)^\s*\uc0ac\uc5c5\ubaa9\uc801"),
]

_DROP_LINE_PATTERNS = [
    re.compile(r"(?im)^\s*[0-9]+\.\s*\uc0ac\uc5c5 \ucf54\ub4dc \uc815\ubcf4\s*$"),
    re.compile(r"(?im)^\s*\u25a1\s*\uc0ac\uc5c5 \ucf54\ub4dc \uc815\ubcf4\s*$"),
    re.compile(r"(?im)^\s*[0-9]+\.\s*\uc0ac\uc5c5 \uc9c0\uc6d0 \ud615\ud0dc \ubc0f \uc9c0\uc6d0\uc728.*$"),
    re.compile(r"(?im)^\s*\u25a1\s*\uc0ac\uc5c5 \uc9c0\uc6d0 \ud615\ud0dc \ubc0f \uc9c0\uc6d0\uc728.*$"),
    re.compile(r"(?im)^\s*\u25a1\s*\uc0ac\uc5c5 \ub2f4\ub2f9\uc790\s*$"),
    re.compile(r"(?im)^\s*\uac00\.\s*\uc608\uc0b0 \ucd1d\uad04\ud45c.*$"),
    re.compile(r"(?im)^\s*\ub098\.\s*\uc0ac\uc5c5\uc124\uba85\uc790\ub8cc\s*$"),
    re.compile(r"(?im)^\s*\uad6c\ubd84\s*$"),
    re.compile(
        r"(?im)^\s*(\ud68c\uacc4|\uc18c\uad00|\uc2e4\uad6d\(\uae30\uad00\)|\uacc4\uc815|\ubd84\uc57c|\ubd80\ubb38|\ucf54\ub4dc|\uba85\uce6d)\s*$"
    ),
]

_TOKEN_RE = re.compile(r"[0-9A-Za-z\uac00-\ud7a3]+")
_VALUE_JOINER = r"[\s\.,\u00b7\u318d\-_()/]*"


def _normalize_metadata_value(raw_value: object) -> str:
    value = str(raw_value or "").strip()
    if value.lower() in {"", "nan", "none"}:
        return ""
    return value


def _find_section_start(text: str) -> tuple[int | None, str]:
    matches: list[int] = []
    for pattern in _SECTION_START_PATTERNS:
        match = pattern.search(text)
        if match:
            matches.append(match.start())

    if not matches:
        return None, "full_text_fallback"

    return min(matches), "purpose_section"


def _drop_header_block(text: str) -> tuple[str, str]:
    start_index, method = _find_section_start(text)
    if start_index is None:
        return text, method
    return text[start_index:], method


def _remove_structured_header_lines(text: str) -> str:
    kept_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept_lines.append("")
            continue
        if any(pattern.search(stripped) for pattern in _DROP_LINE_PATTERNS):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def _build_flexible_value_pattern(value: str) -> re.Pattern[str] | None:
    tokens = _TOKEN_RE.findall(value or "")
    if len("".join(tokens)) < 2:
        return None
    pattern = _VALUE_JOINER.join(re.escape(token) for token in tokens)
    return re.compile(pattern, re.IGNORECASE)


def mask_metadata_values(text: str, values: Iterable[object]) -> tuple[str, int]:
    masked_text = text
    replacements = 0

    for raw_value in values:
        value = _normalize_metadata_value(raw_value)
        if not value:
            continue
        pattern = _build_flexible_value_pattern(value)
        if pattern is None:
            continue
        masked_text, count = pattern.subn(" ", masked_text)
        replacements += count

    return masked_text, replacements


def build_model_text(raw_text: str, metadata: dict[str, object]) -> tuple[str, str]:
    if not str(raw_text or "").strip():
        return "", "empty"

    candidate_text, method = _drop_header_block(str(raw_text))
    candidate_text = _remove_structured_header_lines(candidate_text)
    candidate_text, masked_count = mask_metadata_values(
        candidate_text,
        [
            metadata.get("label", ""),
            metadata.get("budget_field_name", ""),
        ],
    )
    candidate_text = clean_extracted_text(candidate_text)

    if not candidate_text:
        fallback_text, masked_count = mask_metadata_values(
            str(raw_text),
            [
                metadata.get("label", ""),
                metadata.get("budget_field_name", ""),
            ],
        )
        fallback_text = clean_extracted_text(_remove_structured_header_lines(fallback_text))
        if fallback_text:
            suffix = "+masked" if masked_count else ""
            return fallback_text, f"fallback_full_text{suffix}"
        return "", "empty_after_preprocess"

    suffix = "+masked" if masked_count else ""
    return candidate_text, f"{method}{suffix}"
