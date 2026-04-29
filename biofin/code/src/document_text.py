from __future__ import annotations

import re
import struct
import unicodedata
import zlib
from pathlib import Path

import olefile
from pypdf import PdfReader


_HWP_PARA_TEXT_TAG = 67
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_DISALLOWED_CHAR_RE = re.compile(
    r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ\s\.,;:!?%()/\-\[\]{}<>&+*'\"“”‘’·,~_=#@○△▲▽▼□■※ㆍ]"
)
_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def clean_extracted_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _CONTROL_CHAR_RE.sub(" ", normalized)
    normalized = _DISALLOWED_CHAR_RE.sub(" ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    normalized = _MULTI_NEWLINE_RE.sub("\n\n", normalized)
    return normalized.strip()


def count_words(text: str) -> int:
    return len(_TOKEN_RE.findall(text or ""))


def _read_hwp_body_sections(path: Path) -> list[tuple[str, bytes]]:
    with olefile.OleFileIO(str(path)) as ole:
        header = ole.openstream("FileHeader").read()
        flags = struct.unpack("<I", header[36:40])[0]
        is_compressed = bool(flags & 1)

        sections = [
            entry
            for entry in ole.listdir()
            if len(entry) == 2 and entry[0] == "BodyText" and entry[1].startswith("Section")
        ]
        sections = sorted(sections, key=lambda entry: int(entry[1].replace("Section", "")))

        result: list[tuple[str, bytes]] = []
        for entry in sections:
            stream_name = "/".join(entry)
            payload = ole.openstream(stream_name).read()
            if is_compressed:
                payload = zlib.decompress(payload, -15)
            result.append((stream_name, payload))
        return result


def _extract_text_from_hwp_section(section_bytes: bytes) -> str:
    fragments: list[str] = []
    offset = 0

    while offset < len(section_bytes):
        header = struct.unpack_from("<I", section_bytes, offset)[0]
        tag_id = header & 0x3FF
        size = (header >> 20) & 0xFFF
        offset += 4

        if size == 0xFFF:
            size = struct.unpack_from("<I", section_bytes, offset)[0]
            offset += 4

        payload = section_bytes[offset : offset + size]
        if tag_id == _HWP_PARA_TEXT_TAG and payload:
            fragments.append(payload.decode("utf-16le", errors="ignore"))
        offset += size

    return "\n".join(fragment for fragment in fragments if fragment.strip())


def _extract_hwp_preview_text(path: Path) -> str:
    with olefile.OleFileIO(str(path)) as ole:
        if not ole.exists("PrvText"):
            return ""
        preview = ole.openstream("PrvText").read()
        for encoding in ("utf-16le", "utf-8", "cp949"):
            try:
                return preview.decode(encoding, errors="ignore")
            except UnicodeDecodeError:
                continue
    return ""


def extract_hwp_text(path: Path) -> tuple[str, str]:
    sections = _read_hwp_body_sections(path)
    fragments = [_extract_text_from_hwp_section(payload) for _, payload in sections]
    combined = clean_extracted_text("\n".join(fragment for fragment in fragments if fragment.strip()))
    if combined:
        return combined, "hwp_bodytext"

    preview = clean_extracted_text(_extract_hwp_preview_text(path))
    if preview:
        return preview, "hwp_preview"

    return "", "hwp_empty"


def extract_pdf_text(path: Path) -> tuple[str, str]:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return clean_extracted_text("\n".join(pages)), "pdf_pypdf"


def extract_document_text(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".hwp":
        return extract_hwp_text(path)
    if suffix == ".pdf":
        return extract_pdf_text(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")
