"""
Build a document-text dataset for biodiversity budget classification.

For each labeled budget row, this script tries to read the matched HWP/PDF
document from the 열린재정 document folder. If no usable document text is
available, it falls back to budget metadata such as 소관명, 분야명, and 세부사업명.

The output always includes clean_document_text and biodiv_label. Extra audit
columns are kept by default so extraction failures can be reviewed later.

Usage:
    python src/build_biodiv_text_dataset.py
    python src/build_biodiv_text_dataset.py --minimal-output
    python src/build_biodiv_text_dataset.py --docs-dir "C:/path/to/docs"
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import unicodedata
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

try:
    import olefile
except ImportError:  # pragma: no cover - environment check
    olefile = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - environment check
    PdfReader = None


BASE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = BASE_DIR / "data" / "사업별결산세출지출현황_2024년도_biodiv_labeled.csv"
DEFAULT_DOCS_DIR = REPO_DIR / "국가생물다양성_열린재정 데이터"
DEFAULT_OUTPUT_CSV = BASE_DIR / "data" / "biodiv_document_text_dataset.csv"
DEFAULT_SUMMARY_JSON = BASE_DIR / "data" / "biodiv_document_text_dataset_summary.json"

LABEL_COL = "biodiv_label"
DOCUMENT_TEXT_COL = "clean_document_text"
MATCHED_FILENAME_COL = "matched_filename"
METADATA_COLUMNS = ["소관명", "분야명", "부문명", "프로그램명", "단위사업명", "세부사업명"]
SUPPORTED_EXTENSIONS = {".hwp", ".pdf"}
PLACEHOLDER_FILENAMES = {"", "x", "X", "-", "없음", "미입력", "na", "n/a", "nan", "none"}

_HWP_PARA_TEXT_TAG = 67
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_DISALLOWED_CHAR_RE = re.compile(
    r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ\s\.,;:!?%()/\-\[\]{}<>&+*'\"“”‘’·,~_=#@○△▲▽▼□■※ㆍ]"
)
_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
_PURPOSE_START_PATTERNS = [
    re.compile(r"(?im)^\s*(?:[0-9]+[.)]\s*)?(?:사업\s*)?목적\s*(?:[·ㆍ.\-/]\s*내용)?"),
    re.compile(r"(?im)^\s*(?:[0-9]+[.)]\s*)?사업\s*목적\s*(?:[·ㆍ.\-/]\s*내용)?"),
    re.compile(r"(?im)^\s*[□○◦ㅇ\-]\s*사업\s*목적\s*(?:[·ㆍ.\-/]\s*내용)?"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HWP/PDF 문서를 읽어 생물다양성 분류용 clean_document_text 데이터셋을 만듭니다."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--limit", type=int, default=0, help="처리할 최대 행 수 (0=전체)")
    parser.add_argument(
        "--minimal-output",
        action="store_true",
        help="clean_document_text와 biodiv_label 두 컬럼만 저장합니다.",
    )
    parser.add_argument(
        "--add-source-prefix",
        action="store_true",
        help="입력 앞에 '입력유형: 문서/메타데이터'를 붙입니다.",
    )
    parser.add_argument(
        "--metadata-fallback-on-missing-anchor",
        action="store_true",
        help="문서는 있으나 사업목적 앵커를 못 찾으면 전체 문서 대신 메타데이터를 사용합니다.",
    )
    return parser.parse_args()


def clean_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


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


def normalize_key(value: object) -> str:
    text = unicodedata.normalize("NFKC", clean_cell(value)).lower()
    return re.sub(r"[^0-9a-z가-힣]+", "", text)


def parse_label(value: object) -> int | None:
    text = clean_cell(value)
    if not text:
        return None
    try:
        label = int(float(text))
    except ValueError:
        return None
    return label if label in {0, 1} else None


def split_matched_filenames(value: object) -> list[str]:
    filenames: list[str] = []
    for part in clean_cell(value).split("|"):
        filename = part.strip()
        if filename and filename not in PLACEHOLDER_FILENAMES:
            filenames.append(filename)
    return filenames


def scan_document_files(docs_dir: Path) -> dict[str, list[Path]]:
    if not docs_dir.exists():
        raise FileNotFoundError(f"문서 폴더를 찾을 수 없습니다: {docs_dir}")

    files_by_name: dict[str, list[Path]] = defaultdict(list)
    for path in docs_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        files_by_name[path.name].append(path.resolve())
    return files_by_name


def filename_variants(filename: str) -> list[str]:
    variants = [filename]
    decoded = unquote(filename)
    if decoded != filename:
        variants.append(decoded)
    return variants


def resolve_document_path(
    filename: str,
    row: dict[str, object],
    files_by_name: dict[str, list[Path]],
) -> tuple[Path | None, str]:
    candidates: list[Path] = []
    for variant in filename_variants(filename):
        candidates.extend(files_by_name.get(variant, []))

    candidates = sorted(set(candidates))
    if not candidates:
        return None, "filename_not_found"
    if len(candidates) == 1:
        return candidates[0], "unique_filename"

    field_key = normalize_key(row.get("분야명", ""))
    field_matches = [path for path in candidates if normalize_key(path.parent.name) == field_key]
    if len(field_matches) == 1:
        return field_matches[0], "filename_and_field_folder"
    if field_matches:
        return sorted(field_matches)[0], "multiple_field_candidates_used_first"
    return candidates[0], "multiple_candidates_used_first"


def _read_hwp_body_sections(path: Path) -> list[tuple[str, bytes]]:
    if olefile is None:
        raise RuntimeError("HWP 추출을 위해 olefile 패키지가 필요합니다.")

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
    if olefile is None:
        raise RuntimeError("HWP 추출을 위해 olefile 패키지가 필요합니다.")

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
    if PdfReader is None:
        raise RuntimeError("PDF 추출을 위해 pypdf 패키지가 필요합니다.")

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return clean_extracted_text("\n".join(pages)), "pdf_pypdf"


def extract_document_text(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".hwp":
        return extract_hwp_text(path)
    if suffix == ".pdf":
        return extract_pdf_text(path)
    raise ValueError(f"지원하지 않는 문서 형식입니다: {path.suffix}")


def trim_to_purpose_section(text: str) -> tuple[str, bool]:
    starts: list[int] = []
    for pattern in _PURPOSE_START_PATTERNS:
        match = pattern.search(text)
        if match:
            starts.append(match.start())

    if not starts:
        return text, False
    return text[min(starts) :], True


def build_metadata_text(row: dict[str, object]) -> str:
    lines = []
    for column in METADATA_COLUMNS:
        value = clean_cell(row.get(column, ""))
        if value:
            lines.append(f"{column}: {value}")
    return clean_extracted_text("\n".join(lines))


def with_source_prefix(text: str, source: str, enabled: bool) -> str:
    if not enabled or not text:
        return text
    label = "문서" if source == "document" else "메타데이터"
    return f"입력유형: {label}\n{text}"


def build_document_text_for_row(
    row: dict[str, object],
    docs_dir: Path,
    files_by_name: dict[str, list[Path]],
    add_source_prefix: bool,
    metadata_fallback_on_missing_anchor: bool,
) -> dict[str, object]:
    filenames = split_matched_filenames(row.get(MATCHED_FILENAME_COL, ""))
    extracted_texts: list[str] = []
    resolved_paths: list[Path] = []
    resolution_statuses: list[str] = []
    extract_methods: list[str] = []
    extract_errors: list[str] = []

    if not filenames:
        status = "no_matched_filename"
    else:
        status = "document_empty"

    for filename in filenames:
        path, resolution_status = resolve_document_path(filename, row, files_by_name)
        resolution_statuses.append(f"{filename}:{resolution_status}")
        if path is None:
            status = "missing_file"
            continue

        resolved_paths.append(path)
        try:
            text, method = extract_document_text(path)
            extract_methods.append(f"{path.name}:{method}")
            if text:
                extracted_texts.append(text)
                status = "document_ok"
        except Exception as exc:  # pragma: no cover - file level error path
            extract_errors.append(f"{path.name}: {exc}")
            status = "extract_error"

    combined_text = clean_extracted_text("\n\n".join(extracted_texts))
    clean_document_text = ""
    text_source = "metadata"
    purpose_anchor_found = False

    if combined_text:
        trimmed_text, purpose_anchor_found = trim_to_purpose_section(combined_text)
        trimmed_text = clean_extracted_text(trimmed_text)
        if purpose_anchor_found or not metadata_fallback_on_missing_anchor:
            clean_document_text = trimmed_text
            text_source = "document"
            status = "document_ok" if purpose_anchor_found else "document_full_text_no_anchor"

    if not clean_document_text:
        clean_document_text = build_metadata_text(row)
        text_source = "metadata"
        status = f"{status}_metadata_fallback"

    clean_document_text = with_source_prefix(clean_document_text, text_source, add_source_prefix)

    return {
        DOCUMENT_TEXT_COL: clean_document_text,
        "text_source": text_source,
        "document_status": status,
        "purpose_anchor_found": purpose_anchor_found,
        "resolved_paths": " | ".join(str(path) for path in resolved_paths),
        "relative_paths": " | ".join(safe_relative_path(path, docs_dir) for path in resolved_paths),
        "resolution_statuses": " | ".join(resolution_statuses),
        "extract_methods": " | ".join(extract_methods),
        "extract_errors": " | ".join(extract_errors),
        "clean_text_char_count": len(clean_document_text),
        "clean_text_word_count": count_words(clean_document_text),
    }


def safe_relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_rows(
    input_rows: list[dict[str, object]],
    docs_dir: Path,
    files_by_name: dict[str, list[Path]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = len(input_rows)

    for index, row in enumerate(input_rows, start=1):
        label = parse_label(row.get(LABEL_COL, ""))
        if label is None:
            continue

        if index == 1 or index == total or index % 50 == 0:
            print(f"[{index}/{total}] processing {clean_cell(row.get(MATCHED_FILENAME_COL, ''))}")

        text_info = build_document_text_for_row(
            row=row,
            docs_dir=docs_dir,
            files_by_name=files_by_name,
            add_source_prefix=args.add_source_prefix,
            metadata_fallback_on_missing_anchor=args.metadata_fallback_on_missing_anchor,
        )

        output_row = {
            "No.": row.get("No.", ""),
            MATCHED_FILENAME_COL: row.get(MATCHED_FILENAME_COL, ""),
            LABEL_COL: label,
            **{column: row.get(column, "") for column in METADATA_COLUMNS},
            **text_info,
        }
        rows.append(output_row)

    return rows


def read_csv_rows(path: Path, limit: int) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    if limit > 0:
        return rows[:limit]
    return rows


def write_csv_rows(path: Path, rows: list[dict[str, object]], minimal_output: bool) -> None:
    if minimal_output:
        fieldnames = [DOCUMENT_TEXT_COL, LABEL_COL]
    else:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_csv, rows, args.minimal_output)

    summary = {
        "output_csv": str(output_csv),
        "row_count": len(rows),
        "label_counts": dict(Counter(row[LABEL_COL] for row in rows)),
        "text_source_counts": dict(Counter(row["text_source"] for row in rows)),
        "document_status_counts": dict(Counter(row["document_status"] for row in rows)),
        "purpose_anchor_counts": dict(Counter(str(row["purpose_anchor_found"]) for row in rows)),
    }

    summary_json = args.summary_json.resolve()
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved dataset: {output_csv}")
    print(f"Saved summary: {summary_json}")


def run(args: argparse.Namespace) -> int:
    input_csv = args.input_csv.resolve()
    docs_dir = args.docs_dir.resolve()

    input_rows = read_csv_rows(input_csv, args.limit)
    files_by_name = scan_document_files(docs_dir)
    rows = build_rows(input_rows, docs_dir, files_by_name, args)
    write_outputs(rows, args)
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
