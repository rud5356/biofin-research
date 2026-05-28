"""
HWP/PDF 문서에서 텍스트를 추출해 생물다양성 분류용 데이터셋을 만드는 스크립트.

입력: 생물다양성 라벨이 붙은 CSV (biodiv_labeled.csv)
      + 사업별 HWP/PDF 원문 문서들이 있는 폴더

출력: clean_document_text 컬럼이 추가된 데이터셋 CSV
      + 통계 요약 JSON

문서 텍스트 추출 우선순위:
    1. HWP BodyText 스트림 (본문 전체)
    2. HWP PrvText 스트림 (미리보기 텍스트, 본문 추출 실패 시)
    3. PDF 전체 페이지 텍스트
    4. 메타데이터 컬럼 (문서를 찾지 못한 경우 fallback)

HWP 파일 구조:
    HWP는 OLE2(Compound Document) 형식으로 저장됩니다.
    olefile 라이브러리로 OLE 구조를 열고,
    BodyText/SectionN 스트림에서 본문 텍스트를 추출합니다.
    스트림 데이터는 zlib 압축되어 있을 수 있으며,
    파라그래프 텍스트는 UTF-16 LE 인코딩입니다.
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
    import olefile  # HWP 파일 파싱용 OLE2 라이브러리
except ImportError:
    olefile = None

try:
    from pypdf import PdfReader  # PDF 텍스트 추출 라이브러리
except ImportError:
    PdfReader = None

from config import (
    BIODIV_LABELED_CSV,
    BIODIV_TEXT_DATASET_CSV,
    BIODIV_TEXT_DATASET_SUMMARY_JSON,
    DOCUMENT_TEXT_COLUMN,
    LABEL_COLUMN,
    MATCHED_FILENAME_COLUMN,
    METADATA_COLUMNS,
    SOURCE_DOCS_DIR,
)


# ─── 기본 경로 ────────────────────────────────────────────────────────────────
DEFAULT_INPUT_CSV    = BIODIV_LABELED_CSV
DEFAULT_DOCS_DIR     = SOURCE_DOCS_DIR
DEFAULT_OUTPUT_CSV   = BIODIV_TEXT_DATASET_CSV
DEFAULT_SUMMARY_JSON = BIODIV_TEXT_DATASET_SUMMARY_JSON

# 처리 가능한 문서 확장자
SUPPORTED_EXTENSIONS = {".hwp", ".pdf"}

# 파일명으로 인식하지 않을 placeholder 값들
PLACEHOLDER_FILENAMES = {"", "x", "X", "-", "없음", "미입력", "na", "n/a", "nan", "none"}

# HWP 파라그래프 텍스트 레코드 태그 ID (HWP 5.0 규격)
_HWP_PARA_TEXT_TAG = 67

# 텍스트 정제용 정규표현식 패턴들
_CONTROL_CHAR_RE   = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # 제어 문자 (탭/줄바꿈 제외)
_DISALLOWED_CHAR_RE = re.compile(                                # 허용 문자 외 제거
    r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ\s\.,;:!?%()/\-\[\]{}<>&+*'\"""''·,~_=#@○△▲▽▼□■※ㆍ]"
)
_WHITESPACE_RE    = re.compile(r"[ \t\f\v]+")   # 연속 공백 → 단일 공백
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")        # 3개 이상 줄바꿈 → 2개로 축소
_TOKEN_RE         = re.compile(r"[0-9A-Za-z가-힣]+")  # 단어 토큰 (단어 수 계산용)

# '사업목적' 섹션 시작을 찾는 패턴 (다양한 표기 방식 처리)
_PURPOSE_START_PATTERNS = [
    re.compile(r"(?im)^\s*(?:[0-9]+[.)]\s*)?(?:사업\s*)?목적\s*(?:[·ㆍ.\-/]\s*내용)?"),
    re.compile(r"(?im)^\s*[□○◦ㅇ\-]\s*사업\s*목적\s*(?:[·ㆍ.\-/]\s*내용)?"),
]


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="HWP/PDF 문서를 읽어 생물다양성 분류용 clean_document_text 데이터셋을 만듭니다."
    )
    parser.add_argument("--input-csv",   type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--docs-dir",    type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--output-csv",  type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-json",type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--limit",       type=int,  default=0, help="처리할 최대 행 수 (0=전체)")
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
    """셀 값을 문자열로 변환하고 NaN/None은 빈 문자열로 반환합니다."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def clean_extracted_text(text: str) -> str:
    """
    추출된 원시 텍스트를 정제합니다.

    처리 순서:
    1. NFKC 유니코드 정규화 (전각 문자 → 반각, 호환 한자 통일)
    2. 캐리지 리턴(\r) → 줄바꿈(\n) 통일
    3. 제어 문자 제거
    4. 허용 문자 외 특수문자 제거
    5. 줄바꿈 주변 공백 정리
    6. 연속 공백 → 단일 공백
    7. 3개 이상 빈 줄 → 2개로 축소
    """
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _CONTROL_CHAR_RE.sub(" ", normalized)
    normalized = _DISALLOWED_CHAR_RE.sub(" ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    normalized = _MULTI_NEWLINE_RE.sub("\n\n", normalized)
    return normalized.strip()


def count_words(text: str) -> int:
    """텍스트에서 한글·영문·숫자 토큰의 수를 셉니다."""
    return len(_TOKEN_RE.findall(text or ""))


def normalize_key(value: object) -> str:
    """
    매칭 키 생성을 위해 값을 소문자로 변환하고 비알파벳 문자를 제거합니다.

    예: "제주특별자치도 (환경)" → "제주특별자치도환경"
    """
    text = unicodedata.normalize("NFKC", clean_cell(value)).lower()
    return re.sub(r"[^0-9a-z가-힣]+", "", text)


def parse_label(value: object) -> int | None:
    """라벨 컬럼 값을 0 또는 1로 변환하고, 유효하지 않으면 None을 반환합니다."""
    text = clean_cell(value)
    if not text:
        return None
    try:
        label = int(float(text))
    except ValueError:
        return None
    return label if label in {0, 1} else None


def split_matched_filenames(value: object) -> list[str]:
    """
    '|' 구분자로 연결된 파일명 문자열을 개별 파일명 목록으로 분리합니다.

    placeholder 값(없음, x, nan 등)은 제외합니다.
    예: "사업설명.hwp|첨부자료.pdf" → ["사업설명.hwp", "첨부자료.pdf"]
    """
    filenames: list[str] = []
    for part in clean_cell(value).split("|"):
        filename = part.strip()
        if filename and filename not in PLACEHOLDER_FILENAMES:
            filenames.append(filename)
    return filenames


def scan_document_files(docs_dir: Path) -> dict[str, list[Path]]:
    """
    문서 폴더를 재귀 탐색해 파일명 → 절대경로 목록 인덱스를 만듭니다.

    동일 파일명이 여러 하위 폴더에 존재할 수 있으므로 list[Path]를 값으로 씁니다.
    """
    if not docs_dir.exists():
        raise FileNotFoundError(f"문서 폴더를 찾을 수 없습니다: {docs_dir}")

    files_by_name: dict[str, list[Path]] = defaultdict(list)
    for path in docs_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        files_by_name[path.name].append(path.resolve())
    return files_by_name


def filename_variants(filename: str) -> list[str]:
    """
    URL 인코딩된 파일명 변형도 함께 반환합니다.

    예: "%ED%95%9C%EA%B8%80.hwp" → ["%ED%95%9C%EA%B8%80.hwp", "한글.hwp"]
    """
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
    """
    파일명으로 실제 문서 경로를 찾습니다.

    1단계: 파일명 완전 일치 탐색
    2단계: 후보가 여러 개이면 분야명 폴더로 좁힘
    3단계: 그래도 여러 개이면 알파벳 순서 첫 번째 사용

    반환값: (경로 또는 None, 해결 상태 문자열)
    """
    candidates: list[Path] = []
    for variant in filename_variants(filename):
        candidates.extend(files_by_name.get(variant, []))

    candidates = sorted(set(candidates))
    if not candidates:
        return None, "filename_not_found"
    if len(candidates) == 1:
        return candidates[0], "unique_filename"

    # 분야명으로 폴더 좁히기
    field_key     = normalize_key(row.get("분야명", ""))
    field_matches = [path for path in candidates if normalize_key(path.parent.name) == field_key]
    if len(field_matches) == 1:
        return field_matches[0], "filename_and_field_folder"
    if field_matches:
        return sorted(field_matches)[0], "multiple_field_candidates_used_first"
    return candidates[0], "multiple_candidates_used_first"


def _read_hwp_body_sections(path: Path) -> list[tuple[str, bytes]]:
    """
    HWP 파일에서 BodyText/SectionN 스트림 데이터를 읽습니다.

    HWP OLE 구조:
    - FileHeader: 파일 헤더 (압축 여부 등 플래그 포함)
    - BodyText/Section0, Section1...: 실제 본문 데이터

    FileHeader offset 36~40: 4바이트 플래그 (비트 0 = 압축 여부)
    압축된 경우 zlib.decompress(..., -15)로 해제합니다.
    """
    if olefile is None:
        raise RuntimeError("HWP 추출을 위해 olefile 패키지가 필요합니다.")

    with olefile.OleFileIO(str(path)) as ole:
        # 파일 헤더에서 압축 플래그 읽기 (little-endian 4바이트 정수)
        header         = ole.openstream("FileHeader").read()
        flags          = struct.unpack("<I", header[36:40])[0]
        is_compressed  = bool(flags & 1)  # 비트 0이 1이면 압축됨

        # BodyText 하위의 Section* 스트림 목록 수집
        sections = [
            entry
            for entry in ole.listdir()
            if len(entry) == 2 and entry[0] == "BodyText" and entry[1].startswith("Section")
        ]
        # 섹션 번호 순으로 정렬 (Section0, Section1, ...)
        sections = sorted(sections, key=lambda entry: int(entry[1].replace("Section", "")))

        result: list[tuple[str, bytes]] = []
        for entry in sections:
            stream_name = "/".join(entry)
            payload     = ole.openstream(stream_name).read()
            if is_compressed:
                # zlib raw deflate: wbits=-15 (헤더 없는 raw 스트림)
                payload = zlib.decompress(payload, -15)
            result.append((stream_name, payload))
        return result


def _extract_text_from_hwp_section(section_bytes: bytes) -> str:
    """
    HWP 섹션 바이너리 데이터에서 파라그래프 텍스트를 추출합니다.

    HWP 레코드 구조:
    - 헤더(4바이트): [태그 ID(10비트) | 레벨(10비트) | 크기(12비트)]
    - 크기가 0xFFF이면 다음 4바이트가 실제 크기
    - 태그 ID 67 (HWPTAG_PARA_TEXT): 파라그래프 텍스트
    - 텍스트는 UTF-16 LE 인코딩
    """
    fragments: list[str] = []
    offset = 0

    while offset < len(section_bytes):
        # 레코드 헤더 읽기: little-endian 4바이트
        header = struct.unpack_from("<I", section_bytes, offset)[0]
        tag_id = header & 0x3FF        # 하위 10비트: 태그 ID
        size   = (header >> 20) & 0xFFF  # 상위 12비트: 페이로드 크기
        offset += 4

        # 크기가 0xFFF (4095)이면 다음 4바이트에 실제 크기가 있음 (확장 크기)
        if size == 0xFFF:
            size    = struct.unpack_from("<I", section_bytes, offset)[0]
            offset += 4

        payload = section_bytes[offset : offset + size]
        # 태그 67 = HWPTAG_PARA_TEXT: 한 문단의 텍스트
        if tag_id == _HWP_PARA_TEXT_TAG and payload:
            fragments.append(payload.decode("utf-16le", errors="ignore"))
        offset += size

    return "\n".join(fragment for fragment in fragments if fragment.strip())


def _extract_hwp_preview_text(path: Path) -> str:
    """
    HWP 파일의 PrvText 스트림에서 미리보기 텍스트를 추출합니다.

    PrvText는 HWP 파일에 포함된 텍스트 미리보기로,
    BodyText 추출에 실패했을 때 fallback으로 사용합니다.
    UTF-16 LE 인코딩입니다.
    """
    if olefile is None:
        raise RuntimeError("HWP 추출을 위해 olefile 패키지가 필요합니다.")

    with olefile.OleFileIO(str(path)) as ole:
        if not ole.exists("PrvText"):
            return ""
        return ole.openstream("PrvText").read().decode("utf-16le", errors="ignore")


def extract_hwp_text(path: Path) -> tuple[str, str]:
    """
    HWP 파일에서 본문 텍스트를 추출합니다.

    우선순위:
    1. BodyText 스트림 → 전체 본문
    2. PrvText 스트림 → 미리보기 (본문 추출 실패 시)
    3. 둘 다 없으면 빈 문자열

    반환값: (정제된 텍스트, 출처 표시)
    """
    sections  = _read_hwp_body_sections(path)
    fragments = [_extract_text_from_hwp_section(payload) for _, payload in sections]
    combined  = clean_extracted_text("\n".join(fragment for fragment in fragments if fragment.strip()))
    if combined:
        return combined, "hwp_bodytext"

    # BodyText가 비어있으면 PrvText(미리보기)로 fallback
    preview = clean_extracted_text(_extract_hwp_preview_text(path))
    if preview:
        return preview, "hwp_preview"

    return "", "hwp_empty"


def extract_pdf_text(path: Path) -> tuple[str, str]:
    """
    PDF 파일에서 전체 페이지 텍스트를 추출합니다.

    pypdf.PdfReader를 사용해 페이지별 텍스트를 연결합니다.
    반환값: (정제된 텍스트, "pdf_pypdf")
    """
    if PdfReader is None:
        raise RuntimeError("PDF 추출을 위해 pypdf 패키지가 필요합니다.")

    reader = PdfReader(str(path))
    pages  = [page.extract_text() or "" for page in reader.pages]
    return clean_extracted_text("\n".join(pages)), "pdf_pypdf"


def extract_document_text(path: Path) -> tuple[str, str]:
    """확장자에 따라 적합한 추출 함수를 호출합니다."""
    suffix = path.suffix.lower()
    if suffix == ".hwp":
        return extract_hwp_text(path)
    if suffix == ".pdf":
        return extract_pdf_text(path)
    raise ValueError(f"지원하지 않는 문서 형식입니다: {path.suffix}")


def trim_to_purpose_section(text: str) -> tuple[str, bool]:
    """
    문서에서 '사업목적' 섹션 이후 텍스트만 잘라냅니다.

    사업목적/목적 등의 제목을 앵커(anchor)로 삼아
    그 이전의 사업 번호, 목차, 표지 등의 노이즈를 제거합니다.

    반환값: (잘라낸 텍스트, 앵커 발견 여부)
    앵커를 찾지 못하면 원본 전체를 반환합니다.
    """
    starts: list[int] = []
    for pattern in _PURPOSE_START_PATTERNS:
        match = pattern.search(text)
        if match:
            starts.append(match.start())

    if not starts:
        return text, False
    # 여러 패턴이 매칭된 경우 가장 앞에 있는 위치부터 자르기
    return text[min(starts):], True


def build_metadata_text(row: dict[str, object]) -> str:
    """
    메타데이터 컬럼들을 '{컬럼명}: {값}' 형식으로 이어붙입니다.

    문서를 찾지 못하거나 텍스트 추출에 실패했을 때
    모델 입력의 fallback으로 사용됩니다.
    """
    lines = []
    for column in METADATA_COLUMNS:
        value = clean_cell(row.get(column, ""))
        if value:
            lines.append(f"{column}: {value}")
    return clean_extracted_text("\n".join(lines))


def with_source_prefix(text: str, source: str, enabled: bool) -> str:
    """--add-source-prefix 옵션이 활성화된 경우 텍스트 앞에 입력 유형을 명시합니다."""
    if not enabled or not text:
        return text
    label = "문서" if source == "document" else "메타데이터"
    return f"입력유형: {label}\n{text}"


def safe_relative_path(path: Path, root: Path) -> str:
    """경로가 root 아래에 없으면 절대경로를 그대로 반환합니다."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_document_text_for_row(
    row: dict[str, object],
    docs_dir: Path,
    files_by_name: dict[str, list[Path]],
    add_source_prefix: bool,
    metadata_fallback_on_missing_anchor: bool,
) -> dict[str, object]:
    """
    한 행에 대한 문서 텍스트를 구성하고 추출 메타데이터를 반환합니다.

    처리 흐름:
    1. matched_filename 컬럼에서 파일명 목록 파싱
    2. 각 파일명으로 실제 경로 찾기
    3. HWP/PDF 텍스트 추출
    4. 사업목적 섹션 이후로 자르기
    5. 텍스트가 없으면 메타데이터 fallback
    """
    filenames = split_matched_filenames(row.get(MATCHED_FILENAME_COLUMN, ""))
    extracted_texts:      list[str]  = []
    resolved_paths:       list[Path] = []
    resolution_statuses:  list[str]  = []
    extract_methods:      list[str]  = []
    extract_errors:       list[str]  = []

    status = "no_matched_filename" if not filenames else "document_empty"

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
        except Exception as exc:
            extract_errors.append(f"{path.name}: {exc}")
            status = "extract_error"

    combined_text         = clean_extracted_text("\n\n".join(extracted_texts))
    clean_document_text   = ""
    text_source           = "metadata"
    purpose_anchor_found  = False

    if combined_text:
        trimmed_text, purpose_anchor_found = trim_to_purpose_section(combined_text)
        trimmed_text = clean_extracted_text(trimmed_text)
        # 앵커를 찾았거나 metadata_fallback_on_missing_anchor가 꺼져 있으면 문서 텍스트 사용
        if purpose_anchor_found or not metadata_fallback_on_missing_anchor:
            clean_document_text = trimmed_text
            text_source = "document"
            status = "document_ok" if purpose_anchor_found else "document_full_text_no_anchor"

    # 문서 텍스트가 없으면 메타데이터로 대체
    if not clean_document_text:
        clean_document_text = build_metadata_text(row)
        text_source = "metadata"
        status = f"{status}_metadata_fallback"

    clean_document_text = with_source_prefix(clean_document_text, text_source, add_source_prefix)

    return {
        DOCUMENT_TEXT_COLUMN:    clean_document_text,
        "text_source":           text_source,
        "document_status":       status,
        "purpose_anchor_found":  purpose_anchor_found,
        "resolved_paths":        " | ".join(str(path) for path in resolved_paths),
        "relative_paths":        " | ".join(safe_relative_path(path, docs_dir) for path in resolved_paths),
        "resolution_statuses":   " | ".join(resolution_statuses),
        "extract_methods":       " | ".join(extract_methods),
        "extract_errors":        " | ".join(extract_errors),
        "clean_text_char_count": len(clean_document_text),
        "clean_text_word_count": count_words(clean_document_text),
    }


def build_rows(
    input_rows: list[dict[str, object]],
    docs_dir: Path,
    files_by_name: dict[str, list[Path]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """
    입력 행 목록 전체를 처리해 출력 행 목록을 반환합니다.

    유효한 라벨(0 또는 1)이 없는 행은 건너뜁니다.
    50행마다 진행 상황을 출력합니다.
    """
    rows:  list[dict[str, object]] = []
    total = len(input_rows)

    for index, row in enumerate(input_rows, start=1):
        label = parse_label(row.get(LABEL_COLUMN, ""))
        if label is None:
            continue

        # 첫 행, 마지막 행, 50행마다 진행 상황 출력
        if index == 1 or index == total or index % 50 == 0:
            print(f"[{index}/{total}] processing {clean_cell(row.get(MATCHED_FILENAME_COLUMN, ''))}")

        text_info = build_document_text_for_row(
            row=row,
            docs_dir=docs_dir,
            files_by_name=files_by_name,
            add_source_prefix=args.add_source_prefix,
            metadata_fallback_on_missing_anchor=args.metadata_fallback_on_missing_anchor,
        )

        output_row = {
            "No.": row.get("No.", ""),
            MATCHED_FILENAME_COLUMN: row.get(MATCHED_FILENAME_COLUMN, ""),
            LABEL_COLUMN: label,
            **{column: row.get(column, "") for column in METADATA_COLUMNS},
            **text_info,
        }
        rows.append(output_row)

    return rows


def read_csv_rows(path: Path, limit: int) -> list[dict[str, object]]:
    """CSV 파일을 읽어 행 딕셔너리 목록으로 반환합니다. limit > 0이면 앞 N행만 반환합니다."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows   = list(reader)
    if limit > 0:
        return rows[:limit]
    return rows


def write_csv_rows(path: Path, rows: list[dict[str, object]], minimal_output: bool) -> None:
    """
    행 목록을 CSV로 저장합니다.

    minimal_output=True: clean_document_text, label 두 컬럼만 저장
    minimal_output=False: 모든 컬럼 저장
    extrasaction="ignore": 필드명에 없는 키는 무시
    """
    if minimal_output:
        fieldnames = [DOCUMENT_TEXT_COLUMN, LABEL_COLUMN]
    else:
        # 행 전체에서 컬럼 이름을 수집 (dict.fromkeys로 순서 유지 + 중복 제거)
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    """데이터셋 CSV와 통계 요약 JSON을 저장합니다."""
    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_csv, rows, args.minimal_output)

    summary = {
        "output_csv":              str(output_csv),
        "row_count":               len(rows),
        "label_counts":            dict(Counter(row[LABEL_COLUMN] for row in rows)),
        "text_source_counts":      dict(Counter(row["text_source"] for row in rows)),
        "document_status_counts":  dict(Counter(row["document_status"] for row in rows)),
        "purpose_anchor_counts":   dict(Counter(str(row["purpose_anchor_found"]) for row in rows)),
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
    """데이터셋 빌드 파이프라인 실행."""
    input_csv = args.input_csv.resolve()
    docs_dir  = args.docs_dir.resolve()

    input_rows    = read_csv_rows(input_csv, args.limit)
    files_by_name = scan_document_files(docs_dir)
    rows          = build_rows(input_rows, docs_dir, files_by_name, args)
    write_outputs(rows, args)
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
