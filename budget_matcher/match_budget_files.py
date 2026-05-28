"""
예산 CSV의 각 행을 '열린재정 데이터' 폴더의 HWP/PDF 파일과 매칭하는 스크립트.

매칭 전략:
    1. 정확 매칭(exact): 분야명 + 소관명 + 세부사업명이 파일명과 완전히 일치
    2. 정규화 매칭(normalized): NFKC 정규화 + 특수문자/공백 제거 후 일치
    3. 후보 검토(review_candidate): 유사도 점수로 상위 N개 후보 제안
    4. 매칭 불가(unmatched): 후보도 없는 경우

파일명 형식 (열린재정 폴더):
    파일은 '소관명_세부사업명.hwp' 또는 '번호_소관명_세부사업명.hwp' 형식입니다.
    '_'를 구분자로 소관명과 사업명을 분리합니다.

유사도 계산:
    SequenceMatcher.ratio(): 두 문자열의 편집 거리 기반 유사도 (0~1)
    token_overlap(): 공통 토큰 / 전체 토큰 (Jaccard 유사도)
    두 지표를 가중합해 최종 점수(0~100)를 산출합니다.

파일명 자동 변경:
    정확 매칭된 파일은 '매칭번호_원본파일명' 형식으로 이름을 변경합니다.
    이후 다른 스크립트에서 번호로 파일을 찾을 수 있게 됩니다.

사용 예:
    python match_budget_files.py
    python match_budget_files.py --field 환경
    python match_budget_files.py --budget-root C:\\데이터폴더 --output-dir C:\\출력폴더
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote


# ─── 상수 ─────────────────────────────────────────────────────────────────────
# CSV 읽기 시도할 인코딩 순서
CSV_ENCODINGS = ("utf-8-sig", "cp949", "utf-8")

# 처리할 파일 확장자
DEFAULT_EXTENSIONS = (".hwp", ".pdf")

# 각 미매칭 행에 제안할 최대 후보 수
TOP_CANDIDATES = 5

# 알려진 예산 분야명 목록 (폴더명과 매칭에 사용)
EXPECTED_FIELD_NAMES = (
    "공공질서및 안전",
    "과학기술",
    "교육",
    "교통및물류",
    "국방",
    "국토및지역개발",
    "농림수산",
    "문화및관광",
    "보건",
    "사회복지",
    "산업/중소기업및에너지",
    "예비비",
    "일반/지방행정",
    "통신",
    "통일외교",
    "환경",
)

# R&D 표기 방식 통일 (URL 인코딩, 한글 표기 등)
_RND_ALIASES = {
    "r&d":     "rnd",
    "r %26 d": "rnd",
    "r %26d":  "rnd",
    "r and d": "rnd",
}


@dataclass(frozen=True)
class BudgetRow:
    """
    예산 CSV의 행 하나를 나타냅니다.

    frozen=True: 생성 후 수정 불가 (딕셔너리 키로 사용 가능)
    match_key: (분야명, 소관명, 세부사업명) 정규화 튜플 (매칭 인덱스 키로 사용)
    """
    row_no:       int
    source_no:    str
    fiscal_year:  str
    field_raw:    str    # 분야명 원문
    field_norm:   str    # 분야명 정규화
    ministry_raw: str    # 소관명 원문
    ministry_norm: str   # 소관명 정규화
    project_raw:  str    # 세부사업명 원문
    project_norm: str    # 세부사업명 정규화
    raw:          dict[str, str]  # 원본 CSV 행 전체

    @property
    def match_key(self) -> tuple[str, str, str]:
        return self.field_norm, self.ministry_norm, self.project_norm


@dataclass(frozen=True)
class FileRecord:
    """
    폴더에서 발견된 파일 하나를 나타냅니다.

    파일명에서 소관명과 사업명을 파싱해 매칭 인덱스를 구성합니다.
    """
    field_raw:    str    # 상위 폴더명 (분야명)
    field_norm:   str
    ministry_raw: str    # 파일명에서 파싱한 소관명
    ministry_norm: str
    project_raw:  str    # 파일명에서 파싱한 사업명
    project_norm: str
    filename:     str    # 파일명
    path:         str    # 절대 경로

    @property
    def match_key(self) -> tuple[str, str, str]:
        return self.field_norm, self.ministry_norm, self.project_norm


@dataclass(frozen=True)
class Candidate:
    """
    유사도 매칭 후보 하나를 나타냅니다.

    score: 0~100 범위의 종합 점수 (높을수록 좋음)
    reason: 점수에 기여한 요소 (same_field, same_ministry, token_overlap 등)
    """
    file_record:         FileRecord
    score:               float
    project_similarity:  float   # 세부사업명 문자열 유사도 (0~1)
    ministry_similarity: float   # 소관명 문자열 유사도 (0~1)
    full_similarity:     float   # (소관명+사업명) 합친 유사도 (0~1)
    token_overlap:       float   # 토큰 Jaccard 유사도 (0~1)
    reason:              str


@dataclass
class MatchResults:
    """매칭 결과 분류를 담는 컨테이너."""
    matched_exact:          list[dict[str, object]] = field(default_factory=list)
    matched_normalized:     list[dict[str, object]] = field(default_factory=list)
    review_candidates:      list[dict[str, object]] = field(default_factory=list)
    unmatched_no_candidates: list[dict[str, object]] = field(default_factory=list)
    out_of_scope_rows:      list[dict[str, object]] = field(default_factory=list)
    single_result_rows:     list[dict[str, object]] = field(default_factory=list)
    matched_file_paths:     set[str]                = field(default_factory=set)
    in_scope_rows:          int                     = 0


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="예산 CSV와 열린재정 폴더의 파일을 교차 매칭합니다."
    )
    parser.add_argument(
        "--csv", type=Path, default=find_default_csv(),
        help="예산 CSV 경로. 기본값: 스크립트 폴더의 첫 번째 CSV",
    )
    parser.add_argument(
        "--budget-root", type=Path,
        default=Path(r"C:\Yuna\국가생물다양성_열린재정 데이터"),
        help="분야별 폴더가 있는 열린재정 루트 디렉토리",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="결과 파일 저장 폴더. 기본값: file_match/output/<타임스탬프>",
    )
    parser.add_argument(
        "--extensions", nargs="+", default=list(DEFAULT_EXTENSIONS),
        help="탐색할 파일 확장자. 기본값: .hwp .pdf",
    )
    parser.add_argument(
        "--top-candidates", type=int, default=TOP_CANDIDATES,
        help="미매칭 행마다 제안할 후보 파일 수",
    )
    parser.add_argument(
        "--field", type=str, default=None,
        help="분야명 필터 (예: 농림수산). 지정하면 해당 분야 행만 처리합니다.",
    )
    return parser.parse_args()


def find_default_csv() -> Path:
    """스크립트 폴더에서 첫 번째 CSV 파일을 찾습니다."""
    script_dir = Path(__file__).resolve().parent
    csv_files  = sorted(script_dir.glob("*.csv"))
    if not csv_files:
        return script_dir / "budget.csv"
    return csv_files[0]


def normalize_text(value: str | None) -> str:
    """
    텍스트를 매칭 키로 사용하기 위해 정규화합니다.

    처리 순서:
    1. NFKC 정규화 (전각 → 반각, 공백 통일)
    2. URL 디코딩 (%ED%95%9C → 한)
    3. 말미의 '(숫자)' 제거 (중복 항목 접미사)
    4. 소문자 변환
    5. R&D 표기 통일 (r&d → rnd)
    6. 한글·영문·숫자 외 모든 문자 제거
    """
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = unquote(normalized)
    # '사업명 (2)' 같은 중복 접미사 제거
    normalized = re.sub(r"\s+\((\d+)\)$", "", normalized)
    lowered    = normalized.lower()
    for alias, canonical in _RND_ALIASES.items():
        lowered = lowered.replace(alias, canonical)
    lowered = re.sub(r"[^0-9a-z가-힣]+", "", lowered)
    return lowered


# 정규화된 분야명 → 표준 분야명 역방향 인덱스 (예: "환경" → "환경")
CANONICAL_FIELD_NAME_BY_KEY = {
    normalize_text(field_name): field_name for field_name in EXPECTED_FIELD_NAMES
}


def canonical_field_name(value: str | None) -> str:
    """분야명을 표준 분야명으로 변환합니다. 알 수 없는 분야명은 그대로 반환합니다."""
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", value).strip()
    canonical  = CANONICAL_FIELD_NAME_BY_KEY.get(normalize_text(normalized))
    return canonical if canonical is not None else normalized


def normalize_field_name(value: str | None) -> str:
    """분야명을 표준 분야명으로 변환한 뒤 normalize_text를 적용합니다."""
    return normalize_text(canonical_field_name(value))


def normalize_exact_text(value: str | None) -> str:
    """
    정확 매칭용 정규화: 소문자 변환 없이 공백/특수문자만 정리합니다.

    normalize_text와 달리 대소문자를 보존합니다.
    정확 매칭 인덱스에 사용됩니다.
    """
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", unquote(value)).strip()
    normalized = re.sub(r"\s+\((\d+)\)$", "", normalized)
    return normalized


def tokenize_text(value: str) -> set[str]:
    """
    텍스트를 한글·영문·숫자 토큰 집합으로 변환합니다.

    토큰 Jaccard 유사도(token_overlap) 계산에 사용됩니다.
    """
    decoded = unicodedata.normalize("NFKC", unquote(value or "")).lower()
    decoded = re.sub(r"\s+\((\d+)\)$", "", decoded)
    pieces  = re.findall(r"[0-9a-z가-힣]+", decoded)
    return {piece for piece in pieces if piece}


def similarity(left: str, right: str) -> float:
    """
    두 문자열의 편집 거리 기반 유사도를 반환합니다 (0.0~1.0).

    SequenceMatcher.ratio(): 공통 부분 문자열의 비율
    둘 중 하나가 비어있으면 0.0을 반환합니다.
    """
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def token_overlap(left: str, right: str) -> float:
    """
    두 문자열의 토큰 Jaccard 유사도를 반환합니다 (0.0~1.0).

    Jaccard = |공통 토큰| / |전체 토큰 합집합|
    토큰 순서에 무관하므로 어순이 다른 경우에도 유사도를 측정할 수 있습니다.
    """
    left_tokens  = tokenize_text(left)
    right_tokens = tokenize_text(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def read_budget_rows(csv_path: Path) -> list[BudgetRow]:
    """
    예산 CSV를 읽어 BudgetRow 목록으로 반환합니다.

    여러 인코딩을 순서대로 시도합니다.
    필수 컬럼(분야명, 소관명, 세부사업명)이 없으면 ValueError를 발생시킵니다.
    """
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as handle:
                reader  = csv.DictReader(handle)
                required = {"분야명", "소관명", "세부사업명"}
                header   = set(reader.fieldnames or [])
                missing  = required - header
                if missing:
                    raise ValueError(f"CSV에 필수 컬럼이 없습니다: {', '.join(sorted(missing))}")

                rows: list[BudgetRow] = []
                for index, raw in enumerate(reader, start=1):
                    field_raw    = (raw.get("분야명")    or "").strip()
                    ministry_raw = (raw.get("소관명")    or "").strip()
                    project_raw  = (raw.get("세부사업명") or "").strip()
                    rows.append(BudgetRow(
                        row_no=index,
                        source_no=(raw.get("No.") or "").strip(),
                        fiscal_year=(raw.get("회계연도") or "").strip(),
                        field_raw=field_raw,
                        field_norm=normalize_field_name(field_raw),
                        ministry_raw=ministry_raw,
                        ministry_norm=normalize_text(ministry_raw),
                        project_raw=project_raw,
                        project_norm=normalize_text(project_raw),
                        raw=raw,
                    ))
                return rows
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"CSV 읽기 실패: {csv_path}") from last_error


def parse_file_stem(stem: str) -> tuple[str, str] | None:
    """
    파일명(확장자 제외)에서 소관명과 사업명을 파싱합니다.

    지원 형식:
    - '소관명_세부사업명'
    - '번호_소관명_세부사업명' (앞에 번호 접두사가 붙은 경우)

    '_'가 없거나 소관명/사업명이 비어있으면 None을 반환합니다.
    """
    cleaned = unicodedata.normalize("NFKC", unquote(stem)).strip()
    cleaned = re.sub(r"\s+\((\d+)\)$", "", cleaned)
    # '번호_소관명_사업명' 형식 처리: 번호 부분 제거
    prefixed = re.match(r"^(\d+)_(.+)$", cleaned)
    if prefixed and "_" in prefixed.group(2):
        cleaned = prefixed.group(2)
    if "_" not in cleaned:
        return None
    ministry_raw, project_raw = cleaned.split("_", 1)
    ministry_raw = ministry_raw.strip()
    project_raw  = project_raw.strip()
    if not ministry_raw or not project_raw:
        return None
    return ministry_raw, project_raw


def scan_file_records(
    budget_root: Path,
    extensions: Iterable[str],
) -> tuple[list[FileRecord], list[str], dict[str, list[str]]]:
    """
    열린재정 폴더를 탐색해 FileRecord 목록을 만듭니다.

    반환값:
        records: 파싱에 성공한 파일 목록
        unparsed_paths: '_' 구분자가 없어 파싱에 실패한 파일 경로 목록
        available_field_dirs: 정규화된 분야명 → 실제 폴더명 목록

    폴더 구조 가정:
        budget_root/
            환경/
                환경부_멸종위기종복원사업.hwp
            농림수산/
                해양수산부_어업자원조사.pdf
    """
    normalized_exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    records:              list[FileRecord]          = []
    unparsed_paths:       list[str]                 = []
    available_field_dirs: dict[str, list[str]]      = defaultdict(list)

    if not budget_root.exists():
        raise FileNotFoundError(f"열린재정 루트 폴더가 없습니다: {budget_root}")

    for field_dir in sorted(path for path in budget_root.iterdir() if path.is_dir()):
        field_raw  = field_dir.name
        field_norm = normalize_field_name(field_raw)
        available_field_dirs[field_norm].append(field_raw)

        for file_path in sorted(path for path in field_dir.iterdir() if path.is_file()):
            if file_path.suffix.lower() not in normalized_exts:
                continue
            parsed = parse_file_stem(file_path.stem)
            if parsed is None:
                unparsed_paths.append(str(file_path))
                continue
            ministry_raw, project_raw = parsed
            records.append(FileRecord(
                field_raw=field_raw,
                field_norm=field_norm,
                ministry_raw=ministry_raw,
                ministry_norm=normalize_text(ministry_raw),
                project_raw=project_raw,
                project_norm=normalize_text(project_raw),
                filename=file_path.name,
                path=str(file_path),
            ))

    return records, unparsed_paths, dict(available_field_dirs)


def build_file_indexes(
    file_records: list[FileRecord],
) -> tuple[
    dict[tuple[str, str, str], list[FileRecord]],
    dict[tuple[str, str, str], list[FileRecord]],
    dict[str, list[FileRecord]],
]:
    """
    파일 레코드를 3가지 인덱스로 구성합니다.

    exact_key_index:      (분야, 소관명, 사업명) 원문 기반 → 정확 매칭용
    normalized_key_index: (분야, 소관명, 사업명) 정규화 기반 → 정규화 매칭용
    records_by_field:     분야 정규화키 → 해당 분야 파일 목록 → 후보 탐색 범위 좁히기용
    """
    exact_key_index:      dict[tuple[str, str, str], list[FileRecord]] = defaultdict(list)
    normalized_key_index: dict[tuple[str, str, str], list[FileRecord]] = defaultdict(list)
    records_by_field:     dict[str, list[FileRecord]]                  = defaultdict(list)

    for record in file_records:
        # 정확 매칭: 대소문자/공백 보존 (normalize_exact_text만 적용)
        exact_key_index[(
            record.field_norm,
            normalize_exact_text(record.ministry_raw),
            normalize_exact_text(record.project_raw),
        )].append(record)
        # 정규화 매칭: 소문자 + 특수문자 제거
        normalized_key_index[record.match_key].append(record)
        # 분야별 인덱스: 후보 탐색 시 같은 분야 파일로 범위 좁히기
        records_by_field[record.field_norm].append(record)

    return exact_key_index, normalized_key_index, records_by_field


def score_candidate(row: BudgetRow, file_record: FileRecord) -> Candidate:
    """
    CSV 행과 파일 레코드 간의 종합 유사도 점수를 계산합니다.

    점수 구성 (최대 ~125점):
    - 세부사업명 유사도 × 60  (가장 중요)
    - (소관명+사업명) 합산 유사도 × 20
    - 토큰 Jaccard 유사도 × 20
    - 같은 분야: +10 보너스
    - 같은 소관명: +10 보너스 (다르면 소관명 유사도 × 5)
    """
    same_field    = row.field_norm    == file_record.field_norm
    same_ministry = row.ministry_norm == file_record.ministry_norm

    project_ratio  = similarity(row.project_norm, file_record.project_norm)
    ministry_ratio = similarity(row.ministry_norm, file_record.ministry_norm)
    full_ratio     = similarity(
        row.ministry_norm + row.project_norm,
        file_record.ministry_norm + file_record.project_norm,
    )
    overlap = token_overlap(row.project_raw, file_record.project_raw)

    score  = project_ratio * 60
    score += full_ratio    * 20
    score += overlap       * 20
    if same_field:
        score += 10
    if same_ministry:
        score += 10
    else:
        score += ministry_ratio * 5

    reason_parts: list[str] = []
    if same_field:
        reason_parts.append("same_field")
    if same_ministry:
        reason_parts.append("same_ministry")
    if overlap >= 0.2:
        reason_parts.append("token_overlap")
    if project_ratio >= 0.65:
        reason_parts.append("project_name_close")
    reason = ",".join(reason_parts) if reason_parts else "low_confidence"

    return Candidate(
        file_record=file_record,
        score=round(score, 2),
        project_similarity=round(project_ratio, 4),
        ministry_similarity=round(ministry_ratio, 4),
        full_similarity=round(full_ratio, 4),
        token_overlap=round(overlap, 4),
        reason=reason,
    )


def choose_candidate_pool(
    row: BudgetRow,
    records_by_field: dict[str, list[FileRecord]],
    all_records: list[FileRecord],
) -> tuple[str, list[FileRecord]]:
    """
    후보 탐색 범위를 선택합니다.

    우선순위: 같은 분야+소관명 → 같은 분야만 → 전체 파일
    범위가 좁을수록 의미없는 후보가 줄어들고 임계값도 다르게 적용됩니다.
    """
    same_field_records = records_by_field.get(row.field_norm, [])
    if not same_field_records:
        return "global", all_records

    same_ministry_records = [
        record for record in same_field_records if record.ministry_norm == row.ministry_norm
    ]
    if same_ministry_records:
        return "same_field_same_ministry", same_ministry_records
    return "same_field", same_field_records


def _candidate_passes_threshold(candidate: Candidate, pool_type: str) -> bool:
    """
    후보가 풀 타입에 맞는 최소 유사도 임계값을 통과하는지 확인합니다.

    풀이 좁을수록 (same_field_same_ministry) 임계값을 낮춥니다.
    같은 분야+소관명이면 project_similarity 0.35만 넘어도 후보로 포함합니다.
    전체 풀(global)에서는 높은 기준(0.45 + 0.7)을 적용합니다.
    """
    if pool_type == "same_field_same_ministry":
        return (
            candidate.project_similarity >= 0.35
            or candidate.token_overlap    >= 0.15
            or candidate.full_similarity  >= 0.5
        )
    if pool_type == "same_field":
        return (
            candidate.project_similarity >= 0.45
            or candidate.token_overlap    >= 0.2
            or candidate.full_similarity  >= 0.6
        )
    # global: 오탐(false positive) 방지를 위해 더 엄격한 기준 적용
    return candidate.project_similarity >= 0.45 and candidate.full_similarity >= 0.7


def find_candidates(
    row: BudgetRow,
    records_by_field: dict[str, list[FileRecord]],
    all_records: list[FileRecord],
    top_n: int,
) -> list[Candidate]:
    """
    CSV 행에 대한 상위 N개 유사 파일 후보를 찾습니다.

    풀 선택 → 전체 점수 계산 → 점수 내림차순 정렬 → 임계값 필터 → 상위 N개 반환
    """
    pool_type, pool = choose_candidate_pool(row, records_by_field, all_records)
    scored = [score_candidate(row, file_record) for file_record in pool]
    scored.sort(
        key=lambda item: (
            item.score,
            item.project_similarity,
            item.full_similarity,
            item.file_record.filename,
        ),
        reverse=True,
    )
    return [c for c in scored if _candidate_passes_threshold(c, pool_type)][:top_n]


def ensure_output_dir(output_dir: Path | None, script_dir: Path) -> Path:
    """출력 폴더를 확인하거나 기본 경로로 생성합니다."""
    final_dir = output_dir if output_dir is not None else script_dir / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    return final_dir


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """행 목록을 CSV로 저장합니다. 행이 없으면 빈 파일을 만듭니다."""
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    # dict.fromkeys로 모든 행의 컬럼 이름을 순서 유지하며 수집
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_row(row: BudgetRow) -> dict[str, object]:
    """BudgetRow를 출력용 딕셔너리로 변환합니다."""
    flat: dict[str, object] = {
        "row_no":   row.row_no,
        "source_no": row.source_no,
        "회계연도":  row.fiscal_year,
        "분야명":    row.field_raw,
        "소관명":    row.ministry_raw,
        "세부사업명": row.project_raw,
    }
    flat.update(row.raw)  # 원본 CSV의 모든 컬럼 포함
    return flat


def annotate_result_row(
    data: dict[str, object],
    row: BudgetRow,
    result_status: str,
    result_reason: str = "",
) -> dict[str, object]:
    """결과 행에 canonical_field_name, result_status, result_reason을 추가합니다."""
    data["canonical_field_name"] = canonical_field_name(row.field_raw)
    data["result_status"]        = result_status
    data["result_reason"]        = result_reason
    return data


def build_match_row(
    row: BudgetRow,
    file_record: FileRecord,
    match_type: str,
) -> dict[str, object]:
    """정확/정규화 매칭 성공 행을 생성합니다."""
    data = flatten_row(row)
    data.update({
        "match_type":        match_type,
        "matched_field":     file_record.field_raw,
        "matched_ministry":  file_record.ministry_raw,
        "matched_project":   file_record.project_raw,
        "matched_filename":  file_record.filename,
        "matched_path":      file_record.path,
    })
    return annotate_result_row(data, row, match_type)


def _merge_reason(primary_reason: str, secondary_reason: str) -> str:
    """두 이유 문자열을 ','로 합칩니다. 빈 문자열은 무시합니다."""
    parts = [part for part in (primary_reason, secondary_reason) if part]
    return ",".join(parts)


def build_duplicate_candidates(
    row: BudgetRow,
    file_records: list[FileRecord],
    duplicate_reason: str,
) -> list[Candidate]:
    """
    동일 키에 파일이 여러 개인 경우 (중복 키) 후보 목록을 만듭니다.

    중복 파일을 그대로 통과시키지 않고 검토 대상으로 분류합니다.
    """
    candidates: list[Candidate] = []
    for file_record in sorted(file_records, key=lambda record: (record.filename, record.path)):
        base_candidate = score_candidate(row, file_record)
        candidates.append(Candidate(
            file_record=file_record,
            score=base_candidate.score,
            project_similarity=base_candidate.project_similarity,
            ministry_similarity=base_candidate.ministry_similarity,
            full_similarity=base_candidate.full_similarity,
            token_overlap=base_candidate.token_overlap,
            reason=_merge_reason(duplicate_reason, base_candidate.reason),
        ))
    return candidates


def build_candidate_row(
    row: BudgetRow,
    candidates: list[Candidate],
    top_n: int,
    result_reason: str = "",
) -> dict[str, object]:
    """
    검토 필요 행을 생성합니다.

    상위 top_n개 후보의 점수와 파일 정보를 candidate_1~candidate_N 컬럼에 기록합니다.
    후보가 top_n보다 적으면 남은 칸은 빈 문자열로 채웁니다.
    """
    data = flatten_row(row)
    data["candidate_count"] = len(candidates)
    for index in range(top_n):
        prefix = f"candidate_{index + 1}"
        c      = candidates[index] if index < len(candidates) else None

        def cv(attr: str, obj: object = c) -> object:
            """후보가 없으면 빈 문자열을 반환하는 헬퍼 함수."""
            return "" if obj is None else getattr(obj, attr)

        data.update({
            f"{prefix}_score":               cv("score"),
            f"{prefix}_reason":              cv("reason"),
            f"{prefix}_project_similarity":  cv("project_similarity"),
            f"{prefix}_ministry_similarity": cv("ministry_similarity"),
            f"{prefix}_full_similarity":     cv("full_similarity"),
            f"{prefix}_token_overlap":       cv("token_overlap"),
            f"{prefix}_field":    "" if c is None else c.file_record.field_raw,
            f"{prefix}_ministry": "" if c is None else c.file_record.ministry_raw,
            f"{prefix}_project":  "" if c is None else c.file_record.project_raw,
            f"{prefix}_filename": "" if c is None else c.file_record.filename,
            f"{prefix}_path":     "" if c is None else c.file_record.path,
        })
    return annotate_result_row(data, row, "review_candidate", result_reason)


def build_no_candidate_row(row: BudgetRow) -> dict[str, object]:
    """후보를 찾지 못한 행을 생성합니다."""
    data = flatten_row(row)
    return annotate_result_row(data, row, "unmatched_no_candidate", "no_candidate_found")


def build_out_of_scope_row(row: BudgetRow) -> dict[str, object]:
    """해당 분야 폴더가 아직 없어 비교 대상에서 제외된 행을 생성합니다."""
    data = flatten_row(row)
    return annotate_result_row(data, row, "out_of_scope_field", "field_folder_not_found_yet")


def build_file_only_row(file_record: FileRecord) -> dict[str, object]:
    """CSV와 매칭되지 않은 파일 전용 행을 생성합니다."""
    return {
        "분야명":    file_record.field_raw,
        "소관명":    file_record.ministry_raw,
        "세부사업명": file_record.project_raw,
        "filename":  file_record.filename,
        "path":      file_record.path,
    }


def build_match_workfile_row(
    result_row: dict[str, object],
    original_fieldnames: list[str],
) -> dict[str, object]:
    """
    워크파일용 행을 생성합니다.

    원본 CSV 컬럼을 유지하면서 오른쪽에 최소한의 작업용 컬럼만 추가합니다.
    정확/정규화 매칭된 경우에만 matched_filename/matched_path를 채웁니다.
    review_note: 수동 검토 메모용 빈 컬럼
    """
    auto_matched = str(result_row.get("result_status", "")) in {"exact", "normalized"}
    base_row: dict[str, object] = {
        "row_no":    result_row.get("row_no",    ""),
        "source_no": result_row.get("source_no", ""),
    }
    for fieldname in original_fieldnames:
        base_row[fieldname] = result_row.get(fieldname, "")
    return {
        **base_row,
        "auto_match_status":  result_row.get("result_status", ""),
        "auto_match_reason":  result_row.get("result_reason", ""),
        "matched_filename":   result_row.get("matched_filename", "") if auto_matched else "",
        "matched_path":       result_row.get("matched_path",     "") if auto_matched else "",
        "review_note":        "",
    }


def build_prefixed_filename(match_number: str, filename: str) -> str:
    """
    파일명 앞에 매칭 번호 접두사를 붙입니다.

    이미 동일 번호로 시작하면 그대로 반환합니다.
    '번호_소관명_사업명' 형식에서 번호 부분을 교체합니다.

    예: ('42', '환경부_멸종위기종.hwp') → '42_환경부_멸종위기종.hwp'
    """
    if not match_number or not filename:
        return filename
    normalized_filename = filename
    # 이미 '번호_' 접두사가 있으면 번호 부분만 제거
    prefixed = re.match(r"^(\d+)_(.+)$", filename)
    if prefixed and "_" in prefixed.group(2):
        normalized_filename = prefixed.group(2)
    return (
        normalized_filename
        if normalized_filename.startswith(f"{match_number}_")
        else f"{match_number}_{normalized_filename}"
    )


def match_number_from_row_dict(row: dict[str, object]) -> str:
    """행 딕셔너리에서 매칭 번호(source_no 또는 row_no)를 추출합니다."""
    return str(row.get("source_no", "") or row.get("row_no", "")).strip()


def update_row_paths_for_rename(
    row: dict[str, object],
    rename_map: dict[str, tuple[str, str]],
) -> None:
    """
    파일명 변경 맵을 적용해 행의 경로/파일명 컬럼을 업데이트합니다.

    matched_path 및 '*_path' 패턴의 모든 컬럼을 업데이트합니다.
    파일명이 변경되면 결과 파일의 경로도 일관되게 반영됩니다.
    """
    matched_path = str(row.get("matched_path", "")).strip()
    if matched_path in rename_map:
        new_path, new_filename = rename_map[matched_path]
        row["matched_path"]     = new_path
        row["matched_filename"] = new_filename

    for key, value in list(row.items()):
        if not key.endswith("_path"):
            continue
        current_path = str(value).strip()
        if current_path not in rename_map:
            continue
        new_path, new_filename = rename_map[current_path]
        row[key] = new_path
        filename_key = f"{key[:-5]}_filename"  # '_path' → '_filename'
        if filename_key in row:
            row[filename_key] = new_filename


def apply_auto_match_file_renames(results: MatchResults) -> list[dict[str, object]]:
    """
    정확 매칭된 파일을 '매칭번호_원본파일명' 형식으로 이름을 변경합니다.

    처리 로직:
    - 같은 파일이 여러 번호에 매칭된 경우(중복): 이름 변경 건너뜀
    - 대상 파일이 없는 경우: missing_source_file 기록
    - 이미 올바른 이름인 경우: already_prefixed 기록
    - 대상 이름의 파일이 이미 있는 경우: skipped_target_exists 기록
    - 정상 변경: renamed 기록

    rename_map으로 변경 전후 경로를 추적해 결과 파일의 경로도 업데이트합니다.
    """
    matched_rows = results.matched_exact
    # 파일 경로 → 해당 경로를 참조하는 매칭 행 목록
    rows_by_path: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in matched_rows:
        matched_path = str(row.get("matched_path", "")).strip()
        if matched_path:
            rows_by_path[matched_path].append(row)

    rename_map:  dict[str, tuple[str, str]] = {}
    rename_logs: list[dict[str, object]]    = []

    for old_path in sorted(rows_by_path):
        related_rows    = rows_by_path[old_path]
        old_path_obj    = Path(old_path)
        old_filename    = old_path_obj.name
        related_numbers = sorted(
            {number for number in (match_number_from_row_dict(row) for row in related_rows) if number}
        )
        related_row_nos = sorted({str(row.get("row_no", "")).strip() for row in related_rows if row.get("row_no", "")})

        log_row: dict[str, object] = {
            "old_path":           old_path,
            "old_filename":       old_filename,
            "related_source_nos": " | ".join(related_numbers),
            "related_row_nos":    " | ".join(related_row_nos),
            "matched_row_count":  len(related_rows),
            "status":             "",
            "new_path":           old_path,
            "new_filename":       old_filename,
            "note":               "",
        }

        if not old_path_obj.exists():
            log_row["status"] = "missing_source_file"
            log_row["note"]   = "matched_path_not_found"
            rename_logs.append(log_row)
            continue

        # 하나의 파일이 여러 번호에 매칭되면 어느 번호를 붙여야 할지 알 수 없음
        if len(related_numbers) != 1:
            log_row["status"] = "skipped_multiple_match_numbers"
            log_row["note"]   = "same_file_matched_to_multiple_numbers"
            rename_logs.append(log_row)
            continue

        match_number  = related_numbers[0]
        new_filename  = build_prefixed_filename(match_number, old_filename)
        new_path_obj  = old_path_obj.with_name(new_filename)

        if new_path_obj == old_path_obj:
            # 이미 올바른 이름이면 실제 변경 없이 맵에만 기록
            log_row["status"]      = "already_prefixed"
            log_row["new_path"]    = str(new_path_obj)
            log_row["new_filename"] = new_filename
            rename_map[old_path]   = (str(new_path_obj), new_filename)
            rename_logs.append(log_row)
            continue

        if new_path_obj.exists():
            log_row["status"]      = "skipped_target_exists"
            log_row["note"]        = "target_filename_already_exists"
            log_row["new_path"]    = str(new_path_obj)
            log_row["new_filename"] = new_filename
            rename_logs.append(log_row)
            continue

        # 실제 파일명 변경
        old_path_obj.rename(new_path_obj)
        log_row["status"]    = "renamed"
        log_row["new_path"]  = str(new_path_obj)
        log_row["new_filename"] = new_filename
        rename_map[old_path] = (str(new_path_obj), new_filename)
        rename_logs.append(log_row)

    # 변경된 경로를 single_result_rows에도 반영
    for row in results.single_result_rows:
        update_row_paths_for_rename(row, rename_map)

    return rename_logs


def build_field_inventory_rows(
    available_field_dirs: dict[str, list[str]],
    file_records: list[FileRecord],
) -> list[dict[str, object]]:
    """
    분야별 폴더 존재 여부와 파일 수를 요약합니다.

    EXPECTED_FIELD_NAMES에 있는 분야는 폴더가 없어도 행이 생성됩니다.
    예상 목록에 없는 폴더는 is_expected_field=N으로 별도 표시됩니다.
    """
    file_count_by_field: dict[str, int] = defaultdict(int)
    for file_record in file_records:
        file_count_by_field[file_record.field_norm] += 1

    rows: list[dict[str, object]] = []
    expected_keys = {normalize_text(field_name) for field_name in EXPECTED_FIELD_NAMES}

    for field_name in EXPECTED_FIELD_NAMES:
        field_key   = normalize_text(field_name)
        actual_dirs = sorted(available_field_dirs.get(field_key, []))
        rows.append({
            "canonical_field_name": field_name,
            "field_key":            field_key,
            "folder_present":       "Y" if actual_dirs else "N",
            "actual_folder_names":  " | ".join(actual_dirs),
            "parsed_file_count":    file_count_by_field.get(field_key, 0),
            "is_expected_field":    "Y",
        })

    # 예상 목록에 없는 추가 폴더도 기록
    for field_key in sorted(set(available_field_dirs) - expected_keys):
        actual_dirs = sorted(available_field_dirs[field_key])
        rows.append({
            "canonical_field_name": actual_dirs[0] if actual_dirs else field_key,
            "field_key":            field_key,
            "folder_present":       "Y",
            "actual_folder_names":  " | ".join(actual_dirs),
            "parsed_file_count":    file_count_by_field.get(field_key, 0),
            "is_expected_field":    "N",
        })

    return rows


def build_output_guide_text() -> str:
    """결과 파일 설명 텍스트(한글)를 생성합니다."""
    lines = [
        "결과 파일 설명",
        "",
        "아래 파일들은 output 폴더에 함께 생성됩니다.",
        "",
        "1. matched_exact.csv",
        "파일명이 정확히 일치해서 자동 매칭된 항목입니다.",
        "",
        "2. matched_normalized.csv",
        "공백, 특수문자, 표기 차이를 정규화한 뒤 자동 매칭된 항목입니다.",
        "",
        "3. review_candidates.csv",
        "자동 확정은 못 했지만 검토할 후보 파일이 있는 항목입니다.",
        "candidate_1 ~ candidate_N 컬럼을 보고 수동으로 판단하면 됩니다.",
        "",
        "4. unmatched_no_candidates.csv",
        "후보 파일도 찾지 못한 항목입니다.",
        "",
        "5. csv_out_of_scope_fields.csv",
        "현재 시점에 해당 분야 폴더가 아직 없어서 비교 대상에서 빠진 항목입니다.",
        "",
        "6. file_only_unmatched.csv",
        "열린재정 폴더에는 있지만 CSV 쪽과 자동 매칭되지 않은 파일 목록입니다.",
        "",
        "7. field_inventory.csv",
        "현재 감지된 상위 분야 폴더 현황과 분야별 파싱 파일 수를 보여줍니다.",
        "",
        "8. single_result.csv",
        "CSV 전체 행을 기준으로 exact, normalized, review_candidate, unmatched_no_candidate, out_of_scope_field 상태를 모두 붙인 통합 결과입니다.",
        "",
        "9. match_workfile.csv",
        "원본 사업별결산세출지출현황 행을 그대로 유지하고, 오른쪽에 작업용 컬럼만 최소한으로 붙인 파일입니다.",
        "auto_match_status 컬럼으로 자동 매칭 여부와 상태를 확인합니다.",
        "matched_filename / matched_path 컬럼은 자동 매칭된 경우 자동으로 채워지고,",
        "exact로 자동 매칭된 파일만 실제 파일명도 '매칭번호_원본파일명' 형식으로 변경됩니다.",
        "normalized 매칭 파일은 실제 파일명을 바꾸지 않습니다.",
        "자동 매칭이 아닌 경우에는 같은 칸에 직접 파일명과 경로를 입력하면 됩니다.",
        "review_note 컬럼은 수동 검토 메모용입니다.",
        "",
        "10. auto_rename_log.csv",
        "exact 자동 매칭 파일의 실제 rename 결과를 기록합니다.",
        "renamed, already_prefixed, skipped_multiple_match_numbers, skipped_target_exists 같은 상태를 확인할 수 있습니다.",
        "",
        "11. summary.json",
        "이번 실행의 전체 건수와 요약 통계를 담은 파일입니다.",
        "",
        "12. unparsed_files.txt",
        "파일명 규칙을 해석하지 못한 파일이 있으면 이 목록에 기록됩니다.",
        "",
        "권장 확인 순서",
        "1. match_workfile.csv",
        "2. auto_rename_log.csv",
        "3. review_candidates.csv",
        "4. unmatched_no_candidates.csv",
        "5. field_inventory.csv",
        "6. summary.json",
    ]
    return "\n".join(lines) + "\n"


def _classify_rows(
    budget_rows: list[BudgetRow],
    available_field_norms: set[str],
    exact_index:      dict[tuple[str, str, str], list[FileRecord]],
    normalized_index: dict[tuple[str, str, str], list[FileRecord]],
    records_by_field: dict[str, list[FileRecord]],
    all_records:      list[FileRecord],
    top_candidates:   int,
) -> MatchResults:
    """
    각 CSV 행을 4가지 결과로 분류합니다.

    분류 순서:
    1. 해당 분야 폴더 없음 → out_of_scope
    2. 정확 매칭 (중복이면 → review_candidate)
    3. 정규화 매칭 (중복이면 → review_candidate)
    4. 유사도 후보 검색
       - 후보 있음 → review_candidates
       - 후보 없음 → unmatched_no_candidates
    """
    results = MatchResults()

    for row in budget_rows:
        # 분야 폴더가 없으면 비교 대상이 없으므로 범위 밖으로 분류
        if row.field_norm not in available_field_norms:
            row_dict = build_out_of_scope_row(row)
            results.out_of_scope_rows.append(row_dict)
            results.single_result_rows.append(row_dict)
            continue

        results.in_scope_rows += 1

        # 정확 매칭 시도 (normalize_exact_text로 공백/특수문자만 정리)
        exact_matches = exact_index.get((
            row.field_norm,
            normalize_exact_text(row.ministry_raw),
            normalize_exact_text(row.project_raw),
        ), [])
        if len(exact_matches) > 1:
            # 동일 키에 파일이 여러 개: 검토 필요
            row_dict = build_candidate_row(
                row, build_duplicate_candidates(row, exact_matches, "duplicate_exact_key"),
                top_candidates, "duplicate_exact_key",
            )
            results.review_candidates.append(row_dict)
            results.single_result_rows.append(row_dict)
            continue
        if exact_matches:
            row_dict = build_match_row(row, exact_matches[0], "exact")
            results.matched_exact.append(row_dict)
            results.single_result_rows.append(row_dict)
            results.matched_file_paths.add(exact_matches[0].path)
            continue

        # 정규화 매칭 시도 (소문자 + 특수문자 제거)
        normalized_matches = normalized_index.get(row.match_key, [])
        if len(normalized_matches) > 1:
            row_dict = build_candidate_row(
                row, build_duplicate_candidates(row, normalized_matches, "duplicate_normalized_key"),
                top_candidates, "duplicate_normalized_key",
            )
            results.review_candidates.append(row_dict)
            results.single_result_rows.append(row_dict)
            continue
        if normalized_matches:
            row_dict = build_match_row(row, normalized_matches[0], "normalized")
            results.matched_normalized.append(row_dict)
            results.single_result_rows.append(row_dict)
            results.matched_file_paths.add(normalized_matches[0].path)
            continue

        # 유사도 기반 후보 탐색
        candidates = find_candidates(
            row=row,
            records_by_field=records_by_field,
            all_records=all_records,
            top_n=top_candidates,
        )
        if candidates:
            row_dict = build_candidate_row(row, candidates, top_candidates)
            results.review_candidates.append(row_dict)
        else:
            row_dict = build_no_candidate_row(row)
            results.unmatched_no_candidates.append(row_dict)
        results.single_result_rows.append(row_dict)

    return results


def run(args: argparse.Namespace) -> int:
    """매칭 파이프라인 실행."""
    script_dir  = Path(__file__).resolve().parent
    csv_path    = args.csv.resolve()
    budget_root = args.budget_root.resolve()
    output_dir  = ensure_output_dir(args.output_dir, script_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_path}")

    # CSV 읽기 및 분야명 필터 적용
    budget_rows = read_budget_rows(csv_path)
    if args.field:
        filter_norm = normalize_field_name(args.field)
        budget_rows = [row for row in budget_rows if row.field_norm == filter_norm]
        if not budget_rows:
            print(f"ERROR: '{args.field}' 분야에 해당하는 행이 없습니다.", file=sys.stderr)
            return 1
    original_fieldnames = list(budget_rows[0].raw.keys()) if budget_rows else []

    # 파일 탐색 및 인덱스 구성
    file_records, unparsed_paths, available_field_dirs = scan_file_records(budget_root, args.extensions)
    exact_index, normalized_index, records_by_field    = build_file_indexes(file_records)
    field_inventory_rows = build_field_inventory_rows(available_field_dirs, file_records)

    # 매칭 실행
    results          = _classify_rows(
        budget_rows=budget_rows,
        available_field_norms=set(available_field_dirs),
        exact_index=exact_index,
        normalized_index=normalized_index,
        records_by_field=records_by_field,
        all_records=file_records,
        top_candidates=args.top_candidates,
    )
    auto_rename_logs = apply_auto_match_file_renames(results)

    # CSV 매칭 안 된 파일과 워크파일 생성
    file_only_unmatched = [
        build_file_only_row(fr) for fr in file_records if fr.path not in results.matched_file_paths
    ]
    match_workfile_rows = [
        build_match_workfile_row(row, original_fieldnames) for row in results.single_result_rows
    ]
    duplicate_key_candidate_count = sum(
        1 for row in results.review_candidates
        if str(row.get("result_reason", "")).startswith("duplicate_")
    )

    # 결과 파일 저장
    write_csv(output_dir / "matched_exact.csv",          results.matched_exact)
    write_csv(output_dir / "matched_normalized.csv",     results.matched_normalized)
    write_csv(output_dir / "review_candidates.csv",      results.review_candidates)
    write_csv(output_dir / "unmatched_no_candidates.csv", results.unmatched_no_candidates)
    write_csv(output_dir / "csv_out_of_scope_fields.csv", results.out_of_scope_rows)
    write_csv(output_dir / "file_only_unmatched.csv",    file_only_unmatched)
    write_csv(output_dir / "field_inventory.csv",        field_inventory_rows)
    write_csv(output_dir / "single_result.csv",          results.single_result_rows)
    write_csv(output_dir / "match_workfile.csv",         match_workfile_rows)
    write_csv(output_dir / "auto_rename_log.csv",        auto_rename_logs)
    (output_dir / "파일설명_한글.txt").write_text(build_output_guide_text(), encoding="utf-8")

    # 파일명 변경 상태별 건수 집계
    rename_status_counts: dict[str, int] = defaultdict(int)
    for log in auto_rename_logs:
        rename_status_counts[str(log.get("status", ""))] += 1

    summary = {
        "scan_timestamp":                          datetime.now().isoformat(timespec="seconds"),
        "csv_path":                                str(csv_path),
        "budget_root":                             str(budget_root),
        "output_dir":                              str(output_dir),
        "total_csv_rows":                          len(budget_rows),
        "csv_rows_in_scope":                       results.in_scope_rows,
        "csv_rows_out_of_scope":                   len(results.out_of_scope_rows),
        "configured_field_count":                  len(EXPECTED_FIELD_NAMES),
        "detected_field_dir_count":                len(available_field_dirs),
        "total_files_scanned":                     len(file_records),
        "unparsed_file_count":                     len(unparsed_paths),
        "matched_exact_count":                     len(results.matched_exact),
        "matched_normalized_count":                len(results.matched_normalized),
        "total_unmatched_after_auto_match":        len(results.review_candidates) + len(results.unmatched_no_candidates),
        "review_candidate_count":                  len(results.review_candidates),
        "duplicate_key_candidate_count":           duplicate_key_candidate_count,
        "unmatched_no_candidate_count":            len(results.unmatched_no_candidates),
        "file_only_unmatched_count":               len(file_only_unmatched),
        "single_result_count":                     len(results.single_result_rows),
        "match_workfile_count":                    len(match_workfile_rows),
        "auto_rename_log_count":                   len(auto_rename_logs),
        "auto_renamed_count":                      rename_status_counts.get("renamed", 0),
        "auto_already_prefixed_count":             rename_status_counts.get("already_prefixed", 0),
        "auto_rename_skipped_multiple_numbers_count": rename_status_counts.get("skipped_multiple_match_numbers", 0),
        "auto_rename_skipped_target_exists_count": rename_status_counts.get("skipped_target_exists", 0),
        "auto_rename_missing_source_file_count":   rename_status_counts.get("missing_source_file", 0),
        "extensions":                              list(args.extensions),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output_dir / "unparsed_files.txt").write_text("\n".join(unparsed_paths), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
