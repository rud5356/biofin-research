"""
예산 문서 텍스트를 모델 학습에 적합하게 가공하는 모듈.

핵심 기능:
1. 문서 앞부분의 구조화된 헤더(사업코드, 지원형태 등 메타데이터 표)를 제거하고
   실제 사업 내용이 담긴 '사업목적·내용' 섹션부터 텍스트를 추출합니다.
2. 텍스트에서 분야명, 라벨명 등 정답 정보를 마스킹하여
   모델이 레이블 단어 자체를 외우는 것을 방지합니다.
"""

from __future__ import annotations

import re
from typing import Iterable

from document_text import clean_extracted_text


# ─── 섹션 구분 문자 패턴 ─────────────────────────────────────────────────────
# '사업목적 내용' 처럼 가운데 점(·, ㆍ)이나 마침표가 구분자로 쓰이는 경우를 허용
_SECTION_SEPARATOR = r"[\.·ㆍ]?"

# ─── '사업목적 내용' 섹션 시작 위치를 찾는 패턴 ──────────────────────────────
# 다양한 표기 방식을 모두 커버합니다:
#   "1) 사업목적·내용", "4. 사업목적·내용", "사업목적 내용", "사업목적" 등
# (?im): 대소문자 무시(i), 각 줄의 시작(m)에서 매칭
_SECTION_START_PATTERNS = [
    re.compile(r"(?im)^\s*1\)\s*사업목적\s*" + _SECTION_SEPARATOR + r"\s*내용"),
    re.compile(r"(?im)^\s*4\.\s*사업목적\s*" + _SECTION_SEPARATOR + r"\s*내용"),
    re.compile(r"(?im)^\s*4\.\s*사업목적"),
    re.compile(r"(?im)^\s*사업목적\s*" + _SECTION_SEPARATOR + r"\s*내용"),
    re.compile(r"(?im)^\s*사업목적"),
]

# ─── 제거할 헤더 줄 패턴 ─────────────────────────────────────────────────────
# 사업코드 정보, 지원형태, 담당자 등 내용 파악에 불필요한 구조화된 줄들
_DROP_LINE_PATTERNS = [
    re.compile(r"(?im)^\s*[0-9]+\.\s*사업 코드 정보\s*$"),
    re.compile(r"(?im)^\s*□\s*사업 코드 정보\s*$"),
    re.compile(r"(?im)^\s*[0-9]+\.\s*사업 지원 형태 및 지원율.*$"),
    re.compile(r"(?im)^\s*□\s*사업 지원 형태 및 지원율.*$"),
    re.compile(r"(?im)^\s*□\s*사업 담당자\s*$"),
    re.compile(r"(?im)^\s*가\.\s*예산 총괄표.*$"),
    re.compile(r"(?im)^\s*나\.\s*사업설명자료\s*$"),
    re.compile(r"(?im)^\s*구분\s*$"),
    re.compile(
        r"(?im)^\s*(회계|소관|실국\(기관\)|계정|분야|부문|코드|명칭)\s*$"
    ),
]

# 단어 토큰(숫자·영문·한글) 추출용 패턴
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")

# 메타데이터 값 사이에 들어올 수 있는 구분 문자 (공백, 쉼표, 가운데점 등)
_VALUE_JOINER = r"[\s\.,·ㆍ\-_()/]*"


def _normalize_metadata_value(raw_value: object) -> str:
    """메타데이터 값을 문자열로 변환하고, 빈 값(NaN, None 등)은 빈 문자열로 처리합니다."""
    value = str(raw_value or "").strip()
    if value.lower() in {"", "nan", "none"}:
        return ""
    return value


def _find_section_start(text: str) -> tuple[int | None, str]:
    """
    텍스트에서 '사업목적·내용' 섹션이 시작하는 위치를 찾습니다.

    여러 패턴 중 가장 앞에 나타나는 것을 사용합니다.
    패턴이 없으면 None을 반환해 전체 텍스트를 사용하도록 합니다.
    """
    match_positions: list[int] = []
    for pattern in _SECTION_START_PATTERNS:
        match = pattern.search(text)
        if match:
            match_positions.append(match.start())

    if not match_positions:
        # 섹션을 찾지 못한 경우: 전체 텍스트를 그대로 사용 (fallback)
        return None, "full_text_fallback"

    # 여러 패턴이 매칭된 경우 가장 앞에 있는 위치를 선택
    return min(match_positions), "purpose_section"


def _drop_header_block(text: str) -> tuple[str, str]:
    """
    '사업목적·내용' 섹션 이전의 헤더 블록(메타데이터 표 등)을 잘라냅니다.

    반환값: (잘라낸 후 텍스트, 처리 방법 레이블)
    """
    start_index, method = _find_section_start(text)
    if start_index is None:
        return text, method
    # 섹션 시작 위치부터 끝까지만 남김
    return text[start_index:], method


def _remove_structured_header_lines(text: str) -> str:
    """
    텍스트에서 구조화된 헤더 줄(사업코드표, 구분 등)을 한 줄씩 검사하여 제거합니다.
    빈 줄은 그대로 유지합니다.
    """
    kept_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept_lines.append("")
            continue
        # 제거 패턴에 매칭되는 줄은 건너뜀
        if any(pattern.search(stripped) for pattern in _DROP_LINE_PATTERNS):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def _build_flexible_value_pattern(value: str) -> re.Pattern[str] | None:
    """
    메타데이터 값(예: 분야명 "농림수산")에서 단어 토큰을 추출하고
    토큰 사이에 구분 문자를 허용하는 유연한 정규식 패턴을 만듭니다.

    예: "농림·수산" → r"농림[\s\.,·\-_()/]*수산" (기호가 섞여도 매칭)

    너무 짧은 값(2자 미만)은 오매칭 위험이 높아 None을 반환합니다.
    """
    tokens = _TOKEN_RE.findall(value or "")
    if len("".join(tokens)) < 2:
        return None
    # 각 토큰 사이에 구분 문자를 허용하는 패턴으로 연결
    pattern = _VALUE_JOINER.join(re.escape(token) for token in tokens)
    return re.compile(pattern, re.IGNORECASE)


def mask_metadata_values(text: str, values: Iterable[object]) -> tuple[str, int]:
    """
    텍스트에서 메타데이터 값(분야명, 라벨명 등)을 공백으로 마스킹합니다.

    모델이 '환경' 같은 분야명 단어를 보고 정답을 단순 외우는 것을 방지합니다.

    반환값: (마스킹된 텍스트, 총 교체 횟수)
    """
    masked_text = text
    total_replacements = 0

    for raw_value in values:
        value = _normalize_metadata_value(raw_value)
        if not value:
            continue
        pattern = _build_flexible_value_pattern(value)
        if pattern is None:
            continue
        # pattern.subn(): 매칭된 부분을 공백으로 교체하고 교체 횟수도 반환
        masked_text, count = pattern.subn(" ", masked_text)
        total_replacements += count

    return masked_text, total_replacements


def build_model_text(raw_text: str, metadata: dict[str, object]) -> tuple[str, str]:
    """
    문서 텍스트를 모델 학습에 적합하게 가공합니다.

    처리 순서:
    1. '사업목적·내용' 섹션 이전 헤더 블록 제거
    2. 구조화된 헤더 줄 제거
    3. 분야명·라벨명 마스킹
    4. 텍스트 정제 (공백 통일 등)

    가공 후 텍스트가 비어 있으면 원본 전체 텍스트로 fallback합니다.

    반환값: (가공된 텍스트, 처리 방법 레이블)
    """
    if not str(raw_text or "").strip():
        return "", "empty"

    # 1~2단계: 섹션 추출 및 헤더 줄 제거
    candidate_text, method = _drop_header_block(str(raw_text))
    candidate_text = _remove_structured_header_lines(candidate_text)

    # 3단계: 분야명, 라벨명 마스킹 (정답 유출 방지)
    candidate_text, masked_count = mask_metadata_values(
        candidate_text,
        [
            metadata.get("label", ""),
            metadata.get("budget_field_name", ""),
        ],
    )

    # 4단계: 최종 정제
    candidate_text = clean_extracted_text(candidate_text)

    if not candidate_text:
        # 섹션 추출 후 텍스트가 비어있으면 원본 전체로 fallback
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

    # 마스킹이 일어났으면 레이블에 '+masked' 접미사 추가 (추적용)
    suffix = "+masked" if masked_count else ""
    return candidate_text, f"{method}{suffix}"
