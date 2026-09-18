"""
Ollama LLM으로 예산 사업을 BIOFIN 1차 카테고리 0~9로 분류합니다.

SYSTEM_PROMPT에는 한국 BIOFIN 카테고리별 분류 기준과 대표사업을
반영했습니다.

사용 예:
    python llm/v1/classify_biofin_category_with_ollama.py --dry-run
    python llm/v1/classify_biofin_category_with_ollama.py --limit-keys 10
    python llm/v1/classify_biofin_category_with_ollama.py --overwrite
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from http.client import RemoteDisconnected
import json
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

LLM_DIR = Path(__file__).resolve().parents[1]
if str(LLM_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_DIR))
from document_parser import DocumentParseError, extract_document  # noqa: E402
from business_purpose import extract_business_purpose  # noqa: E402


DEFAULT_MODEL = "gemma3:12b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_INPUT_FILE = Path("document/2023biofin_label_matched.csv")
DEFAULT_LABEL_COLUMN = "LLM BIOFIN 1차 카테고리"
DEFAULT_GOLD_LABEL_COLUMN = "BIOFIN 1차 카테고리"
PROMPT_VERSION = "kr-biofin-category-2026-09-18-purpose-only-v1"
VALID_LABELS = set(range(10))
ENCODINGS = ("utf-8-sig", "cp949", "utf-8")

KEY_COLUMNS = (
    "소관명",
    "분야명",
    "부문명",
    "프로그램명",
    "단위사업명",
    "세부사업명",
)

# Classification rules: llm guideline v3 (2026-09-08).
SYSTEM_PROMPT = """\
너는 대한민국 정부 예산사업을 「분류기준_지침_v4_2026.09.15」에 따라 BIOFIN 1차 카테고리로 분류하는 전문 분류자다.

목표는 모든 사업을 억지로 1~9에 배정하는 것이 아니다.

먼저 해당 사업이 BIOFIN 생물다양성 관련 지출인지 판정한다.
관련성이 충분히 확인되지 않거나 명시적 제외조건에 해당하면 반드시 0(비해당)으로 분류한다.

관련성이 확인된 경우에만 1~9 중 정확히 하나를 선택한다.

키워드는 판정의 단서일 뿐 포함 근거가 아니다.
사업의 실제 목적, 대상, 활동, 법적 근거와 기관 기능을 기준으로 판정한다.

# 출력값

0 = 비해당
1 = 유전자원 접근 및 이익 공유(ABS)
2 = 인식 제고 및 연구
3 = 생물안전성
4 = 녹색경제와 생물다양성
5 = 생물다양성 기획 및 재정
6 = 오염 관리
7 = 보호지역 및 기타 보전조치
8 = 복원
9 = 지속 가능한 이용과 생물다양성

# 판정 원칙

다음 순서를 반드시 지킨다.

## STEP 1. BIOFIN 관련성 판정

먼저 다음 질문을 판단한다.

"이 사업의 실제 목적 또는 활동이 생물다양성, 생태계, 자연자원, 환경오염 저감, 생물자원의 지속가능한 이용·관리 등 지침에서 인정하는 활동과 직접 연결되는가?"

YES → STEP 2로 이동한다.
NO → 즉시 0으로 분류한다.
UNKNOWN → 아래 명시적 포함규칙으로 관련성을 확인할 수 없다면 0으로 분류한다.

단순히 다음 단어가 있다는 이유만으로 YES로 판단하지 않는다.

* 생태계
* 바이오
* 생물
* 친환경
* 녹색
* 탄소중립
* 지속가능
* 연구
* R&D
* 정보화
* 해양
* 산림
* 농업
* 에너지
* 환경

예를 들어 산업생태계·창업생태계 등 비유적 표현은 생물다양성과 무관하다.

## STEP 2. 비해당 여부를 먼저 검사

다음에 해당하면 원칙적으로 0이다.

* 생물다양성과 관련 없는 일반 산업지원
* 일반 생산량 확대
* 일반 유통·가공·판매 지원
* 일반 기업지원·창업지원
* 산업 또는 행정 데이터 정보화
* 인간 보건만을 목적으로 하는 생물자원 관리·연구
* 반려동물·유기동물 보호 및 복지 정책
* 인간의 인명·재산 보호만을 위한 재난대응
* 일반 하천 치수
* 상수 공급 인프라
* 인간 생활환경만을 위한 소음대책
* 일반 주택·건물 신재생에너지 설비 지원
* 생물다양성 공존조치가 없는 수소·암모니아·연료전지·풍력 등 에너지 산업 인프라
* 사용후핵연료 관리·원전해체 등 핵연료주기 기술개발
* 일반 청사 신축·재건축
* 지침에서 인정하지 않은 기관의 일반 경비
* 지속가능 요소가 확인되지 않는 일반 양식·농업·수산업 육성
* 입력자료만으로 인정 활동을 확인할 수 없는 사업

위 조건에 해당하면 다른 카테고리와 키워드가 일부 겹치더라도 0을 우선한다.

단, 지침에 명시된 개별 포함 예외가 있으면 해당 예외를 우선한다.

## STEP 3. 최우선 특례 확인

다음 규칙은 일반 카테고리 정의보다 우선한다.

### A. R&D

사업이 생물다양성 관련성이 인정되고 (R&D) 사업이면 2로 분류한다.

R&D라는 이유만으로 관련성을 인정하지 않는다.

즉:

관련 R&D → 2
무관 R&D → 0

핵연료주기 기술개발은 제외하여 0이다.

### B. 지정 연구기관·수목원

다음 기관의 운영·인력·시설관리·출연·조성·건립·유지는 2로 분류한다.

* 국립수목원
* 국립산림과학원
* 국립생물자원관
* 국립생태원
* 낙동강생물자원관
* 해양생물자원관
* 한국수목원정원관리원
* 새만금수목원 조성

다른 기관에 임의 확대하지 않는다.

### C. 경상경비

기본경비·인건비·전산운영경비 등 경상경비는 기관·부서·기금의 핵심 기능으로 판단한다.

명시된 기관 규칙을 우선하며, 기관 기능이 생물다양성과 관련된다고 확인할 수 없으면 0이다.

순수 청사 신축·재건축에는 이 규칙을 적용하지 않는다.

## STEP 4. 사업의 '기능'을 판정

관련성이 인정되면 사업명에 포함된 명사보다 실제 수행하는 동작을 찾는다.

다음 질문 중 가장 잘 맞는 하나를 선택한다.

유전자원에 접근·탐사하거나 ABS/나고야 이행인가? → 1

연구·조사·통계·자연자원 모니터링·교육·홍보인가? → 2

외래종·병해충·수생질병의 국경 유입 차단 또는 GMO/LMO 안전관리인가? → 3

경제·에너지·공급망·관광·교통·도시의 녹색 전환인가? → 4

정책·법률·계획·재정·조정·공간계획·국제협력인가? → 5

오염을 측정·예방·저감·정화·처리·관리하는가? → 6

보호지역 또는 야생종을 현재 상태에서 보호·보전하는가? → 7

이미 훼손된 생태계·서식지를 원래 또는 개선된 생태상태로 회복시키는가? → 8

농업·임업·어업·양식·담수·해양 등 자연자원을 장기적으로 지속가능하게 생산·이용·관리하는가? → 9

# 주요 경쟁 카테고리 판별 규칙

## 1 vs 2 vs 9

유전자원의 탐사·발굴·접근·ABS·나고야 이행 → 1

관련 소재·생명자원 R&D → 2

종자·품종·가축유전자원 등 농업생물다양성의 관리·보전 → 9

단순히 "유전자원", "생물자원", "바이오"라는 단어만으로 1에 넣지 않는다.

## 2 vs 5 vs 6 정보·조사 사업

조사·정보화라는 형식이 아니라 '무엇을 조사하는지'를 본다.

자연자원·생태계·산림·일반 해양환경 등의 과학 조사·통계·DB·공간정보 → 2

국토·농지·해양 공간계획 또는 전략환경평가를 위한 정보 → 5

오염을 직접 측정·감시하는 정보 → 6

구체적으로:

수질·토양 오염 측정망·감시·환경기초조사 → 6

연안·해양 오염 측정·감시 → 6

오염 관련 정책연구·홍보·다매체 지도점검·배출정보 → 6

일반 자연자원 조사·생태 모니터링 → 2

공간계획·SEA 정보화 → 5

산업·행정 데이터 정보화 → 0

## 2 vs 9 교육

범용 생물다양성·자연체험 교육 → 2

농·어·임업 종사자에게 지속가능한 생산·자원관리 기술을 교육·지도하는 사업 → 9

일반 농업인 교육이나 일반 학위·공무원 직무교육 → 0

## 3 vs 7 vs 9

외래 병해충·수생질병의 국경 유입 차단·검역 → 3

GMO/LMO 안전관리 → 3

야생동식물 보호·구조·질병관리 → 7

가축 전염병 방역·수의 조치 → 9

산림병해충의 지속적 임업 관리 → 9

## 4 vs 6

자원순환형 공급망·물류·생산체계의 녹색 전환 → 4

폐기물 자체의 수거·처리·처분 시설 → 6

친환경 선박의 건조·교체·설비 전환 → 4

친환경 선박 R&D → 2

해양폐기물·폐어구 수거 → 6

## 4 지속가능 에너지의 엄격한 조건

에너지 사업은 다음 중 하나가 명확히 확인될 때만 4로 분류한다.

1. 바이오가스·바이오에너지 등 바이오 기반 에너지 이용
2. 지침이 인정하는 재생에너지 이용·소비 전환 융자·보증·발전차액·1차산업 연계 설비
3. 해양환경 모니터링·수산업 상생 등 생물다양성 공존조치가 명시된 에너지 시스템

위 조건이 없으면 단순 신재생·수소·풍력·연료전지·탄소중립 키워드만으로 4에 넣지 않는다.

일반 에너지 산업 인프라는 0이다.

## 5 vs 6 vs 7 vs 8 vs 9

사업의 실행단계를 구분한다.

정책·법률·계획·재정·제도 설계 → 5

오염의 실제 저감·정화·처리 → 6

생태계·보호지역·종의 현재 상태 보호 → 7

훼손된 생태계·서식지의 회복 → 8

농림수산업 및 자연자원의 장기적인 지속가능 이용·관리 → 9

## 7 vs 8

"보호·보전"이라는 단어만으로 판단하지 않는다.

현재 존재하는 생태계·지역·종을 보호하는 것이 목적 → 7

훼손된 상태를 회복시키기 위한 조성·복원·재생·재도입 → 8

일반 시설 유지관리 → 0 또는 해당 기능을 별도 판단

## 8 vs 9

훼손된 생태계·서식지를 회복시키는 것이 주목적 → 8

생산·이용을 계속하면서 자원을 지속가능하게 관리하는 것이 주목적 → 9

예:
산림복원 → 8
숲가꾸기 → 9

생태하천복원 → 8
농업용수관리 → 9

수산자원 조성·서식지 회복 → 8
TAC·감척·자율관리어업 → 9

# 0 판정 원칙

다음 중 하나이면 0으로 판정한다.

1. 지침의 1~9 포함 활동에 해당하지 않는다.
2. 지침에 명시적으로 제외되어 있다.
3. 관련 키워드는 있으나 실제 활동이 생물다양성과 무관하다.
4. 일반 산업·생산·유통·행정 지원이다.
5. 입력자료만으로 생물다양성 관련 활동을 확인할 수 없다.

정보 부족 상태에서 1~9를 추정하지 않는다.

"가능성이 있다"는 이유로 포함하지 않는다.

포함 근거가 명확하지 않으면 0을 선택한다.

# 혼합사업

한 사업에 여러 활동이 있으면 다음 순서로 판단한다.

1. 생물다양성 관련 하위 요소를 찾는다.
2. 관련 하위 요소가 명시되어 있지 않으면 0이다.
3. 여러 관련 활동이 있으면 사업의 주목적을 우선한다.
4. 주목적이 불분명하면 확인 가능한 예산 비중이 큰 관련 활동을 선택한다.
5. 입력에 없는 예산 비중이나 사업 중요도를 추정하지 않는다.
6. 최종적으로 하나의 카테고리만 선택한다.

# 증거 사용 규칙

판단에는 다음 순서로 정보를 사용한다.

1. 사업설명자료의 사업 목적
2. 구체적인 수행 활동
3. 법적 근거
4. 세부사업명
5. 단위사업명
6. 프로그램명
7. 회계명·계정명·소관기관

사업명 또는 키워드 하나만으로 판정하지 않는다.

입력에 존재하지 않는 목적·활동·기관 기능을 만들어내지 않는다.

# 최종 자기검증

출력하기 전에 내부적으로 다음을 확인한다.

1. 이 사업이 정말 BIOFIN 관련 사업인가?
2. 명시적 제외조건은 없는가?
3. 키워드 하나 때문에 포함시키고 있지는 않은가?
4. 0일 가능성을 충분히 검토했는가?
5. 경쟁하는 카테고리는 무엇인가?
6. 그 경쟁 카테고리를 제외할 명확한 근거가 있는가?
7. 입력자료에 없는 내용을 추정하지 않았는가?

위 검증 후 최종 카테고리를 결정한다.

# 최종 출력

반드시 JSON 하나만 반환한다.

{
"label": 0,
"reason": "판정 근거",
"evidence": ["입력에 실제 존재하는 표현"],
"confidence": 0.00
}

label은 0~9의 정수만 사용한다.

reason에는 다음을 포함한다.

* 실제 핵심 활동
* 적용한 포함 또는 제외 기준
* 가장 혼동될 수 있는 경쟁 카테고리와 그것을 배제한 이유

0인 경우 reason에서 반드시 다음 둘 중 하나를 명시한다.

* "명시적 제외": 지침의 제외조건에 해당
* "근거 부족": 입력자료에서 1~9의 인정 활동을 확인할 수 없음

evidence에는 입력자료에 실제 존재하는 문구만 사용한다.
입력에 없는 내용을 생성하거나 추론하여 evidence로 작성하지 않는다.

confidence는 0~1 사이 숫자로 반환한다.

BAR는 분류기준 또는 출력대상이 아니므로 카테고리 판정에 사용하지 않는다.

"""

# 시스템 프롬프트와 별도로 유지하는 입출력 형식입니다.
# SYSTEM_PROMPT가 비어 있어도 0~9 중 하나를 JSON으로 반환하게 합니다.
PROMPT_TEMPLATE = """\
{classification_prompt}

다음 예산 사업을 BIOFIN 1차 카테고리 0~9 중 하나로 분류하라.

소관명: {소관명}
회계명: {회계명}
계정명: {계정명}
분야명: {분야명}
부문명: {부문명}
프로그램명: {프로그램명}
단위사업명: {단위사업명}
세부사업명: {세부사업명}

사업설명자료의 사업목적 항목(다른 항목은 제공하지 않음):
{document_text}

반드시 아래 JSON 객체만 반환하라.
{{
  "label": 0,
  "confidence": 0.0,
  "reason": "",
  "evidence": ""
}}

label은 0부터 9까지의 정수여야 하고 confidence는 0.0부터 1.0까지다.
"""

CACHE_FIELDS = (
    "key_hash",
    "label",
    "confidence",
    "reason",
    "evidence",
    "model",
    "prompt_version",
    "input_text",
    "document_status",
    "document_path",
    "document_chars",
    "raw_response",
    "updated_at",
)
EXTRA_OUTPUT_COLUMNS = (
    "confidence", "reason", "evidence",
    "document_status", "document_path", "document_chars",
)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="예산 사업을 Ollama로 BIOFIN 1차 카테고리 0~9로 분류합니다."
    )
    parser.add_argument(
        "--input-file", type=Path, default=project_dir / DEFAULT_INPUT_FILE
    )
    parser.add_argument("--output-dir", type=Path, default=project_dir / "outputs/llm/v1")
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--cache-csv", type=Path, default=None)
    parser.add_argument("--audit-csv", type=Path, default=None)
    parser.add_argument("--review-csv", type=Path, default=None)
    parser.add_argument("--label-col", default=DEFAULT_LABEL_COLUMN)
    parser.add_argument(
        "--gold-label-col",
        default=None,
        help="정확도 평가용 정답 컬럼. 생략 시 알려진 컬럼명을 자동 탐색",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument(
        "--doc-dir",
        type=Path,
        default=project_dir / "document/2023/사업설명자료",
        help="사업설명자료 파일 폴더",
    )
    parser.add_argument(
        "--max-document-chars",
        type=int,
        default=16000,
        help="프롬프트에 포함할 사업목적 최대 문자 수",
    )
    parser.add_argument(
        "--no-document-text",
        action="store_true",
        help="사업설명자료를 읽지 않고 예산 메타데이터만 사용",
    )
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="완료 N건마다 캐시 저장. 기본값 1은 중단 시 완료 결과를 모두 보존",
    )
    parser.add_argument("--review-threshold", type=float, default=0.7)
    parser.add_argument("--limit-keys", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-json-format", action="store_true")
    return parser.parse_args()


def set_default_paths(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_file is None:
        args.output_file = args.output_dir / f"{args.input_file.stem}_llm_classified.csv"
    if args.cache_csv is None:
        args.cache_csv = args.output_dir / "category_label_cache.csv"
    if args.audit_csv is None:
        args.audit_csv = args.output_dir / "category_label_audit.csv"
    if args.review_csv is None:
        args.review_csv = args.output_dir / "review_needed.csv"


def clean_cell(value: Any) -> str:
    return re.sub(r"\s+", " ", str("" if value is None else value).strip())


def resolve_gold_label_column(headers: list[str], requested: str | None) -> str:
    if requested is not None:
        if requested not in headers:
            raise ValueError(f"지정한 정답 컬럼이 없습니다: {requested}")
        return requested
    candidates = [
        name for name in (DEFAULT_GOLD_LABEL_COLUMN, "1차 카테고리", "1차")
        if name in headers
    ]
    if len(candidates) > 1:
        raise ValueError(
            f"정답 컬럼 후보가 여러 개입니다: {candidates}. --gold-label-col로 지정하세요."
        )
    if candidates:
        return candidates[0]
    print("정답 컬럼 없음: 예측은 진행하지만 정확도 평가는 할 수 없습니다.")
    return DEFAULT_GOLD_LABEL_COLUMN


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                reader = csv.DictReader(file)
                if not reader.fieldnames:
                    raise ValueError("CSV 헤더가 없습니다.")
                return list(reader.fieldnames), [dict(row) for row in reader], encoding
        except UnicodeError as exc:
            last_error = exc
    raise RuntimeError(f"CSV를 읽을 수 없습니다: {path}") from last_error


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_key(row: dict[str, str]) -> str:
    business_key = clean_cell(row.get("business_key"))
    if business_key:
        return f"business_key:{business_key}"
    return "␟".join(clean_cell(row.get(column)) for column in KEY_COLUMNS)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def prompt_values(row: dict[str, str]) -> dict[str, str]:
    columns = (
        "소관명",
        "회계명",
        "계정명",
        "분야명",
        "부문명",
        "프로그램명",
        "단위사업명",
        "세부사업명",
    )
    return {column: clean_cell(row.get(column)) for column in columns}


def build_input_text(row: dict[str, str]) -> str:
    values = prompt_values(row)
    return " | ".join(f"{key}: {value}" for key, value in values.items() if value)


def build_prompt(row: dict[str, str], document_text: str) -> str:
    return PROMPT_TEMPLATE.format(
        classification_prompt=SYSTEM_PROMPT.strip(),
        document_text=document_text,
        **prompt_values(row),
    ).strip()


def resolve_document_path(row: dict[str, str], args: argparse.Namespace) -> Path | None:
    """CSV의 절대·상대경로와 파일명을 이용해 실제 사업설명자료를 찾습니다."""
    project_dir = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    absolute = clean_cell(row.get("사업설명자료_절대경로"))
    relative = clean_cell(row.get("사업설명자료_상대경로"))
    filename = clean_cell(row.get("사업설명자료_파일명"))
    if absolute:
        candidates.append(Path(absolute))
    if relative:
        candidates.extend(
            [
                args.input_file.parent / relative,
                project_dir / "document/2023" / relative,
                args.doc_dir.parent / relative,
            ]
        )
    if filename:
        candidates.append(args.doc_dir / filename)
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            # 다른 OS에서 생성된 절대경로, 너무 긴 경로, 접근 불가 경로는
            # 무시하고 상대경로/파일명 후보를 계속 확인합니다.
            continue
    return None


def truncate_document(text: str, max_chars: int) -> str:
    """긴 문서는 앞부분과 뒷부분을 함께 남겨 프롬프트 크기를 제한합니다."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return (
        text[:head]
        + "\n\n[... 문서 중간 부분 생략 ...]\n\n"
        + text[-tail:]
    )


def load_document_for_prompt(
    row: dict[str, str], args: argparse.Namespace
) -> tuple[str, str, str, int]:
    if args.no_document_text:
        return "[사업설명자료 사용 안 함]", "DISABLED", "", 0
    path = resolve_document_path(row, args)
    if path is None:
        return "[사업설명자료 파일을 찾지 못함]", "NOT_FOUND", "", 0
    try:
        full_text = extract_document(path)
        purpose = extract_business_purpose(full_text)
        if not purpose:
            return (
                "[사업목적 항목 또는 종료 경계를 찾지 못함. 문서 본문 미사용]",
                "PURPOSE_NOT_FOUND", str(path), 0,
            )
        prompt_text = purpose[:args.max_document_chars]
        return prompt_text, "PURPOSE_EXTRACTED", str(path), len(prompt_text)
    except (DocumentParseError, RuntimeError, OSError) as exc:
        return (
            f"[사업설명자료 파싱 실패: {clean_cell(exc)[:300]}]",
            "PARSE_FAILED",
            str(path),
            0,
        )


def parse_jsonish_response(text: str) -> dict[str, Any]:
    if not text.strip():
        raise ValueError("LLM 최종 응답이 비어 있습니다.")
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            label_match = re.search(r"(?<!\d)([0-9])(?!\d)", raw)
            if not label_match:
                raise ValueError("응답에서 0~9 라벨을 찾지 못했습니다.")
            data = {
                "label": int(label_match.group(1)),
                "confidence": 0.5,
                "reason": "JSON 외 응답에서 라벨 추출",
                "evidence": "",
            }
        else:
            data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("LLM 응답이 JSON 객체가 아닙니다.")
    value = data.get("label")
    label = None if isinstance(value, bool) else parse_valid_label(value)
    if label is None:
        raise ValueError(f"label은 0~9 정수여야 합니다: {value!r}")

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "label": label,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": clean_cell(data.get("reason"))[:500],
        "evidence": clean_cell(data.get("evidence"))[:500],
        "raw_response": text,
    }


class OllamaResponseError(ValueError):
    def __init__(self, message: str, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


def call_ollama(prompt: str, args: argparse.Namespace) -> str:
    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "top_p": 0.1, "num_ctx": args.num_ctx},
    }
    if not args.no_json_format:
        payload["format"] = "json"
    req = request.Request(
        f"{args.ollama_url.rstrip('/')}/api/generate",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=args.timeout) as response:
        raw_body = response.read().decode("utf-8")
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise OllamaResponseError("Ollama HTTP 응답이 JSON이 아닙니다.", raw_body) from exc
    if not isinstance(body, dict):
        raise OllamaResponseError("Ollama HTTP 응답이 JSON 객체가 아닙니다.", raw_body)
    if body.get("error"):
        raise OllamaResponseError(f"Ollama 서버 오류: {body['error']}", raw_body)
    text = body.get("response")
    if not isinstance(text, str) or not text.strip():
        raise OllamaResponseError(
            "Ollama 최종 response가 없거나 비어 있습니다. "
            f"응답 필드={list(body)}, done_reason={body.get('done_reason')!r}, "
            f"thinking 존재={bool(body.get('thinking'))}",
            raw_body,
        )
    return text


def classify(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    document_text, document_status, document_path, document_chars = (
        load_document_for_prompt(row, args)
    )
    prompt = build_prompt(row, document_text)
    last_error: Exception | None = None
    raw_response = ""
    for attempt in range(args.retries + 1):
        try:
            raw_response = ""
            raw_response = call_ollama(prompt, args)
            result = parse_jsonish_response(raw_response)
            result.update(
                {
                    "document_status": document_status,
                    "document_path": document_path,
                    "document_chars": document_chars,
                }
            )
            if args.delay > 0:
                time.sleep(args.delay)
            return result
        except (
            HTTPError,
            URLError,
            TimeoutError,
            RemoteDisconnected,
            ConnectionError,
            OSError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            last_error = exc
            if isinstance(exc, OllamaResponseError):
                raw_response = exc.raw_response
            if attempt < args.retries:
                time.sleep(args.retry_delay)
    return {
        "label": "",
        "confidence": 0.0,
        "reason": f"분류 실패: {last_error}",
        "evidence": "",
        "document_status": document_status,
        "document_path": document_path,
        "document_chars": document_chars,
        "raw_response": raw_response,
    }


def load_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    _, rows, _ = read_csv(path)
    return {row["key_hash"]: row for row in rows if row.get("key_hash")}


def save_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    """캐시를 임시 파일에 먼저 쓴 뒤 교체해 중단 시 파일 손상을 방지합니다."""
    temporary_path = path.with_name(f"{path.name}.tmp")
    write_csv(
        temporary_path,
        list(CACHE_FIELDS),
        [cache[key] for key in sorted(cache)],
    )
    temporary_path.replace(path)


def collect_items(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for row in rows:
        key_hash = hash_key(build_key(row))
        if key_hash not in items:
            items[key_hash] = {
                "row": row,
                "input_text": build_input_text(row),
                "row_count": 0,
            }
        items[key_hash]["row_count"] += 1
    return items


def print_document_match_summary(
    rows: list[dict[str, str]], args: argparse.Namespace
) -> None:
    """LLM 호출 전에 실제 사업설명자료 파일 연결 여부를 점검합니다."""
    if args.no_document_text:
        print("사업설명자료: 사용 안 함(--no-document-text)")
        return
    resolved = 0
    missing = 0
    matched_rows = 0
    matched_but_missing = 0
    for row in rows:
        declared_matched = clean_cell(row.get("문서매칭상태")).upper() == "MATCHED"
        if declared_matched:
            matched_rows += 1
        if resolve_document_path(row, args) is not None:
            resolved += 1
        else:
            missing += 1
            if declared_matched:
                matched_but_missing += 1
    print(f"사업설명자료 실제 연결: {resolved:,}건 / 미발견: {missing:,}건")
    print(
        f"CSV상 MATCHED: {matched_rows:,}건 / "
        f"MATCHED지만 실제 파일 미발견: {matched_but_missing:,}건"
    )
    print(f"사업설명자료 탐색 폴더: {args.doc_dir}")


def valid_cached_label(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    if record.get("document_status") not in {
        "PURPOSE_EXTRACTED", "PURPOSE_NOT_FOUND", "NOT_FOUND", "PARSE_FAILED", "DISABLED"
    }:
        return False
    if record.get("prompt_version") != PROMPT_VERSION:
        return False
    try:
        return int(record.get("label", -1)) in VALID_LABELS
    except (TypeError, ValueError):
        return False


def parse_valid_label(value: Any) -> int | None:
    """값을 0~9 정수 라벨로 변환하며 유효하지 않으면 None을 반환합니다."""
    text = clean_cell(value)
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not number.is_integer():
        return None
    label = int(number)
    return label if label in VALID_LABELS else None


def evaluate_predictions(
    rows: list[dict[str, Any]],
    gold_column: str,
    pred_column: str,
    output_dir: Path,
) -> dict[str, Any] | None:
    """정답과 예측을 비교해 정확도·클래스별 지표·혼동행렬을 저장합니다."""
    evaluated: list[tuple[int, int, dict[str, Any]]] = []
    skipped_missing_gold = 0
    skipped_missing_prediction = 0

    for row in rows:
        gold = parse_valid_label(row.get(gold_column))
        pred = parse_valid_label(row.get(pred_column))
        if gold is None:
            skipped_missing_gold += 1
            continue
        if pred is None:
            skipped_missing_prediction += 1
            continue
        evaluated.append((gold, pred, row))

    if not evaluated:
        print(
            f"정확도 평가 생략: '{gold_column}'과 유효한 예측이 함께 있는 행이 없습니다."
        )
        return None

    matrix = [[0 for _ in range(10)] for _ in range(10)]
    incorrect_rows: list[dict[str, Any]] = []
    correct = 0
    for gold, pred, row in evaluated:
        matrix[gold][pred] += 1
        if gold == pred:
            correct += 1
        else:
            incorrect_rows.append(
                {
                    "gold_label": gold,
                    "pred_label": pred,
                    "confidence": row.get("confidence", ""),
                    "reason": row.get("reason", ""),
                    "evidence": row.get("evidence", ""),
                    "소관명": row.get("소관명", ""),
                    "프로그램명": row.get("프로그램명", ""),
                    "단위사업명": row.get("단위사업명", ""),
                    "세부사업명": row.get("세부사업명", ""),
                    "business_key": row.get("business_key", ""),
                }
            )

    per_class: dict[str, dict[str, Any]] = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    for label in range(10):
        tp = matrix[label][label]
        support = sum(matrix[label])
        predicted = sum(matrix[gold][label] for gold in range(10))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[str(label)] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
            "predicted": predicted,
        }
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    total = len(evaluated)
    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gold_label_column": gold_column,
        "prediction_column": pred_column,
        "evaluated_rows": total,
        "correct_rows": correct,
        "incorrect_rows": total - correct,
        "accuracy": round(correct / total, 6),
        "macro_precision": round(macro_precision / 10, 6),
        "macro_recall": round(macro_recall / 10, 6),
        "macro_f1": round(macro_f1 / 10, 6),
        "skipped_missing_gold": skipped_missing_gold,
        "skipped_missing_prediction": skipped_missing_prediction,
        "per_class": per_class,
    }
    metrics_path = output_dir / "evaluation_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    confusion_rows = []
    for gold in range(10):
        confusion_rows.append(
            {
                "gold_label": gold,
                **{f"pred_{pred}": matrix[gold][pred] for pred in range(10)},
            }
        )
    write_csv(
        output_dir / "confusion_matrix.csv",
        ["gold_label", *[f"pred_{label}" for label in range(10)]],
        confusion_rows,
    )
    write_csv(
        output_dir / "incorrect_predictions.csv",
        [
            "gold_label",
            "pred_label",
            "confidence",
            "reason",
            "evidence",
            "소관명",
            "프로그램명",
            "단위사업명",
            "세부사업명",
            "business_key",
        ],
        incorrect_rows,
    )
    print(
        f"정확도: {metrics['accuracy']:.4f} "
        f"({correct:,}/{total:,}), macro F1: {metrics['macro_f1']:.4f}"
    )
    print(f"평가 지표: {metrics_path}")
    return metrics


def classify_items(
    items: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    selected = list(items.items())
    if args.limit_keys > 0:
        selected = selected[: args.limit_keys]
    pending = [
        (key_hash, item)
        for key_hash, item in selected
        if args.overwrite or not valid_cached_label(cache.get(key_hash))
    ]
    print(f"LLM 분류 대상: {len(pending):,}개 (workers={args.workers})")

    lock = threading.Lock()
    completed = 0

    def process(key_hash: str, item: dict[str, Any]) -> None:
        nonlocal completed
        result = classify(item["row"], args)
        record = {
            "key_hash": key_hash,
            "label": result["label"],
            "confidence": f"{float(result['confidence']):.3f}",
            "reason": result["reason"],
            "evidence": result["evidence"],
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
            "input_text": item["input_text"],
            "document_status": result["document_status"],
            "document_path": result["document_path"],
            "document_chars": result["document_chars"],
            "raw_response": result["raw_response"],
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        with lock:
            cache[key_hash] = record
            completed += 1
            print(
                f"[{completed:,}/{len(pending):,}] "
                f"label={record['label']} conf={record['confidence']} "
                f"{item['input_text'][:70]}"
            )
            if not valid_cached_label(record):
                print(f"  {record['reason']}", file=sys.stderr, flush=True)
            if args.save_every > 0 and completed % args.save_every == 0:
                save_cache(args.cache_csv, cache)

    executor = ThreadPoolExecutor(max_workers=max(1, args.workers))
    futures = [executor.submit(process, key_hash, item) for key_hash, item in pending]
    try:
        for future in as_completed(futures):
            future.result()
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        print("\n중단됨: 현재 캐시를 저장합니다.", file=sys.stderr)
        save_cache(args.cache_csv, cache)
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
        save_cache(args.cache_csv, cache)


def write_outputs(
    headers: list[str],
    rows: list[dict[str, str]],
    items: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    excluded = {args.label_col, *EXTRA_OUTPUT_COLUMNS}
    output_headers = [
        column for column in headers if column not in excluded
    ] + [args.label_col, *EXTRA_OUTPUT_COLUMNS]

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        result = dict(row)
        cached = cache.get(hash_key(build_key(row)), {})
        result[args.label_col] = cached.get("label", "")
        for column in EXTRA_OUTPUT_COLUMNS:
            result[column] = cached.get(column, "")
        output_rows.append(result)
    write_csv(args.output_file, output_headers, output_rows)
    evaluation = evaluate_predictions(
        output_rows,
        args.gold_label_col,
        args.label_col,
        args.output_dir,
    )

    audit_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for key_hash, item in items.items():
        cached = cache.get(key_hash, {})
        audit = {
            "key_hash": key_hash,
            "row_count": item["row_count"],
            "label": cached.get("label", ""),
            "confidence": cached.get("confidence", ""),
            "reason": cached.get("reason", ""),
            "evidence": cached.get("evidence", ""),
            "input_text": item["input_text"],
            "document_status": cached.get("document_status", ""),
            "document_path": cached.get("document_path", ""),
            "document_chars": cached.get("document_chars", ""),
            "raw_response": cached.get("raw_response", ""),
        }
        audit_rows.append(audit)
        try:
            confidence = float(audit["confidence"])
        except (TypeError, ValueError):
            confidence = 0.0
        if not valid_cached_label(cached) or confidence < args.review_threshold:
            review_rows.append(audit)

    audit_headers = [
        "key_hash", "row_count", "label", "confidence",
        "reason", "evidence", "input_text", "document_status",
        "document_path", "document_chars", "raw_response",
    ]
    write_csv(args.audit_csv, audit_headers, audit_rows)
    write_csv(args.review_csv, audit_headers, review_rows)

    counts = Counter(str(row.get("label", "")) for row in cache.values())
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "input_file": str(args.input_file),
        "input_rows": len(rows),
        "unique_businesses": len(items),
        "label_counts_in_cache": dict(sorted(counts.items())),
        "output_file": str(args.output_file),
        "evaluation": evaluation,
    }
    summary_path = args.output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"분류 결과: {args.output_file}")
    print(f"검수 결과: {args.audit_csv}")
    print(f"확인 필요: {args.review_csv} ({len(review_rows):,}건)")


def main() -> int:
    args = parse_args()
    if args.max_document_chars < 1:
        raise ValueError("--max-document-chars는 1 이상이어야 합니다.")
    if args.num_ctx < 1024:
        raise ValueError("--num-ctx는 1024 이상이어야 합니다.")
    set_default_paths(args)
    headers, rows, encoding = read_csv(args.input_file)
    args.gold_label_col = resolve_gold_label_column(headers, args.gold_label_col)
    print(f"평가용 정답 컬럼: {args.gold_label_col}")
    missing = [column for column in KEY_COLUMNS if column not in headers]
    if missing and "business_key" not in headers:
        raise ValueError(f"고유 사업 키 컬럼이 부족합니다: {', '.join(missing)}")

    items = collect_items(rows)
    print(f"입력: {args.input_file} ({len(rows):,}행, {encoding})")
    print(f"고유 사업: {len(items):,}개")
    print_document_match_summary(rows, args)
    if args.dry_run:
        print("--dry-run: Ollama 호출 및 파일 저장 없이 종료합니다.")
        return 0

    cache = load_cache(args.cache_csv)
    print(f"기존 캐시: {len(cache):,}개")
    classify_items(items, cache, args)
    write_outputs(headers, rows, items, cache, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
