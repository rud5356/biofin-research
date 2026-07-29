"""
Ollama LLM으로 예산 사업을 BIOFIN 1차 카테고리 0~9로 분류합니다.

SYSTEM_PROMPT는 의도적으로 비워 두었습니다. 분류 기준 프롬프트가
확정되면 해당 변수에 내용을 넣어 사용하세요.

사용 예:
    python classify_biofin_category_with_ollama.py --dry-run
    python classify_biofin_category_with_ollama.py --limit-keys 10
    python classify_biofin_category_with_ollama.py --overwrite
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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


DEFAULT_MODEL = "gemma3:12b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_INPUT_FILE = Path("document/2023biofin_label_matched.csv")
DEFAULT_LABEL_COLUMN = "LLM BIOFIN 1차 카테고리"
DEFAULT_GOLD_LABEL_COLUMN = "BIOFIN 1차 카테고리"
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

# BIOFIN/GLOBE 분류표(clip_20260724142552447 (1).bmp)를 바탕으로 작성한
# BIOFIN 1차 카테고리 분류 기준입니다.
SYSTEM_PROMPT = """\
너는 대한민국 정부 예산사업을 BIOFIN/GLOBE 생물다양성 지출 기준에 따라
BIOFIN 1차 카테고리 0~9 중 정확히 하나로 분류하는 전문 분류자다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[분류 목적과 기본 원칙]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 사업의 실제 또는 부수적 효과가 아니라 사업에 명시된 최종 목적과 의도를
   기준으로 판단한다. 자연환경에 긍정적일 가능성만으로 생물다양성 지출로
   분류하지 않는다.
2. 세부사업명, 단위사업명, 프로그램명 등 입력에 실제로 제시된 정보만 사용한다.
   판단 우선순위는 세부사업명 → 단위사업명 → 프로그램명이다.
3. 소관 부처, 분야 또는 '친환경·녹색·지속가능·생태' 같은 일반 키워드만으로
   분류하지 않는다. 해당 활동과 생물다양성 목적의 연결이 확인되어야 한다.
4. 아래 1~9 중 어느 범주에도 명확히 해당하지 않거나 정보가 부족하면 0으로
   분류한다. 일반 행정, 인건비, 기관운영, 일반 정보화, 일반 시설관리,
   일반 산업지원 등도 구체적인 생물다양성 목적이 없으면 0이다.
5. 여러 범주가 가능하면 사업의 핵심 목적과 예산의 주된 활동을 가장 구체적으로
   설명하는 한 범주만 선택한다. 단순히 관련될 수 있는 범주를 나열하지 않는다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[상위 카테고리 0~9]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

0. 비해당

아래 1~9의 목적이 입력에 명시되거나 충분히 확인되지 않는 사업이다.
일반 경제·사회·환경 사업의 부수적인 생물다양성 편익은 0으로 분류한다.

1. 유전자원 접근 및 이익 공유(ABS)

유전자원 및 원주민·지역사회(IPLC)의 전통지식 이용에서 발생한 혜택을
공정하고 공평하게 공유하는 활동이다. 기술 이전, 로열티 배분, 관련 기금
출연·관리도 포함한다.

- 1.01 생물다양성 지역·유전자원 스크리닝 및 접근 허가
- 1.02 유전자원 접근·이용 계약 체결
- 1.03 유전자원 이익공유 메커니즘
- 1.04 나고야의정서 이행
- 유전자원 정보 접근권 보장

주의: 나고야의정서 등 협약의 국가 이행 법률·정책·조정 자체가 주목적이면
5번을 우선한다. 실제 유전자원 접근과 이익공유의 실행이면 1번이다.

2. 생물다양성 인식 제고 및 연구

생물다양성 교육, 훈련, 대중 인식과 소통, 과학 연구, 조사, 모니터링,
데이터 수집·평가·공유 활동이다.

- 2.01 정규 생물다양성 교육
- 2.02 비정규 교육 및 기술 훈련
- 2.03 대중 인식 제고 및 소통
- 2.04 생물다양성 과학 연구
- 2.05 원주민·지역사회 지식 보전
- 2.06 CBD 정보공유체계

주의: 직업 훈련을 제외한 학교 교육과정은 2.01, 일반 시민 대상 교육은
2.02로 본다. 시민과학(Citizen Science)과 생물다양성 데이터 수집도
2번에 포함한다. 대상이 생물다양성임이 확인되지 않는 일반 교육·연구·홍보·
정보화 사업은 0이다.

3. 생물안전성

생명공학 결과물의 안전한 취급과 유입경로 모니터링, 침입외래종의 유입 차단,
격리, 통제, 박멸을 위한 활동이다.

- 3.01 침입외래종(IAS) 관리
- 3.02 유전자변형생물체(GMO/LMO) 관리

포함: 길고양이·유기견 등 반려동물이 야생동물과 서식지에 주는 피해를
저감하는 조치. 제외: 생산 목적의 가축·작물 질병 방지와 일반 방역은
생물안전성으로 보지 않으며, 지속가능한 생산 목적이 명확하면 9번을 검토한다.

4. 친환경 경제 전환

생물다양성 압력을 줄이도록 공급망, 채굴, 소비, 에너지, 관광, 교통,
도시·농촌의 경제활동과 기반시설을 전환하는 활동이다.

- 4.01 녹색 공급망 구축
- 4.02 지속가능한 광업
- 4.03 지속가능한 소비
- 4.04 지속가능한 에너지
- 4.05 지속가능한 관광
- 4.06 지속가능한 교통
- 4.07 지속가능한 도시·지역

가장 중요한 조건: 반드시 생물다양성 목적 또는 생태계·서식지 압력 저감과의
명시적인 연결이 있어야 한다. 단순 기후변화 완화·적응, 탄소중립, 재생에너지,
친환경 교통, 녹색도시만으로는 4번이 아니며 0으로 분류한다. 특정 인프라
건설·개발 프로젝트 단위의 환경영향평가(EIA)는 4번에 포함할 수 있다.

5. 생물다양성 기획 및 재정

생물다양성 관련 법률, 정책, 전략, 계획, 재정, 부처 간 조정, 공간계획,
전략환경평가와 국제협약 이행체계를 구축하는 활동이다.

- 5.01 생물다양성 법률 및 정책
- 5.02 다른 부처·섹터의 생물다양성 법률·정책
- 5.03 부처 간 조정 및 관리
- 5.04 생물다양성 재정
- 5.05 전략환경평가(SEA) 체계
- 5.06 공간계획
- 5.07 다자간 환경협약(MEA)
- 5.08 자연환경·의사결정 정보 접근 및 FPIC

포함: 국가생물다양성전략 및 행동계획(NBSAP), 정책·보조금 개혁,
생물다양성 계획 수립과 복원 제도, 법률 위반 단속·집행 체계.
주의: 공여 프로젝트의 설계·계획 단계까지 포함할 수 있으나 실제 현장 집행,
보전 지원, 단속 활동은 해당 활동을 직접 설명하는 다른 범주를 우선한다.
다자간 환경협약 이행 지원은 5번이다. 보호지역 내부의 경계 획정과
보호지역 고유의 집행·관리는 7번을 우선한다.

6. 오염 저감

오염 물질이 환경·생태계로 유입되는 것을 막거나, 오염의 양·독성·확산을
줄이고 제거하여 생태계 기능을 유지하는 활동이다.

- 6.01 토양 및 수질 오염
- 6.02 대기 및 기후 오염
- 6.03 하수 및 폐기물 관리
- 6.04 연안·해양 쓰레기 및 오염 잔해
- 6.05 기타 오염(빛·소음·중금속 등)
- 6.06 오염 영향 관리의 기반 조성

주의: 인간 건강 개선이 주목적이면 6번에서 제외한다. 생태계 기능 유지와
생물다양성 압력 저감이 최종 목적이어야 한다. 대기·기후 오염 분야는
명확한 생물다양성 목적이 명시된 경우에만 포함한다. 일반 폐기물·하수처리,
대기질 개선, 온실가스 감축은 생물다양성 목적이 없으면 0이다.

7. 보호지역 및 보전조치(PA & OECM)

지정 보호구역 내부의 인프라 운영·정비, 보호구역 외부 완충지와 연결지역,
생태통로·회랑·경관·해양경관, 현지 내 종 보전, 현지 외 종자·유전자원
보전과 야생종 보호를 위한 활동이다.

- 7.01 보호지역(PA) 및 ICCA 관리·확장
- 7.02 보호구역 외부 완충지 관리
- 7.03 기타 효과적 지역기반 보전조치(OECM)
- 7.04 야생종·이동성 종 보전조치

주의: 보호지역 담당 부처의 일반 행정이나 보호지역과 고유하게 연결되지 않은
일반 기술 도입·운영은 제외한다. 생산성과 경제 목적의 농림수산 관리는
7번이 아니라 9번을 우선 검토한다.

8. 생태계 복원

파괴·훼손된 생태계의 구조와 기능을 물리적으로 회복하고, 종을 재도입·
이식하며, 복원 완료 부지를 사후 관리하는 활동이다. 멸종종·위기종 복원,
자연재해·화재 이후 생태복원도 포함한다.

- 8.01 종의 재도입 및 이식
- 8.02 훼손 부지 재개발 및 공학적 복원
- 8.03 복원 완료 부지 사후 유지관리

주의: 훼손 기업의 법적 의무 이행이나 생물다양성 상쇄(Offset)·순손실 없음
(No Net Loss) 사업은 명확한 순편익(Net Gain)이 없으면 제외한다.
단순 시설복구와 재해복구는 생태계 구조·기능 회복 목적이 없으면 0이다.

9. 지속가능한 자연 이용

생물다양성 구성요소를 장기적 감소가 일어나지 않는 방식과 속도로 이용하고,
생태계서비스와 공급·생산 활동을 지속가능하게 관리하는 활동이다.

- 9.01 농업생물다양성 관리
- 9.02 지속가능한 농업
- 9.03 지속가능한 양식업
- 9.04 지속가능한 어업
- 9.05 지속가능한 임업
- 9.06 지속가능한 토지관리(UNCCD)
- 9.07 지속가능한 해안·해양 관리
- 9.08 지속가능한 방목지 관리
- 9.09 지속가능한 야생동물 관리·사냥

주의: 생물다양성 목적의 숲·농지·어장 관리, 토양유실 등 인간 활동을 포함한
생산경관 관리는 9번이다. 형질·품종 개량과 일반 생산성 향상은 지속가능한
이용의 생물다양성 목적이 없으면 제외한다. 대체재 육종은 저서식·다양성
훼손 저감 등 생물다양성 목적이 명시된 경우만 포함한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[직접성 및 BAR 참고]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 1, 3, 7, 8: 직접 지출 성격, 일반적인 BAR 참고 범위 75~100%
- 5, 9: 직접 또는 간접 지출, 일반적인 BAR 참고 범위 25~75%
- 2, 4, 6: 간접 지출 성격, 일반적인 BAR 참고 범위 25~50%

이 범위는 분류의 보조 정보일 뿐이며, BAR 수치만으로 카테고리를 선택하지 않는다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[범주 간 우선 판단]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 조사·연구·교육·데이터가 핵심 산출물이면 2번.
- 법률·전략·재정·조정·SEA 제도가 핵심이면 5번.
- 특정 개발사업의 EIA와 경제활동 전환이 핵심이면 4번.
- 오염 자체의 생태계 압력 저감이 핵심이면 6번.
- 보호구역·OECM·야생종의 현재 보전이 핵심이면 7번.
- 이미 훼손된 생태계·서식지의 회복이 핵심이면 8번.
- 농림수산·토지·해양 자원의 장기적 생산·이용 관리가 핵심이면 9번.
- 침입외래종 또는 GMO/LMO의 생물안전 관리가 핵심이면 3번.
- 유전자원 이용 이익의 공정한 공유가 핵심이면 1번.

최종 응답의 reason에는 선택한 범주의 핵심 기준과 다른 유력 범주를 배제한
이유를 간결하게 설명한다. evidence에는 입력 사업명에 실제로 등장한
근거 표현만 적는다. 입력에 없는 활동이나 목적을 만들어내지 않는다.
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
    "input_text",
    "raw_response",
    "updated_at",
)
EXTRA_OUTPUT_COLUMNS = ("confidence", "reason", "evidence")


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="예산 사업을 Ollama로 BIOFIN 1차 카테고리 0~9로 분류합니다."
    )
    parser.add_argument(
        "--input-file", type=Path, default=project_dir / DEFAULT_INPUT_FILE
    )
    parser.add_argument("--output-dir", type=Path, default=project_dir / "outputs/llm")
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--cache-csv", type=Path, default=None)
    parser.add_argument("--audit-csv", type=Path, default=None)
    parser.add_argument("--review-csv", type=Path, default=None)
    parser.add_argument("--label-col", default=DEFAULT_LABEL_COLUMN)
    parser.add_argument(
        "--gold-label-col",
        default=DEFAULT_GOLD_LABEL_COLUMN,
        help="정확도 평가에 사용할 원래 정답 컬럼",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
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
    return re.sub(r"\s+", " ", str(value or "").strip())


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


def build_prompt(row: dict[str, str]) -> str:
    return PROMPT_TEMPLATE.format(
        classification_prompt=SYSTEM_PROMPT.strip(),
        **prompt_values(row),
    ).strip()


def parse_jsonish_response(text: str) -> dict[str, Any]:
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

    try:
        label = int(data["label"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("label이 정수가 아닙니다.") from exc
    if label not in VALID_LABELS:
        raise ValueError(f"label 범위 오류: {label}")

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


def call_ollama(prompt: str, args: argparse.Namespace) -> str:
    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "top_p": 0.1, "num_ctx": 4096},
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
        body = json.loads(response.read().decode("utf-8"))
    return str(body.get("response", ""))


def classify(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            result = parse_jsonish_response(call_ollama(build_prompt(row), args))
            if args.delay > 0:
                time.sleep(args.delay)
            return result
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt < args.retries:
                time.sleep(args.retry_delay)
    return {
        "label": "",
        "confidence": 0.0,
        "reason": f"분류 실패: {last_error}",
        "evidence": "",
        "raw_response": "",
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


def valid_cached_label(record: dict[str, Any] | None) -> bool:
    if not record:
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
            "input_text": item["input_text"],
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
        "reason", "evidence", "input_text", "raw_response",
    ]
    write_csv(args.audit_csv, audit_headers, audit_rows)
    write_csv(args.review_csv, audit_headers, review_rows)

    counts = Counter(str(row.get("label", "")) for row in cache.values())
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
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
    set_default_paths(args)
    headers, rows, encoding = read_csv(args.input_file)
    missing = [column for column in KEY_COLUMNS if column not in headers]
    if missing and "business_key" not in headers:
        raise ValueError(f"고유 사업 키 컬럼이 부족합니다: {', '.join(missing)}")

    items = collect_items(rows)
    print(f"입력: {args.input_file} ({len(rows):,}행, {encoding})")
    print(f"고유 사업: {len(items):,}개")
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
