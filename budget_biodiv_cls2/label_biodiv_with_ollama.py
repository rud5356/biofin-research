"""
Ollama 로컬 LLM을 사용해 예산 사업을 생물다양성 관련 여부로 라벨링하는 스크립트.

핵심 설계:
    중복 라벨링 방지: KEY_COLUMNS(소관명, 분야명, 부문명, 프로그램명, 단위사업명, 세부사업명)
    로 고유 사업 조합을 식별하고, SHA256 해시(24자)로 캐시 키를 만듭니다.
    캐시(label_cache.csv)를 사용하므로 중단 후 재실행해도 처음부터 다시 시작하지 않습니다.

LLM 응답 처리:
    Ollama는 format=json으로 요청하더라도 마크다운 코드블록이나
    JSON 외 텍스트를 포함한 응답을 반환할 수 있습니다.
    parse_jsonish_response()는 여러 단계로 응답을 파싱합니다.

사용 예:
    python label_biodiv_with_ollama.py
    python label_biodiv_with_ollama.py --dry-run           # Ollama 호출 없이 구조 확인
    python label_biodiv_with_ollama.py --overwrite         # 기존 캐시 무시하고 재라벨링
    python label_biodiv_with_ollama.py --limit-keys 50     # 50개 사업만 테스트
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
from urllib.error import URLError


# ─── 기본 설정값 ─────────────────────────────────────────────────────────────
DEFAULT_MODEL       = "llama3.1:8b"
DEFAULT_OLLAMA_URL  = "http://localhost:11434"
DEFAULT_INPUT_GLOB  = "세부사업 예산편성현황(총액)_*.csv"
DEFAULT_LABEL_COLUMN = "biodiv_label"

# CSV 읽기 시도할 인코딩 순서 (EUC-KR 기반 파일이 많음)
ENCODINGS = ("utf-8-sig", "cp949", "utf-8")

# 고유 사업 조합을 정의하는 키 컬럼들
# 같은 조합의 행은 연도가 달라도 동일 사업으로 간주해 캐시를 재사용합니다.
KEY_COLUMNS = (
    "소관명",
    "분야명",
    "부문명",
    "프로그램명",
    "단위사업명",
    "세부사업명",
)

# 캐시에 저장하는 LLM 출력 컬럼들
OUTPUT_COLUMNS = (
    "label",
    "confidence",
    "reason",
    "evidence",
    "raw_response",
)


# LLM에게 보내는 분류 프롬프트 템플릿
# temperature=0, top_p=0.1로 재현성 높은 결과를 얻습니다.
PROMPT_TEMPLATE = """\
너는 대한민국 재정사업이 생물다양성(Biodiversity)과 관련될 가능성이 있는지 판단하는 BIOFIN 전문 분류자이다.

목적은 사업이 생물다양성 보전, 지속가능한 이용, 생태계 복원, 생태계 관리, 생물자원 관리, 생태계서비스 유지 등에 기여할 가능성이 있는지를 폭넓게 식별하는 것이다.

직접적으로 생물다양성을 목적으로 하는 사업뿐 아니라, 생태계·서식지·야생생물·산림·습지·해양·연안·생물자원·생태복원·생태계서비스 등에 간접적으로 기여할 가능성이 있는 사업도 1로 분류한다.

보수적으로 제외하지 말고, 생물다양성과의 연관 가능성이 존재하면 1로 판단한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[판단 우선순위]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

아래 순서대로 종합 판단한다.

1순위 : BIOFIN 9대 범주
2순위 : UNDP(2018) BIOFIN 원문 분류 내용
3순위 : KEI BIOFIN 세부 분류 기준
4순위 : 생물다양성 및 자연경관 보호 키워드

상위 기준과 하위 기준이 동시에 충족되면 높은 신뢰도로 1을 부여한다.

키워드는 보조 신호일 뿐이며, 키워드가 존재하더라도 사업 맥락상 생물다양성과 무관하면 0이 될 수 있다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[1순위 : BIOFIN 9대 범주]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

① 보호구역 및 기타 보전 조치

* 보호구역 지정·확대·관리
* 국립공원
* 생태경관보전지역
* 자연보호구역
* 생물권보전지역
* 람사르습지
* 야생생물 보호
* 생태축 및 생태통로
* 종은행
* 서식지 보호

② 생태계 복원

* 산림 복원
* 습지 복원
* 하천 복원
* 연안 복원
* 갯벌 복원
* 자연성 회복
* 생태통로 조성
* 멸종위기종 복원
* 종 복원
* 훼손지 복원

③ 유전자원 접근 및 이익공유(ABS)

* 나고야의정서
* 유전자원 등록
* 유전자원 관리
* 이익공유 체계 구축

④ 지속가능한 이용 및 생물안전

* 지속가능 농업
* 지속가능 임업
* 지속가능 어업
* 지속가능 양식업
* 생물자원 관리
* 외래종 방제
* 외래종 예찰
* 외래종 퇴치
* GMO 관리
* LMO 관리
* 생물안전

⑤ 오염관리

* 수질오염 저감
* 토양오염 저감
* 해양오염 저감
* 폐수관리
* 폐기물관리
* 비점오염 관리
* 수생태계 건강성 회복
* 지하수 오염관리
* 대기오염 관리

⑥ 생물다양성 인식 제고 및 지식

* 생물다양성 조사
* 생태계 조사
* 모니터링
* DB 구축
* 정보시스템 구축
* 연구
* 교육
* 홍보
* 시민참여
* 생태계서비스 가치평가
* 기술개발

⑦ 녹색경제

* 친환경 농업
* 친환경 임업
* 친환경 수산업
* 유기농업
* 생태관광
* 지속가능 관광
* 지속가능 산림경영
* 친환경 인증
* 생물자원 기반 산업

⑧ 생물다양성 및 개발계획

* 국가생물다양성전략
* 생물다양성 정책
* 생물다양성 법률
* 생물다양성 계획
* 전략환경영향평가
* 공간계획
* 생물다양성 재정
* 다자간 환경협약

⑨ 기타 생물다양성 관련 활동

* 전통 생태지식
* 지역 생태공동체
* 기타 생물다양성 보전 기여 사업

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[2순위 : UNDP(2018) BIOFIN 원문 분류 내용]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

① 생물다양성 인식개선과 지식 공유

* 생물다양성 관련 DB 구축 및 공간 연계
* 생물다양성 관련 교육
* 세미나
* 토론회
* 대국민 캠페인
* 생태공원 방문 및 견학 프로그램
* 생물다양성 관련 연구(R&D)
* 생태계서비스 가치 추정
* 전통지식 활용

② 녹색경제

* 환경영향평가
* 생물자원을 이용한 온실가스 저감
* 지속가능한 채굴 및 벌목
* 생물자원을 이용한 지속가능 에너지
* 지속가능 관광(생태관광)

③ 오염저감

* 토양오염관리
* 지하수 오염관리
* 하천 오염관리

④ 지속가능한 이용

* 농업분야 관련 생물다양성
* 지속가능 농업
* 지속가능 어업·양식업
* 지속가능 임업
* 지속가능한 해양 및 해안관리
* 수변관리

⑤ 생태안전

* 외래종 방역·방제·예찰
* 유전자변형물질 관리

⑥ 서식지 보호와 다른 보존 수단

* 보호구역 확대와 관리
* 야생동물 불법 거래 금지
* 서식지 관리
* 생태통로 및 생태축 관리
* 종은행

⑦ 생태계 복원

* 종 복원
* 훼손 지역 복원
* 재해 복구

⑧ 자연환경 접근 및 참여, ABS, 경관 특성 강화

* 생물다양성 이용에 대한 재정 지불
* 녹지 확대 및 접근 확대
* 휴양림 등 이용 확대
* 경관 특성 강화

⑨ 생물다양성 정책, 계획, 국제협력

* 생물다양성법
* 정책
* 계획
* 전략환경영향평가
* 공간계획
* ODA
* 다자간 환경 협약

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[3순위 : KEI BIOFIN 세부 분류 기준]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

① 생물다양성 인식 및 지식 증진

* 서식지 및 생물종 데이터 생성
* 공간정보 연계
* 교육
* 캠페인
* 세미나
* 토론회
* 생태공원 견학
* 연구개발(R&D)
* 생태계서비스 가치 추정
* 전통지식 활용
* CBD Clearing House Mechanism

② 오염저감

* 토양오염관리
* 지하수오염관리
* 지표수오염관리
* 대기오염관리
* 기후변화관리
* 폐기물관리
* 폐수관리

③ 녹색경제

* 기업의 사회적 책임
* 환경영향평가
* 온실가스 저감
* 녹색물류
* 지속가능한 채굴
* 지속가능한 벌목
* 지속가능한 소비
* 지속가능한 에너지(재생에너지)
* 지속가능한 투자
* 지속가능한 관광
* 지속가능한 교통
* 지속가능한 인프라
* 지속가능한 도심 관리

④ 지속가능한 이용

* 농업생물다양성
* 지속가능 농업
* 지속가능 어업
* 지속가능 양식업
* 지속가능 임업
* 지속가능 토지관리
* 지속가능 해양 및 연안관리
* 수변관리

⑤ 바이오세이프티

* 외래종 방역
* 외래종 방제
* 외래종 예찰
* GMO 관리

⑥ 서식지 보호

* 보호구역 확대
* 경관관리
* 서식지관리
* 야생동물 보호
* 생태통로
* 생태축
* 식물원
* 정원
* 종은행

⑦ 복원

* 종 복원
* 훼손지역 복원
* 재해 이후 생태복원

⑧ 이익에 대한 공평한 접근(ABS)

* 생물다양성 이용 이익공유
* 나고야의정서

⑨ 생물다양성 및 개발계획

* 생물다양성 법률
* 정책
* 계획
* 생물다양성 재정
* 전략환경영향평가
* 공간계획
* 다자간 환경협약

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[4순위 : 생물다양성 및 자연경관 보호 키워드]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

대표 키워드

* 생태하천
* 숲
* 조림
* 파크
* 하천생태계

확장 키워드

* 경제림
* 고사목
* 공원
* 군락
* 그늘막
* 나무
* 노거수
* 녹지
* 동식물
* 드론
* 따오기
* 무인감시
* 밀렵
* 바이오매스
* 방재
* 방제
* 방제사업
* 백두대간
* 벌레
* 벌채
* 병해
* 병해충
* 보호구역
* 보호수
* 사방
* 산림
* 산불
* 산사태
* 생물다양성
* 생태
* 서식
* 세계유산
* 소나무
* 수목식재
* 수목원
* 숲가꾸기
* 쉼터
* 습지
* 식목일
* 식물
* 식물원
* 식재
* 씨
* 야생
* 양묘
* 양식
* 어구
* 어류
* 어업
* 어장
* 어종
* 어초
* 에코센터
* 에코스쿨
* 연안
* 예취기
* 우포늪
* 유네스코
* 유류피해
* 자어
* 자연
* 재선충
* 적조
* 정책숲가꾸기
* 제초
* 종묘
* 종자
* 진화
* 천연기념물
* 철새
* 청정
* 치어
* 해변
* 해양
* 해충
* 해파리
* 헬기

위 키워드는 생물다양성 관련 가능성을 높이는 강한 양성 신호로 활용한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[0으로 판단하는 경우]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

아래에만 해당하고 생물다양성과의 연관성이 확인되지 않는 경우에만 0이다.

* 기본경비
* 인건비
* 청사운영
* 위원회운영
* 전출금
* 예치금
* 여유자금 운용
* 일반 행정
* 일반 복지
* 일반 의료
* 일반 교육
* 일반 건축
* 일반 도로
* 일반 교통
* 단순 에너지사업

단, 위 사업이라도 자연환경·생태계·생물자원·산림·습지·해양 관리 목적이 포함되면 1로 판단한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[판단 절차]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 세부사업명 → 단위사업명 → 프로그램명 순으로 검토한다.
2. BIOFIN 9개 범주와의 관련성을 판단한다.
3. UNDP 원문 분류 내용과의 관련성을 판단한다.
4. KEI 세부 분류 기준과의 관련성을 판단한다.
5. 생물다양성 키워드 존재 여부를 확인한다.
6. 직접 관련이 없어도 생태계·서식지·생물종·산림·습지·해양 보전에 기여 가능성이 있으면 1이다.
7. 확실히 무관한 경우만 0이다.
8. 애매한 경계 사례는 1로 분류하되 confidence를 0.50~0.60 수준으로 부여한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[입력 사업 정보]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

소관명: {소관명}
회계코드명: {회계코드명}
계정명: {계정명}
분야명: {분야명}
부문명: {부문명}
프로그램명: {프로그램명}
단위사업명: {단위사업명}
세부사업명: {세부사업명}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[출력 형식]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

JSON만 출력하라.

{{
"label": 0 또는 1,
"biofin_category": "해당 BIOFIN 범주 번호 및 명칭, 해당 없으면 null",
"confidence": 0.0~1.0,
"reason": "판단 근거를 2~3문장으로 서술",
"evidence": "판단에 결정적으로 작용한 사업명 또는 키워드"
}}
"""


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="CSV 예산 사업 행을 Ollama LLM으로 생물다양성 관련 여부 라벨링합니다."
    )
    parser.add_argument("--input-dir",         type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--input-glob",        default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--output-dir",        type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--model",             default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url",        default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--label-col",         default=DEFAULT_LABEL_COLUMN)
    parser.add_argument("--cache-csv",         type=Path, default=None,
                        help="라벨 캐시 저장 경로 (기본: output-dir/label_cache.csv)")
    parser.add_argument("--audit-csv",         type=Path, default=None,
                        help="전체 검수 결과 저장 경로")
    parser.add_argument("--review-csv",        type=Path, default=None,
                        help="낮은 신뢰도 항목 별도 저장 경로")
    parser.add_argument("--review-threshold",  type=float, default=0.7,
                        help="이 값 미만의 confidence는 review_needed.csv에 저장")
    parser.add_argument("--delay",             type=float, default=0.05,
                        help="Ollama 호출 간 대기 시간(초)")
    parser.add_argument("--timeout",           type=int,   default=60,
                        help="Ollama 응답 대기 최대 시간(초). CPU에서는 30~60 권장")
    parser.add_argument("--retries",           type=int,   default=1)
    parser.add_argument("--retry-delay",       type=float, default=1.0)
    parser.add_argument("--workers",           type=int,   default=3,
                        help="동시 Ollama 호출 스레드 수 (기본: 3)")
    parser.add_argument("--save-every",        type=int,   default=20,
                        help="N개 라벨링마다 캐시를 중간 저장")
    parser.add_argument("--limit-keys",        type=int,   default=0,
                        help="테스트용: 앞 N개 고유 사업 조합만 라벨링")
    parser.add_argument("--dry-run",           action="store_true",
                        help="Ollama 호출 없이 입력 구조와 중복 키만 확인")
    parser.add_argument("--overwrite",         action="store_true",
                        help="기존 캐시 라벨도 다시 생성")
    parser.add_argument("--no-json-format",    action="store_true",
                        help="Ollama format=json 옵션을 끕니다")
    return parser.parse_args()


def read_csv_file(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    """
    여러 인코딩을 순서대로 시도해 CSV를 읽습니다.

    반환값: (헤더 목록, 행 딕셔너리 목록, 성공한 인코딩)
    모든 인코딩이 실패하면 RuntimeError를 발생시킵니다.
    """
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                reader = csv.DictReader(file)
                if not reader.fieldnames:
                    raise ValueError("CSV 헤더가 없습니다.")
                rows = [dict(row) for row in reader]
            return list(reader.fieldnames), rows, encoding
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"CSV 읽기 실패: {path}") from last_error


def clean_surrogates(value: Any) -> str:
    """
    UTF-16 서로게이트 쌍을 올바른 유니코드 문자로 변환합니다.

    Windows에서 특수 문자가 포함된 한국어 파일을 읽으면
    서로게이트 쌍(0xD800~0xDFFF)이 남아있을 수 있습니다.
    이를 변환하지 않으면 JSON/CSV 저장 시 오류가 발생합니다.

    상위 서로게이트(0xD800~0xDBFF) + 하위 서로게이트(0xDC00~0xDFFF) 쌍을
    하나의 유니코드 문자(U+10000 이상)로 결합합니다.
    """
    text    = str(value or "")
    cleaned: list[str] = []
    index = 0

    while index < len(text):
        code = ord(text[index])

        if 0xD800 <= code <= 0xDBFF:
            # 상위 서로게이트: 다음 문자가 하위 서로게이트이면 쌍 처리
            if index + 1 < len(text):
                low = ord(text[index + 1])
                if 0xDC00 <= low <= 0xDFFF:
                    # UTF-16 서로게이트 쌍 → 실제 유니코드 코드포인트 계산
                    cleaned.append(chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)))
                    index += 2
                    continue
            # 쌍을 이루지 않는 상위 서로게이트는 제거
            index += 1
            continue

        if 0xDC00 <= code <= 0xDFFF:
            # 고아 하위 서로게이트 제거
            index += 1
            continue

        cleaned.append(text[index])
        index += 1

    return "".join(cleaned)


def write_csv_file(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """
    행 목록을 UTF-8 BOM CSV로 저장합니다.

    모든 값에 clean_surrogates를 적용해 서로게이트 관련 인코딩 오류를 방지합니다.
    extrasaction="ignore": fieldnames에 없는 키는 무시합니다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {key: clean_surrogates(value) for key, value in row.items()}
            for row in rows
        )


def clean_cell(value: Any) -> str:
    """값을 문자열로 변환하고 연속 공백을 단일 공백으로 정리합니다."""
    return re.sub(r"\s+", " ", clean_surrogates(value).strip())


def build_prompt_values(row: dict[str, str]) -> dict[str, str]:
    """프롬프트 템플릿에 삽입할 컬럼별 값을 추출합니다."""
    columns = (
        "회계연도", "소관명", "회계코드명", "계정명",
        "분야명", "부문명", "프로그램명", "단위사업명", "세부사업명",
    )
    return {column: clean_cell(row.get(column, "")) for column in columns}


def build_input_text(row: dict[str, str]) -> str:
    """캐시에 저장할 사람이 읽기 쉬운 요약 텍스트를 생성합니다."""
    values = build_prompt_values(row)
    return " | ".join(f"{key}: {value}" for key, value in values.items() if value)


def build_key(row: dict[str, str]) -> str:
    """
    KEY_COLUMNS 값을 구분자(U+241F)로 연결해 고유 사업 키를 만듭니다.

    U+241F(UNIT SEPARATOR)는 실제 데이터에 거의 등장하지 않아
    컬럼 값 구분자로 안전하게 사용할 수 있습니다.
    """
    values = [clean_cell(row.get(column, "")) for column in KEY_COLUMNS]
    return "␟".join(values)


def hash_key(key: str) -> str:
    """
    키 문자열을 SHA256 해시값(24자)으로 변환합니다.

    24자는 충돌 가능성이 극히 낮으면서도 CSV 저장에 적당한 길이입니다.
    (SHA256 전체 64자를 저장하면 불필요하게 파일이 커짐)
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def build_prompt(row: dict[str, str]) -> str:
    """행 데이터를 PROMPT_TEMPLATE에 삽입해 완성된 프롬프트를 반환합니다."""
    values = build_prompt_values(row)
    return PROMPT_TEMPLATE.format(**values)


def parse_jsonish_response(text: str) -> dict[str, Any]:
    """
    LLM 응답에서 JSON을 파싱합니다. 여러 형태의 응답을 처리합니다.

    파싱 시도 순서:
    1. 마크다운 코드블록(```json ... ```) 제거 후 직접 파싱
    2. 실패하면 정규표현식으로 '{...}' 부분 추출 후 파싱
    3. 그래도 실패하면 텍스트에서 0 또는 1 숫자 추출
    4. 모두 실패하면 label=-1 오류 딕셔너리 반환

    confidence는 0.0~1.0 범위로 클리핑합니다.
    reason/evidence는 240자로 잘라 저장 공간을 제한합니다.
    """
    raw = text.strip()
    # 마크다운 코드블록 제거 (```json 또는 ``` 시작)
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 직접 파싱 실패: JSON 객체 부분만 추출 시도
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            # JSON 없음: 텍스트에서 0/1 숫자만 추출
            label_match = re.search(r"\b([01])\b", raw)
            if not label_match:
                # 응답은 왔지만 파싱 불가 → 보수적으로 0 처리
                return {
                    "label": 0, "confidence": 0.0,
                    "reason": "응답 파싱 실패 후 0 fallback", "evidence": "",
                    "raw_response": text,
                }
            return {
                "label": int(label_match.group(1)), "confidence": 0.5,
                "reason": "JSON이 아닌 응답에서 숫자만 추출", "evidence": "",
                "raw_response": text,
            }
        data = json.loads(match.group(0))

    # label 값 정수 변환 (0 또는 1만 허용, 아니면 보수적으로 0)
    label = data.get("label", 0)
    try:
        label = int(label)
    except (TypeError, ValueError):
        label = 0
    if label not in {0, 1}:
        label = 0

    # confidence 0.0~1.0 범위로 클리핑
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "label":        label,
        "confidence":   confidence,
        "reason":       clean_cell(data.get("reason",   ""))[:240],
        "evidence":     clean_cell(data.get("evidence", ""))[:240],
        "raw_response": text,
    }


def call_ollama(
    prompt: str,
    model: str,
    ollama_url: str,
    timeout: int,
    use_json_format: bool,
) -> str:
    """
    Ollama REST API를 호출해 LLM 응답을 반환합니다.

    stream=False: 전체 응답을 한 번에 받습니다 (스트리밍 비활성화).
    temperature=0: 항상 같은 결과를 출력 (재현성 확보).
    top_p=0.1: 상위 10% 확률의 토큰만 사용 (보수적 응답).
    num_ctx=4096: 컨텍스트 창 크기 (프롬프트가 이 크기를 초과하면 잘림).
    format=json: JSON 형식 출력 강제 (--no-json-format으로 끌 수 있음).
    """
    payload: dict[str, Any] = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p":       0.1,
            "num_ctx":     4096,
        },
    }
    if use_json_format:
        payload["format"] = "json"

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req  = request.Request(
        f"{ollama_url.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body.get("response", ""))


def classify_with_retries(
    row: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Ollama를 호출해 분류하고, 실패하면 args.retries 회 재시도합니다.

    유효한 label(0 또는 1)을 받으면 즉시 반환합니다.
    모든 시도가 실패하면 label=0 으로 fallback합니다.
    (타임아웃 = 생물다양성 해당 확신 없음 → 보수적으로 0 처리)
    """
    prompt      = build_prompt(row)
    last_error: Exception | None = None

    for attempt in range(args.retries + 1):
        try:
            raw_response = call_ollama(
                prompt=prompt, model=args.model,
                ollama_url=args.ollama_url, timeout=args.timeout,
                use_json_format=not args.no_json_format,
            )
            result = parse_jsonish_response(raw_response)
            if result["label"] in {0, 1}:
                return result
            last_error = RuntimeError("유효하지 않은 label 응답")
        except (URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc

        if attempt < args.retries:
            time.sleep(args.retry_delay)

    # 모든 재시도 소진 — 확신 없으면 0으로 보수적 처리
    return {
        "label": 0, "confidence": 0.0,
        "reason": f"타임아웃 후 0 fallback: {last_error}",
        "evidence": "", "raw_response": "",
    }


def default_output_paths(args: argparse.Namespace) -> None:
    """출력 경로가 지정되지 않은 경우 기본값을 설정합니다."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_csv  is None:
        args.cache_csv  = args.output_dir / "label_cache.csv"
    if args.audit_csv  is None:
        args.audit_csv  = args.output_dir / "label_audit.csv"
    if args.review_csv is None:
        args.review_csv = args.output_dir / "review_needed.csv"


def load_cache(path: Path) -> dict[str, dict[str, str]]:
    """
    기존 라벨 캐시를 읽어 key_hash → 캐시 레코드 딕셔너리로 반환합니다.

    캐시가 없으면 빈 딕셔너리를 반환합니다.
    이 캐시 덕분에 중단 후 재실행해도 이미 라벨링된 항목은 건너뜁니다.
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return {row["key_hash"]: dict(row) for row in reader if row.get("key_hash")}


def save_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    """캐시를 key_hash 순으로 정렬해 CSV로 저장합니다."""
    fieldnames = [
        "key_hash", "label", "confidence", "reason",
        "evidence", "model", "input_text", "raw_response", "updated_at",
    ]
    rows = sorted(cache.values(), key=lambda row: row.get("key_hash", ""))
    write_csv_file(path, fieldnames, rows)


def collect_inputs(
    args: argparse.Namespace,
) -> tuple[dict[Path, dict[str, Any]], dict[str, dict[str, Any]]]:
    """
    입력 디렉토리에서 glob 패턴에 맞는 CSV 파일들을 읽어
    파일별 정보와 고유 사업 키 맵을 반환합니다.

    key_map: key_hash → {key_hash, key, row, input_text, count}
    count: 해당 사업 조합이 몇 개 행에 나타나는지 (연도별 중복 감지)
    """
    csv_paths = sorted(
        path
        for path in args.input_dir.glob(args.input_glob)
        # 출력 폴더 내 파일과 이미 라벨링된 파일 제외
        if path.is_file()
        and args.output_dir not in path.parents
        and not path.name.endswith("_labeled.csv")
    )
    if not csv_paths:
        raise FileNotFoundError(f"입력 CSV를 찾지 못했습니다: {args.input_dir / args.input_glob}")

    files:   dict[Path, dict[str, Any]] = {}
    key_map: dict[str, dict[str, Any]] = {}

    for path in csv_paths:
        headers, rows, encoding = read_csv_file(path)
        # 필수 컬럼이 있는지 확인
        missing = [column for column in KEY_COLUMNS if column not in headers]
        if missing:
            raise ValueError(f"{path.name}에 필수 컬럼이 없습니다: {', '.join(missing)}")

        files[path] = {"headers": headers, "rows": rows, "encoding": encoding}

        for row in rows:
            key      = build_key(row)
            key_hash = hash_key(key)
            if key_hash not in key_map:
                key_map[key_hash] = {
                    "key_hash":   key_hash,
                    "key":        key,
                    "row":        row,
                    "input_text": build_input_text(row),
                    "count":      0,
                }
            key_map[key_hash]["count"] += 1

    return files, key_map


def print_input_summary(
    files:   dict[Path, dict[str, Any]],
    key_map: dict[str, dict[str, Any]],
) -> None:
    """입력 파일과 중복 사업 조합 통계를 출력합니다."""
    total_rows = sum(len(item["rows"]) for item in files.values())
    print("입력 CSV")
    for path, item in files.items():
        print(f"  - {path.name}: {len(item['rows']):,}행, encoding={item['encoding']}")
    print(f"전체 행 수: {total_rows:,}")
    print(f"고유 사업 조합: {len(key_map):,}")

    reuse_counts = Counter(int(item["count"]) for item in key_map.values())
    reused_keys  = sum(1 for item in key_map.values() if int(item["count"]) > 1)
    reused_rows  = sum(int(item["count"]) for item in key_map.values() if int(item["count"]) > 1)
    print(f"2회 이상 재사용되는 조합: {reused_keys:,}개 / {reused_rows:,}행")
    print(f"재사용 분포 상위: {reuse_counts.most_common(5)}")


def build_cache_record(
    key_hash:  str,
    key_item:  dict[str, Any],
    result:    dict[str, Any],
    model:     str,
) -> dict[str, Any]:
    """LLM 결과를 캐시 레코드 형태로 변환합니다."""
    return {
        "key_hash":     key_hash,
        "label":        str(result["label"]),
        "confidence":   f"{float(result.get('confidence', 0.0)):.3f}",
        "reason":       result.get("reason", ""),
        "evidence":     result.get("evidence", ""),
        "model":        model,
        "input_text":   key_item["input_text"],
        "raw_response": result.get("raw_response", ""),
        "updated_at":   datetime.now().isoformat(timespec="seconds"),
    }


def label_unique_keys(
    key_map: dict[str, dict[str, Any]],
    cache:   dict[str, dict[str, Any]],
    args:    argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    """
    캐시에 없는(또는 --overwrite인) 고유 사업 조합을 Ollama로 라벨링합니다.

    --workers 개의 스레드가 동시에 Ollama를 호출합니다.
    Ollama가 CPU 전용일 때는 내부적으로 순차 처리하지만,
    타임아웃 대기 중 다른 요청을 미리 전송해 전체 대기 시간을 줄입니다.

    --save-every: N개 완료마다 캐시 중간 저장 (락으로 보호)
    중단 시 KeyboardInterrupt를 잡아 캐시를 저장하고 재발생시킵니다.
    """
    keys = list(key_map.items())
    if args.limit_keys > 0:
        keys = keys[: args.limit_keys]

    # 아직 유효한 라벨(0/1)이 없는 항목만 라벨링 대상으로 선택
    pending = [
        (key_hash, item)
        for key_hash, item in keys
        if args.overwrite or key_hash not in cache
        or str(cache[key_hash].get("label", "")) not in {"0", "1"}
    ]
    total = len(pending)
    print(f"라벨링 대상 고유 조합: {total:,}개  (동시 호출: {args.workers})")
    if args.limit_keys > 0:
        print(f"주의: --limit-keys {args.limit_keys} 적용 중")

    # 캐시와 카운터를 여러 스레드가 공유하므로 락으로 보호
    lock        = threading.Lock()
    done_count  = [0]   # 리스트로 감싸서 클로저에서 수정 가능하게

    def process_one(key_hash: str, item: dict[str, Any]) -> None:
        """단일 항목을 분류하고 캐시에 저장합니다 (스레드 1개가 실행)."""
        result = classify_with_retries(item["row"], args)
        record = build_cache_record(key_hash, item, result, args.model)

        with lock:
            cache[key_hash] = record
            done_count[0] += 1
            idx        = done_count[0]
            label      = record["label"]
            confidence = record["confidence"]
            print(f"[{idx:,}/{total:,}] {label} conf={confidence} {item['input_text'][:80]}")

            # N개마다 캐시 중간 저장 (락 안에서 수행해 파일 충돌 방지)
            if args.save_every > 0 and idx % args.save_every == 0:
                save_cache(args.cache_csv, cache)

    # Ctrl+C 신호를 받으면 True로 설정 — 스레드들이 이 플래그를 확인해 조기 종료
    stop_flag = threading.Event()

    def process_one_guarded(key_hash: str, item: dict[str, Any]) -> None:
        """stop_flag가 설정되면 즉시 반환합니다."""
        if stop_flag.is_set():
            return
        process_one(key_hash, item)

    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = {
        executor.submit(process_one_guarded, key_hash, item): key_hash
        for key_hash, item in pending
    }

    try:
        for future in as_completed(futures):
            future.result()

    except KeyboardInterrupt:
        print(f"\nWARN: Ctrl+C 감지 — 현재 실행 중인 요청 완료 후 종료합니다...", file=sys.stderr)
        stop_flag.set()
        # 대기 중인 미실행 future 취소
        for f in futures:
            f.cancel()
        # 실행 중인 스레드가 끝날 때까지 대기 (timeout 이내로 종료됨)
        executor.shutdown(wait=True, cancel_futures=True)
        print(f"WARN: 캐시 저장 중: {args.cache_csv}", file=sys.stderr)
        try:
            with lock:
                save_cache(args.cache_csv, cache)
            print("WARN: 캐시 저장 완료. 재실행하면 이어서 시작합니다.", file=sys.stderr)
        except Exception as save_exc:
            print(f"ERROR: 캐시 저장 실패: {save_exc}", file=sys.stderr)
        # os._exit로 스레드 블로킹 없이 즉시 종료
        os._exit(0)

    except Exception as exc:
        stop_flag.set()
        executor.shutdown(wait=False, cancel_futures=True)
        print(f"\nWARN: 오류 발생; 캐시 저장 중: {args.cache_csv}", file=sys.stderr)
        try:
            with lock:
                save_cache(args.cache_csv, cache)
        except Exception as save_exc:
            print(f"ERROR: 캐시 저장 실패: {save_exc}", file=sys.stderr)
        raise exc

    else:
        executor.shutdown(wait=True)
        save_cache(args.cache_csv, cache)

    return cache


EXTRA_COLS = ("confidence", "reason", "evidence")

def output_headers(headers: list[str], label_col: str) -> list[str]:
    """기존 헤더에서 label_col과 부가 컬럼을 제거하고 맨 끝에 추가한 목록을 반환합니다."""
    exclude = {label_col} | set(EXTRA_COLS)
    return [col for col in headers if col not in exclude] + [label_col] + list(EXTRA_COLS)


def write_labeled_outputs(
    files:   dict[Path, dict[str, Any]],
    key_map: dict[str, dict[str, Any]],
    cache:   dict[str, dict[str, Any]],
    args:    argparse.Namespace,
) -> None:
    """
    각 입력 CSV에 라벨 컬럼을 추가해 '{원본파일명}_labeled.csv'로 저장합니다.

    key_hash로 캐시를 조회해 라벨을 채웁니다.
    캐시에 없거나 유효하지 않은 라벨은 빈 문자열로 저장합니다.
    """
    missing_labels = 0
    for path, item in files.items():
        headers    = output_headers(item["headers"], args.label_col)
        rows_out:  list[dict[str, Any]] = []
        for row in item["rows"]:
            row_out  = dict(row)
            key_hash = hash_key(build_key(row))
            cached   = cache.get(key_hash)
            label    = str(cached.get("label", "")) if cached else ""
            if label not in {"0", "1"}:
                missing_labels += 1
                label = ""
            row_out[args.label_col] = label
            for col in EXTRA_COLS:
                row_out[col] = cached.get(col, "") if cached else ""
            rows_out.append(row_out)

        output_path = args.output_dir / f"{path.stem}_labeled.csv"
        write_csv_file(output_path, headers, rows_out)
        print(f"저장: {output_path}")

    if missing_labels:
        print(f"WARN: 라벨이 비어 있는 행 {missing_labels:,}개가 있습니다.")


def write_audit_files(
    key_map: dict[str, dict[str, Any]],
    cache:   dict[str, dict[str, Any]],
    args:    argparse.Namespace,
) -> None:
    """
    전체 검수 파일(label_audit.csv)과 확인 필요 파일(review_needed.csv)을 저장합니다.

    review_needed.csv 조건:
    - label이 0/1이 아닌 경우
    - confidence가 review_threshold 미만인 경우
    낮은 신뢰도 항목은 사람이 직접 검토해야 합니다.
    """
    audit_rows:  list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for key_hash, item in sorted(key_map.items(), key=lambda kv: kv[1]["input_text"]):
        cached = cache.get(key_hash, {})
        row    = {
            "key_hash":     key_hash,
            "row_count":    item["count"],
            "label":        cached.get("label", ""),
            "confidence":   cached.get("confidence", ""),
            "reason":       cached.get("reason", ""),
            "evidence":     cached.get("evidence", ""),
            "input_text":   item["input_text"],
            "raw_response": cached.get("raw_response", ""),
        }
        audit_rows.append(row)

        label = str(row["label"])
        try:
            confidence = float(row["confidence"])
        except (TypeError, ValueError):
            confidence = 0.0
        if label not in {"0", "1"} or confidence < args.review_threshold:
            review_rows.append(row)

    fieldnames = [
        "key_hash", "row_count", "label", "confidence",
        "reason", "evidence", "input_text", "raw_response",
    ]
    write_csv_file(args.audit_csv,  fieldnames, audit_rows)
    write_csv_file(args.review_csv, fieldnames, review_rows)
    print(f"검수 파일: {args.audit_csv}")
    print(f"확인 필요: {args.review_csv} ({len(review_rows):,}건)")


def write_summary(
    files: dict[Path, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    args:  argparse.Namespace,
) -> None:
    """실행 요약 JSON을 output-dir/run_summary.json으로 저장합니다."""
    counts  = Counter(str(row.get("label", "")) for row in cache.values())
    summary = {
        "created_at":          datetime.now().isoformat(timespec="seconds"),
        "model":               args.model,
        "input_files":         {path.name: len(item["rows"]) for path, item in files.items()},
        "cache_rows":          len(cache),
        "label_counts_in_cache": dict(sorted(counts.items())),
        "output_dir":          str(args.output_dir),
        "label_column":        args.label_col,
    }
    path = args.output_dir / "run_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"요약: {path}")


def main() -> int:
    """라벨링 파이프라인 실행."""
    args = parse_args()
    default_output_paths(args)

    files, key_map = collect_inputs(args)
    print_input_summary(files, key_map)

    if args.dry_run:
        print("\n--dry-run: Ollama 호출과 파일 저장 없이 종료합니다.")
        return 0

    cache = load_cache(args.cache_csv)
    print(f"기존 캐시: {len(cache):,}개 ({args.cache_csv})")

    cache = label_unique_keys(key_map, cache, args)
    write_labeled_outputs(files, key_map, cache, args)
    write_audit_files(key_map, cache, args)
    write_summary(files, cache, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
