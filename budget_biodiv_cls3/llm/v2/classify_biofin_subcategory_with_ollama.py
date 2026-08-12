"""
Ollama LLM으로 예산 사업을 BIOFIN 하위 카테고리 코드로 분류합니다.

SYSTEM_PROMPT에는 한국 BIOFIN 카테고리별 분류 기준과 대표사업을
반영했습니다.

사용 예:
    python llm/v2/classify_biofin_subcategory_with_ollama.py --dry-run
    python llm/v2/classify_biofin_subcategory_with_ollama.py --limit-keys 10
    python llm/v2/classify_biofin_subcategory_with_ollama.py --overwrite
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


DEFAULT_MODEL = "gemma3:12b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_INPUT_FILE = Path("document/2023biofin_label_matched.csv")
DEFAULT_LABEL_COLUMN = "LLM BIOFIN 하위 카테고리"
DEFAULT_GOLD_LABEL_COLUMN = "하위 카테고리"
PROMPT_VERSION = "kr-biofin-subcategory-2026-08-12-v2"
VALID_LABELS = (
    "0",
    "1.01", "1.02", "1.03", "1.04",
    "2.01", "2.02", "2.03", "2.04", "2.05", "2.06",
    "3.01", "3.02",
    "4.01", "4.02", "4.03", "4.04", "4.05", "4.06", "4.07",
    "5.01", "5.02", "5.03", "5.04", "5.05", "5.06", "5.07", "5.08",
    "6.01", "6.02", "6.03", "6.04", "6.05", "6.06",
    "7.01", "7.02", "7.03", "7.04",
    "8.01", "8.02", "8.03",
    "9.01", "9.02", "9.03", "9.04", "9.05", "9.06", "9.07", "9.08", "9.09",
)
VALID_LABEL_SET = set(VALID_LABELS)
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
# BIOFIN 하위 카테고리 분류 기준입니다.
SYSTEM_PROMPT = """\
너는 대한민국 정부 예산사업을 BIOFIN/GLOBE 생물다양성 지출 기준에 따라
BIOFIN 하위 카테고리 코드 중 정확히 하나로 분류하는 전문 분류자다.

반드시 먼저 상위 범주를 판단한 뒤, 그 범주 아래에서 사업의 핵심 목적과
가장 구체적으로 일치하는 하위 코드 하나를 선택한다. 상위 번호 1~9만
출력해서는 안 된다. 생물다양성 비해당은 "0"으로 출력한다.

허용되는 최종 코드는 다음뿐이다.
0,
1.01, 1.02, 1.03, 1.04,
2.01, 2.02, 2.03, 2.04, 2.05, 2.06,
3.01, 3.02,
4.01, 4.02, 4.03, 4.04, 4.05, 4.06, 4.07,
5.01, 5.02, 5.03, 5.04, 5.05, 5.06, 5.07, 5.08,
6.01, 6.02, 6.03, 6.04, 6.05, 6.06,
7.01, 7.02, 7.03, 7.04,
8.01, 8.02, 8.03,
9.01, 9.02, 9.03, 9.04, 9.05, 9.06, 9.07, 9.08, 9.09.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[분류 목적과 기본 원칙]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 사업의 핵심 활동과 정책 목적을 판단하되 직접적인 보전지출뿐 아니라 오염
   저감, 저탄소 전환, 지속가능한 1차산업처럼 생물다양성에 간접 성과를 내는
   지출도 아래 기준에 명시된 경우 포함한다.
2. 사업설명자료 본문을 최우선으로 사용하고, 세부사업명 → 단위사업명 →
   프로그램명 순으로 보완한다. 입력에 없는 활동을 임의로 만들어내지 않는다.
3. 소관 부처만으로 판단하지 않는다. 다만 '신재생에너지 보급', '하수관로 정비',
   '공익직불', '숲가꾸기'처럼 아래 기준에서 명시적으로 인정한 활동은 사업명과
   본문에서 해당 활동이 확인되면 생물다양성이라는 단어가 없어도 포함한다.
4. 아래 1~9 중 어느 범주에도 명확히 해당하지 않거나 정보가 부족하면 0으로
   분류한다. 일반 행정, 인건비, 기관운영, 일반 정보화, 일반 시설관리,
   일반 산업지원도 아래에서 인정한 직접·간접 활동이 확인되지 않으면 0이다.
5. 여러 범주가 가능하면 사업의 핵심 목적과 예산의 주된 활동을 가장 구체적으로
   설명하는 한 범주만 선택한다. 단순히 관련될 수 있는 범주를 나열하지 않는다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[상위 카테고리 0~9]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

0. 비해당

아래 1~9의 직접 또는 간접 생물다양성 활동에 해당하지 않는 사업이다.
일반 행정·인건비·기관운영·채무상환·단순 소득지원처럼 인정 활동이 확인되지
않는 사업은 0으로 분류한다. 단, 아래 기준에 명시된 간접 편익 사업을 단지
생물다양성 목적이 명시되지 않았다는 이유만으로 0으로 분류하지 않는다.

1. 유전자원 접근 및 이익 공유(ABS)

나고야의정서 이행 등 유전자원의 접근과 이익공유 기반을 구축·운영하는
활동이다. 국가 유전자원·병원체자원 은행, 품종보호·채종원 관리, 야생생물
유전자원 활용 기반, 기술이전·로열티·관련 기금과 유전자원 관련 ODA를 포함한다.

- 1.01 생물다양성 지역·유전자원 스크리닝 및 접근 허가
- 1.02 유전자원 접근·이용 계약 체결
- 1.03 유전자원 이익공유 메커니즘
- 1.04 나고야의정서 이행
- 유전자원 정보 접근권 보장

대표 포함 사례: 감염병표준실험실운영, 산림품종보호·채종원관리,
야생생물 유전자원 활용지원기반 구축. 일반 생명공학 연구만 수행하고
유전자원 접근·은행·품종보호·이익공유 기반이 없으면 2번을 검토한다.

2. 생물다양성 인식 제고 및 연구

생물다양성 인식을 높이는 조사·연구·모니터링·전시·교육·홍보 활동이다.
생태·생물자원 조사연구기관 출연, 생명공학 R&D와 환경·저탄소 전환 관련
핵심기술 연구개발을 포함한다.

- 2.01 정규 생물다양성 교육
- 2.02 비정규 교육 및 기술 훈련
- 2.03 대중 인식 제고 및 소통
- 2.04 생물다양성 과학 연구
- 2.05 원주민·지역사회 지식 보전
- 2.06 CBD 정보공유체계

대표 포함 사례: 신재생에너지핵심기술개발(R&D), 한국원자력연구원 연구
운영비 지원(R&D), 국립생태원 출연. 연구·기술개발 자체가 핵심 산출물이면
2번, 기술의 보급·융자·시설 구축은 4번이다. 보호·복원사업의 현장 성과
모니터링은 각각 7번 또는 8번을 우선한다.

3. 생물안전성

외래 병해충·수생질병·침입외래종의 유입 차단, 국경검역, 예찰·방제와
검역기술 고도화 R&D, GMO/LMO 안전관리를 포함한다.

- 3.01 침입외래종(IAS) 관리
- 3.02 유전자변형생물체(GMO/LMO) 관리

대표 포함 사례: 식물검역검사및수출촉진, 수산물품질관리 중 검역·질병관리,
수산물검역검사. 국경 유입 차단·검역체계·광역 예찰은 3번, 보호지역 내부의
부지 기반 방제는 7번, 훼손지 회복을 위한 방제는 8번을 우선한다.

4. 친환경 경제 전환

순환경제와 기후 대응을 위한 저탄소 경제 전환이다. 신재생에너지 보급·융자·
발전차액지원, 수소경제 인프라, 녹색 교통, 도시 그린 인프라와 지속가능한
공급망·소비·관광·광업 투자를 포함한다.

- 4.01 녹색 공급망 구축
- 4.02 지속가능한 광업
- 4.03 지속가능한 소비
- 4.04 지속가능한 에너지
- 4.05 지속가능한 관광
- 4.06 지속가능한 교통
- 4.07 지속가능한 도시·지역

대표 포함 사례: 신재생에너지금융지원(융자), 신재생에너지보급지원,
신재생에너지발전차액지원. 이러한 보급·금융·인프라 활동은 생물다양성이라는
단어가 없어도 간접 편익을 인정한다. 연구개발이 핵심이면 2번, 단순 기관운영·
일반 산업보조로서 녹색전환 활동이 확인되지 않으면 0번이다.

5. 생물다양성 기획 및 재정

생물다양성 계획·정책·재정·법률·집행과 국토·연안 공간계획을 위한 활동이다.
개발제한구역 지정·관리, 간척지 개발의 계획·관리, 환경 부담금 징수,
세입징수비용 교부와 관련 기금 운영을 포함한다.

- 5.01 생물다양성 법률 및 정책
- 5.02 다른 부처·섹터의 생물다양성 법률·정책
- 5.03 부처 간 조정 및 관리
- 5.04 생물다양성 재정
- 5.05 전략환경평가(SEA) 체계
- 5.06 공간계획
- 5.07 다자간 환경협약(MEA)
- 5.08 자연환경·의사결정 정보 접근 및 FPIC

대표 포함 사례: 새만금지구개발 중 계획·관리, 개발제한구역관리,
세입징수비용교부금. 계획·지정·재정·제도·기금이 핵심이면 5번이고,
실제 오염시설·보호지역 관리·복원공사·농림수산 이용관리는 각각 6·7·8·9번을
우선한다.

6. 오염 저감

오염 물질이 환경·생태계로 유입되는 것을 막거나, 오염의 양·독성·확산을
줄이고 제거하여 생태계 기능을 유지하는 활동이다.

- 6.01 토양 및 수질 오염
- 6.02 대기 및 기후 오염
- 6.03 하수 및 폐기물 관리
- 6.04 연안·해양 쓰레기 및 오염 잔해
- 6.05 기타 오염(빛·소음·중금속 등)
- 6.06 오염 영향 관리의 기반 조성

일반 오염관리의 간접적인 생물다양성 편익을 인정한다. 하수관로 정비,
하수처리장 설치·운영, 하수관로 BTL 임대료, 수질·대기·토양 오염 대응,
해양 오염 대응과 오염 측정·감시 기반을 포함한다. 대표 포함 사례는
하수관로 정비, 하수처리장 설치, 하수관로정비 BTL사업 임대료 지급이다.
온실가스 감축·재생에너지 전환은 오염처리보다 4번을 우선한다.

7. 보호지역 및 보전조치(PA & OECM)

생태계 건전성·대표성 유지, 멸종위기종 보호와 공·사유지의 생물다양성 손실
방지를 위한 활동이다. 국립공원·지질공원 등 보호지역 운영·관리, 공단 출연,
토지 매수, 보호지역 인프라와 부지 기반 관리, 보전 성과 모니터링을 포함한다.

- 7.01 보호지역(PA) 및 ICCA 관리·확장
- 7.02 보호구역 외부 완충지 관리
- 7.03 기타 효과적 지역기반 보전조치(OECM)
- 7.04 야생종·이동성 종 보전조치

대표 포함 사례: 국립공원 및 지질공원사업, 국립공원공단출연,
토지등의 매수. 생산경관의 지속가능한 농림수산 관리는 9번을 우선한다.

8. 생태계 복원

훼손된 생태계의 복원·재활이 주목적인 활동이다. 댐 유역 생태계 복원,
바다숲·산란서식장 조성, 청정어장 재생, 수산자원 조성, 훼손요인 제거와
서식지 재조성, 복원 후 관리를 포함한다.

- 8.01 종의 재도입 및 이식
- 8.02 훼손 부지 재개발 및 공학적 복원
- 8.03 복원 완료 부지 사후 유지관리

대표 포함 사례: 생태복원 내용이 포함된 댐 운영관리, 수산자원조성사업지원,
친환경양식어업육성 중 어장·서식지 재생 활동. 훼손된 생태계의 회복이면 8번,
장기적인 생산방식 개선이면 9번이다. 단순 시설 유지·재해복구는 제외한다.

9. 지속가능한 자연 이용

생물다양성에 직간접 성과를 내는 육상·담수·해양 1차산업의 지속가능성
활동이다. 공익직불, 숲가꾸기·사방사업, 친환경 농림수산, 유역·어장 관리,
토양유실 방지와 산림 공익기능 관리를 포함한다.

- 9.01 농업생물다양성 관리
- 9.02 지속가능한 농업
- 9.03 지속가능한 양식업
- 9.04 지속가능한 어업
- 9.05 지속가능한 임업
- 9.06 지속가능한 토지관리(UNCCD)
- 9.07 지속가능한 해안·해양 관리
- 9.08 지속가능한 방목지 관리
- 9.09 지속가능한 야생동물 관리·사냥

대표 포함 사례: 공익기능증진직불, 숲가꾸기, 사방사업. 생산경관을 지속가능하게
유지·관리하면 9번, 훼손 생태계의 원상 회복은 8번, 보호지역·야생종 보전은
7번이다. 환경·공익 기능이 없는 단순 생산량 확대·가격지원·소득보전은 0번이다.

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

- 조사·연구·교육·데이터·기술개발 R&D가 핵심 산출물이면 2번.
- 법률·전략·재정·조정·SEA 제도가 핵심이면 5번.
- 저탄소 기술의 보급·금융·인프라와 순환경제 전환은 4번.
- 하수·폐기물·수질·대기·토양·해양 오염 저감은 6번.
- 보호구역·OECM·야생종의 현재 보전이 핵심이면 7번.
- 이미 훼손된 생태계·서식지의 회복이 핵심이면 8번.
- 농림수산·토지·해양 자원의 장기적 생산·이용 관리가 핵심이면 9번.
- 외래 병해충·수생질병 국경검역, 침입외래종 또는 GMO/LMO 관리는 3번.
- 유전자원 은행·품종보호·채종원·접근·이익공유 기반은 1번.

대표사업과 부처 정보는 유사 활동을 이해하는 예시일 뿐이다. 부처명이나
사전에 제시된 사업 수·금액·비중을 근거로 분류하거나 목표 비율을 맞추지 않는다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[한국 대표 활동의 하위 코드 적용]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 유전자원·병원체자원 은행, 자원 수집·보존·접근 기반은 1.01.
- 유전자원 계약·이용조건은 1.02, 이익공유·기술이전·로열티는 1.03,
  나고야의정서 이행과 관련 국제협력·ODA 역량강화는 1.04.
- 생태·생물자원·생명공학·환경·저탄소 핵심기술 R&D와 연구기관 출연은 2.04.
- 외래 병해충·수생질병의 국경검역·예찰·방제·검역기술 R&D는 3.01.
- 신재생에너지 보급·융자·발전차액·수소경제 인프라는 4.04,
  도시 그린 인프라는 4.07로 분류한다.
- 환경 부담금·세입징수·관련 기금은 5.04, 국토·연안계획·개발제한구역·
  간척지 계획관리는 5.06을 우선한다.
- 수질·토양 오염은 6.01, 대기오염은 6.02, 하수관로·하수처리장·폐기물은
  6.03, 해양오염은 6.04로 분류한다.
- 국립공원·지질공원 관리·공단 출연·보호지역 토지매수는 7.01,
  보호지역 밖의 효과적 지역보전 토지매수는 7.03을 검토한다.
- 바다숲·산란서식장·청정어장 재생·댐 유역 생태복원 같은 물리적 복원은
  8.02, 복원 완료 부지·어장의 사후관리는 8.03.
- 공익직불·친환경 농업은 9.02, 친환경 양식은 9.03, 지속가능한 어업·
  어장관리는 9.04, 숲가꾸기·산림 공익기능은 9.05, 사방·토양유실 방지는
  9.06, 연안·해양 이용관리는 9.07을 우선한다.

최종 응답의 reason에는 선택한 상위 범주와 하위 코드의 핵심 기준, 다른 유력
범주를 배제한 이유를 간결하게 설명한다. evidence에는 사업명 또는
사업설명자료 본문에 실제로 등장한 근거 표현만 적는다. 입력에 없는 활동이나
목적을 만들어내지 않는다.
"""

# 시스템 프롬프트와 별도로 유지하는 입출력 형식입니다.
# 하위 코드 하나를 JSON으로 반환하게 합니다.
PROMPT_TEMPLATE = """\
{classification_prompt}

다음 예산 사업을 BIOFIN 하위 카테고리 코드 중 하나로 분류하라.

소관명: {소관명}
회계명: {회계명}
계정명: {계정명}
분야명: {분야명}
부문명: {부문명}
프로그램명: {프로그램명}
단위사업명: {단위사업명}
세부사업명: {세부사업명}

사업설명자료 본문:
{document_text}

반드시 아래 JSON 객체만 반환하라.
{{
  "label": "0",
  "confidence": 0.0,
  "reason": "",
  "evidence": ""
}}

label은 허용된 하위 코드 문자열 또는 비해당 "0"이어야 한다.
reason에는 선택한 상위 범주와 하위 코드의 판단 근거를 모두 설명한다.
confidence는 0.0부터 1.0까지다.
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
        description="예산 사업을 Ollama로 BIOFIN 하위 카테고리 코드로 분류합니다."
    )
    parser.add_argument(
        "--input-file", type=Path, default=project_dir / DEFAULT_INPUT_FILE
    )
    parser.add_argument("--output-dir", type=Path, default=project_dir / "outputs/llm/v2")
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
        help="프롬프트에 포함할 사업설명자료 본문 최대 문자 수",
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
        args.cache_csv = args.output_dir / "subcategory_label_cache.csv"
    if args.audit_csv is None:
        args.audit_csv = args.output_dir / "subcategory_label_audit.csv"
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
        prompt_text = truncate_document(full_text, args.max_document_chars)
        return prompt_text, "PARSED", str(path), len(prompt_text)
    except (DocumentParseError, RuntimeError, OSError) as exc:
        return (
            f"[사업설명자료 파싱 실패: {clean_cell(exc)[:300]}]",
            "PARSE_FAILED",
            str(path),
            0,
        )


def parse_jsonish_response(text: str) -> dict[str, Any]:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            label_match = re.search(
                r"(?<![\d.])(0|[1-9]\.0[1-9])(?![\d.])",
                raw,
            )
            if not label_match:
                raise ValueError("응답에서 유효한 BIOFIN 하위 코드를 찾지 못했습니다.")
            data = {
                "label": label_match.group(1),
                "confidence": 0.5,
                "reason": "JSON 외 응답에서 라벨 추출",
                "evidence": "",
            }
        else:
            data = json.loads(match.group(0))

    label = parse_valid_label(data.get("label"))
    if label is None:
        raise ValueError(f"허용되지 않은 하위 카테고리: {data.get('label')}")

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
        body = json.loads(response.read().decode("utf-8"))
    return str(body.get("response", ""))


def classify(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    document_text, document_status, document_path, document_chars = (
        load_document_for_prompt(row, args)
    )
    prompt = build_prompt(row, document_text)
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            result = parse_jsonish_response(call_ollama(prompt, args))
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
        "PARSED", "NOT_FOUND", "PARSE_FAILED", "DISABLED"
    }:
        return False
    if record.get("prompt_version") != PROMPT_VERSION:
        return False
    return parse_valid_label(record.get("label")) is not None


def parse_valid_label(value: Any) -> str | None:
    """값을 허용된 BIOFIN 하위 코드로 정규화합니다."""
    text = clean_cell(value)
    if not text:
        return None
    if re.fullmatch(r"0(?:\.0+)?", text):
        return "0"
    match = re.match(r"^(0|[1-9]\.0[1-9])(?:\s|$)", text)
    if match and match.group(1) in VALID_LABEL_SET:
        return match.group(1)
    return None


def evaluate_predictions(
    rows: list[dict[str, Any]],
    gold_column: str,
    pred_column: str,
    output_dir: Path,
) -> dict[str, Any] | None:
    """정답과 예측을 비교해 정확도·클래스별 지표·혼동행렬을 저장합니다."""
    evaluated: list[tuple[str, str, dict[str, Any]]] = []
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

    label_to_index = {label: index for index, label in enumerate(VALID_LABELS)}
    label_count = len(VALID_LABELS)
    matrix = [[0 for _ in range(label_count)] for _ in range(label_count)]
    incorrect_rows: list[dict[str, Any]] = []
    correct = 0
    for gold, pred, row in evaluated:
        matrix[label_to_index[gold]][label_to_index[pred]] += 1
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
    active_label_count = 0
    for label in VALID_LABELS:
        index = label_to_index[label]
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(matrix[row][index] for row in range(label_count))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
            "predicted": predicted,
        }
        if support or predicted:
            macro_precision += precision
            macro_recall += recall
            macro_f1 += f1
            active_label_count += 1

    total = len(evaluated)
    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gold_label_column": gold_column,
        "prediction_column": pred_column,
        "evaluated_rows": total,
        "correct_rows": correct,
        "incorrect_rows": total - correct,
        "accuracy": round(correct / total, 6),
        "macro_precision": round(macro_precision / active_label_count, 6),
        "macro_recall": round(macro_recall / active_label_count, 6),
        "macro_f1": round(macro_f1 / active_label_count, 6),
        "active_label_count": active_label_count,
        "skipped_missing_gold": skipped_missing_gold,
        "skipped_missing_prediction": skipped_missing_prediction,
        "per_class": per_class,
    }
    metrics_path = output_dir / "evaluation_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    confusion_rows = []
    for gold_index, gold in enumerate(VALID_LABELS):
        confusion_rows.append(
            {
                "gold_label": gold,
                **{
                    f"pred_{pred}": matrix[gold_index][pred_index]
                    for pred_index, pred in enumerate(VALID_LABELS)
                },
            }
        )
    write_csv(
        output_dir / "confusion_matrix.csv",
        ["gold_label", *[f"pred_{label}" for label in VALID_LABELS]],
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
