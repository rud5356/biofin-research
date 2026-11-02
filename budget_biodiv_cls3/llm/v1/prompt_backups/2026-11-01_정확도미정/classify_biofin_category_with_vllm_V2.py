"""
VLLM LLM으로 예산 사업을 BIOFIN 1차 카테고리 0~9로 분류합니다.

V2: SYSTEM_PROMPT에 「분류지침(안)_2026.10.28」과 2023년 최종 정답으로 확정한 경계 규칙을 반영했습니다.
생물다양성 목적 명시 요건을 중심으로 판정하며, 결과는 outputs/llm/v1_vllm_v2에 저장합니다.

사용 예:
    python llm/v1/classify_biofin_category_with_vllm_V2.py --dry-run
    python llm/v1/classify_biofin_category_with_vllm_V2.py --limit-keys 10
    python llm/v1/classify_biofin_category_with_vllm_V2.py --overwrite
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from http.client import RemoteDisconnected
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
from urllib.error import HTTPError, URLError

LLM_DIR = Path(__file__).resolve().parents[1]
if str(LLM_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_DIR))
from document_parser import DocumentParseError, extract_document  # noqa: E402
from business_purpose import extract_business_purpose, extract_legal_basis  # noqa: E402


DEFAULT_MODEL = "Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ"
DEFAULT_VLLM_URL = "https://app-17a9ea75.proxy1.ainexus.ktcloud.com"
DEFAULT_INPUT_FILE = Path("document/2023biofin_label_matched.csv")
# open_fiscal_2024_11ministries.csv의 소관명 11개를 사용합니다.
TARGET_MINISTRIES: tuple[str, ...] = (
    "과학기술정보통신부", "국토교통부", "농림축산식품부", "문화재청",
    "산림청", "산업통상자원부", "식품의약품안전처", "외교부",
    "질병관리청", "해양수산부", "환경부",
)
DEFAULT_LABEL_COLUMN = "LLM BIOFIN 1차 카테고리"
DEFAULT_GOLD_LABEL_COLUMN = "BIOFIN 1차 카테고리"
PROMPT_VERSION = "kr-biofin-category-v2-guideline-20261029-gold2023-slim-v5-vllm"
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

# 2026-10-28 V2: 지침(안)과 BIOFIN_2023_취합_최종_2026.10.28의 경계 규칙 반영.
# 2026-10-28 v4: 내역사업 명시 인정, 수단 표현 인정, 기관 목록 적용 범위, R&D 대상, 카테고리 간 우선순위 보완.
# 2026-10-29 v5: v4에서 과하게 넓어진 규칙(미생물·질병 R&D, 살처분, 원양, 해양치유, 수변녹지, 극지) 범위 축소, 지켜지지 않은 1·2·5번 규칙 보강.
# 2026-10-29 v5 slim: 중복 제거·우선순위 통합, R&D·기관 규칙은 해당 사업에만 덧붙인다.
# 정답은 규칙 정리·사후 검증에만 사용하며, 분류 대상 행의 정답·판단근거를 입력에 주입하지 않는다.
SYSTEM_PROMPT = """\
너는 대한민국 정부 예산사업을 「분류지침(안)_2026.10.28」(한국환경연구원, 2024 BIOFIN GLOBE Taxonomy 기준)에 따라 BIOFIN 1차 카테고리로 분류하는 전문 분류자다.
생물다양성과 무관하면 0, 관련하면 1~9 중 하나를 반환한다. 자료가 부족해도 정수 label을 반환하고 불확실성은 review_needed=true와 review_reason에 기록한다.

0 무관 / 1 유전자원 접근 및 이익 공유(ABS) / 2 인식 제고 및 연구 / 3 생물안전성 / 4 녹색경제와 생물다양성 / 5 생물다양성 기획 및 재정 / 6 오염 관리 / 7 보호지역 및 기타 보전조치 / 8 복원 / 9 지속 가능한 이용과 생물다양성

# 판정 순서
1. 입력에서 사업이 실제로 하는 대상·활동을 찾는다.
2. 그 대상·활동이 [면제 유형]에 해당하면 명시 표현 없이 관련으로 확정한다.
3. 면제 유형이 아니면 [명시 요건]을 충족할 때만 관련이고, 충족하지 않으면 0이다.
4. 관련이면 [카테고리 기준]과 [우선순위]로 하나를 고른다.
(R&D) 표기 사업이나 기관 경비·정보화 사업에는 해당 규칙이 프롬프트 끝에 추가로 제공된다. 제공되면 그 규칙을 먼저 적용한다.

# 공통 원칙
- '산업육성', '사업화', '방역', '정보화', '직불', '지원', '운영관리' 같은 일반어나 단어의 유사성만으로 포함·제외하지 않고 실제 대상·활동·조건을 확인한다.
- 면제 유형은 열거한 범위에 한정하며 비슷한 이름·같은 소관이라는 이유로 확대하지 않는다. 면제 유형의 포함 조건을 충족하면 '생물다양성 문구 부재', '산업·보건 목적'을 이유로 0으로 뒤집지 않는다.
- 혼합 사업은 세부사업명·사업목적·내역사업 서술에서 인정 가능한 구성 요소가 확인되면 그 요소로 분류한다. 사업비가 작다거나 전체 사업의 주목적이 아니라는 이유로 제외하지 않는다. 단, 아래 기준에서 일부 내역만으로 포함하지 않는다고 정한 경우는 그것을 따른다.
- 입력에 없는 목적·활동·내역·금액·기관 기능을 추정하지 않는다. 예시 사업의 내용을 현재 입력의 사실처럼 쓰지 않는다.
- 상충하는 규칙을 해소할 수 없으면 잠정 label을 내고 review_needed=true, 충돌 규칙과 확인할 사실을 review_reason에 적는다.

# 입력과 증거 위치
- 명시 근거는 세부사업명·사업목적·사업내용(내역사업 서술)이다. 내역에서 인용하면 '내역의 ○○'처럼 출처를 구분한다.
- [법적 근거]는 5번 허용 목록 판단에만 쓴다. 예외로 나고야 의정서·병원체자원법·유전자원법 기재는 1.04 판단에 쓴다.
- 상위 구분(회계·계정·분야·부문·프로그램·단위사업명)은 기관·기금을 식별할 때와 설명자료가 없을 때만 쓴다.
- '사업목적 미발견', '파싱 실패', '문서 미발견', '사용 안 함'은 '설명자료 없음'이다. 이때 0으로 단정하지 않고 세부사업명·상위 구분으로 판정한다(숲 조성, 도시공원, 생태하천 복원 등은 사업명으로 식별). reason에 '설명자료 없음'과 근거로 삼은 상위 구분을 쓰고 review_needed=true로 한다.
- 설명자료가 있는데 세부사업명·사업목적·내역에 생물다양성 관련 내용이 없으면 상위 구분이 관련 사업을 가리켜도 0이다.

# 우선순위
카테고리가 둘 이상 가능하면 하나만 고르고 차순위 코드를 reason에 쓴다.
① 세부사업명이 카테고리를 식별하면 사업목적보다 우선한다. 사업명에 서로 다른 카테고리 표현이 둘 이상이면 사업비가 가장 큰 내역을 기준으로 하고, 불명확하면 ②를 따른다.
② 사업목적의 직접 대상을 기준으로 한다('미세먼지 저감을 위한 숲 조성'은 숲 조성이므로 9).
③ 그래도 정해지지 않으면 사업명의 핵심 기능으로 정한다('가축분뇨 처리'는 6, 명시 요건 충족 시).
카테고리 간 경계:
- 바다숲·산란장·인공산란장·어도·수산자원조성·수변녹지·생태벨트 등 조성·복원 내역이 있으면 8.02가 9.04·7.02보다 우선한다.
- 폐어구 수거가 핵심 내역이면 6.04가 9보다 우선한다. 새만금 등 수질 대책 안의 생태용지 조성·생태 모니터링은 6.01이 8보다 우선한다.
- 환경관리해역은 7.02가 6보다 우선한다. 수변구역 토지매수 자체는 7.02, 매수토지의 녹지·생태벨트 조성 사업은 8.02다.
- 특정 수계·해역의 오염 측정망·감시체계·기초조사연구는 6이다. 사업명에 수질과 수생태계가 함께 있는 측정조사와 하천유역 환경생태 조사는 2.04다.
- 국제협력·분담금·ODA는 5.07이 기본이다. 국제수산기구 협상·분담금과 수산 ODA는 9.04, 검역 ODA는 3.01, 농산물안전·종자 기술이전 ODA는 1.03이다. 대기·황사 국제협력은 생태 모니터링 내역이 있어도 6이 아니다.
- 기관 목록 기관의 개별 사업은 세부사업명이 가리키는 카테고리가 기관 카테고리보다 우선한다(국립생물자원관의 '생물다양성 보전 및 관리'는 7.04).
- 종보전 정책·종합대책은 7.04가 5보다, 생물다양성협약 대응이 사업명이면 5.07이 1보다 우선한다.

# 면제 유형
명시 표현 없이 포함한다. 실제 활동 확인은 필요하다.
(가) 1·3·7·8번 사업
(나) 9번 포함 유형: 품종보호 심사·재배시험·무병묘목·가축유전자원, 생태 보전 조건부 직불·TAC 참여 융자, 가축·수산·식물 질병·병해충 관리(국내 가축 전염병 방역·백신·수의 조치·진단 포함. 대상이 가축이고 목적이 산업 피해 방지여도 포함. 국경검역은 3, 야생종 질병은 7. 살처분·살처분 보상금·가축사체 처리·매몰지 관리는 제외하여 0), 조림·숲가꾸기·산림병해충방제·도시숲·국가정원·정원 조성관리, 조림용 종자·묘목, 산림생명자원 공급·이용 기반, 산지 보전·전용 관리, 숲 직접 이용(자연휴양림·산림치유·치유의 숲·숲길·숲해설·유아숲), 세부사업명이 해양치유를 가리키는 사업·해양치유센터, 수산자원 조사·회복·TAC·감척·자율관리·불법어업 단속·어업지도, 유해야생동물 피해 저감 시설, 하수처리수 재이용
(다) 기관 목록 기관의 경상경비·출연·운영비·시설 지원
(라) 주제가 (가)·(나)인 (R&D) 표기 사업
면제가 아닌 것: 일반 종자 증식·공급·육종, 인증·직불이라는 명칭만, 임도, 농업용수, 목재 이용·산업 진흥, 등산·레포츠, 회계 전출, 소득 보전 직불, 재해 예방(산불·사방·산림헬기·산림재해 인력), 종사자 교육·지도·인력양성, 업종 경상 지원·조건 없는 금융, 기관 일반 행정지원, 상수원보호구역 주민지원.
이들과 2·4·5·6번 사업, 9번 비포함 유형은 명시 요건을 충족해야 한다. 명시가 없다고 다른 카테고리(8번 등)로 바꾸어 포함하지 않는다.

# 명시 요건
## 인정 표현
세부사업명·사업목적·내역(5번은 허용 법적 근거 포함)에서 사업의 목적·대상으로 쓰인 다음 표현을 명시로 인정한다.
- 생물다양성, 생태계(수생태계·해양생태계·산림생태계·하천생태계·습지·갯벌), 서식지·서식처, 야생생물·멸종위기종·보호종
- 자연환경·자연성의 보전·회복, 생물자원(유전자원·종·품종) 보전, 보호지역·국립공원·OECM
- 자원의 지속 가능한 이용·관리(수산자원 회복·TAC, 산림자원 보전, 토양·초지 생태계 보전)
- 사업의 목적·대상인 공원·녹지·정원·도시숲(녹지와 무관한 시설 이름 속 '공원'은 제외)
- '생태적으로 건강한', '물환경관리기본계획' 등 생태계 복원 용어
- 협약: CBD, 나고야 의정서, 카르타헤나 의정서, CITES, 람사르, CMS, UNCCD, 선박평형수 관리 협약, NOWPAP, 스톡홀름, 미나마타
추가 기준:
- 1차산업·생물자원을 직접 수식하는 '지속 가능한 농업·어업·산림경영'은 인정한다. '지속 가능한 발전', 지명·해역·국토·경제의 지속 가능성은 인정하지 않는다.
- 생태계·생물이 조사·측정·관리 대상이면 인정한다(가뭄의 생태학적 영향 분석, 생태계 변화유발물질 측정 등).
- 인정 표현의 대상을 사업이 실제로 조성·관리·조사·보호하면 다른 목적의 수단이어도 인정한다(수질개선을 위한 수변생태벨트 조성, 벌채의 생태 영향 최소화). 비유적 용법·관용구·기대효과 문구는 인정하지 않는다.
- '친환경' 등 이름과 함께 토양환경 보전·지속 가능한 농업(농업환경)이 목적으로 기재되면 인정한다.
## 그것만으로는 불인정
수질·대기질·환경 개선 / 위생·보건·건강 / 재난·안전·치수 / 탄소중립·에너지 전환·UNFCCC·탄소배출권·몬트리올 의정서 / 소득·생산성·경제 활성화 / 관광 / '산업·창업·기술·의료 생태계' 같은 비유 / '친환경'·'유기농업'이라는 이름, 인증·직불·융자라는 형식 / '환경 보전'만 있는 조건 / '목재의 지속가능한 이용' 자체 / 자원 이름만의 서술(산림자원 육성·생산, 수자원, 인적자원) / '해양환경 보전·관리', '물환경 개선', '저서환경 고려' / 바젤·로테르담·런던·MARPOL 이름만.
이 표현만으로 판정하지 말고 다른 인정 근거가 함께 있는지 확인한다.
## 재해 예방
인명·재산 보호 위주의 재난대응·경계·피난은 0이다. 산불 예방·사방·토양유실 방지는 '생태', '생물다양성', '산림자원 보전·보호', '자연생태계 보호', '지속 가능 실천'이 사업명·사업목적에 명시된 경우에만 9이며, 식생 복구·녹화 같은 수단만으로는 인정하지 않는다. 재해 조사·모니터링에 생태학적 영향 분석이 있으면 2다.
## 인간 보건
인간 보건 목적의 일반 생물자원 관리·정화는 0이다. 나고야 대응(1.04)과 숲 직접 이용·해양치유 사업(9)은 예외다.

# 카테고리 기준
## 1 유전자원 접근 및 이익 공유
- 1.01 유전자원·생물소재 탐사·발굴·정보관리·활용기반, 생물자원 활용 인증센터·성과 사업화 지원. 해양바이오 등 생물자원을 활용하는 산업화 기반(기업 육성·인증지원센터)은 '산업 생태계' 표현이 있어도 1.01이다.
- 1.03 비금전적 이익공유 성격의 ODA(기술이전·연수). 종자 외에 농산물 안전관리 기술이전 ODA(농산물안전성조사 ODA 등)도 ABS 언급 없이 포함한다.
- 1.04 나고야 의정서 대응: 세부사업명이 나고야·유전자원 대응 자체인 사업, 대응 기관 운영·자원은행. 채종원 관리가 포함된 산림품종보호·채종원 사업과 병원체자원 수집·관리·분양(병원체자원법, 감염병 표준실험실)은 사업명에 나고야가 없어도 1.04다. 사업명이 생물다양성협약 대응(5.07), 생물다양성 보전·관리(7.04), 생명자원 조사(2.04, 해양생명자원조사 등)이고 나고야가 목적·내역의 일부이면 사업명의 카테고리를 따른다. 일반 감염병 감시·의료로 확대하지 않는다.
## 2 인식 제고 및 연구
- 2.01 학교 교육과정 연계 생물다양성 교육. 2.02 범용 생물다양성 교육·자연체험학습. 1차산업 종사자 교육은 명시 시 해당 9, 일반 교육훈련·공무원 교육기관·교육기관 출연은 0.
- 2.03 생물다양성·자연·해양환경 인식제고, 보전단체 지원, 전시관 운영. 오염 홍보는 6.
- 2.04 자연자원 조사·통계·모니터링·DB·정보화(수질·수생태계 측정조사, 하천유역 환경생태 조사, 산림자원정보 관리 시스템, 해양생명자원·철새·가뭄 생태영향 조사), 기관 목록 연구기관·수목원. 일반 행정·산업정보는 제외.
- 2.05 전통지식 연구·문서화. 2.06 GBIF 등 CHM 참여(GBIF가 일부인 과학기술 국제기구 부담금은 0).
## 3 생물안전성
- 3.01 국경 검역·방역(동식물·수산물 검역, 수산생물방역, 검역 ODA), 선박평형수 관리(미세먼지·온실가스 감축 내역과 함께 있어도 3), 침입외래종 협약 이행. 식품 위생 검사만이면 3이 아니다. 국내 정착종 제거는 8을 검토한다.
- 3.02 LMO 안전관리·표시, 카르타헤나 의정서 이행·분담금.
## 4 녹색경제와 생물다양성
- 4.01 자원순환·리사이클링 기반. 친환경 선박·항만은 해양생태계 보전이 명시될 때만. 폐기물 처리시설은 6.
- 4.04 생태계 보전 조치가 명시된 에너지 시스템만. 신재생 보급·보조·융자, 수소·풍력 산업시설, 전력계통, 원자력은 0.
- 4.05 명시 요건을 충족한 생태관광. 관광·여가만이면 0.
- 4.06 야생동물 이동통로·로드킬 저감 시설이 목적에 명시된 도로 사업. 도로 건설·교통안전 일반은 0.
- 4.07 도시공원·녹지·그린인프라 보전·조성·유지, 장기미집행공원, 오염부지 공원화, 도시농업(활성화 지원 포함). 국가정원·정원 조성관리는 9.05, 법적 보호지역은 7. 공원 조성은 관광·도시녹지면 4, 보호지역 관리면 7, 조림·숲가꾸기면 9.
- 개별 사업의 환경영향평가는 그 사업이 속한 4번 하위 카테고리.
## 5 생물다양성 기획 및 재정
허용 법적 근거(조문·내용이 확인되어야 함): 환경정책기본법 제27조(지구환경 보전·생물자원 감소 대응), 자연환경보전법 제46조(생태계보전부담금), 개발제한구역법 제26조(훼손지 복구·공원화·조림), 해양생태계법 제38조(해양생물다양성 국제협력), 해양공간계획법(환경·생태계관리구역), 환경영향평가법·자연환경보전법(자연생태 평가), 추진 경위의 NOWPAP, 가축전염병예방법·식물신품종보호법의 OIE·UPOV 분담금. 목록 밖 근거만 있으면 잠정 0, review_needed=true. 간척지 개발은 0.
- 5.01 생물다양성을 직접 목적으로 하는 법·계획. 5.02 생물다양성에 영향을 주는 기타 법·계획·정책연구(명시 요건 적용. 토양·식생 등 산림생태 조사를 기반으로 하는 산림경영계획 작성은 5.02).
- 5.03 이해관계자 조율, 환경부 자연보전정책관실·산림청 산림보호국 경상경비.
- 5.04 생물다양성 재원 제도·징수(생태계보전부담금이 포함된 세입징수비용교부금), 보전 목적이 명시된 수계관리기금 지출. 수계관리기금의 주민지원·정수비용·징수비용 보전, 목적 없는 기금 운영은 0.
- 5.05 전략환경평가·기초생태조사·환경성평가지도.
- 5.06 공간계획·정보화(생태자연도가 포함된 환경정보화, 환경·생태계관리구역을 다루는 해양공간정보, 무인도서, 개발제한구역).
- 5.07 인정 협약 대응·분담금·ODA, REDD+, 생태계 보전 국제산림협력, NOWPAP 해양과학 국제협력, 생태계 조사가 포함된 국제 공동연구, 화학물질 협약 분담금, OIE·UPOV 분담금. 허용 법적 근거가 확인되면 다른 협약 분담금이 함께 있어도 5.07.
- 5.08 환경정책 정보접근·의사결정 참여.
## 6 오염 관리
세부사업명·사업목적·내역에 생태계·서식지·생물다양성의 보전·회복·건강성·교란 방지가 명시된 오염 사업만 포함한다(중소유역 수생태 복원 내역이 있는 상수원관리지역관리, '생태적으로 건강한 하천'을 목적으로 한 환경기초조사연구는 6.01). 처리시설·하수관로·정화·원자력 관련 사업도 같은 요건을 적용한다. 오염퇴적물 정화·해양환경 감시체계는 '해양환경 보전'만 있으면 0이며, 감시 사업 안의 환경관리해역 개선 내역 일부로 전체를 포함하지 않는다. 수계관리기금의 민간 수질보전활동 지원은 0.
- 6.01 수질·토양(비점오염·오염총량·토양정화·수계 조사·측정망). 상수 공급은 0.
- 6.02 대기. 대기질 개선·온실가스 감축만이면 0.
- 6.03 하수·폐기물 처리시설·가축분뇨. 탄소중립·에너지 회수만이면 0.
- 6.04 해양폐기물·하구쓰레기, 서식지 개선 목적 폐어구 수거, 해양오염 방제·감시·해양환경정보 정보화. 부표·방제선·침몰선박만이면 0.
- 6.05 화학물질·POPs·중금속·빛·소음의 국내 이행. 분담금·국제협력은 5.07, 인간 생활환경 소음대책은 0.
- 6.06 오염 관련 정책연구·감시·지도점검·홍보·정보화.
## 7 보호지역 및 기타 보전조치
- 7.01 법정 보호지역: 천연기념물(지질·지형), 명승, 국립·자연공원, 보호구역(산림유전자원보호구역 포함), 세계유산(자연유산) 등재·보존관리, 관리기관 출연.
- 7.02 보호지역 외 관리: 수변구역 토지매수(수질 목적이 함께 있어도), 국토생태네트워크, 환경관리해역, 보호지역 감시 인력, 법적 보호지역이 아닌 보호·보전 구역.
- 7.03 OECM: 국유림 매수·보전, 해양정원 등 보전 성과형 구역.
- 7.04 종의 보전: 야생동식물 보호·구조·질병관리, 종보전 정책·종합대책, 종보전·연구 기능이 있는 전시시설(해양생명과학관이 포함된 시설 사업), CITES·CMS 이행. 수목원·연구기관은 2, 가축 유전자원은 9, 반려·유기동물·동물복지인증은 0.
'보호', '공원', '조성'이라는 단어만으로 법적 보호지역을 추정하지 않는다.
## 8 복원
- 8.01 특정 종의 사육·야생 방사·서식지 이전.
- 8.02 생태하천·수변녹지(토지매수 지역의 녹지·생태벨트 조성 포함)·갯벌·서식지·산림 복원, 바다숲·산란서식장·인공산란장·어도, 내수면·수산자원조성, 청정어장 재생, 댐유역 생태계·옛 물길·수생태 연속성 복원, UNCCD 토지 복원. 인명·재산 재해복구는 0.
- 8.03 복원 이후 유지관리.
'조성', '공원화'라는 말만으로 훼손·복원을 추정하지 않는다.
## 9 지속 가능한 이용과 생물다양성
면제 유형이 아니고 인명·재산·생산성·소득·보건·여가만 확인되면 0이다.
- 9.01 품종보호·무병묘목·국립종자원 경비·품질관리 시설, 농업유산, 가축·곤충·미생물 유전자원(곤충·미생물 산업화 기반 포함), 종자관리 제도 이행 교육 인프라, 실제 활동이 품종보호 심사인 수산종자산업육성. 일반 증식·공급·육종·수출·동물복지 인증은 0.
- 9.02 토양 보전·지속 가능한 농업이 명시된 농자재·비료·농약 관리, 생태 조건부 직불, 유해야생동물 피해 저감, 화분매개 보전 목적 양봉, 가축 방역·백신. '친환경' 이름·인증·유통만이면 0.
- 9.03 지속 가능한 생산기반이 목적에 명시된 양식, 양식생물 질병 관리. '친환경' 이름만이면 0.
- 9.04 수산자원 조사·회복·TAC·감척·자율관리, 불법어업 단속·어업지도선·어업지도정보화·어업관리단 경비, 조건부 직불·융자, 국제수산기구 협상·분담금·수산 ODA, 지속 가능한 어업생산이 목적인 체계. 원양어업 활성화 등 원양산업 육성은 옵서버·국제규제 대응 내역이 있어도 0. 유통·가공은 0.
- 9.05 조림·숲가꾸기·산림병해충방제, 조림용 종묘, 산림생명자원, 도시숲·국가정원·정원, 산지 보전·전용 관리, 숲 직접 이용, 지방산림청·국립자연휴양림관리소 경비, 벌채 생태 영향 최소화가 명시된 목재생산관리, 합법목재·산림보전이 명시된 수출 지원, 조림·병해충 중심 남북산림협력. 임도·목재 산업·임산물 유통·한국임업진흥원 지원은 명시 없으면 0. 산림 정책개발은 5, 산림복원은 8.
- 9.06 하수처리수 재이용, 수생태 결합 통합하천 조성. 일반 용수 공급은 0.
- 9.07 세부사업명이 해양치유인 사업·해양치유센터(관광 목적이 함께 있어도). 관광 사업 안의 일부 해양치유 내역과 헬스케어 R&D는 0. 그 밖에 지속 가능한 해양생태계 이용이 명시된 사업.
- 9.08 자연 초지 보호. 9.09 야생생물의 지속 가능한 이용(보전 주목적은 7, 관광 목적은 4).
현금 지원: 생태 보전·수산자원 회복 등 구체적 조건이 확인된 직불·융자만 해당 업의 9다. '공익직불', 환경 준수사항, 친환경 한정 융자, 조건 없는 업종 금융은 0이다.

# 일관성
- 사업명·활동·목적이 같다고 확인된 사업은 소관이 달라도 같은 카테고리를, 같은 기관의 경비에는 같은 카테고리를 배정한다.
- 같은 이름만으로 다른 행의 근거를 전파하지 않는다. 기존 레이블이나 카테고리 비율에 맞추지 않는다.

# 최종 자기검증
다음 모순이 있으면 고친 뒤 출력한다.
- 면제 유형을 인정하고도 문구 부재나 산업·보건 목적으로 0을 골랐는가?
- 내역에 명시가 있는데 사업목적에 없다는 이유로, 또는 수단이라는 이유로 0을 골랐는가?
- 불인정 표현 하나만 보고 다른 인정 근거를 확인하지 않은 채 0을 골랐는가?
- 일부 내역의 키워드 하나로 면제 유형을 인정하면서 사업 전체의 대상·활동을 확인하지 않았는가?
- 조성·복원 내역이 있는데 9·7을 골랐는가? 폐어구 수거가 핵심인데 9를 골랐는가?
- 프롬프트에 없는 제외 조건을 만들었는가? reason의 결론과 label이 다른가?

# 최종 출력
JSON 하나만 반환한다.
{"label": 0, "classification_status": "무관", "reason": "", "evidence": ["입력에 실제 있는 표현"], "confidence": 0.00, "review_needed": false, "review_reason": ""}
- label은 0~9 정수, classification_status는 0이면 '무관', 1~9면 '관련'이다. review_needed는 boolean, review_reason은 150자 이내이며 검토가 필요 없으면 빈 문자열이다.
- reason은 300자 이내로 머리말 없이 쓴다: 핵심 활동과 분류 근거 → 면제 사유(1·3·7·8번 / 9번 포함 유형 / 기관 목록 / R&D 주제) 또는 명시 문구와 출처(예: 사업목적의 '수생태계 보전'), 명시가 없어 0이면 그 사실 → 혼합 사업이면 기준 구성 요소와 차순위 → 설명자료 없음 여부 → 실제 경쟁 카테고리가 있으면 배제 이유. 판단 과정을 반복해 쓰지 않는다.
- evidence에는 입력의 실제 문구만 쓴다. 입력에 없는 내용은 만들지 않는다.
"""

# 세부사업명에 (R&D) 표기가 있을 때만 프롬프트에 붙이는 규칙.
RND_RULES = """\
# R&D 규칙(세부사업명에 (R&D) 표기)
관련성을 먼저 판단하고, 관련이면 예외 없이 2로 분류한다. 표기 자체는 포함 근거가 아니다.
- 연구 대상의 주제가 1·3·7·8번이나 9번 포함 유형이면 명시 요건 없이 2다. 그 밖의 주제(2·4·5·6번 등)는 명시 요건을 충족하면 2, 아니면 0이다.
- 산업화·상용화 목적이 함께 있어도 2인 연구: 사업명이 생물자원·유전자원을 가리키는 확보·소재화·원료화 연구(해양생명자원, 극지 유전자원, 국가생명연구자원, 해양생물자원 전략소재 상용화), 사업명이 질병·병해충 대응을 가리키는 연구(작물바이러스, 가축·수산생물 질병의 방역·백신·진단), LMO 위해성평가, 선체부착생물·선박평형수 등 외래 해양생물의 이동 차단, 도시숲·정원 식물·산림병해충 등 9번 포함 유형을 다루는 산림 연구(목적이 미세먼지 저감·재난 대응이어도), 수산자원 관리(TAC 등) 시스템.
- 0인 연구: 일반 육종·품종개발·생산성·인력양성(디지털육종 등), 인간 질환 치료 목적의 미생물(마이크로바이옴) 연구, 범용 바이오 데이터센터·데이터 인프라, 생물자원·생태계를 직접 다루지 않는 공학(인공구조물·소재·위성·생태모방·의료·헬스케어. 블루카본·생태·해양치유 효과가 기재되어도 0), 생태 보전 조치가 명시되지 않은 에너지 기술, 원자력·방사선·핵연료주기, 극지연구소 운영지원·시설 지원과 극지및대양과학연구 등 극지연구소의 기관 고유 연구, GBIF가 일부인 과학기술 국제기구 부담금, '산업·기술·의료 생태계 조성' 같은 비관련 활동.
- 여러 분야의 실용화 기술을 묶은 R&D에서 질병 등 포함 유형이 일부 과제로만 기재되면 그 과제만으로 2로 하지 않는다. 해양·극지 기초원천 기술개발 R&D는 사업목적에 해양·극지 생물자원 발굴·이용이 연구 과제로 기재되면 2다.
- 오염 저감 R&D는 생태계 보전 목적이 명시된 경우에만 2다. 국립환경과학원 연구사업(R&D)은 수생태계·생태독성·생물 등의 연구 내역이 있으면 2다. 한국임업진흥원 R&D는 표기만으로 포함하지 않는다.
- 연구 대상을 먼저 식별하고 산업화·사업화라는 표현 자체를 제외 사유로 삼지 않는다.
"""

# 기관명이나 경비·정보화 관련 표현이 있을 때만 프롬프트에 붙이는 규칙.
INSTITUTION_RULES = """\
# 기관·경상경비 규칙
- 경상경비(기본경비·인건비·전산운영경비 등)는 그 기관·부서·기금의 핵심 기능으로 분류한다. 세부사업명이 '운영경비' 같은 일반어이면 상위 구분이 가리키는 기관·기금으로 판단한다. 핵심 기능이 생물다양성이 아닌 부처 본청·소속기관의 경비·인건비는 0이다. 순수 청사 신축·재건축은 0이다.
- 아래 기관 목록 기관의 경상경비(기본경비·인건비·공익요원경비·전산운영경비·기관 정보화(정보화) 사업·수입대체경비·운영지원·연금부담금·연금기금 전출·건강보험부담금)와 기관 출연·운영비·시설 지원·조성은 명시 요건 없이 해당 카테고리다((R&D)·(정보화) 표기가 붙어도 같다). 기관 고유 기능과 무관한 개별 사업은 사업명·사업목적으로 판단하며, 사업명이 카테고리를 가리키면 그것을 우선한다.
  1: 국립호남권생물자원관, 국가생약자원관리센터
  2: 국립생물자원관, 국립낙동강생물자원관, 국립생태원, 국립해양생물자원관, 국립수목원(새만금수목원·국립난대수목원 조성 포함), 국립산림과학원, 한국수목원정원관리원, 국립수산과학원, 한국해양과학기술원(운영지원·시설 지원), 한국환경보전원(출연)
  3: 농림축산검역본부
  5: 환경부 자연보전정책관실, 산림청 산림보호국
  7: 국립공원공단, 국립야생동물질병관리원
  9: 국립종자원(시설현대화·품질관리 포함), 남해·동해·서해어업관리단(어업지도선 관리·운영, 어업지도정보화 포함), 지방산림청(국유림관리소 포함), 국립산림품종관리센터, 한국산림복지진흥원, 국립자연휴양림관리소(운영·조성, 해당 회계의 연금부담금·손익·자본계정 전출 포함)
- 목록에 없는 기관은 자동 포함하지 않지만 그 기관의 사업이 모두 무관하다는 뜻은 아니다. 실제 활동으로 판단한다.
- 생물다양성 목적이 명시된 경우에만 포함: 한국원자력환경공단·방사성폐기물관리기금·원자력기금 원자력안전규제계정·한국원자력연구원, 4대강 수계관리기금 운영경비(징수비용보전은 0), 수산사무소 수산어업행정지원, 한국임업진흥원 지원·기획평가관리비.
- 경비·인건비가 0인 기관: 산림청 본청(산림보호국 외)·산림항공본부, 산림교육원·해양수산인재개발원 등 공무원 교육기관, 농림수산식품교육문화정보원, 국립환경과학원 기본경비·인건비, 국립해양조사원·국립해양측위정보원, 유역·지방환경청, 박물관·과학관 기관 경비(종보전·연구 기능이 확인된 전시시설만 7), 부처 회계 전체의 연금 전출·건강보험부담금, 극지연구소 운영지원·시설 지원.
"""

RND_RE = re.compile(r"[（(]\s*R\s*&\s*D\s*[)）]", re.I)
# 기관 규칙이 필요한 사업을 고르는 표현. 판정이 아니라 프롬프트에 규칙을 붙일지만 정한다.
INSTITUTION_HINTS = (
    "경비", "인건비", "정보화", "운영지원", "출연", "연금", "건강보험", "공익요원",
    "수입대체", "전출", "시설", "청사", "기관운영",
    "생물자원관", "생약자원", "생태원", "수목원", "산림과학원", "수산과학원",
    "해양과학기술원", "환경보전원", "검역본부", "자연보전정책", "산림보호국",
    "국립공원", "야생동물질병", "종자원", "어업관리단", "산림청", "국유림",
    "산림품종관리", "산림복지", "자연휴양림", "극지연구소", "환경과학원",
    "임업진흥원", "원자력", "방사성폐기물", "수계관리기금", "수산사무소",
    "교육원", "인재개발원", "교육문화정보원", "해양조사원", "측위정보원",
    "환경청", "박물관", "과학관",
)

# 시스템 프롬프트와 별도로 유지하는 입출력 형식입니다.
# 카테고리와 별도 검토 표시의 JSON 형식을 지정합니다.
PROMPT_TEMPLATE = """\
{classification_prompt}

다음 예산 사업을 BIOFIN 1차 카테고리 0~9 중 하나로 분류하고 검토 필요 여부를 별도로 표시하라.

소관명: {소관명}
회계명: {회계명}
계정명: {계정명}
분야명: {분야명}
부문명: {부문명}
프로그램명: {프로그램명}
단위사업명: {단위사업명}
세부사업명: {세부사업명}

사업설명자료의 사업목적 또는 사업목적·내용, 법적 근거(추출된 항목만 제공. 법적 근거는 5번 판정에만 사용):
{document_text}

반드시 아래 JSON 객체만 반환하라.
{{
  "label": 0,
  "classification_status": "무관",
  "confidence": 0.0,
  "reason": "",
  "evidence": [],
  "review_needed": false,
  "review_reason": ""
}}

label은 반드시 0~9 정수다. classification_status는 '관련'(1~9) 또는 '무관'(0)이다. review_needed는 true/false, review_reason은 별도 검토 사유다. confidence는 0.0부터 1.0까지다.
"""


REASON_PREFIX_RE = re.compile(r"^\s*(?:명시적\s*제외\s*[:：]|잠정\s*분류\s*[:：](?:\s*포함\s*근거\s*미확인\s*[.:：]?)?)\s*")


def strip_reason_prefix(reason: str) -> str:
    """Drop status headers such as '명시적 제외:' so reason holds only the model's rationale."""
    return REASON_PREFIX_RE.sub("", str(reason or ""), count=1)


# Keep the first model answer separate from the rule-adjusted final result.
PROVENANCE_FIELDS = ("model_original_label", "model_original_reason",
                     "model_original_response", "rule_adjustment_reason",
                     "consistency_check", "consistency_response")


def explicit_reason_labels(reason: str) -> set[int]:
    """Only explicit conclusions, never arbitrary category numbers in prose."""
    patterns = (
        r"(?:최종\s*(?:label|레이블|분류)|label|레이블)\s*(?:은|는|:|=)\s*([0-9])",
        r"([0-9])(?:번)?(?:을|를)\s*(?:최종적으로\s*)?적용(?:한다|합니다)",
        r"([0-9])(?:번)?(?:으로|로)\s*최종\s*분류",
    )
    return {int(m.group(1)) for p in patterns for m in re.finditer(p, reason, re.I)}


def enforce_result_consistency(result: dict[str, Any], row: dict[str, str],
                               prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    original = dict(result)
    original_data = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", result["raw_response"].strip())) if result["raw_response"].strip().startswith("{") else {}
    reason = str(original_data.get("reason", result["reason"]))
    # Use the business title only; parent programs and quoted documents may mention unrelated R&D.
    is_rnd = is_rnd_business(row)
    target = 2 if is_rnd and result["label"] != 0 else result["label"]
    adjustment = "관련 사업으로 판단된 명시적 R&D에 2 우선 규칙 적용" if target != result["label"] else ""
    conclusions = explicit_reason_labels(reason)
    conflict = bool(conclusions - {target})
    result.update({"model_original_label": original["label"], "model_original_reason": reason,
                   "model_original_response": original["raw_response"],
                   "rule_adjustment_reason": adjustment, "consistency_check": "통과",
                   "consistency_response": ""})
    if not adjustment and not conflict:
        return result
    # One targeted recheck, independent of HTTP retries. Never change relevance based on a number in reason.
    final_label = target
    correction = ("\n\n[최종 일관성 검증]\n관련 R&D는 다른 기능보다 2를 우선한다. "
                  f"코드가 확정한 label={final_label}을 유지하고 classification_status를 맞춰라. "
                  "최초 답변의 모순을 해소하고 실제 입력에 근거한 최종 이유를 300자 이내로 작성하라. 명시 또는 면제 근거를 우선하고 혼합 사업이면 기준 구성 요소와 차순위를 기록하라. "
                  "원래 출력 형식의 완전한 JSON을 반환하라. 다른 레이블로 결론을 번복하지 마라.\n"
                  + json.dumps({"previous_label": original["label"], "previous_reason": reason}, ensure_ascii=False))
    try:
        raw = call_vllm(prompt + correction, args)
        result["consistency_response"] = raw
        checked = parse_jsonish_response(raw)
        full_reason = str(json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())).get("reason", ""))
        if checked["label"] != final_label or explicit_reason_labels(full_reason) - {final_label}:
            raise ValueError("재검증 응답의 레이블·근거 모순 지속")
        for field in ("reason", "evidence", "confidence"):
            result[field] = checked[field]
        result["review_needed"] = original["review_needed"] or checked["review_needed"]
        result["review_reason"] = "; ".join(filter(None, [original["review_reason"], checked["review_reason"]]))
        result["consistency_check"] = "재검증 통과"
    except (ValueError, OSError, TimeoutError, RemoteDisconnected) as exc:
        if isinstance(exc, VLLMResponseError):
            result["consistency_response"] = exc.raw_response
        result["consistency_check"] = "재검증 미해결"
        result["review_needed"] = True
        result["review_reason"] = (original["review_reason"] + "; 근거 재검증 필요: " + str(exc))[:500]
        result["reason"] = ("모델이 관련으로 판단한 R&D 사업에 코드의 2 우선 규칙을 적용함. 상세 근거 재검증 필요."
                            if adjustment else "최초 모델의 숫자 분류를 유지했으나 근거와의 모순을 해결하지 못함. 별도 검토 필요.")
        result["confidence"] = 0.0
    result["label"] = final_label
    result["classification_status"] = "무관" if final_label == 0 else "관련"
    result["검토필요"] = "검토 필요" if result["review_needed"] else ""
    return result


CACHE_FIELDS = (
    *PROVENANCE_FIELDS,
    "key_hash",
    "label",
    "classification_status",
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
    "review_needed", "review_reason", "검토필요",
)
EXCLUDED_CSV_COLUMNS = frozenset({
    "사업설명자료_파일명",
    "사업설명자료_상대경로",
    "사업설명자료_절대경로",
    "문서매칭상태",
    "문서매칭방식",
    "문서매칭후보수",
    "document_status",
    "document_path",
    "document_chars",
})
# 결과 CSV에는 사업 식별 정보와 판정 결과만 남긴다. 세부 근거·원응답은 audit CSV에 있다.
OUTPUT_INFO_COLUMNS = (
    "회계연도", "소관명", "회계명", "분야명", "부문명",
    "프로그램명", "단위사업명", "세부사업명", "예산액",
)
OUTPUT_RESULT_COLUMNS = ("reason", "evidence", "review_reason", "confidence")
EXTRA_OUTPUT_COLUMNS = (
    *PROVENANCE_FIELDS,
    "classification_status",
    "confidence", "reason", "evidence",
    "document_status", "document_path", "document_chars",
    "review_needed", "review_reason", "검토필요",
)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="예산 사업을 VLLM로 BIOFIN 1차 카테고리 0~9로 분류합니다."
    )
    parser.add_argument(
        "--input-file", type=Path, default=project_dir / DEFAULT_INPUT_FILE
    )
    parser.add_argument("--output-dir", type=Path, default=project_dir / "outputs/llm/v1_vllm_v2")
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--cache-csv", type=Path, default=None)
    parser.add_argument("--audit-csv", type=Path, default=None)
    parser.add_argument("--review-csv", type=Path, default=None)
    parser.add_argument("--label-col", default=DEFAULT_LABEL_COLUMN)
    parser.add_argument(
        "--target-ministries", nargs=11, default=TARGET_MINISTRIES,
        help="LLM으로 분류할 소관명 11개. 나머지 소관기관은 LLM 호출 없이 0 처리",
    )
    parser.add_argument(
        "--gold-label-col",
        default=None,
        help="정확도 평가용 정답 컬럼. 생략 시 알려진 컬럼명을 자동 탐색",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--vllm-url", "--ollama-url", dest="vllm_url", default=DEFAULT_VLLM_URL,
        help="vLLM 서버 주소. /v1 포함 또는 생략 가능. --ollama-url은 호환 별칭",
    )
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
        help="프롬프트에 포함할 사업목적·법적 근거 최대 문자 수",
    )
    parser.add_argument(
        "--no-document-text",
        action="store_true",
        help="사업설명자료를 읽지 않고 예산 메타데이터만 사용",
    )
    parser.add_argument(
        "--num-ctx", type=int, default=None,
        help="기존 명령 호환용. vLLM 문맥 한도는 서버에서 설정하며 이 옵션은 무시",
    )
    parser.add_argument("--max-tokens", type=int, default=2048, help="응답 최대 토큰 수")
    parser.add_argument(
        "--enable-thinking", action="store_true",
        help="Qwen thinking 활성화. 기본값은 비활성화",
    )
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
                # 편집기가 .csv를 탭 구분 텍스트로 다시 저장한 경우도 읽는다.
                header_line = file.readline()
                delimiter = "\t" if "\t" in header_line else ","
                file.seek(0)
                reader = csv.DictReader(file, delimiter=delimiter)
                if not reader.fieldnames:
                    raise ValueError("CSV 헤더가 없습니다.")
                rows = [dict(row) for row in reader]
                if any(None in row or any(value is None for value in row.values()) for row in rows):
                    raise ValueError(f"CSV 컬럼 수가 헤더와 다릅니다: {path}")
                return list(reader.fieldnames), rows, encoding
        except UnicodeError as exc:
            last_error = exc
    raise RuntimeError(f"CSV를 읽을 수 없습니다: {path}") from last_error


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    # 문서 메타데이터는 분류에만 사용하고 CSV에는 저장하지 않습니다.
    fieldnames = [name for name in fieldnames if name not in EXCLUDED_CSV_COLUMNS]
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


def is_rnd_business(row: dict[str, str]) -> bool:
    return bool(RND_RE.search(clean_cell(row.get("세부사업명"))))


def needs_institution_rules(row: dict[str, str]) -> bool:
    text = " ".join(clean_cell(row.get(field)) for field in (
        "소관명", "회계명", "계정명", "프로그램명", "단위사업명", "세부사업명"))
    return any(hint in text for hint in INSTITUTION_HINTS)


def classification_prompt_for(row: dict[str, str]) -> str:
    """공통 규칙에 해당 사업에 필요한 R&D·기관 규칙만 덧붙인다."""
    blocks = [SYSTEM_PROMPT.strip()]
    if is_rnd_business(row):
        blocks.append(RND_RULES.strip())
    if needs_institution_rules(row):
        blocks.append(INSTITUTION_RULES.strip())
    return "\n\n".join(blocks)


def build_prompt(row: dict[str, str], document_text: str) -> str:
    return PROMPT_TEMPLATE.format(
        classification_prompt=classification_prompt_for(row),
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
        legal_basis = extract_legal_basis(full_text)
        if not purpose and not legal_basis:
            return (
                "[사업목적·법적 근거 항목 또는 종료 경계를 찾지 못함. 문서 본문 미사용]",
                "PURPOSE_NOT_FOUND", str(path), 0,
            )
        # 두 항목이 길어도 법적 근거가 목적 뒤에서 전부 잘리지 않게 배분한다.
        if purpose and legal_basis:
            separator = "\n\n[법적 근거(5번 판정 전용)]\n"
            budget = max(0, args.max_document_chars - len(separator))
            legal_limit = min(len(legal_basis), budget // 3)
            purpose_text = purpose[:budget - legal_limit]
            legal_limit = budget - len(purpose_text)
            prompt_text = (purpose_text + separator + legal_basis[:legal_limit])[:args.max_document_chars]
        elif legal_basis:
            prompt_text = ("[사업목적 미발견]\n[법적 근거(5번 판정 전용)]\n" + legal_basis)[:args.max_document_chars]
        else:
            prompt_text = purpose[:args.max_document_chars]
        status = "PURPOSE_EXTRACTED" if purpose else "LEGAL_BASIS_EXTRACTED"
        return prompt_text, status, str(path), len(prompt_text)
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
            raise ValueError("응답에 유효한 JSON 객체가 없습니다.")
        else:
            data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("LLM 응답이 JSON 객체가 아닙니다.")
    value = data.get("label")
    status = clean_cell(data.get("classification_status"))
    if type(value) is not int or value not in VALID_LABELS:
        raise ValueError(f"label은 0~9 정수여야 합니다: {value!r}")
    label = value
    expected_status = "무관" if label == 0 else "관련"
    if status and status != expected_status:
        raise ValueError("classification_status와 label이 일치하지 않습니다.")
    status = expected_status
    review_needed = data.get("review_needed", False)
    if not isinstance(review_needed, bool):
        raise ValueError("review_needed는 JSON boolean이어야 합니다.")
    review_reason = clean_cell(data.get("review_reason"))[:500]
    if review_needed and not review_reason:
        raise ValueError("검토가 필요한 응답은 review_reason을 제공해야 합니다.")
    if not review_needed and review_reason:
        raise ValueError("검토가 불필요한 응답의 review_reason은 빈 문자열이어야 합니다.")

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "label": label,
        "classification_status": status,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": strip_reason_prefix(clean_cell(data.get("reason")))[:500],
        "evidence": clean_cell(data.get("evidence"))[:500],
        "review_needed": review_needed,
        "review_reason": review_reason,
        "검토필요": "검토 필요" if review_needed else "",
        "raw_response": text,
    }


class VLLMResponseError(ValueError):
    def __init__(self, message: str, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


# temperature 0에서는 같은 요청을 다시 보내도 같은 반복 루프가 나오므로, 잘린 응답은 반복 억제 설정으로 한 번 다시 요청한다.
TRUNCATION_RETRY_NOTE = (
    "\n\n[출력 길이 제한] 이전 응답이 같은 문장을 반복하다 잘렸다. "
    "reason은 150자 이내 한두 문장으로 결론만 쓰고, 같은 문장을 반복하지 말고 즉시 JSON을 닫아라."
)
TRUNCATION_REPETITION_PENALTY = 1.15


def call_vllm(prompt: str, args: argparse.Namespace, truncation_retry: bool = False) -> str:
    base_url = args.vllm_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    content = prompt + TRUNCATION_RETRY_NOTE if truncation_retry else prompt
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "temperature": 0,
        "top_p": 0.1,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
    }
    if truncation_retry:
        payload["repetition_penalty"] = TRUNCATION_REPETITION_PENALTY
    if not args.no_json_format:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("VLLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=args.timeout) as response:
            raw_body = response.read().decode("utf-8")
    except HTTPError as exc:
        raw_body = exc.read().decode("utf-8", errors="replace")
        raise VLLMResponseError(
            f"vLLM HTTP {exc.code}: {raw_body[:2000]}", raw_body
        ) from exc
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise VLLMResponseError("vLLM HTTP 응답이 JSON이 아닙니다.", raw_body) from exc
    if not isinstance(body, dict):
        raise VLLMResponseError("vLLM HTTP 응답이 JSON 객체가 아닙니다.", raw_body)
    if body.get("error"):
        raise VLLMResponseError(f"vLLM 서버 오류: {body['error']}", raw_body)
    try:
        choice = body["choices"][0]
        text = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VLLMResponseError("vLLM 응답에서 답변을 찾지 못했습니다.", raw_body) from exc
    if choice.get("finish_reason") == "length":
        if not truncation_retry:
            return call_vllm(prompt, args, truncation_retry=True)
        raise VLLMResponseError(
            "반복 억제 재요청 후에도 응답이 토큰 한도로 잘렸습니다. "
            "raw_response에서 반복 여부를 확인하세요.", raw_body,
        )
    if not isinstance(text, str) or not text.strip():
        raise VLLMResponseError("vLLM 최종 답변이 없거나 비어 있습니다.", raw_body)
    return text


def classify(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    document_text, document_status, document_path, document_chars = (
        load_document_for_prompt(row, args)
    )
    prompt = build_prompt(row, document_text) + "\n관련성이 인정된 R&D는 최종 label=2를 우선한다. reason은 최종 label과 일치하는 근거를 300자 이내로 작성한다. 명시 또는 면제 근거를 우선하고 혼합 사업이면 기준 구성 요소와 차순위를 기록한다. 다른 카테고리를 비교하더라도 최종 결론을 번복하지 않는다."
    last_error: Exception | None = None
    raw_response = ""
    for attempt in range(args.retries + 1):
        try:
            raw_response = ""
            raw_response = call_vllm(prompt, args)
            result = parse_jsonish_response(raw_response)
            result = enforce_result_consistency(result, row, prompt, args)
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
            if isinstance(exc, VLLMResponseError):
                raw_response = exc.raw_response
            if attempt < args.retries:
                time.sleep(args.retry_delay)
    return {
        "label": "",
        "classification_status": "실패",
        "review_needed": True,
        "review_reason": f"분류 실패: {last_error}",
        "검토필요": "검토 필요",
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
    # 새 캐시에는 문서 상태를 저장하지 않으므로, 기존 캐시에 있을 때만 검사합니다.
    if "document_status" in record and record["document_status"] not in {
        "PURPOSE_EXTRACTED", "LEGAL_BASIS_EXTRACTED", "PURPOSE_NOT_FOUND", "NOT_FOUND", "PARSE_FAILED", "DISABLED"
    }:
        return False
    if record.get("prompt_version") != PROMPT_VERSION:
        return False
    status = record.get("classification_status")
    label = parse_valid_label(record.get("label"))
    return label is not None and status == ("무관" if label == 0 else "관련")


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
    skipped_review_needed = 0

    for row in rows:
        gold = parse_valid_label(row.get(gold_column))
        pred = parse_valid_label(row.get(pred_column))
        if gold is None:
            skipped_missing_gold += 1
            continue
        if row.get("classification_status") == "검토 필요":
            skipped_review_needed += 1
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
        "backend": "vllm",
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
        "skipped_review_needed": skipped_review_needed,
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


LOCAL_TARGET_FIELDS = (
    "농림해양수산", "문화및관광", "산업ㆍ중소기업및에너지", "일반공공행정",
    "환경", "국토및지역개발", "보건",
)


def normalize_field(value: str) -> str:
    return "".join(clean_cell(value).split()).replace("ㆍ", "·")


def detect_fiscal_type(headers: list[str], rows: list[dict[str, str]]) -> str:
    local_columns = {"지방자치단체코드", "시도코드", "시도명"}.intersection(headers)
    if local_columns:
        if not rows or all(any(clean_cell(row.get(c)) for c in local_columns) for row in rows):
            return "local"
        raise ValueError("지방 식별 열에 값이 없는 행이 있습니다. 중앙·지방 혼합 또는 누락 데이터를 확인하세요.")
    if {"소관명", "분야명"}.issubset(headers):
        return "central"
    raise ValueError("중앙·지방 자동 판별에 필요한 열이 없습니다.")


def filter_rule(args: argparse.Namespace) -> tuple[str, frozenset[str]]:
    if getattr(args, "fiscal_type", "central") == "local":
        return "분야명", frozenset(normalize_field(value) for value in LOCAL_TARGET_FIELDS)
    return "소관명", frozenset(clean_cell(value) for value in getattr(args, "target_ministries", TARGET_MINISTRIES))


def filter_value(row: dict[str, str], column: str) -> str:
    return normalize_field(row.get(column, "")) if column == "분야명" else clean_cell(row.get(column))


def prepare_ministry_labels(
    items: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    column, targets = filter_rule(args)
    selected = []
    for key_hash, item in items.items():
        field = filter_value(item["row"], column)
        if field in targets:
            # 과거 정책 결과는 LLM 캐시로 재사용하지 않는다.
            if cache.get(key_hash, {}).get("model") in {"ministry-filter", "field-filter"}:
                cache.pop(key_hash)
            selected.append((key_hash, item))
        else:
            # --limit-keys 및 기존 캐시와 무관하게 모든 비대상 행을 0으로 고정.
            cache[key_hash] = {
                "key_hash": key_hash, "label": 0, "classification_status": "무관",
                "review_needed": False, "review_reason": "", "검토필요": "",
                "confidence": "1.000", "reason": "",  # LLM 미호출 행은 모델 근거가 없으므로 비워 둔다.
                "evidence": "[]", "model": "ministry-filter", "prompt_version": PROMPT_VERSION,
                "input_text": item["input_text"], "document_status": "DISABLED",
                "document_path": "", "document_chars": 0, "raw_response": "",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
    print(f"{column} 필터: LLM 대상 {len(selected):,}개 / 자동 0 {len(items) - len(selected):,}개")
    return dict(selected)


def classify_items(
    items: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    started_at = getattr(args, "started_at", time.monotonic())
    selected = list(prepare_ministry_labels(items, cache, args).items())
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
            "classification_status": result["classification_status"],
            "review_needed": result["review_needed"],
            "review_reason": result["review_reason"],
            "검토필요": result["검토필요"],
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
        record.update({field: result.get(field, "") for field in PROVENANCE_FIELDS})
        with lock:
            cache[key_hash] = record
            completed += 1
            elapsed = max(0, int(time.monotonic() - started_at))
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(
                f"[경과 {hours:02d}:{minutes:02d}:{seconds:02d}] "
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
    *, preliminary: bool = False,
) -> None:
    gold_columns = [args.gold_label_col] if args.gold_label_col in headers else []
    output_headers = [
        column for column in OUTPUT_INFO_COLUMNS if column in headers
    ] + gold_columns + [args.label_col, *OUTPUT_RESULT_COLUMNS]

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        result = dict(row)
        cached = cache.get(hash_key(build_key(row)), {})
        result[args.label_col] = cached.get("label", "")
        for column in EXTRA_OUTPUT_COLUMNS:
            result[column] = cached.get(column, "")
        output_rows.append(result)
    write_csv(args.output_file, output_headers, output_rows)
    if preliminary:
        return
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
            "classification_status": cached.get("classification_status", ""),
            "review_needed": cached.get("review_needed", ""),
            "review_reason": cached.get("review_reason", ""),
            "검토필요": cached.get("검토필요", ""),
            "confidence": cached.get("confidence", ""),
            "reason": strip_reason_prefix(cached.get("reason", "")),
            "evidence": cached.get("evidence", ""),
            "input_text": item["input_text"],
            "document_status": cached.get("document_status", ""),
            "document_path": cached.get("document_path", ""),
            "document_chars": cached.get("document_chars", ""),
            "raw_response": cached.get("raw_response", ""),
        }
        audit.update({field: cached.get(field, "") for field in PROVENANCE_FIELDS})
        audit_rows.append(audit)
        try:
            confidence = float(audit["confidence"])
        except (TypeError, ValueError):
            confidence = 0.0
        if (not valid_cached_label(cached)
                or clean_cell(cached.get("review_needed")).lower() == "true"
                or confidence < args.review_threshold):
            review_rows.append(audit)

    audit_headers = [
        *PROVENANCE_FIELDS,
        "key_hash", "row_count", "label", "classification_status", "confidence",
        "reason", "evidence", "input_text", "document_status",
        "document_path", "document_chars", "raw_response",
        "review_needed", "review_reason", "검토필요",
    ]
    write_csv(args.audit_csv, audit_headers, audit_rows)
    write_csv(args.review_csv, audit_headers, review_rows)

    counts = Counter(str(row.get("label", "")) for row in cache.values())
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "backend": "vllm",
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "input_file": str(args.input_file),
        "input_rows": len(rows),
        "fiscal_type": getattr(args, "fiscal_type", "central"),
        "filter_column": filter_rule(args)[0],
        "target_local_fields": list(LOCAL_TARGET_FIELDS),
        "target_ministries": list(getattr(args, "target_ministries", TARGET_MINISTRIES)),
        "ministry_filter_zero_rows": sum(
            cache.get(hash_key(build_key(row)), {}).get("model") == "ministry-filter"
            for row in rows
        ),
        "unique_businesses": len(items),
        "label_counts_in_cache": dict(sorted(counts.items())),
        "classification_status_counts": dict(Counter(
            row.get("classification_status", "") for row in cache.values()
        )),
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
    started_at = time.monotonic()
    args = parse_args()
    args.started_at = started_at
    args.target_ministries = tuple(clean_cell(field) for field in args.target_ministries)
    if len(set(args.target_ministries)) != 11 or any(not field for field in args.target_ministries):
        raise ValueError("LLM 대상 소관명 11개를 TARGET_MINISTRIES 또는 --target-ministries로 지정하세요(중복·빈 값 불가).")
    if args.max_document_chars < 1:
        raise ValueError("--max-document-chars는 1 이상이어야 합니다.")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens는 1 이상이어야 합니다.")
    if args.num_ctx is not None:
        print("안내: --num-ctx는 vLLM에 적용되지 않습니다. 문맥 한도는 서버 설정을 따릅니다.")
    set_default_paths(args)
    headers, rows, encoding = read_csv(args.input_file)
    args.fiscal_type = detect_fiscal_type(headers, rows)
    filter_column, filter_targets = filter_rule(args)
    print(f"재정 유형 자동 판별: {args.fiscal_type} / {filter_column} 기준 {len(filter_targets)}개 대상")
    if filter_column not in headers or any(not filter_value(row, filter_column) for row in rows):
        raise ValueError(f"필터 기준 {filter_column} 값이 누락된 행이 있습니다.")
    if "소관명" not in headers or any(not clean_cell(row.get("소관명")) for row in rows):
        raise ValueError("소관명 컬럼과 모든 행의 소관명이 필요합니다. 누락값을 자동 0으로 처리하지 않습니다.")
    # business_key가 서로 다른 소관기관에 중복되면 캐시/출력에서 분류가 섞일 수 있다.
    key_ministries: dict[str, tuple[str, str]] = {}
    for row in rows:
        key, field = build_key(row), (clean_cell(row["소관명"]), filter_value(row, filter_column))
        if key in key_ministries and key_ministries[key] != field:
            raise ValueError(f"서로 다른 소관기관에서 같은 사업 키가 발견되었습니다: {key}")
        key_ministries[key] = field
    args.gold_label_col = resolve_gold_label_column(headers, args.gold_label_col)
    print(f"평가용 정답 컬럼: {args.gold_label_col}")
    missing = [column for column in KEY_COLUMNS if column not in headers]
    if missing and "business_key" not in headers:
        raise ValueError(f"고유 사업 키 컬럼이 부족합니다: {', '.join(missing)}")

    items = collect_items(rows)
    print(f"입력: {args.input_file} ({len(rows):,}행, {encoding})")
    print(f"고유 사업: {len(items):,}개")
    target_rows = [row for row in rows if filter_value(row, filter_column) in filter_targets]
    print(f"전체 {len(rows):,}행 / LLM 대상 {len(target_rows):,}행 / 자동 0 {len(rows) - len(target_rows):,}행")
    if args.dry_run:
        print_document_match_summary(target_rows, args)
        print("--dry-run: VLLM 호출 및 파일 저장 없이 종료합니다.")
        return 0

    cache = load_cache(args.cache_csv)
    cache = {key: record for key, record in cache.items() if key in items}
    print(f"기존 캐시: {len(cache):,}개")
    # Resolve non-target labels and persist the full-row snapshot BEFORE document checks/API calls.
    llm_items = prepare_ministry_labels(items, cache, args)
    zero_results = {key: cache[key] for key in items if key not in llm_items}
    llm_cache = {key: record for key, record in cache.items() if key in llm_items}
    # Do not rewrite thousands of deterministic zeros after every model response.
    save_cache(args.cache_csv, llm_cache)
    write_outputs(headers, rows, items, {**zero_results, **llm_cache}, args, preliminary=True)
    print(f"자동 0 사전 저장 완료: {len(zero_results):,}개 사업 / {args.output_file}", flush=True)
    try:
        classify_items(llm_items, llm_cache, args)
    finally:
        write_outputs(headers, rows, items, {**zero_results, **llm_cache}, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
