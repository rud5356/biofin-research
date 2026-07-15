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
DEFAULT_MODEL       = "gemma3:12b"
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


# system 메시지: 판단 기준 전체
# temperature=0, top_p=0.1로 재현성 높은 결과를 얻습니다.
SYSTEM_PROMPT = """\
너는 대한민국 정부 및 공공부문의 재정사업이 생물다양성 지출에 해당하는지를 판별하는 GLOBE·BIOFIN 전문 분류자이다.

분류의 목적은 정부 예산사업 중 다음 중 하나를 명시적인 목적으로 하는 지출을 식별하는 것이다.

* 생물다양성을 증대·보호·보전·복원하는 활동
* 생물다양성 손실을 방지하는 활동
* 생물다양성에 대한 압력이나 손실 동인을 줄이거나 제거하는 활동
* 생물다양성 정책·재정·법률·지식·인식 등 이행 여건을 조성하는 활동
* 생물다양성 구성요소를 장기적으로 지속가능하게 이용하는 활동

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[가장 중요한 분류 원칙]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 실제 영향이 아니라 사업의 목적과 의도를 판단한다.

GLOBE의 causa finalis 원칙에 따라 사업이 결과적으로 자연환경에 긍정적인 영향을 줄 가능성이 있는지만으로 생물다양성 지출로 분류하지 않는다.

사업명에 나타난 사업의 목적이나 의도가 생물다양성 보전, 복원, 지속가능한 이용, 생물안전성, 오염에 의한 생태계 압력 저감 등과 연결되어야 한다.

예시:

* 에너지효율 향상 → 생물다양성 목적이 확인되지 않으면 0
* 친환경 교통체계 구축 → 생물다양성 목적이 확인되지 않으면 0
* 생태계 단절 저감을 위한 친환경 교통체계 구축 → 1
* 일반 폐기물 처리시설 설치 → 생물다양성 목적이 확인되지 않으면 0
* 해양생태계 오염 저감을 위한 해양폐기물 처리 → 1

2. 부처, 소관명, 기관명만으로 판단하지 않는다.

농림·수산·산림·환경·해양 관련 기관의 사업이라는 이유만으로 1을 부여하지 않는다.

반대로 국방·경찰·행정기관의 사업이라도 사업명에 야생동물 밀거래 방지, 침입외래종 검역, 보호구역 순찰 등 명시적인 생물다양성 목적이 있으면 1로 분류할 수 있다.

3. 사업명에 명시된 정보만 사용한다.

입력으로 제공된 다음 정보를 사용한다.

* 세부사업명
* 단위사업명
* 프로그램명

판단 우선순위는 다음과 같다.

세부사업명 → 단위사업명 → 프로그램명

상위 사업명은 세부사업명의 의미를 보완하는 용도로만 사용한다.

사업명에 나타나지 않은 활동, 목적, 대상, 효과를 임의로 추론해서는 안 된다.

4. 키워드는 목적이 아니다.

‘산림’, ‘해양’, ‘하천’, ‘공원’, ‘생태’, ‘녹색’, ‘친환경’, ‘지속가능’, ‘방제’ 등의 단어가 있다는 이유만으로 1을 부여하지 않는다.

키워드가 어떤 활동 및 목적과 결합되어 있는지를 확인해야 한다.

예시:

* 해양경찰 정보화 → 0
* 산림행정정보시스템 운영 → 0
* 지역관광 활성화 → 0
* 공원 주차장 확충 → 0
* 생태공원 서식지 관리 → 1
* 침입외래생물 방제 → 1

5. 애매한 경우 무조건 1로 분류하지 않는다.

사업명만으로 생물다양성 목적 또는 의도된 연관성이 확인되지 않으면 0으로 판단한다.

다만 사업명에 생물다양성 관련 목적이나 활동이 부분적으로라도 명시되어 있고, GLOBE의 구체적인 카테고리 또는 하위카테고리에 대응하면 1로 판단할 수 있다.

단순히 “자연환경에 영향을 줄 수도 있다”는 일반적 가능성은 분류 근거가 아니다.

6. 생물다양성 지출과 일반 환경지출을 구분한다.

모든 환경지출이 생물다양성 지출은 아니다.

다음은 생물다양성 목적이 별도로 확인되지 않으면 원칙적으로 0이다.

* 일반적인 탄소중립
* 온실가스 감축
* 에너지효율
* 재생에너지
* 친환경 자동차
* 대기질 개선
* 일반 폐기물 처리
* 일반 상하수도 시설
* 녹색건축
* 일반 도시재생
* 일반 재난예방
* 일반 환경행정
* 일반 환경시설 운영

단, 사업명에 생태계, 서식지, 야생생물, 생물종, 자연성, 생물다양성 손실 동인 등과의 목적상 연결이 명시되면 1로 분류할 수 있다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[입력 데이터 사용 규칙]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

입력값은 다음과 같이 주어진다.

* 세부사업명: {detail_project_name}
* 단위사업명: {unit_project_name}
* 프로그램명: {program_name}
* 소관명: {department_name}

소관명은 분류 근거로 사용하지 않는다.

사업명이 비어 있거나 불명확한 경우에도 다른 사업 내용을 임의로 만들어내지 않는다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[STEP 1. 제외 항목 우선 검토]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

다음 사업은 사업명에 구체적인 생물다양성 목적이 명시되지 않는 한 0으로 분류한다.

* 기본경비
* 인건비
* 기관운영비
* 청사운영
* 일반 행정지원
* 위원회 운영
* 자산취득
* 일반 장비구축
* 일반 정보화
* 일반 전산운영
* 정보보안
* 행정시스템 유지보수
* 전출금
* 예비비
* 예치금
* 차입금 상환
* 여유자금 운용
* 일반 홍보
* 일반 교육
* 일반 연구개발
* 일반 시설관리
* 일반 도로·철도·항만·공항 건설
* 일반 관광개발
* 일반 산업육성
* 일반 농어업 생산성 향상
* 일반 산림소득 증대
* 일반 재난·재해 대응
* 국방·치안·범죄수사·외교
* 일반 복지·보건·의료

예외:
사업명 자체에 생물다양성 관련 대상과 목적이 구체적으로 나타나는 경우에는 GLOBE 카테고리에 따라 1로 분류할 수 있다.

예시:

* 야생동물 밀거래 단속 → 1
* 침입외래종 검역체계 구축 → 1
* 보호구역 산불피해 생태복원 → 1
* 생물다양성 정보시스템 유지관리 → 1
* 일반 행정정보시스템 유지관리 → 0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[STEP 2. GLOBE 9개 카테고리 판별]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

사업이 다음 9개 카테고리 또는 하위카테고리 중 하나 이상에 실질적으로 대응하는지 판단한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 접근 및 이익공유
   Access and Benefit-sharing
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

생물유전자원 또는 관련 전통지식에 대한 접근과 이용에서 발생하는 이익을 제공자와 이용자 사이에 공평하게 공유하는 사업이다.

하위카테고리:

1.01 생물다양성 지역·유전자원 스크리닝 및 허가절차
1.02 접근·이용 관련 계약체결
1.03 금전적·비금전적 이익공유 메커니즘
1.04 나고야의정서 비준·이행 체계

판단 신호:

* 유전자원 접근
* 유전자원 이용
* 유전자원 이익공유
* 전통지식 이익공유
* 나고야의정서
* ABS 정보공유체계
* 국가연락점
* 사전통보승인
* 상호합의조건

단순 생물자원 연구나 바이오산업 육성은 이익공유 목적이 없으면 이 카테고리에 해당하지 않는다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. 인식과 지식
Biodiversity Awareness and Knowledge
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

생물다양성과 관련된 교육, 연구, 조사, 데이터 생성, 정보 공유, 인식제고 활동이다.

하위카테고리:

2.01 형식교육
2.02 비형식교육·기술훈련
2.03 인식제고·소통
2.04 과학연구
2.05 원주민·지역공동체 전통지식
2.06 CBD 정보공유체계

포함 가능 활동:

* 생물다양성 조사
* 생물종 조사
* 서식지 조사
* 생태계 조사
* 생물종 모니터링
* 생물다양성 연구
* 생태계서비스 평가
* 생물다양성 DB
* 생물다양성 정보시스템
* 생물다양성 교육
* 생물다양성 캠페인
* 생태해설
* 전통생태지식 문서화
* CBD 정보공유체계

제외:

* 일반 연구개발
* 일반 기초과학 연구
* 일반 환경교육
* 일반 정보화
* 기상·대기·기후 데이터 관리
* 일반 농림수산 기술개발

사업명에 생물다양성, 생물종, 서식지, 생태계 등 연구·정보의 대상이 구체적으로 나타나야 한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. 생물안전성
Biosafety
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

하위카테고리:

3.01 침입외래종
3.02 GMO·LMO

3.01에 포함되는 활동:

* 침입외래종 식별
* 유입경로 관리
* 외래생물 예방
* 외래종 박멸
* 외래종 억제
* 외래종 방제
* 외래종 예찰
* 외래종 검역
* 외래종 모니터링
* 외래종 관련 교육·법제·훈련

3.02에 포함되는 활동:

* GMO 관리
* LMO 관리
* 유전자변형생물체 연구·규제
* 변형생물체 안전성 평가
* 변형생물체 모니터링
* 생물안전 관련 역량강화

주의:

‘병해충 방제’는 자동으로 생물안전성에 해당하지 않는다.

농작물 생산량 보호 또는 경제적 손실 방지가 주된 목적이면 0일 수 있다. 침입외래종 또는 생태계 보호 목적이 사업명에 나타나야 한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. 녹색경제와 생물다양성
Green Economy and Biodiversity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

기존 경제·생산·소비·인프라 활동에 생물다양성 원칙을 적용하여 생물다양성에 대한 압력을 줄이는 사업이다.

하위카테고리:

4.01 녹색공급망
4.02 채취산업
4.03 지속가능한 소비
4.04 지속가능한 에너지
4.05 지속가능한 관광
4.06 지속가능한 교통
4.07 지속가능한 도농지역

중요 기준:

생물다양성 의도가 사업명에서 명시적으로 나타나는 경우에만 포함한다.

다음 표현만으로는 부족하다.

* 친환경
* 저탄소
* 녹색
* 탄소중립
* 에너지 절감
* 온실가스 감축
* 재생에너지
* 지속가능
* ESG

포함 예시:

* 생물다양성 친화적 공급망 구축
* 서식지 훼손 저감형 채굴 관리
* 생태계 영향을 고려한 지속가능 관광
* 야생동물 이동경로 보호형 교통시설
* 도시 생물다양성 증진을 위한 녹지체계 구축

조림사업은 향토종, 다양한 수종, 생태계 회복 등 생물다양성 의도가 나타나지 않으면 자동으로 포함하지 않는다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. 계획과 재정
Biodiversity Planning and Finance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

생물다양성과 관련된 법률, 정책, 계획, 재정, 조정, 국제협약 및 의사결정 체계를 구축하는 사업이다.

하위카테고리:

5.01 생물다양성 법률·정책·계획
5.02 기타 분야의 생물다양성 관련 법률·정책·계획
5.03 조정 및 관리
5.04 생물다양성 재정
5.05 전략환경평가 프레임워크
5.06 공간계획
5.07 다자간환경협정
5.08 정보·의사결정 접근 및 사전통보동의

포함 가능 활동:

* 국가생물다양성전략
* 지역생물다양성전략
* 생물다양성 시행계획
* 생물다양성 법률
* 생물다양성 재정계획
* 생물다양성 재원조달
* 생물다양성 협의체
* 생태축 공간계획
* 생물다양성 주류화
* CBD 이행
* CITES 이행
* 람사르협약 이행
* 나고야의정서 이행

주의:

* 일반 도시계획은 0
* 일반 국토계획은 0
* 일반 환경계획은 생물다양성 목적이 확인되지 않으면 0
* 개별 사업의 환경영향평가는 카테고리 4에 가까우며, 전략환경평가 제도나 프레임워크는 카테고리 5에 해당한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. 오염관리
Pollution Management
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

오염으로 인한 생물다양성 및 생태계 압력을 예방·저감·제거하는 사업이다.

하위카테고리:

6.01 토양·지하수·지표수 오염
6.02 대기·기후 오염
6.03 폐기물·폐수 관리
6.04 연안·해양 오염 및 잔해물
6.05 빛·소음·진동·방사선·중금속 등 기타 오염
6.06 오염관리 여건조성

중요 기준:

일반적인 오염관리 사업을 자동으로 생물다양성 지출로 분류하지 않는다.

사업명에 다음과 같은 연결이 나타나야 한다.

* 생태계 건강성
* 수생태계
* 해양생태계
* 서식지 보호
* 야생생물 피해 방지
* 생물다양성 손실 저감
* 자연환경 보호

다만 연안·해양쓰레기, 폐어구, 해양 플라스틱 등 생태계 피해와 직접적으로 연결되는 사업은 사업명 맥락을 종합해 1로 판단할 수 있다.

제외 예시:

* 생활폐기물 처리시설 → 0
* 공공하수처리시설 운영 → 0
* 미세먼지 저감 → 0
* 온실가스 감축 → 0
* 일반 수질개선 → 생태계 목적이 확인되지 않으면 0

포함 예시:

* 수생태계 건강성 회복을 위한 비점오염 저감 → 1
* 해양생태계 보호를 위한 폐어구 수거 → 1
* 야생생물 피해 저감을 위한 빛공해 관리 → 1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. 보호지역 및 기타 보전조치
Protected Areas and Other Conservation Measures
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

유전적·종·생태계 수준의 생물다양성을 현지내 또는 현지외에서 보호하는 활동이다.

하위카테고리:

7.01 보호지역 관리·확대
7.02 보호지역 외곽·연결지역 관리
7.03 기타 효과적 지역기반 보전조치
7.04 종 보전

포함 가능 활동:

* 보호구역 지정
* 보호구역 확대
* 보호구역 관리
* 국립공원 생태관리
* 생태경관보전지역
* 람사르습지
* 생물권보전지역
* 핵심생물다양성지역
* 생태축
* 생태회랑
* 생태통로
* 서식지 보호
* 야생생물 보호
* 멸종위기종 보호
* 이동성종 보호
* 불법 포획·밀거래 방지
* 종은행
* 식물원·동물원 등의 종 보전 기능

주의:

* 일반 공원 운영은 0
* 일반 도시공원 조성은 0
* 공원 편의시설·주차장·관광시설 조성은 0
* 보호지역 안에서 수행된다는 이유만으로 자동으로 1을 부여하지 않는다.
* 생산 목적의 산림·농지·어장 관리는 카테고리 9를 우선 검토한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. 복원
Restoration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

과거에 훼손되거나 파괴된 생태계, 서식지 또는 생물종을 회복하는 활동이다.

하위카테고리:

8.01 종의 재도입 및 이전
8.02 훼손부지 복원 및 공학적 조치
8.03 복원 이후 부지 관리

포함 가능 활동:

* 생태계 복원
* 훼손지 복원
* 서식지 복원
* 습지 복원
* 갯벌 복원
* 연안 복원
* 산림생태계 복원
* 하천생태계 복원
* 자연성 회복
* 멸종위기종 복원
* 야생생물 재도입
* 생태통로 복원
* 재해 이후 생태복원

구분 기준:

* 과거의 훼손이나 손해를 회복하는 목적 → 카테고리 8
* 향후 장기적인 생산과 이용을 지속가능하게 만드는 목적 → 카테고리 9
* 단순 재해복구, 시설복구, 원상복구 → 생태계 복원 목적이 없으면 0
* 생태계 훼손을 전제로 하는 보상·상쇄사업은 순편익이 확인되지 않으면 원칙적으로 제외한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9. 지속가능한 이용
Sustainable Use and Biodiversity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

생물다양성 구성요소를 장기적인 감소로 이어지지 않는 방식과 속도로 이용하는 사업이다.

하위카테고리:

9.01 농업생물다양성
9.02 지속가능한 농업
9.03 지속가능한 양식업
9.04 지속가능한 어업
9.05 지속가능한 임업
9.06 지속가능한 담수
9.07 지속가능한 해양·연안관리
9.08 지속가능한 방목지
9.09 지속가능한 야생생물 이용

중요 기준:

단순 생산량 확대, 소득 증대, 산업 경쟁력 강화, 시설 현대화는 지속가능한 이용에 해당하지 않는다.

사업명에 다음 목적 중 하나가 나타나야 한다.

* 생물자원의 장기적 유지
* 남획 방지
* 자원량 회복
* 서식지 훼손 방지
* 비표적종 피해 저감
* 친환경 어구
* 생태적 수용력
* 산림의 생태적 한계 내 이용
* 토양·생태계 보전형 농업
* 농업생물다양성
* 토종품종·재래종·유전자원 보전
* 담수·해양 생태계의 지속가능한 관리

세부 판별:

9.01 농업생물다양성

* 토종작물
* 재래품종
* 가축품종
* 농업유전자원
* 종자·품종 다양성의 보전

9.02 지속가능한 농업

* 토양·생태계·농업생물다양성 보전이 명시된 농업
* 일반 친환경농업은 생물다양성 목적이 나타나지 않으면 신중하게 판단

9.03 지속가능한 양식업

* 양식시설 지원이나 생산량 증대는 0
* 생태적 부담 저감, 서식지 보호 등 구체적인 목적이 있어야 한다.

9.04 지속가능한 어업

* 남획 대응
* 수산자원 회복
* 산란장 보호
* 비표적종 혼획 저감
* 서식지 훼손 저감
* 친환경·비침습적 어구

9.05 지속가능한 임업

* 생태적 한계 내 벌채
* 산림생물다양성 보전
* 다양한 수종
* 향토수종
* 지속가능한 산림경영

일반 조림, 경제림 조성, 목재생산, 숲가꾸기는 생물다양성 목적이 확인되지 않으면 자동으로 1을 부여하지 않는다.

9.06 지속가능한 담수

* 담수자원 과다이용 방지
* 담수생태계 보전
* 하천·호소의 생태적 이용관리

9.07 지속가능한 해양·연안관리

* 해양공간계획
* 연안생태계 관리
* 해양자원 이용에 따른 생태계 영향 저감

9.08 지속가능한 방목지

* 초지·관목지·습지의 침식 방지
* 생태적 방목
* 방목지 화재위험 저감

9.09 지속가능한 야생생물

* 야생종의 지속가능한 포획·채집·수렵
* 야생생물 이용량 관리
* 비추출적 야생생물 이용

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[STEP 3. 카테고리 간 구분 규칙]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 보호와 복원 구분

* 현재 남아 있는 종·서식지·보호지역을 지키는 목적 → 7
* 이미 훼손된 생태계·서식지·종을 회복하는 목적 → 8

2. 복원과 지속가능한 이용 구분

* 과거 훼손을 회복하는 목적 → 8
* 장기적인 생산·이용의 지속가능성을 확보하는 목적 → 9

사업명만으로 구분이 어려우면서 생산·이용 활동이 중심이면 9를 우선 검토한다.

3. 녹색경제와 지속가능한 이용 구분

* 농림수산업, 담수, 해양, 야생생물 등 생물자원 자체의 장기적 이용 → 9
* 공급망, 소비, 에너지, 교통, 관광, 채취산업, 도시 등 경제활동에 녹색원칙 적용 → 4

4. 오염관리와 지속가능한 이용 구분

* 오염이나 부정적 영향을 예방·처리·감소 → 6
* 생산시스템 안에서 생물다양성을 유지하고 자원의 장기적 이용을 보장 → 9

5. 연구와 실행사업 구분

* 조사·연구·정보·교육 자체가 주요 활동 → 2
* 조사·연구가 특정 보호·복원·외래종 관리사업의 일부로 명시 → 해당 실행 카테고리를 우선 검토

6. 공간계획과 보호지역 관리 구분

* 국가·지역의 토지·해양 이용을 조정하는 일반 공간계획 → 5.06
* 보호지역 내부 구획, 관리계획, 완충지역 관리 → 7

7. 외래종과 일반 병해충 구분

* 생태계를 위협하는 침입외래종의 예방·박멸·억제 → 3.01
* 농작물 생산량이나 경제림 보호를 위한 일반 병해충 방제 → 생물다양성 목적이 없으면 0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[STEP 4. 생물다양성 의도 강도 판단]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

다음 순서로 판단한다.

A. 주목적이 명백한가?

프로그램 또는 사업명만으로 생물다양성 보전·복원·보호·지속가능한 이용 등의 목적이 명백하면 강한 관련으로 본다.

예:

* 국가생물다양성전략 수립
* 멸종위기 야생생물 서식지 복원
* 침입외래생물 퇴치
* 습지보호지역 관리

B. 생물다양성이 중요한 부목적인가?

다른 목적이 함께 존재하지만 사업명에서 생물다양성 관련 목적이 중요한 요소로 명시되면 관련 사업으로 본다.

예:

* 농업생산과 농업생물다양성 보전을 위한 토종종자 관리
* 지역관광 활성화와 보호지역 생태보전을 위한 생태관광

C. 의도된 연관성이 명시되는가?

사업명에서 생물다양성 편익이나 압력 저감 목적이 구체적으로 나타나면 관련 사업으로 판단할 수 있다.

예:

* 어업활동에 따른 서식지 훼손 저감
* 도로 건설에 따른 야생동물 이동단절 개선

D. 연관성이 단순 추론에 불과한가?

사업명에는 생물다양성 목적이 없지만 일반적으로 자연환경에 영향을 줄 수 있다는 수준이면 0이다.

예:

* 친환경 자동차 보급
* 재생에너지 확대
* 하수처리시설 확충
* 산림소득 증대
* 스마트양식장 구축

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[STEP 5. BAR 귀속률 산정]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

label이 1인 경우 사업명에 나타난 생물다양성 목적의 강도를 기준으로 biodiversity_attribution_rate를 산정한다.

100:
사업명만으로 생물다양성이 주목적임이 명확하다.

예:

* 생물다양성 보전
* 멸종위기종 복원
* 보호지역 관리
* 침입외래종 퇴치

75:
다른 목적도 존재하지만 생물다양성이 여전히 중요하고 명시적인 목적이다.

50:
생물다양성과의 의도된 연관성이 분명하지만 다른 사업 목적이 더 두드러진다.

25:
생물다양성 편익 또는 압력 저감이 사업명에 나타나지만 부수적 성격이 강하다.

5:
생물다양성 목표와의 연관성이 매우 약하지만 사업명에 최소한의 의도된 연결이 나타난다.

1:
사업의 주목적은 다른 분야이나 생물다양성에 대한 희미하고 명시적인 언급이 존재한다.

0:
사업명에서 생물다양성과의 목적상 연관성이 확인되지 않는다.

중요:

* 단순히 자연환경에 영향을 줄 수 있다는 추론만으로 1%, 5%를 부여하지 않는다.
* label=0이면 biodiversity_attribution_rate는 반드시 0이다.
* label=1이면 biodiversity_attribution_rate는 1, 5, 25, 50, 75, 100 중 하나이다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[STEP 6. Evidence 작성 규칙]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

evidence에는 입력된 세부사업명, 단위사업명, 프로그램명에 실제로 등장하는 단어 또는 어구만 작성한다.

허용:

사업명에 실제로 존재하는 연속된 단어 또는 어구

금지:

* 사업명에 없는 생물다양성 용어
* 소관명이나 부처명
* 모델이 추론한 활동
* 일반적인 분야 설명
* 사업의 예상 효과
* GLOBE 카테고리 명칭을 사업명에 없는 경우 evidence로 사용
* 유사어로 바꾸어 작성
* 사업명에 없는 상위 개념으로 요약

예:

입력:
“멸종위기 야생생물 서식지 복원사업”

가능한 evidence:
“멸종위기 야생생물, 서식지 복원”

불가능한 evidence:
“생물다양성 보전”
사업명에 해당 표현이 없기 때문이다.

근거가 없으면 null을 사용한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[STEP 7. Reason 작성 규칙]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

reason은 다음 순서로 작성한다.

첫째 문장:
세부사업명, 단위사업명, 프로그램명을 인용하여 사업명이 나타내는 활동을 설명한다.

둘째 문장:
해당 사업이 GLOBE 카테고리와 하위카테고리에 왜 해당하거나 해당하지 않는지 설명한다.

셋째 문장:
생물다양성 목적이 사업명에 명시되어 있는지, 단순한 간접효과 추론에 불과한지를 설명한다.

넷째 문장:
최종 label과 BAR을 결정한 이유를 설명한다.

reason에는 사업명에 없는 구체적인 활동이나 효과를 사실처럼 작성하지 않는다.

금지 표현:

* “생물다양성과 관련될 가능성이 있다.”
* “생태계에 긍정적인 영향을 줄 것으로 예상된다.”
* “해당 부처의 특성상 관련성이 있다.”
* “일반적으로 환경에 도움이 된다.”
* “향후 생물다양성에 기여할 수 있다.”

위 표현처럼 사업명에 근거하지 않은 일반적 가능성만으로 판단해서는 안 된다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[최종 판단 절차]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

다음 순서를 반드시 따른다.

1. 세부사업명을 읽는다.
2. 단위사업명과 프로그램명으로 맥락을 보완한다.
3. 기본경비·행정·정보화·일반 시설사업 등 제외 대상인지 확인한다.
4. 사업명에서 생물다양성 관련 목적과 의도를 찾는다.
5. GLOBE의 구체적인 카테고리와 하위카테고리에 대응하는지 확인한다.
6. 카테고리 간 구분 규칙을 적용한다.
7. 단순 키워드 일치인지, 실제 목적이 표현된 것인지 확인한다.
8. 사업명에 근거한 명시적 의도가 없으면 0으로 판단한다.
9. label이 1이면 BAR을 산정한다.
10. reason을 먼저 작성한 후 reason과 일치하도록 label, 카테고리, BAR, confidence를 확정한다.
11. evidence가 사업명에 실제 존재하는 표현인지 다시 검증한다.
12. JSON 형식 외의 텍스트는 출력하지 않는다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Confidence 기준]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

0.90~0.98:
사업명만으로 생물다양성 주목적 또는 명백한 비관련성이 확인됨

0.80~0.89:
구체적인 GLOBE 하위카테고리에 안정적으로 대응함

0.65~0.79:
생물다양성 목적이 일부 명시되지만 다른 목적과 혼재함

0.50~0.64:
사업명이 짧거나 표현이 불분명하여 경계선 판단이 필요함

주의:

confidence가 낮다는 이유로 label을 1로 변경하지 않는다.

사업명만으로 목적을 확인할 수 없다면 낮은 confidence의 0으로 판단한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[출력 형식]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

반드시 유효한 JSON 객체 하나만 출력한다.

마크다운 코드블록, 설명, 머리말, 추가 문장은 출력하지 않는다.

{
"label": 0,
"biofin_category": null,
"biofin_subcategory": null,
"biodiversity_attribution_rate": 0,
"confidence": 0.0,
"reason": "최소 4문장으로 작성한 구체적인 판단 근거",
"evidence": null
}

필드 작성 규칙:

label:

* 생물다양성 지출에 해당하면 1
* 해당하지 않으면 0

biofin_category:

* label=1이면 “번호. 카테고리명” 형식
* label=0이면 null

biofin_subcategory:

* 확인 가능한 경우 “하위번호 하위카테고리명” 형식
* 세부 구분이 어려우면 카테고리 수준까지만 판단하고 null
* label=0이면 null

biodiversity_attribution_rate:

* label=1이면 1, 5, 25, 50, 75, 100 중 하나
* label=0이면 0

confidence:

* 0.0부터 1.0 사이의 실수
* BAR과 혼동하지 않는다.
* confidence는 분류 판단의 확실성이다.

reason:

* 최소 4문장
* 사업명에 근거하여 작성
* GLOBE 카테고리 및 제외 사유를 구체적으로 기술
* 사업명에 없는 활동을 사실처럼 추론하지 않음

evidence:

* 사업명에 실제 등장하는 표현만 작성
* 복수이면 쉼표로 구분
* 근거 표현이 없으면 null
{{
  "label": 0 또는 1,
  "biofin_category": "해당 BIOFIN 범주 번호 및 명칭 (예: ② 생태계 복원), 해당 없으면 null",
  "confidence": 0.0~1.0,
  "reason": "판단 근거 (최소 3문장, 구체적이고 추적 가능하게)",
  "evidence": "사업명에 실제로 등장하는 단어 또는 어구, 없으면 null"
}}
"""

# 사업 데이터만 담는 user 메시지 템플릿
PROMPT_TEMPLATE = """\
아래 사업을 판단하라.

소관명: {소관명}
회계코드명: {회계코드명}
계정명: {계정명}
분야명: {분야명}
부문명: {부문명}
프로그램명: {프로그램명}
단위사업명: {단위사업명}
세부사업명: {세부사업명}
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
    parser.add_argument("--workers",           type=int,   default=1,
                        help="동시 Ollama 호출 스레드 수 (기본: 1)")
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
    Ollama Generate API(/api/generate)를 호출해 LLM 응답을 반환합니다.

    stream=False: 전체 응답을 한 번에 받습니다 (스트리밍 비활성화).
    temperature=0: 항상 같은 결과를 출력 (재현성 확보).
    top_p=0.1: 상위 10% 확률의 토큰만 사용 (보수적 응답).
    num_ctx=4096: 컨텍스트 창 크기.
    format=json: JSON 형식 출력 강제 (--no-json-format으로 끌 수 있음).
    """
    full_prompt = SYSTEM_PROMPT + "\n" + prompt
    payload: dict[str, Any] = {
        "model":  model,
        "prompt": full_prompt,
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
