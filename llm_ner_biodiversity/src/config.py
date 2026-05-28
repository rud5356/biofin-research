"""
llm_ner_biodiversity 프로젝트의 경로 및 기본값 설정 파일.

NER(Named Entity Recognition, 개체명 인식)이란:
  텍스트에서 "종명", "지역명", "연구 방법" 같은 특정 범주의 단어를 자동으로 찾아내는 기술입니다.
  이 프로젝트는 LLM(대형 언어 모델)을 사용해 생물다양성 논문 초록에서 개체명을 추출합니다.
"""

from pathlib import Path


# ─── 기본 디렉토리 경로 ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent   # llm_ner_biodiversity/
DATA_DIR = BASE_DIR / "data"                        # 수집된 논문 초록 저장 폴더
PROMPTS_DIR = BASE_DIR / "prompts"                  # LLM에게 보낼 프롬프트 템플릿 폴더
RESULTS_DIR = BASE_DIR / "results"                  # NER 결과 CSV 저장 폴더

# ─── PubMed API 설정 ──────────────────────────────────────────────────────────
# NCBI(미국 국립생물정보센터) API는 이메일 주소를 요구합니다.
# 과도한 요청 방지 및 연락용으로 사용되며, 실제로 이메일이 전송되지는 않습니다.
NCBI_EMAIL = "dbsdkdkssud7@naver.com"

# ─── 논문 수집 기본값 ─────────────────────────────────────────────────────────
# PubMed 검색 키워드 (생물다양성 관련 논문을 찾기 위한 기본 검색어)
DEFAULT_KEYWORD = "Korean mammal species"
# 한 번에 최대 수집할 논문 수
DEFAULT_LIMIT = 100
# NER 처리에 사용할 논문 샘플 수 (전체가 많을 때 일부만 처리)
DEFAULT_SAMPLE_SIZE = 30

# ─── LLM 설정 ────────────────────────────────────────────────────────────────
# Ollama를 통해 로컬에서 실행할 LLM 모델
DEFAULT_MODEL = "llama3.1:8b"
# 프롬프트 방식: "few_shot"(예시 포함), "zero_shot"(예시 없이) 중 선택
DEFAULT_MODE = "few_shot"
# 처리 중 중간 저장 주기 (몇 건마다 결과를 저장할지)
# 오류 발생 시 처음부터 다시 하지 않아도 되도록 체크포인트를 설정합니다.
DEFAULT_CHECKPOINT_EVERY = 10
