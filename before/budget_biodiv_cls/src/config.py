"""
budget_biodiv_cls 프로젝트의 경로 및 열(컬럼) 이름 설정 파일.

이 파일의 상수들을 수정하면 데이터 경로나 컬럼명이 바뀔 때
여러 스크립트를 일일이 고치지 않아도 됩니다.
"""

from __future__ import annotations

from pathlib import Path


# ─── 기본 디렉토리 경로 ───────────────────────────────────────────────────────
# 이 파일(config.py)이 있는 src/ 폴더에서 두 단계 위로 올라가면 프로젝트 루트입니다.
PROJECT_DIR = Path(__file__).resolve().parents[1]   # budget_biodiv_cls/
REPO_DIR = PROJECT_DIR.parent                       # biofin-research/
DATA_DIR = PROJECT_DIR / "data"                     # 처리된 데이터 저장 폴더

# ─── 원본 데이터 경로 (레포지토리 외부에 있는 실제 예산 문서들) ───────────────
# 열린재정 예산 문서 폴더 (HWP, PDF 파일들이 분야별 하위 폴더에 있음)
SOURCE_DOCS_DIR = REPO_DIR / "국가생물다양성_열린재정 데이터"
# 예산 사업과 실제 파일을 매칭한 최종 CSV (budget_matcher가 생성)
SOURCE_MATCHED_CSV = SOURCE_DOCS_DIR / "사업별결산세출지출현황_2024년도_파일매칭_최종.csv"

# ─── 생성되는 데이터 파일 경로 ───────────────────────────────────────────────
# Ollama LLM이 생물다양성 라벨(0 또는 1)을 붙인 CSV
BIODIV_LABELED_CSV = DATA_DIR / "사업별결산세출지출현황_2024년도_biodiv_labeled.csv"
# 각 예산 사업에서 추출한 문서 텍스트 데이터셋
BIODIV_TEXT_DATASET_CSV = DATA_DIR / "biodiv_document_text_dataset.csv"
# 위 데이터셋에 v2 방식으로 라벨이 추가된 버전
BIODIV_TEXT_LABELED_V2_CSV = DATA_DIR / "biodiv_document_text_dataset_labeled_v2.csv"
# 텍스트 데이터셋 통계 요약 (JSON)
BIODIV_TEXT_DATASET_SUMMARY_JSON = DATA_DIR / "biodiv_document_text_dataset_summary.json"

# 학습된 분류 모델이 저장될 폴더
MODEL_DIR = PROJECT_DIR / "model"

# ─── 데이터프레임 열(컬럼) 이름 상수 ─────────────────────────────────────────
# 생물다양성 관련 여부를 나타내는 라벨 열 (0: 비관련, 1: 관련)
LABEL_COLUMN = "biodiv_label"
# 전처리가 완료된 문서 텍스트 열
DOCUMENT_TEXT_COLUMN = "clean_document_text"
# 매칭된 실제 파일명 열
MATCHED_FILENAME_COLUMN = "matched_filename"
# 모델 입력 시 마스킹에 사용할 메타데이터 열 이름들 (개인정보·불필요 정보 제거용)
METADATA_COLUMNS = ("소관명", "분야명", "부문명", "프로그램명", "단위사업명", "세부사업명")
