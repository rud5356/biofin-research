from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

SOURCE_DOCS_DIR = REPO_DIR / "국가생물다양성_열린재정 데이터"
SOURCE_MATCHED_CSV = SOURCE_DOCS_DIR / "사업별결산세출지출현황_2024년도_파일매칭_최종.csv"

BIODIV_LABELED_CSV = DATA_DIR / "사업별결산세출지출현황_2024년도_biodiv_labeled.csv"
BIODIV_TEXT_DATASET_CSV = DATA_DIR / "biodiv_document_text_dataset.csv"
BIODIV_TEXT_LABELED_V2_CSV = DATA_DIR / "biodiv_document_text_dataset_labeled_v2.csv"
BIODIV_TEXT_DATASET_SUMMARY_JSON = DATA_DIR / "biodiv_document_text_dataset_summary.json"

MODEL_DIR = PROJECT_DIR / "model"

LABEL_COLUMN = "biodiv_label"
DOCUMENT_TEXT_COLUMN = "clean_document_text"
MATCHED_FILENAME_COLUMN = "matched_filename"
METADATA_COLUMNS = ("소관명", "분야명", "부문명", "프로그램명", "단위사업명", "세부사업명")
