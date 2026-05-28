"""
budget_field_cls 프로젝트의 경로 및 학습 하이퍼파라미터 설정 파일.

이 파일에 있는 상수들을 수정하면 전체 프로젝트에 영향을 줍니다.
경로 상수들은 이 파일의 위치를 기준으로 자동 계산되므로,
프로젝트 폴더를 옮겨도 그대로 동작합니다.
"""

from pathlib import Path


# ─── 기본 디렉토리 경로 ───────────────────────────────────────────────────────
# Path(__file__)은 현재 파일(config.py)의 경로를 나타냅니다.
# .resolve()는 심볼릭 링크 등을 제거하고 절대 경로로 변환합니다.
# .parent.parent는 src/ → budget_field_cls/ 로 두 단계 올라갑니다.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"        # 데이터 파일들이 저장되는 폴더
MODELS_DIR = BASE_DIR / "models"    # 학습된 모델이 저장되는 폴더

# ─── 데이터셋 파일 경로 ───────────────────────────────────────────────────────
# 예산 문서 메타데이터 (파일명, 분야, 부처명 등)
WORKFILE_DATASET_PATH = DATA_DIR / "workfile_dataset.csv"
# 학습에 사용할 분야 라벨 목록
WORKFILE_LABELS_PATH = DATA_DIR / "workfile_labels.csv"
# 파일 경로를 찾지 못한 항목들 (매칭 실패)
WORKFILE_UNRESOLVED_PATH = DATA_DIR / "workfile_dataset_unresolved.csv"
# 데이터셋 통계 요약 (JSON)
WORKFILE_DATASET_SUMMARY_PATH = DATA_DIR / "workfile_dataset_summary.json"
# 문서에서 추출한 텍스트가 포함된 데이터셋
WORKFILE_TEXT_DATASET_PATH = DATA_DIR / "workfile_text_dataset.csv"

# ─── 분할 데이터셋 경로 (train / val / test) ─────────────────────────────────
WORKFILE_SPLIT_DIR = DATA_DIR / "splits"
WORKFILE_TRAIN_SPLIT_PATH = WORKFILE_SPLIT_DIR / "workfile_train.csv"    # 학습용 (80%)
WORKFILE_VAL_SPLIT_PATH = WORKFILE_SPLIT_DIR / "workfile_val.csv"        # 검증용 (10%)
WORKFILE_TEST_SPLIT_PATH = WORKFILE_SPLIT_DIR / "workfile_test.csv"      # 테스트용 (10%)
# 데이터가 너무 적어서 분할에서 제외된 라벨 목록
WORKFILE_DROPPED_LABELS_PATH = WORKFILE_SPLIT_DIR / "workfile_dropped_labels.csv"
# 분할 결과 통계 요약 (JSON)
WORKFILE_SPLIT_SUMMARY_PATH = WORKFILE_SPLIT_DIR / "workfile_split_summary.json"

# 학습된 KoBERT 분류 모델이 저장될 폴더
WORKFILE_KOBERT_MODEL_DIR = MODELS_DIR / "kobert_workfile_classifier"

# ─── 학습 하이퍼파라미터 기본값 ──────────────────────────────────────────────
# 랜덤 시드: 동일한 숫자로 설정하면 매번 같은 방식으로 데이터가 섞입니다 (재현성 보장)
DEFAULT_SPLIT_SEED = 42

# 데이터셋 분할 비율 (합계가 1.0이 되어야 합니다)
DEFAULT_TRAIN_RATIO = 0.8   # 전체 데이터의 80%를 학습에 사용
DEFAULT_VAL_RATIO = 0.1     # 10%를 학습 중 성능 확인에 사용
DEFAULT_TEST_RATIO = 0.1    # 10%를 최종 평가에 사용

# 분류 라벨(카테고리)당 최소 데이터 수. 이보다 적으면 학습에서 제외합니다.
DEFAULT_MIN_LABEL_COUNT = 3

# 사전 학습된 KoBERT 모델 이름 (HuggingFace 허브에서 자동으로 다운로드됩니다)
DEFAULT_KOBERT_MODEL = "klue/bert-base"

# 텍스트를 모델에 입력할 때 최대 토큰 수 (초과 시 잘림)
DEFAULT_MAX_LENGTH = 512

# 한 번에 처리할 데이터 수 (클수록 빠르지만 GPU 메모리를 많이 씁니다)
DEFAULT_BATCH_SIZE = 8

# 전체 데이터를 몇 번 반복 학습할지 (너무 많으면 과적합 발생 가능)
DEFAULT_NUM_EPOCHS = 3

# 학습률: 모델 가중치를 얼마나 빠르게 조정할지 (너무 크면 학습 불안정)
DEFAULT_LEARNING_RATE = 2e-5

# 가중치 감쇠: 과적합을 방지하는 정규화 기법의 강도
DEFAULT_WEIGHT_DECAY = 0.01

# 모델에 입력할 텍스트가 담긴 데이터프레임의 열(컬럼) 이름
DEFAULT_CLASSIFICATION_TEXT_COLUMN = "model_text"

# 같은 그룹(부처명 + 세부사업명)의 데이터가 train/val/test에 분산되지 않게 묶는 기준 컬럼들
DEFAULT_GROUP_COLUMNS = ("ministry_name", "detail_project_name")
