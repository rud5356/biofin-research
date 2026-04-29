from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROMPTS_DIR = BASE_DIR / "prompts"
RESULTS_DIR = BASE_DIR / "results"

NCBI_EMAIL = "dbsdkdkssud7@naver.com"

DEFAULT_KEYWORD = "Korean mammal species"
DEFAULT_LIMIT = 100
DEFAULT_SAMPLE_SIZE = 30
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_MODE = "few_shot"
DEFAULT_CHECKPOINT_EVERY = 10
