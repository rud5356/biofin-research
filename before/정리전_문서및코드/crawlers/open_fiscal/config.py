"""열린재정 크롤러의 기본 설정."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# 기존 분류 파이프라인과의 호환성을 위해 입출력 기본 경로를 유지한다.
BASE_DIR = REPO_ROOT / "budget_biodiv_cls2" / "outputs"
SAVE_DIR = REPO_ROOT / "crawlers" / "open_fiscal" / "outputs"

SITE_URL = "https://www.openfiscaldata.go.kr/op/ko/bs/UOPKOBSA02"
LIST_API_URL = (
    "https://www.openfiscaldata.go.kr/op/ko/bs/cls/selectSactvSearchList.do"
)
DETAIL_URL = "https://www.openfiscaldata.go.kr/op/ko/bs/cls/UOPKOBSZ01"
DOWNLOAD_BASE_URL = (
    "https://www.openfiscaldata.go.kr/op/ko/cm/downloadFileSayBrkd.do"
)

ALLOWED_EXTENSIONS = {".hwp", ".hwpx", ".pdf", ".zip", ".xlsx"}
