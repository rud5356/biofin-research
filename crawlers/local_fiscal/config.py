"""지방재정365 세부사업별 세출현황 크롤러 설정."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = REPO_ROOT / "crawlers" / "local_fiscal" / "outputs"
SAVE_DIR = BASE_DIR / "사업설명자료"
MANIFEST_PATH = BASE_DIR / "metropolitan_2024_manifest.csv"

LIST_URL = "https://www.lofin365.go.kr/portal/LF3120202.do"
DETAIL_URL = "https://www.lofin365.go.kr/portal/LF3120204.do"

# 지방재정365 지역 선택값. 순서대로 수집한다.
TARGET_REGIONS = {
    "서울": "11",
    "부산": "26",
    "대구": "27",
    "대전": "30",
    "인천": "28",
    "광주": "29",
    "울산": "31",
    "세종": "32",
    "경기": "41",
    "충북": "43",
    "충남": "44",
    "전남": "46",
    "경북": "47",
    "경남": "48",
    "제주": "49",
    # 지방재정365 조회 API는 특별자치도 출범 이후 자료에도 자체 시도
    # 코드(세종 32, 제주 49, 강원 42, 전북 45)를 사용한다.
    "강원": "42",
    "전북": "45",
}

DEFAULT_DATE = "2024-12-31"
DEFAULT_LIMIT = 5

ROWS_PER_PAGE = 100
PAGE_TIMEOUT_MS = 60_000
DETAIL_SETTLE_MS = 800
