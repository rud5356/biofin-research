"""
PubMed API(NCBI Entrez)를 통해 논문 초록을 수집하는 모듈.

생물다양성 논문 필터링 기능 포함:
  - is_biodiversity_paper(): 생물다양성 관련 논문 여부 판별
  - fetch_abstracts()      : 키워드 검색 후 배치 단위로 초록 수집

NCBI API 사용 정책:
  - API 키 없이 초당 3건 이하 요청 (쿨다운: 0.4초)
  - IncompleteRead 오류 시 최대 3회 자동 재시도
"""

import re
import time
from http.client import IncompleteRead
from typing import Any

from Bio import Entrez

from config import NCBI_EMAIL

# Entrez 모듈에 이메일 등록 (NCBI 정책 — 미등록 시 요청 차단 가능)
Entrez.email = NCBI_EMAIL


# ─── 생물다양성 논문 판별 키워드 목록 ───────────────────────────────────────────
# 이 중 하나라도 제목 또는 초록에 포함되면 생물다양성 논문으로 간주
_BIODIVERSITY_KEYWORDS = [
    "species", "biodiversity", "habitat", "distribution", "population",
    "ecology", "wildlife", "fauna", "flora", "taxonomy", "taxonomic",
    "conservation", "amphibian", "reptile", "mammal", "bird", "fish",
    "insect", "plant", "fungus", "specimen", "occurrence", "abundance",
    "richness", "endemic", "native", "invasive", "migration", "breeding",
    "nesting", "foraging", "predator", "prey", "herbivore",
]

# ─── 의생명/의학 논문 제외 키워드 ──────────────────────────────────────────────
# 이 중 하나라도 포함되면 생물다양성 논문이 아닌 것으로 간주
_BIOMEDICAL_EXCLUDE_KEYWORDS = [
    "cancer", "tumor", "tumour", "carcinoma", "apoptosis", "cell line",
    "in vitro", "in vivo", "chemotherapy", "anticancer", "oxidative stress",
    "reactive oxygen species", "mitochondrial", "signaling pathway",
    "clinical trial", "patient", "therapeutic", "drug resistance",
    "gene expression", "protein expression", "cytokine", "inflammation",
    # 신경/세포 분화 관련 (생물다양성과 무관)
    "oligodendrocyte", "myelination", "progenitor cell", "differentiation",
    "encephalomyelitis", "neuronal", "stem cell",
    # 분자생물학 실험 관련
    "western blot", "flow cytometry", "immunostaining", "transfection",
    "knockdown", "overexpression", "crispr",
    # 'species' 오탐 유발 표현 (예: "reactive oxygen species"는 SPECIES가 아님)
    "cross-species", "reactive oxygen species",
]

# 학명 패턴: 대문자로 시작하는 속명 + 소문자 종소명 (이탤릭 HTML 태그 포함 가능)
# 예: "Rana coreana", "<i>Mustela sibirica</i>"
_SCIENTIFIC_NAME_RE = re.compile(
    r"(?:<i>)?[A-Z][a-z]+\s+[a-z]+(?: [a-z]+)?(?:</i>)?",
)


def is_biodiversity_paper(title: str, abstract: str) -> bool:
    """
    제목과 초록을 분석하여 생물다양성 관련 논문 여부를 판별합니다.

    판별 순서:
    1. 의생명 제외 키워드 포함 시 → False (의학/분자생물학 논문 제외)
    2. 생물다양성 포함 키워드 하나 이상 포함 시 → True
    3. 학명 패턴(이탤릭 학명) 발견 시 → True
    4. 해당 없으면 → False
    """
    text = (title + " " + abstract).lower()

    # 의생명 관련 키워드가 있으면 즉시 False
    for kw in _BIOMEDICAL_EXCLUDE_KEYWORDS:
        if kw in text:
            return False

    # 생물다양성 키워드가 하나라도 있으면 True
    for kw in _BIODIVERSITY_KEYWORDS:
        if kw in text:
            return True

    # 학명 형식(대문자 속명 + 소문자 종소명)이 있으면 True
    combined = title + " " + abstract
    if _SCIENTIFIC_NAME_RE.search(combined):
        return True

    return False


# ─── PubMed API 설정 ────────────────────────────────────────────────────────
_BATCH_SIZE    = 20    # 한 번에 가져올 논문 수 (NCBI 권장 최대치)
_REQUEST_DELAY = 0.4   # 요청 간 대기 시간(초) — NCBI 정책: API 키 없을 때 초당 3건 이하
_MAX_RETRIES   = 3     # IncompleteRead 발생 시 최대 재시도 횟수


def _efetch_with_retry(ids: list[str]) -> Any:
    """
    PubMed 에서 논문 상세 정보를 가져옵니다. 연결 오류 시 자동 재시도합니다.

    IncompleteRead: 네트워크 불안정으로 응답이 도중에 끊길 때 발생.
    재시도 간격은 점진적으로 늘어납니다 (2초 → 4초 → 6초).
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            handle  = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            return records
        except IncompleteRead:
            if attempt == _MAX_RETRIES:
                raise
            wait = attempt * 2
            print(f"  [재시도 {attempt}/{_MAX_RETRIES}] 연결 끊김, {wait}초 후 재요청...")
            time.sleep(wait)


def _parse_articles(records: Any) -> list[dict[str, Any]]:
    """
    PubMed XML 응답에서 논문 정보(id, title, abstract)를 추출합니다.

    초록이 없거나 너무 짧은 논문(50자 미만)은 제외합니다.
    초록이 여러 섹션으로 나뉜 경우(list) 공백으로 합칩니다.
    """
    results = []
    for article in records.get("PubmedArticle", []):
        try:
            medline      = article["MedlineCitation"]
            article_data = medline["Article"]
            title        = str(article_data["ArticleTitle"])
            pmid         = str(medline["PMID"])

            abstract_text = ""
            if "Abstract" in article_data:
                parts = article_data["Abstract"]["AbstractText"]
                # 구조화된 초록(배경/방법/결과/결론 등)은 리스트로 반환됨
                abstract_text = " ".join(str(p) for p in parts) if isinstance(parts, list) else str(parts)

            # 초록이 너무 짧은 논문은 NER 처리 의미 없으므로 제외
            if abstract_text and len(abstract_text) > 50:
                results.append({"id": pmid, "title": title, "abstract": abstract_text})
        except (KeyError, IndexError):
            # 필수 필드 누락 시 해당 논문 건너뜀
            continue
    return results


def fetch_abstracts(
    keyword: str,
    limit: int = 200,
    filter_biodiversity: bool = False,
    batch_size: int = _BATCH_SIZE,
) -> list[dict[str, Any]]:
    """
    PubMed에서 키워드로 논문을 검색하고 초록을 수집합니다.

    filter_biodiversity=True 이면 생물다양성 논문만 골라내면서
    limit 건을 채울 때까지 배치 단위로 반복 수집합니다.

    Args:
        keyword             : PubMed 검색어 (예: "amphibian Korea")
        limit               : 목표 수집 건수
        filter_biodiversity : True이면 is_biodiversity_paper() 필터 적용
        batch_size          : 한 번에 가져올 논문 수

    Returns:
        [{"id": PMID, "title": ..., "abstract": ...}] 형태의 목록
    """
    # PubMed에서 해당 키워드의 전체 검색 결과 수 먼저 확인
    handle          = Entrez.esearch(db="pubmed", term=keyword, retmax=0)
    total_available = int(Entrez.read(handle)["Count"])
    handle.close()
    time.sleep(_REQUEST_DELAY)

    if total_available == 0:
        return []

    results: list[dict[str, Any]] = []
    offset = 0

    # 목표 건수에 도달하거나 전체 검색 결과를 모두 소진할 때까지 반복
    while len(results) < limit and offset < total_available:
        fetch_count = min(batch_size, total_available - offset)

        # 현재 배치의 PMID 목록 가져오기
        handle = Entrez.esearch(db="pubmed", term=keyword, retmax=fetch_count, retstart=offset)
        ids    = Entrez.read(handle)["IdList"]
        handle.close()
        offset += fetch_count
        time.sleep(_REQUEST_DELAY)

        if not ids:
            break

        # PMID 목록으로 상세 정보 가져오기 (재시도 포함)
        batch = _parse_articles(_efetch_with_retry(ids))
        time.sleep(_REQUEST_DELAY)

        # 생물다양성 필터 적용 (True이면 관련 없는 논문 제외)
        if filter_biodiversity:
            batch = [a for a in batch if is_biodiversity_paper(a["title"], a["abstract"])]

        results.extend(batch)
        print(f"  수집 중: {len(results)}/{limit}편 (스캔: {offset}/{total_available})")

    # limit 초과분 제거
    return results[:limit]
