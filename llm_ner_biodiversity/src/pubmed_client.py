import re
import time
from typing import Any
from http.client import IncompleteRead
from Bio import Entrez
from config import NCBI_EMAIL

Entrez.email = NCBI_EMAIL

# 생물다양성 논문 판별 키워드 (title 또는 abstract에 하나 이상 포함되어야 함)
_BIODIVERSITY_KEYWORDS = [
    "species", "biodiversity", "habitat", "distribution", "population",
    "ecology", "wildlife", "fauna", "flora", "taxonomy", "taxonomic",
    "conservation", "amphibian", "reptile", "mammal", "bird", "fish",
    "insect", "plant", "fungus", "specimen", "occurrence", "abundance",
    "richness", "endemic", "native", "invasive", "migration", "breeding",
    "nesting", "foraging", "predator", "prey", "herbivore",
]

# 생물다양성과 무관한 의생명/의학 논문 제외 키워드 (title 또는 abstract에 포함 시 제외)
_BIOMEDICAL_EXCLUDE_KEYWORDS = [
    "cancer", "tumor", "tumour", "carcinoma", "apoptosis", "cell line",
    "in vitro", "in vivo", "chemotherapy", "anticancer", "oxidative stress",
    "reactive oxygen species", "mitochondrial", "signaling pathway",
    "clinical trial", "patient", "therapeutic", "drug resistance",
    "gene expression", "protein expression", "cytokine", "inflammation",
    # 신경/세포 분화 관련
    "oligodendrocyte", "myelination", "progenitor cell", "differentiation",
    "encephalomyelitis", "neuronal", "stem cell",
    # 분자생물학 실험 관련
    "western blot", "flow cytometry", "immunostaining", "transfection",
    "knockdown", "overexpression", "crispr",
    # "species" 오탐 유발 표현
    "cross-species", "reactive oxygen species",
]

# 학명 패턴: 대문자로 시작하는 속명 + 소문자 종소명 (이탤릭 태그 포함 가능)
_SCIENTIFIC_NAME_RE = re.compile(
    r"(?:<i>)?[A-Z][a-z]+\s+[a-z]+(?: [a-z]+)?(?:</i>)?",
)


def is_biodiversity_paper(title: str, abstract: str) -> bool:
    """title과 abstract를 바탕으로 생물다양성 논문 여부를 판별한다."""
    text = (title + " " + abstract).lower()

    # 의생명 제외 키워드가 있으면 False
    for kw in _BIOMEDICAL_EXCLUDE_KEYWORDS:
        if kw in text:
            return False

    # 생물다양성 포함 키워드가 하나 이상 있으면 True
    for kw in _BIODIVERSITY_KEYWORDS:
        if kw in text:
            return True

    # 학명 패턴(이탤릭 학명)이 있으면 True
    combined = title + " " + abstract
    if _SCIENTIFIC_NAME_RE.search(combined):
        return True

    return False


_BATCH_SIZE = 20          # NCBI 권장: 요청당 너무 많은 레코드 금지
_REQUEST_DELAY = 0.4      # NCBI 권장: 초당 3건 이하 (API key 없을 때)
_MAX_RETRIES = 3


def _efetch_with_retry(ids: list[str]) -> Any:
    """IncompleteRead 발생 시 최대 _MAX_RETRIES 회 재시도."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
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
    """PubMed XML 응답에서 id/title/abstract 추출."""
    results = []
    for article in records.get("PubmedArticle", []):
        try:
            medline = article["MedlineCitation"]
            article_data = medline["Article"]
            title = str(article_data["ArticleTitle"])
            pmid = str(medline["PMID"])
            abstract_text = ""
            if "Abstract" in article_data:
                parts = article_data["Abstract"]["AbstractText"]
                abstract_text = " ".join(str(p) for p in parts) if isinstance(parts, list) else str(parts)
            if abstract_text and len(abstract_text) > 50:
                results.append({"id": pmid, "title": title, "abstract": abstract_text})
        except (KeyError, IndexError):
            continue
    return results


def fetch_abstracts(
    keyword: str,
    limit: int = 200,
    filter_biodiversity: bool = False,
    batch_size: int = _BATCH_SIZE,
) -> list[dict[str, Any]]:
    """PubMed에서 논문을 수집한다.

    filter_biodiversity=True이면 목표(limit)편을 채울 때까지
    배치 단위로 반복 수집하며 비생물다양성 논문을 건너뛴다.
    """
    # PubMed에서 해당 키워드의 전체 검색 결과 수 확인
    handle = Entrez.esearch(db="pubmed", term=keyword, retmax=0)
    total_available = int(Entrez.read(handle)["Count"])
    handle.close()
    time.sleep(_REQUEST_DELAY)

    if total_available == 0:
        return []

    results: list[dict[str, Any]] = []
    offset = 0

    while len(results) < limit and offset < total_available:
        fetch_count = min(batch_size, total_available - offset)

        # 배치 ID 검색
        handle = Entrez.esearch(db="pubmed", term=keyword, retmax=fetch_count, retstart=offset)
        ids = Entrez.read(handle)["IdList"]
        handle.close()
        offset += fetch_count
        time.sleep(_REQUEST_DELAY)

        if not ids:
            break

        # 상세 정보 가져오기 (재시도 포함)
        batch = _parse_articles(_efetch_with_retry(ids))
        time.sleep(_REQUEST_DELAY)

        if filter_biodiversity:
            batch = [a for a in batch if is_biodiversity_paper(a["title"], a["abstract"])]

        results.extend(batch)
        print(f"  수집 중: {len(results)}/{limit}편 (스캔: {offset}/{total_available})")

    return results[:limit]
