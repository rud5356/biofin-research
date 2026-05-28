"""
GBIF(Global Biodiversity Information Facility) API에서 문헌을 검색하는 클라이언트.

GBIF는 전 세계 생물다양성 데이터를 공유하는 국제 플랫폼입니다 (gbif.org).
이 모듈은 pygbif 라이브러리를 통해 GBIF의 문헌 검색 API에 접근합니다.

주의: GBIF 문헌 API는 PubMed와 달리 논문 초록이 없는 경우가 많습니다.
초록 길이 30자 미만인 항목은 제외합니다.
"""

from typing import Any

import pygbif.literature as gbif_literature


def fetch_abstracts(keyword: str, limit: int = 30) -> list[dict[str, Any]]:
    """
    GBIF 문헌 API에서 키워드로 논문을 검색하고 초록이 있는 항목만 반환합니다.

    Args:
        keyword: 검색 키워드 (예: "Korean mammal species")
        limit: 검색할 최대 문헌 수 (기본값: 30)

    Returns:
        각 항목이 {"title": ..., "abstract": ..., "id": ...} 형태인 딕셔너리 목록
    """
    # GBIF 문헌 API 호출 (q: 검색어, limit: 최대 결과 수)
    search_results = gbif_literature.search(q=keyword, limit=limit)

    abstracts: list[dict[str, Any]] = []
    for item in search_results.get("results", []):
        abstract = item.get("abstract", "")
        title = item.get("title", "")

        # 초록이 없거나 너무 짧은 항목은 NER에 활용하기 어려우므로 제외
        if abstract and len(abstract) > 30:
            abstracts.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "id": item.get("key", ""),  # GBIF 문헌 고유 키
                }
            )

    return abstracts
