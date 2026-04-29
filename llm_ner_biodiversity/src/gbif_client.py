from typing import Any

import pygbif.literature as lit


def fetch_abstracts(keyword: str, limit: int = 30) -> list[dict[str, Any]]:
    results = lit.search(q=keyword, limit=limit)

    abstracts: list[dict[str, Any]] = []
    for item in results.get("results", []):
        abstract = item.get("abstract", "")
        title = item.get("title", "")

        if abstract and len(abstract) > 30:
            abstracts.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "id": item.get("key", ""),
                }
            )

    return abstracts
