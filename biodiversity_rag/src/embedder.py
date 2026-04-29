"""
sentence-transformers로 텍스트를 벡터로 변환한다.
"""
try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError as exc:
    SentenceTransformer = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from config import EMBEDDING_MODEL


_model = None


def get_model() -> SentenceTransformer:
    """모델을 처음 한 번만 로드하고 이후에는 캐싱된 인스턴스를 반환한다."""
    global _model
    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "sentence-transformers가 설치되지 않았습니다. "
            "`conda env create -f biodiversity_rag/environment.yml`로 환경을 만든 뒤 다시 실행하세요."
        ) from _IMPORT_ERROR
    if _model is None:
        print(f"임베딩 모델 로드 중: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """텍스트 목록을 임베딩 벡터 목록으로 변환한다."""
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """질의 문장 하나를 임베딩 벡터로 변환한다."""
    model = get_model()
    return model.encode(query, convert_to_numpy=True).tolist()
