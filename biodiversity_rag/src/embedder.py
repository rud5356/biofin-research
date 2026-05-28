"""
텍스트를 벡터(숫자 배열)로 변환하는 임베딩 모듈.

임베딩(Embedding)이란?
  "생물다양성"이라는 단어를 예를 들면, 컴퓨터는 문자 자체를 이해하지 못합니다.
  임베딩은 텍스트를 [0.3, -0.1, 0.8, ...] 같은 숫자 배열로 변환해
  의미가 비슷한 텍스트는 가까운 벡터로, 다른 텍스트는 먼 벡터로 표현합니다.

이 모듈은 sentence-transformers 라이브러리를 사용하며,
모델을 처음 한 번만 로드하고 이후에는 캐시된 인스턴스를 재사용합니다.
"""

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError as exc:
    # 라이브러리가 없어도 import 자체는 성공하도록 처리
    # 실제 사용 시 get_model()에서 안내 메시지와 함께 오류를 발생시킵니다.
    SentenceTransformer = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from config import EMBEDDING_MODEL


# 로드된 모델을 전역으로 캐시 (첫 호출 이후에는 재사용)
_cached_model = None


def get_model() -> SentenceTransformer:
    """
    임베딩 모델을 반환합니다. 처음 호출 시 로드, 이후에는 캐시된 인스턴스 반환.

    모델 파일은 첫 실행 시 HuggingFace Hub에서 자동으로 다운로드됩니다.
    """
    global _cached_model
    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "sentence-transformers가 설치되지 않았습니다. "
            "`conda env create -f biodiversity_rag/environment.yml`로 환경을 만든 뒤 다시 실행하세요."
        ) from _IMPORT_ERROR
    if _cached_model is None:
        print(f"임베딩 모델 로드 중: {EMBEDDING_MODEL}")
        _cached_model = SentenceTransformer(EMBEDDING_MODEL)
    return _cached_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    여러 텍스트를 한 번에 임베딩 벡터 목록으로 변환합니다.

    show_progress_bar=True: 대량 처리 시 진행률을 터미널에 표시합니다.
    convert_to_numpy=True: 결과를 numpy 배열로 받아 .tolist()로 Python 리스트 변환.
    """
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    질문 문장 하나를 임베딩 벡터로 변환합니다.

    검색 시 사용자의 질문을 벡터로 변환하여
    DB에 저장된 문서 벡터들과 유사도를 비교합니다.
    """
    model = get_model()
    return model.encode(query, convert_to_numpy=True).tolist()
