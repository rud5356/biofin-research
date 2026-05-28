"""
검색된 문서 컨텍스트를 바탕으로 Ollama LLM이 답변을 생성하는 모듈.

RAG 파이프라인에서 "생성(Generation)" 단계를 담당합니다:
  관련 청크 컨텍스트 + 사용자 질문 → LLM → 근거 기반 답변

Ollama는 LLM을 로컬 PC에서 실행할 수 있게 해주는 도구입니다.
별도 설치 및 모델 다운로드가 필요합니다 (ollama.com 참조).
"""

try:
    import ollama
except ModuleNotFoundError as exc:
    # ollama가 없어도 import 자체는 성공하도록 처리
    ollama = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from config import LLM_MODEL, LLM_TEMPERATURE

# LLM에게 역할과 행동 지침을 설명하는 시스템 프롬프트
# 과학 논문 기반 답변만 하도록 제한하여 환각(hallucination)을 줄입니다.
SYSTEM_PROMPT = """You are a biodiversity research assistant.
Answer the user's question using ONLY the provided context from scientific abstracts.
- Cite the PMID when referencing a specific paper (e.g., "According to PMID 12345678, ...").
- If the context does not contain enough information, say so clearly.
- Be concise and accurate. Do not add information not present in the context.
"""


def generate(question: str, context: str) -> str:
    """
    검색된 논문 컨텍스트를 바탕으로 LLM이 질문에 답변하도록 합니다.

    Args:
        question: 사용자의 질문
        context: retriever가 찾아온 관련 논문 청크들 (format_context()로 포맷된 문자열)

    Returns:
        LLM이 생성한 답변 문자열
    """
    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "ollama 파이썬 패키지가 설치되지 않았습니다. "
            "`conda env create -f biodiversity_rag/environment.yml`로 환경을 만든 뒤 다시 실행하세요."
        ) from _IMPORT_ERROR

    # LLM에게 전달할 사용자 메시지: 컨텍스트와 질문을 함께 제공
    user_message = f"""Context:
{context}

Question: {question}

Answer:"""

    # Ollama API 호출: 시스템 프롬프트 + 사용자 메시지로 대화 형식 전송
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        # temperature가 낮을수록 일관된 답변, 높을수록 다양한 답변 생성
        options={"temperature": LLM_TEMPERATURE},
    )
    return response["message"]["content"]
