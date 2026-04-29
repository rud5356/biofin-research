"""
검색된 컨텍스트를 바탕으로 ollama LLM이 답변을 생성한다.
"""
try:
    import ollama
except ModuleNotFoundError as exc:
    ollama = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from config import LLM_MODEL, LLM_TEMPERATURE

SYSTEM_PROMPT = """You are a biodiversity research assistant.
Answer the user's question using ONLY the provided context from scientific abstracts.
- Cite the PMID when referencing a specific paper (e.g., "According to PMID 12345678, ...").
- If the context does not contain enough information, say so clearly.
- Be concise and accurate. Do not add information not present in the context.
"""


def generate(question: str, context: str) -> str:
    """컨텍스트와 질문을 받아 LLM 답변을 반환한다."""
    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "ollama 파이썬 패키지가 설치되지 않았습니다. "
            "`conda env create -f biodiversity_rag/environment.yml`로 환경을 만든 뒤 다시 실행하세요."
        ) from _IMPORT_ERROR

    user_message = f"""Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        options={"temperature": LLM_TEMPERATURE},
    )
    return response["message"]["content"]
