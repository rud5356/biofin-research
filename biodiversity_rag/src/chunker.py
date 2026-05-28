"""
논문 초록(Abstract) 텍스트를 청크(Chunk)로 분할하는 모듈.

청킹(Chunking)이 필요한 이유:
  임베딩 모델에는 한 번에 처리할 수 있는 최대 토큰 수(CHUNK_SIZE)가 있습니다.
  긴 초록을 통째로 넣으면 잘리거나 성능이 저하되므로,
  의미 있는 단위(문장)로 나눠 각각 임베딩합니다.

청킹 전략:
  - 초록 길이가 CHUNK_SIZE 이하: 전체를 하나의 청크로 유지
  - 길면: 문장 단위로 분리 후 CHUNK_SIZE 기준으로 합치되,
          CHUNK_OVERLAP만큼 앞 내용을 다음 청크에 겹쳐 문맥 연속성을 보장
"""

import re
from dataclasses import dataclass

import pandas as pd

from config import CHUNK_OVERLAP, CHUNK_SIZE


@dataclass
class Chunk:
    """
    하나의 청크를 나타내는 데이터 클래스.

    @dataclass: __init__, __repr__ 등을 자동으로 생성해주는 파이썬 데코레이터
    """
    chunk_id: str       # ChromaDB에서 고유 식별자로 사용 (형식: "{pmid}_{index}")
    pmid: str           # PubMed 논문 고유 ID
    title: str          # 논문 제목
    text: str           # 청크 본문 텍스트
    chunk_index: int    # 이 논문에서 몇 번째 청크인지 (0부터 시작)
    total_chunks: int   # 이 논문의 총 청크 수


def _split_sentences(text: str) -> list[str]:
    """
    텍스트를 문장 단위로 분리합니다.

    마침표(.), 느낌표(!), 물음표(?) 뒤에 공백이 오면 문장 경계로 판단합니다.
    (?<=[.!?]): 해당 문자 뒤(lookbehind)에 위치 — 문자 자체는 포함됩니다.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _merge_sentences(sentences: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """
    문장 목록을 max_chars 이하의 청크로 합칩니다.

    청크가 가득 차면:
    1. 현재 문장 모음을 하나의 청크로 저장
    2. 마지막 몇 문장을 다음 청크 시작에 재사용 (overlap_chars 만큼)
       → 문맥이 끊기지 않도록 앞 청크와 약간 겹치게 합니다

    Args:
        sentences: 분리된 문장 목록
        max_chars: 청크 최대 문자 수
        overlap_chars: 인접 청크 간 겹치는 문자 수
    """
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # 현재 청크에 문장을 추가하면 최대 크기를 초과하는 경우
        if current_length + sentence_length + 1 > max_chars and current_sentences:
            # 현재까지의 문장들을 하나의 청크로 저장
            chunks.append(" ".join(current_sentences))

            # 오버랩: 뒤에서부터 overlap_chars 이내의 문장들을 다음 청크 시작으로 가져옴
            overlap_sentences: list[str] = []
            overlap_length = 0
            for s in reversed(current_sentences):
                if overlap_length + len(s) + 1 <= overlap_chars:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s) + 1
                else:
                    break
            current_sentences = overlap_sentences
            current_length = overlap_length

        current_sentences.append(sentence)
        current_length += sentence_length + 1  # +1은 문장 사이 공백

    # 마지막 청크 저장
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """
    문장 경계를 찾을 수 없을 때 문자 수 기준으로 강제 분할합니다.

    step: 각 청크의 시작 위치 간격 (max_chars - overlap_chars)
    예: max_chars=512, overlap_chars=64 → step=448
        0~512, 448~960, 896~1408, ...
    """
    if max_chars <= 0:
        raise ValueError("max_chars는 1 이상이어야 합니다.")

    step = max(max_chars - overlap_chars, 1)
    chunks: list[str] = []

    for start in range(0, len(text), step):
        chunk = text[start : start + max_chars].strip()
        if chunk:
            chunks.append(chunk)
        if start + max_chars >= len(text):
            break

    return chunks


def chunk_abstract(pmid: str, title: str, abstract: str) -> list[Chunk]:
    """
    하나의 논문 초록을 Chunk 리스트로 변환합니다.

    짧은 초록은 단일 청크로, 긴 초록은 문장 단위로 분할합니다.

    Args:
        pmid: PubMed 논문 ID
        title: 논문 제목
        abstract: 초록 본문

    Returns:
        Chunk 객체 리스트 (chunk_index가 0부터 순서대로 부여됨)
    """
    # HTML 태그 제거 (<i>, <b> 등이 포함된 경우 처리)
    cleaned_abstract = re.sub(r"<[^>]+>", "", abstract).strip()

    # 짧은 초록은 분할 없이 하나의 청크로 처리
    if len(cleaned_abstract) <= CHUNK_SIZE:
        return [Chunk(
            chunk_id=f"{pmid}_0",
            pmid=pmid,
            title=title,
            text=cleaned_abstract,
            chunk_index=0,
            total_chunks=1,
        )]

    # 문장 단위로 분리 시도
    sentences = _split_sentences(cleaned_abstract)

    if not sentences:
        # 문장 구분 기호가 없으면 문자 수 기준으로 강제 분할
        text_chunks = _split_long_text(cleaned_abstract, CHUNK_SIZE, CHUNK_OVERLAP)
    elif any(len(sentence) > CHUNK_SIZE for sentence in sentences):
        # 단일 문장이 최대 크기를 초과하는 경우 강제 분할
        text_chunks = _split_long_text(cleaned_abstract, CHUNK_SIZE, CHUNK_OVERLAP)
    else:
        # 일반적인 경우: 문장들을 오버랩과 함께 합침
        text_chunks = _merge_sentences(sentences, CHUNK_SIZE, CHUNK_OVERLAP)

    return [
        Chunk(
            chunk_id=f"{pmid}_{i}",
            pmid=pmid,
            title=title,
            text=chunk_text,
            chunk_index=i,
            total_chunks=len(text_chunks),
        )
        for i, chunk_text in enumerate(text_chunks)
    ]


def chunk_dataframe(df: pd.DataFrame) -> list[Chunk]:
    """
    초록 데이터프레임 전체를 청크 리스트로 변환합니다.

    빈 초록(NaN 또는 빈 문자열)은 건너뜁니다.

    Args:
        df: id, title, abstract 열이 있는 데이터프레임

    Returns:
        모든 논문의 청크를 이어 붙인 리스트
    """
    all_chunks: list[Chunk] = []
    for _, row in df.iterrows():
        raw_abstract = row["abstract"]
        if pd.isna(raw_abstract):
            continue

        abstract = str(raw_abstract).strip()
        if not abstract:
            continue

        chunks = chunk_abstract(
            pmid=str(row["id"]),
            title=str(row["title"]),
            abstract=abstract,
        )
        all_chunks.extend(chunks)

    return all_chunks
