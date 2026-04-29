"""
Abstract 텍스트를 청크로 분할한다.

전략:
- Abstract가 짧으면 (CHUNK_SIZE 이하) 전체를 하나의 청크로 유지
- 길면 문장 단위로 분리 후 CHUNK_SIZE 기준으로 합치되, CHUNK_OVERLAP만큼 앞 내용을 겹침
"""
import re
from dataclasses import dataclass

import pandas as pd

from config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    chunk_id: str       # "{pmid}_{index}"
    pmid: str
    title: str
    text: str
    chunk_index: int
    total_chunks: int


def _split_sentences(text: str) -> list[str]:
    """마침표·느낌표·물음표 기준으로 문장 분리."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _merge_sentences(sentences: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """문장 목록을 max_chars 이하의 청크로 합친다."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)

        if current_len + sent_len + 1 > max_chars and current:
            chunks.append(" ".join(current))
            # overlap: 마지막 문장들을 다음 청크의 시작으로 재사용
            overlap: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) + 1 <= overlap_chars:
                    overlap.insert(0, s)
                    overlap_len += len(s) + 1
                else:
                    break
            current = overlap
            current_len = overlap_len

        current.append(sent)
        current_len += sent_len + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """문장 경계가 없을 때 문자 수 기준으로 강제 분할한다."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    step = max(max_chars - overlap_chars, 1)
    chunks: list[str] = []

    for start in range(0, len(text), step):
        chunk = text[start:start + max_chars].strip()
        if chunk:
            chunks.append(chunk)
        if start + max_chars >= len(text):
            break

    return chunks


def chunk_abstract(pmid: str, title: str, abstract: str) -> list[Chunk]:
    """하나의 abstract를 Chunk 리스트로 변환한다."""
    # HTML 태그 제거
    clean = re.sub(r"<[^>]+>", "", abstract).strip()

    if len(clean) <= CHUNK_SIZE:
        return [Chunk(
            chunk_id=f"{pmid}_0",
            pmid=pmid,
            title=title,
            text=clean,
            chunk_index=0,
            total_chunks=1,
        )]

    sentences = _split_sentences(clean)
    if not sentences:
        texts = _split_long_text(clean, CHUNK_SIZE, CHUNK_OVERLAP)
    elif any(len(sentence) > CHUNK_SIZE for sentence in sentences):
        texts = _split_long_text(clean, CHUNK_SIZE, CHUNK_OVERLAP)
    else:
        texts = _merge_sentences(sentences, CHUNK_SIZE, CHUNK_OVERLAP)

    return [
        Chunk(
            chunk_id=f"{pmid}_{i}",
            pmid=pmid,
            title=title,
            text=t,
            chunk_index=i,
            total_chunks=len(texts),
        )
        for i, t in enumerate(texts)
    ]


def chunk_dataframe(df) -> list[Chunk]:
    """abstracts DataFrame 전체를 청킹한다."""
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
