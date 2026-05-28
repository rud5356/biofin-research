"""
LLM을 이용한 NER(개체명 인식) 배치 처리 파이프라인 모듈.

주요 기능:
  - run_ner()   : 논문 초록 하나에서 개체명 추출 (LLM 호출 + JSON 파싱)
  - run_batch() : 여러 초록을 순서대로 처리하고 체크포인트 저장

NER 결과 형식:
  {"entities": [{"type": "SPECIES"|"LOCATION"|"DATE", "text": "..."}]}
"""

import json
import re
import time
from pathlib import Path
from typing import Any

import ollama
import pandas as pd

from prompting import build_prompt


def ensure_directory(path: Path) -> Path:
    """
    경로가 없으면 생성하고, 해당 경로를 반환합니다.

    parents=True  : 중간 디렉토리도 함께 생성
    exist_ok=True : 이미 존재해도 오류 없이 통과
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_output(raw: str) -> dict[str, Any]:
    """
    LLM 응답 문자열에서 JSON을 파싱합니다.

    LLM이 JSON 주변에 설명 텍스트를 추가할 수 있으므로 두 단계로 시도합니다:
    1단계: 전체 문자열을 그대로 JSON으로 파싱
    2단계: 중괄호 {} 범위를 정규식으로 찾아 JSON 파싱
    두 단계 모두 실패하면 parse_error=True 와 함께 raw 텍스트 일부를 반환합니다.
    """
    # 1단계: 직접 파싱 시도
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # 2단계: {...} 블록 추출 후 파싱 (re.DOTALL: 줄바꿈 포함하여 매칭)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # 파싱 실패: 오류 플래그와 함께 원본 텍스트 일부 보존
    return {"entities": [], "parse_error": True, "raw": raw[:200]}


def run_ner(text: str, model: str = "llama3.1:8b", mode: str = "few_shot") -> dict[str, Any]:
    """
    논문 초록 하나에 대해 LLM NER을 수행합니다.

    Args:
        text  : 개체명을 추출할 논문 초록 텍스트
        model : Ollama 로컬 모델 이름 (기본값: llama3.1:8b)
        mode  : 프롬프팅 방식 ('few_shot' 또는 'zero_shot')

    Returns:
        {"entities": [...]} 형태의 딕셔너리
        오류 시 {"entities": [], "error": "오류 메시지"}
    """
    prompt = build_prompt(text, mode=mode)

    try:
        # Ollama 로컬 LLM 호출
        # temperature=0: 매번 동일한 결과 생성 (재현성 확보)
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        raw_output = response["message"]["content"]
        return parse_output(raw_output)
    except Exception as exc:
        # 네트워크 오류, 모델 미설치 등 예외 처리
        return {"entities": [], "error": str(exc)}


def run_batch(
    abstracts: list[dict[str, Any]],
    output_dir: Path,
    model: str = "llama3.1:8b",
    mode: str = "few_shot",
    sample_size: int = 30,
    checkpoint_every: int = 10,
) -> pd.DataFrame:
    """
    여러 논문 초록을 순서대로 처리하고 결과를 CSV로 저장합니다.

    Args:
        abstracts       : [{"id": PMID, "title": ..., "abstract": ...}] 형태의 목록
        output_dir      : 결과 CSV를 저장할 폴더
        model           : 사용할 Ollama 모델 이름
        mode            : 프롬프팅 방식 ('few_shot' / 'zero_shot')
        sample_size     : 처리할 최대 초록 수 (abstracts 앞에서부터)
        checkpoint_every: N건 처리마다 중간 저장 (0이면 비활성화)

    Returns:
        처리 결과가 담긴 DataFrame
    """
    results: list[dict[str, Any]] = []
    sample = abstracts[:sample_size]   # 앞에서부터 sample_size 건만 처리
    ensure_directory(output_dir)

    for index, item in enumerate(sample, start=1):
        print(f"[{index}/{len(sample)}] Processing abstract...")

        # 처리 시간 측정
        start_time = time.time()
        ner_result = run_ner(item["abstract"], model=model, mode=mode)
        elapsed    = time.time() - start_time

        results.append({
            "id":          item["id"],
            "title":       item["title"],
            "abstract":    item["abstract"],
            # entities 리스트를 JSON 문자열로 직렬화하여 CSV 셀에 저장
            "entities":    json.dumps(ner_result.get("entities", []), ensure_ascii=False),
            "parse_error": ner_result.get("parse_error", False),
            "error":       ner_result.get("error", ""),
            "elapsed_sec": round(elapsed, 2),
        })

        # N건마다 중간 저장 (처리 중 중단되어도 진행 상황 보존)
        if checkpoint_every > 0 and index % checkpoint_every == 0:
            checkpoint_path = output_dir / f"ner_results_checkpoint_{index}.csv"
            pd.DataFrame(results).to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
            print(f"  체크포인트 저장: {checkpoint_path.name}")

    # 최종 결과 저장 (파일명에 모델명과 모드 포함)
    df = pd.DataFrame(results)
    # Windows 파일명에서 콜론(:)은 사용 불가 → 하이픈(-)으로 대체
    safe_model  = model.replace(":", "-")
    result_path = output_dir / f"ner_results_{safe_model}_{mode}.csv"
    df.to_csv(result_path, index=False, encoding="utf-8-sig")
    print(f"\n최종 결과 저장: {result_path}")
    return df
