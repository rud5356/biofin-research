"""
NER(개체명 인식) 결과를 분석하고 통계를 출력하는 모듈.

NER 파이프라인이 생성한 결과 CSV를 읽어
추출된 종명(SPECIES), 지역명(LOCATION) 등의 개체명 통계를 요약합니다.
"""

import json

import pandas as pd


def extract_by_type(df: pd.DataFrame, entity_type: str) -> list[str]:
    """
    결과 데이터프레임에서 특정 유형의 개체명만 추출하여 리스트로 반환합니다.

    각 행의 'entities' 열에는 JSON 문자열로 개체명 목록이 저장되어 있습니다.
    예시: '[{"type": "SPECIES", "text": "Amur leopard"}, {"type": "LOCATION", "text": "Korea"}]'

    Args:
        df: NER 결과 데이터프레임 (entities 열 포함)
        entity_type: 추출할 개체명 유형 (예: "SPECIES", "LOCATION", "HABITAT")

    Returns:
        해당 유형의 개체명 텍스트 목록
    """
    all_entities: list[str] = []
    for entities_json in df["entities"]:
        try:
            # JSON 문자열을 파이썬 리스트로 파싱
            entities = json.loads(entities_json)
        except (TypeError, json.JSONDecodeError):
            # JSON 파싱 실패 (빈 값이나 형식 오류) 시 해당 행을 건너뜀
            continue

        for entity in entities:
            if entity.get("type") == entity_type:
                all_entities.append(entity.get("text", ""))

    return all_entities


def summarize_results(df: pd.DataFrame) -> None:
    """
    NER 결과 전체의 요약 통계를 출력합니다.

    출력 항목:
    - 추출된 종명(SPECIES) 수 및 상위 10개
    - 추출된 지역명(LOCATION) 수
    - JSON 파싱 실패율 (parse_error 비율)
    - 평균 처리 시간 (초)
    """
    if df.empty:
        print("결과가 없습니다.")
        return

    species_list = extract_by_type(df, "SPECIES")
    location_list = extract_by_type(df, "LOCATION")

    print(f"\n추출된 종명(SPECIES): {len(species_list)}개")
    print(f"추출된 지역명(LOCATION): {len(location_list)}개")

    if species_list:
        print("종명 상위 10개:")
        # value_counts(): 각 값의 등장 횟수를 세어 많은 순으로 정렬
        print(pd.Series(species_list).value_counts().head(10))

    # parse_error가 True인 비율: JSON 파싱에 실패한 행의 비율
    print(f"\nJSON 파싱 실패율: {df['parse_error'].mean():.1%}")
    print(f"평균 처리 시간: {df['elapsed_sec'].mean():.1f}초")
