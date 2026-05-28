"""
NER 결과에서 추출된 LOCATION 엔티티를 인터랙티브 지도에 시각화하는 모듈.

동작 순서:
  1. NER 결과 CSV에서 LOCATION 엔티티 추출 및 출현 빈도 계산
  2. Nominatim(OpenStreetMap)으로 지명 → 위경도 변환 (결과 캐시)
  3. folium으로 인터랙티브 HTML 지도 생성

실행 방법:
    python visualize_locations.py
    python visualize_locations.py --results-file results/ner_results_llama3.1-8b_few_shot.csv
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import folium
import pandas as pd
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

from config import RESULTS_DIR


# Nominatim 사용 정책: 초당 1건 이하로 요청
_GEOCODE_DELAY   = 1.1   # 요청 간 대기 시간(초)
_GEOCODE_TIMEOUT = 10    # 응답 대기 제한(초)


def extract_locations(df: pd.DataFrame) -> tuple[Counter, dict[str, list[str]]]:
    """
    NER 결과 DataFrame에서 LOCATION 엔티티를 추출합니다.

    Args:
        df: 'entities' 컬럼에 JSON 문자열이 있는 NER 결과 DataFrame

    Returns:
        (Counter: 지명 → 출현 횟수, dict: 지명 → 관련 논문 제목 목록)
    """
    counter: Counter = Counter()
    # 지명 → 해당 지명이 등장한 논문 제목 목록
    paper_map: dict[str, list[str]] = defaultdict(list)

    for _, row in df.iterrows():
        try:
            # entities 컬럼은 JSON 문자열로 저장되어 있음
            entities = json.loads(row["entities"])
        except (TypeError, json.JSONDecodeError):
            continue

        for entity in entities:
            if entity.get("type") == "LOCATION":
                name = entity.get("text", "").strip()
                if name:
                    counter[name] += 1
                    # 관련 논문 제목 수집 (중복 방지, 최대 60자 표시)
                    title = str(row.get("title", ""))[:60]
                    if title not in paper_map[name]:
                        paper_map[name].append(title)

    return counter, paper_map


def geocode_locations(
    location_counter: Counter,
    cache_path: Path,
) -> dict[str, tuple[float, float]]:
    """
    지명을 위경도 좌표로 변환합니다. 결과는 JSON 파일에 캐싱하여 중복 요청을 방지합니다.

    Nominatim API: OpenStreetMap 기반 무료 지오코딩 서비스.
    캐시 파일이 있으면 저장된 좌표를 재사용합니다.

    Args:
        location_counter: 지명 → 출현 횟수 Counter
        cache_path      : 캐시 JSON 파일 경로

    Returns:
        dict: 지명 → (위도, 경도) 튜플
    """
    coords: dict[str, tuple[float, float]] = {}

    # 기존 캐시 로드
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        coords = {k: tuple(v) for k, v in cached.items()}  # type: ignore

    # 캐시에 없는 지명만 새로 지오코딩
    geolocator = Nominatim(user_agent="llm_ner_biodiversity")
    to_geocode  = [loc for loc in location_counter if loc not in coords]

    if to_geocode:
        print(f"지오코딩 중: {len(to_geocode)}개 지명 (캐시: {len(coords)}개)")
        for loc in to_geocode:
            try:
                result = geolocator.geocode(loc, timeout=_GEOCODE_TIMEOUT)
                if result:
                    coords[loc] = (result.latitude, result.longitude)
                    print(f"  ✓ {loc} → ({result.latitude:.3f}, {result.longitude:.3f})")
                else:
                    print(f"  ✗ {loc} → 지명 없음")
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                print(f"  ✗ {loc} → 오류: {e}")
            # Nominatim 정책 준수: 초당 1건 이하
            time.sleep(_GEOCODE_DELAY)

        # 새 좌표를 캐시 파일에 저장
        cache_path.write_text(
            json.dumps(coords, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        print(f"모든 지명이 캐시에 있습니다 ({len(coords)}개)")

    return coords


def build_map(
    location_counter: Counter,
    coords: dict[str, tuple[float, float]],
    paper_map: dict[str, list[str]],
) -> folium.Map:
    """
    folium 인터랙티브 지도를 생성하고 지명 마커를 추가합니다.

    마커 크기: 출현 횟수에 비례 (최소 6px, 최대 18px)
    MarkerCluster: 마커가 많을 때 자동으로 클러스터링하여 가독성 향상

    Args:
        location_counter: 지명 → 출현 횟수
        coords          : 지명 → (위도, 경도)
        paper_map       : 지명 → 관련 논문 제목 목록

    Returns:
        folium.Map 객체
    """
    # 지도 중심: 좌표가 있는 지명들의 위경도 평균
    valid = [(lat, lon) for loc, (lat, lon) in coords.items() if loc in location_counter]
    if valid:
        center_lat = sum(lat for lat, _ in valid) / len(valid)
        center_lon = sum(lon for _, lon in valid) / len(valid)
    else:
        center_lat, center_lon = 36.0, 127.5  # 기본값: 한반도 중심

    # CartoDB positron: 배경이 깔끔한 무채색 지도 타일
    m       = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="CartoDB positron")
    cluster = MarkerCluster().add_to(m)

    max_count = max(location_counter.values()) if location_counter else 1

    for loc, count in location_counter.most_common():
        if loc not in coords:
            continue

        lat, lon = coords[loc]

        # 출현 횟수에 비례한 마커 반지름 (최소 6, 최대 18)
        radius = 6 + int(12 * count / max_count)

        # 팝업에 관련 논문 최대 3편 표시
        papers_html = "".join(
            f"<li style='font-size:11px'>{t}...</li>"
            for t in paper_map[loc][:3]
        )
        popup_html = f"""
        <b>{loc}</b><br>
        출현 횟수: <b>{count}</b>회<br>
        <hr style='margin:4px 0'>
        관련 논문:<ul style='padding-left:14px;margin:4px 0'>{papers_html}</ul>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color="#2c7bb6",
            fill=True,
            fill_color="#2c7bb6",
            fill_opacity=0.6,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{loc} ({count}회)",
        ).add_to(cluster)

    # 지도 좌측 하단에 범례 추가 (인라인 HTML)
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:10px 14px;border-radius:6px;border:1px solid #ccc;font-size:12px">
        <b>지명 출현 빈도</b><br>
        <span style="color:#2c7bb6">●</span> 원 크기 = 출현 횟수<br>
        <span style="color:#888">클릭하면 상세 정보</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description="NER 결과의 LOCATION 엔티티를 지도에 시각화합니다.")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=RESULTS_DIR / "ner_results_llama3.1-8b_few_shot.csv",
        help="NER 결과 CSV 파일 경로",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "location_map.html",
        help="출력 HTML 지도 파일 경로",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=RESULTS_DIR / "geocode_cache.json",
        help="지오코딩 캐시 파일 경로 (재실행 시 API 중복 호출 방지)",
    )
    return parser.parse_args()


def main() -> None:
    """지도 시각화 파이프라인 실행: 추출 → 지오코딩 → 지도 저장."""
    args = parse_args()

    if not args.results_file.exists():
        print(f"ERROR: 결과 파일을 찾을 수 없습니다: {args.results_file}")
        return

    print(f"결과 파일 로드: {args.results_file}")
    df = pd.read_csv(args.results_file, encoding="utf-8-sig")

    print("LOCATION 엔티티 추출 중...")
    location_counter, paper_map = extract_locations(df)
    print(f"  고유 지명: {len(location_counter)}개, 총 출현: {sum(location_counter.values())}회")
    print(f"  상위 10개: {location_counter.most_common(10)}")

    coords = geocode_locations(location_counter, args.cache)

    matched = sum(1 for loc in location_counter if loc in coords)
    print(f"\n지도에 표시할 지명: {matched}/{len(location_counter)}개")

    m = build_map(location_counter, coords, paper_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(args.output))
    print(f"\n지도 저장 완료: {args.output}")


if __name__ == "__main__":
    main()
