"""
NER 결과에서 추출된 LOCATION 엔티티를 지도에 시각화한다.
- Nominatim(OpenStreetMap)으로 지명 → 위경도 변환
- folium으로 인터랙티브 HTML 지도 생성
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


# Nominatim 사용 제한: 초당 1건
_GEOCODE_DELAY = 1.1
_GEOCODE_TIMEOUT = 10


def extract_locations(df: pd.DataFrame) -> tuple[Counter, dict[str, list[str]]]:
    """NER 결과 DataFrame에서 LOCATION 빈도와 논문 매핑을 반환한다."""
    counter: Counter = Counter()
    paper_map: dict[str, list[str]] = defaultdict(list)

    for _, row in df.iterrows():
        try:
            entities = json.loads(row["entities"])
        except (TypeError, json.JSONDecodeError):
            continue

        for entity in entities:
            if entity.get("type") == "LOCATION":
                name = entity.get("text", "").strip()
                if name:
                    counter[name] += 1
                    title = str(row.get("title", ""))[:60]
                    if title not in paper_map[name]:
                        paper_map[name].append(title)

    return counter, paper_map


def geocode_locations(
    location_counter: Counter,
    cache_path: Path,
) -> dict[str, tuple[float, float]]:
    """지명을 위경도로 변환한다. 결과는 JSON 파일에 캐싱한다."""
    # 캐시 로드
    coords: dict[str, tuple[float, float]] = {}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        coords = {k: tuple(v) for k, v in cached.items()}  # type: ignore

    geolocator = Nominatim(user_agent="llm_ner_biodiversity")
    to_geocode = [loc for loc in location_counter if loc not in coords]

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
            time.sleep(_GEOCODE_DELAY)

        # 캐시 저장
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
    """folium 지도를 생성하고 마커를 추가한다."""
    # 지도 중심: 좌표가 있는 지명들의 평균
    valid = [(lat, lon) for loc, (lat, lon) in coords.items() if loc in location_counter]
    if valid:
        center_lat = sum(lat for lat, _ in valid) / len(valid)
        center_lon = sum(lon for _, lon in valid) / len(valid)
    else:
        center_lat, center_lon = 36.0, 127.5  # 한반도 중심

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="CartoDB positron")
    cluster = MarkerCluster().add_to(m)

    max_count = max(location_counter.values()) if location_counter else 1

    for loc, count in location_counter.most_common():
        if loc not in coords:
            continue

        lat, lon = coords[loc]

        # 빈도에 따라 마커 크기 조절 (최소 6, 최대 18)
        radius = 6 + int(12 * count / max_count)

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

    # 범례 추가
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
    parser = argparse.ArgumentParser(description="NER 결과의 LOCATION을 지도에 시각화한다.")
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
        help="지오코딩 캐시 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
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
