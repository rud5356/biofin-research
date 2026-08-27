from __future__ import annotations

import argparse
import csv
import re
import shutil
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


YEARS = range(2019, 2024)
MINISTRIES = {
    "환경부",
    "농림축산식품부",
    "산림청",
    "해양수산부",
    "산업통상자원부",
    "과학기술정보통신부",
    "국토교통부",
    "문화재청",
    "질병관리청",
    "식품의약품안전처",
}
IDENTITY_COLUMNS = (
    "소관명",
    "회계명",
    "분야명",
    "부문명",
    "프로그램명",
    "단위사업명",
    "세부사업명",
)

# 단독으로도 행정성 지출임이 명확한 표현만 자동 제외한다. '운영', '정보화',
# '교육'처럼 환경ㆍ연구 사업에도 나타날 수 있는 일반어는 의도적으로 제외했다.
EXCLUSION_RULES = (
    ("인건비", re.compile(r"(?:^|[ (])(?:총액|비총액)?인건비(?:$|[ )])")),
    ("기본경비", re.compile(r"기본경비")),
    ("보수", re.compile(r"^(?:기타직\s*)?보수$")),
    ("연금부담금", re.compile(r"연금부담금")),
    ("퇴직급여", re.compile(r"퇴직급여")),
    ("업무추진비", re.compile(r"업무추진비")),
    ("복리후생비", re.compile(r"복리후생(?:비)?$")),
    ("여비", re.compile(r"^(?:국내|국외)?여비$")),
)

# 종료사업에서 BIOFIN 직접 범위 표현 없이 아래 비관련 분야 표현만 확인된 경우 제외한다.
ENDED_NON_BIOFIN_PATTERN = re.compile(
    r"과학관|전시관운영|드론|G-First|IT활용|백신|반도체|디스플레이|나노제품|"
    r"과학기술 부담금|전문인력활용지원|연구개발특구|국민생활연구실증사업화|"
    r"공공기술사업화|이공계전문기술인력|지역신산업선도인력|기업지원허브|"
    r"산업전문인력|용접도장전문인력|조선업 생산기술 인력|조선업퇴직자|"
    r"시험연구인력|해운물류전문인력|Giga|기가인터넷|"
    r"인터넷서비스|인터넷진흥|네트워크|Wi-Fi|데이터요금|특수번호|휴대전화|"
    r"5G|6G|블록체인|인공지능|AI|빅데이터|로봇|우주|위성|원자력|핵융합|"
    r"양자|사이버|정보보호|보안|암호|해킹|ICT|소프트웨어|SW|콘텐츠|방송|"
    r"통신|주파수|전파|의료|의약|신약|치료제|병원|헬스케어|도로건설|"
    r"국도건설|고속도로|철도|공항|주택|건축|부동산|도시재생|자동차|조선산업|"
    r"도로|교통|산단|물류|항공|과학문화|기초원천|가속기|정보화|디지털|"
    r"연구성과|기술사업화|나노|소재|우편|기획심사평가|기획평가|위원회 운영|"
    r"인력양성|인력지원|교육훈련|마케팅|"
    r"무역|수출지원|관광산업|창업지원|벤처|청년정착|운영지원|운영비지원|운영비 지원"
)
BIOFIN_SCOPE_PATTERN = re.compile(
    r"생물|생태|유전자원|ABS|나고야|종자|품종|LMO|GMO|검역|병해충|야생|곤충|"
    r"미생물|산림|숲|임업|목재|수목원|녹색|저탄소|탄소|기후|재생에너지|신재생|"
    r"청정에너지|에너지효율|에너지자립|온실가스|지속가능|순환경제|자원순환|친환경|"
    r"유기농|농업|농촌|수산|어업|양식|해양|갯벌|연안|습지|하천|홍수|가뭄|"
    r"환경(?:오염|보전|관리|개선|기술|정책|연구|영향|기초|산업|협력|교육|조사|복원|생태|친화)|"
    r"오염|미세먼지|녹조|폐기물|"
    r"재활용|수질|대기|상수도|하수|수자원|물관리|자연|보전|복원|문화재|"
    r"문화유산|궁궐|왕릉|국립공원|지질공원|정원|공원|사방|"
    r"생물보호|야생동물보호|자연보호|보호구역|문화재보호|품종보호|개발제한구역"
)
ALWAYS_NON_BIOFIN_PATTERN = re.compile(
    r"ICT융합Industry|공공연구성과 기반 BIG 선도모델|국민생활안전긴급대응연구|"
    r"국제IT협력|예수(?:금)?이자상환|차입금이자상환|예수원금.*상환|기금간예탁|"
    r"비통화금융기관예치|국공채매입|여유자금운용|예탁|예치|전출|전입|"
    r"기금간거래|회계간거래|차입금원금상환|국민체감형 자율주행서비스|"
    r"긴급구조용지능형정밀측위|디지털.?트윈\s*기반.*화재.?재난|스마트빌리지|"
    r"인터넷 이용환경 고도화|자율주행솔루션|지능정보사회 이용자보호|"
    r"차세대인터넷비즈니스|차세대초소형IoT|기타경비|농지은행인적역량강화|"
    r"농지이용관리지원|농지제도개선홍보|사업관리비|스마트 농정 통계체계|"
    r"주력산업정책수립운영|통상협력정책지원|사회복무제도지원"
)


@dataclass
class YearRecord:
    year: int
    identity: tuple[str, ...]
    rows: list[dict[str, str]]


def clean(value: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or ""))).strip()


def identity(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(clean(row.get(column)) for column in IDENTITY_COLUMNS)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def consecutive_segments(records: list[YearRecord]) -> list[list[YearRecord]]:
    records = sorted(records, key=lambda record: record.year)
    if not records:
        return []
    segments: list[list[YearRecord]] = [[records[0]]]
    for record in records[1:]:
        if record.year == segments[-1][-1].year + 1:
            segments[-1].append(record)
        else:
            segments.append([record])
    return segments


def parse_budget(value: str) -> int | None:
    text = clean(value).replace(",", "")
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def safe_filename(value: str, limit: int = 90) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", clean(value)).rstrip(". ")
    return (value or "이름없음")[:limit]


def exclusion_reason(segment: list[YearRecord]) -> tuple[str, str, str] | None:
    # 세부사업명을 우선 판정하고, 단위사업명만 보조적으로 확인한다.
    # 프로그램명에 행정성 표현이 있어도 실제 세부사업은 관련 사업일 수 있으므로 제외한다.
    values = (
        ("세부사업명", segment[0].identity[6]),
        ("단위사업명", segment[0].identity[5]),
    )
    for column, value in values:
        for rule_name, pattern in EXCLUSION_RULES:
            match = pattern.search(value)
            if match:
                return rule_name, column, match.group(0)
    return None


def ended_criteria_exclusion(segment: list[YearRecord]) -> str | None:
    text = " ".join(segment[0].identity[4:7])
    always_match = ALWAYS_NON_BIOFIN_PATTERN.search(text)
    if always_match:
        return always_match.group(0)
    negative_match = ENDED_NON_BIOFIN_PATTERN.search(text)
    if negative_match and not BIOFIN_SCOPE_PATTERN.search(text):
        return negative_match.group(0)
    if not BIOFIN_SCOPE_PATTERN.search(text):
        return "BIOFIN 1~9 직접 범위 표현 없음"
    return None


def similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    # 같은 부처 안에서 세부사업명을 가장 강하게 보고 상위 사업명으로 보완한다.
    if left[0] != right[0]:
        return 0.0
    ratios = [
        SequenceMatcher(None, left[6], right[6]).ratio(),
        SequenceMatcher(None, left[5], right[5]).ratio(),
        SequenceMatcher(None, left[4], right[4]).ratio(),
    ]
    return 0.60 * ratios[0] + 0.25 * ratios[1] + 0.15 * ratios[2]


def document_source(outputs_dir: Path, row: dict[str, str]) -> Path | None:
    relative = clean(row.get("사업설명자료_상대경로"))
    if not relative:
        return None
    candidate = outputs_dir / clean(row.get("회계연도")) / Path(relative.replace("/", "\\"))
    return candidate if candidate.is_file() else None


def copy_documents(
    segment: list[YearRecord], outputs_dir: Path, destination: Path
) -> tuple[dict[int, list[str]], list[int]]:
    copied: dict[int, list[str]] = defaultdict(list)
    missing: list[int] = []
    destination.mkdir(parents=True, exist_ok=True)
    for record in segment:
        found = False
        for row in record.rows:
            source = document_source(outputs_dir, row)
            if source is None:
                continue
            found = True
            key = safe_filename(row.get("business_key", ""), 50)
            detail = safe_filename(row.get("세부사업명", ""))
            ministry = safe_filename(row.get("소관명", ""), 30)
            target = destination / f"{record.year}_{ministry}_{detail}_{key}{source.suffix.lower()}"
            counter = 2
            while target.exists() and target.resolve() != source.resolve():
                target = destination / f"{record.year}_{ministry}_{detail}_{key}_{counter}{source.suffix.lower()}"
                counter += 1
            if not target.exists():
                shutil.copy2(source, target)
            copied[record.year].append(target.name)
        if not found:
            missing.append(record.year)
    return copied, missing


def budget_for(record: YearRecord) -> str:
    values = [parse_budget(row.get("예산액", "")) for row in record.rows]
    numbers = [value for value in values if value is not None]
    return str(sum(numbers)) if numbers else ""


def base_output_row(segment: list[YearRecord]) -> dict[str, object]:
    first = segment[0]
    result: dict[str, object] = dict(zip(IDENTITY_COLUMNS, first.identity))
    years = [record.year for record in segment]
    result.update(
        {
            "존재연도": ", ".join(map(str, years)),
            "연속시작연도": years[0],
            "연속종료연도": years[-1],
            "연속기간": len(years),
        }
    )
    by_year = {record.year: record for record in segment}
    for year in YEARS:
        result[f"예산액_{year}"] = budget_for(by_year[year]) if year in by_year else ""
        result[f"business_key_{year}"] = (
            "; ".join(clean(row.get("business_key")) for row in by_year[year].rows)
            if year in by_year
            else ""
        )
    return result


def add_document_fields(
    output_row: dict[str, object],
    document_folder: str,
    copied: dict[int, list[str]],
    missing: list[int],
) -> None:
    for year in YEARS:
        output_row[f"사업설명자료_{year}"] = "; ".join(
            f"{document_folder}/{name}" for name in copied.get(year, [])
        )
    output_row["설명자료_수"] = sum(len(items) for items in copied.values())
    output_row["설명자료_누락연도"] = ", ".join(map(str, missing))


def output_columns(extra: Iterable[str]) -> list[str]:
    return [
        *IDENTITY_COLUMNS,
        "존재연도",
        "연속시작연도",
        "연속종료연도",
        "연속기간",
        *(f"예산액_{year}" for year in YEARS),
        *(f"business_key_{year}" for year in YEARS),
        *extra,
        *(f"사업설명자료_{year}" for year in YEARS),
        "설명자료_수",
        "설명자료_누락연도",
    ]


def build(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    outputs_dir = Path(args.outputs_dir).resolve() if args.outputs_dir else script_dir / "outputs"
    label_path = (
        Path(args.label_csv).resolve()
        if args.label_csv
        else script_dir.parents[1] / "budget_biodiv_cls3" / "document" / "2023biofin_label.csv"
    )
    result_dir = Path(args.result_dir).resolve() if args.result_dir else script_dir / "biofin_continuity_outputs"

    # 재실행 시 이 스크립트가 직전에 만든 문서 사본만 정리한다. 원본 outputs는 건드리지 않는다.
    for generated_dir in ("continuing_documents", "ended_documents"):
        target = result_dir / generated_dir
        if target.is_dir():
            shutil.rmtree(target)
    # 이전 버전에서 만든 좁은 범위의 제외 목록은 전체 감사 파일로 대체한다.
    old_exclusion_file = result_dir / "excluded_obviously_non_biofin.csv"
    if old_exclusion_file.is_file():
        old_exclusion_file.unlink()

    grouped: dict[tuple[str, ...], dict[int, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for year in YEARS:
        input_path = outputs_dir / str(year) / f"open_fiscal_{year}.csv"
        for row in read_csv(input_path):
            if clean(row.get("소관명")) in MINISTRIES:
                grouped[identity(row)][year].append(row)

    label_rows = [
        row
        for row in read_csv(label_path)
        if clean(row.get("소관명")) in MINISTRIES and clean(row.get("BIOFIN분류")) == "1"
    ]
    positive_2023: dict[tuple[str, ...], dict[str, str]] = {
        identity(row): row for row in label_rows
    }
    all_2023_labels: dict[tuple[str, ...], dict[str, str]] = {
        identity(row): row
        for row in read_csv(label_path)
        if clean(row.get("소관명")) in MINISTRIES
    }

    all_segments: list[list[YearRecord]] = []
    for key, yearly_rows in grouped.items():
        records = [YearRecord(year, key, rows) for year, rows in yearly_rows.items()]
        all_segments.extend(consecutive_segments(records))

    continuing: list[dict[str, object]] = []
    ended: list[dict[str, object]] = []
    excluded_all: list[dict[str, object]] = []
    ambiguous: list[dict[str, object]] = []

    for segment in all_segments:
        key = segment[0].identity
        ends_in_2023 = segment[-1].year == 2023

        # 2023년에 처음 생긴 사업은 2019~2022 사업의 계속 여부 비교 대상이 아니다.
        if ends_in_2023 and segment[0].year < 2023 and key in positive_2023:
            row = base_output_row(segment)
            row.update(
                {
                    "2023매칭방식": "정확일치",
                    "2023BIOFIN분류": "1",
                    "2023BIOFIN_1차_카테고리": clean(
                        positive_2023[key].get("BIOFIN 1차 카테고리")
                    ),
                    "2023하위_카테고리": clean(positive_2023[key].get("하위 카테고리")),
                }
            )
            copied, missing = copy_documents(
                segment, outputs_dir, result_dir / "continuing_documents"
            )
            add_document_fields(row, "continuing_documents", copied, missing)
            continuing.append(row)
            continue

        if ends_in_2023:
            row = base_output_row(segment)
            if segment[0].year == 2023:
                reason_text = "2023년 신규사업: 2019~2022 계속·종료 비교대상 아님"
            else:
                reason_text = "2023년까지 연속되었으나 2023 BIOFIN분류가 1이 아님"
            row.update(
                {
                    "제외사유": reason_text,
                    "제외규칙": "",
                    "일치컬럼": "",
                    "일치표현": "",
                    "2023BIOFIN분류": clean(
                        all_2023_labels.get(key, {}).get("BIOFIN분류")
                    ),
                }
            )
            excluded_all.append(row)
            continue

        reason = exclusion_reason(segment)
        if reason:
            row = base_output_row(segment)
            row.update(
                {
                    "제외사유": "2023년 이전 종료사업 중 명백한 비관련 사업",
                    "제외규칙": reason[0],
                    "일치컬럼": reason[1],
                    "일치표현": reason[2],
                    "2023BIOFIN분류": "",
                }
            )
            excluded_all.append(row)
            continue

        criteria_match = ended_criteria_exclusion(segment)
        if criteria_match:
            row = base_output_row(segment)
            row.update(
                {
                    "제외사유": "BIOFIN 직접 범위 표현 없이 비관련 분야 표현이 확인된 종료사업",
                    "제외규칙": "BIOFIN 1~9 범위 검토",
                    "일치컬럼": "프로그램명·단위사업명·세부사업명",
                    "일치표현": criteria_match,
                    "2023BIOFIN분류": "",
                }
            )
            excluded_all.append(row)
            continue

        candidates = sorted(
            ((similarity(key, candidate), candidate) for candidate in positive_2023),
            reverse=True,
        )
        best_score, best_key = candidates[0] if candidates else (0.0, tuple())
        second_score = candidates[1][0] if len(candidates) > 1 else 0.0
        row = base_output_row(segment)
        row.update(
            {
                "BIOFIN분류": "",  # 종료사업에는 새 레이블을 달지 않는다.
                "2023유사사업_소관명": best_key[0],
                "2023유사사업_프로그램명": best_key[4],
                "2023유사사업_단위사업명": best_key[5],
                "2023유사사업_세부사업명": best_key[6],
                "2023유사도": f"{best_score:.4f}",
                "2023차순위유사도": f"{second_score:.4f}",
                "2023유사매칭기준통과": (
                    "예" if best_score >= args.similarity_threshold else "아니오"
                ),
                "검토상태": "미검토",
            }
        )
        if (
            best_score >= args.similarity_threshold
            and best_score - second_score < args.ambiguity_margin
        ):
            ambiguous.append(dict(row))
        copied, missing = copy_documents(segment, outputs_dir, result_dir / "ended_documents")
        add_document_fields(row, "ended_documents", copied, missing)
        ended.append(row)

    continuing.sort(key=lambda row: (row["소관명"], row["세부사업명"], row["연속시작연도"]))
    ended.sort(key=lambda row: (row["소관명"], row["세부사업명"], row["연속시작연도"]))
    excluded_all.sort(
        key=lambda row: (row["제외사유"], row["소관명"], row["세부사업명"], row["연속시작연도"])
    )

    continuing_extra = ["2023매칭방식", "2023BIOFIN분류", "2023BIOFIN_1차_카테고리", "2023하위_카테고리"]
    ended_extra = [
        "BIOFIN분류",
        "2023유사사업_소관명",
        "2023유사사업_프로그램명",
        "2023유사사업_단위사업명",
        "2023유사사업_세부사업명",
        "2023유사도",
        "2023차순위유사도",
        "2023유사매칭기준통과",
        "검토상태",
    ]
    write_csv(
        result_dir / "biofin_continuing_through_2023.csv",
        continuing,
        output_columns(continuing_extra),
    )
    write_csv(
        result_dir / "biofin_ended_before_2023.csv", ended, output_columns(ended_extra)
    )
    write_csv(
        result_dir / "excluded_from_biofin_outputs.csv",
        excluded_all,
        output_columns(["제외사유", "제외규칙", "일치컬럼", "일치표현", "2023BIOFIN분류"]),
    )
    write_csv(
        result_dir / "ambiguous_matches.csv", ambiguous, output_columns(ended_extra)
    )
    write_csv(
        result_dir / "exclusion_rules.csv",
        [
            {
                "규칙명": name,
                "적용컬럼": "세부사업명 우선, 단위사업명 보조",
                "정규식": pattern.pattern,
            }
            for name, pattern in EXCLUSION_RULES
        ],
        ["규칙명", "적용컬럼", "정규식"],
    )

    print(f"입력 폴더: {outputs_dir}")
    print(f"2023 레이블: {label_path}")
    print(f"결과 폴더: {result_dir}")
    print(f"2023년까지 연속된 BIOFIN 사업: {len(continuing):,}개 구간")
    print(f"2023년 이전 종료 BIOFIN 후보: {len(ended):,}개 구간")
    print(f"최종 결과에서 제외되어 감사 파일에 기록: {len(excluded_all):,}개 구간")
    print(f"유사매칭 검토 필요: {len(ambiguous):,}개 구간")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2019~2023 열린재정 사업을 연속 구간으로 나누고 BIOFIN 결과와 설명자료를 모읍니다."
    )
    parser.add_argument("--outputs-dir", help="open_fiscal/outputs 폴더")
    parser.add_argument("--label-csv", help="BIOFIN분류 컬럼이 있는 2023 레이블 CSV")
    parser.add_argument("--result-dir", help="결과를 저장할 폴더")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.72,
        help="종료사업의 2023 유사매칭 참고 기준 (포함 여부에는 영향 없음, 기본값: 0.72)",
    )
    parser.add_argument(
        "--ambiguity-margin",
        type=float,
        default=0.05,
        help="1ㆍ2순위 점수 차가 이 값 미만이면 ambiguous_matches.csv에 기록 (기본값: 0.05)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
