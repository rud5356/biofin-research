"""
사업별결산세출지출현황_2024년도_파일매칭.csv 에 매칭실패여부 컬럼 추가.

- 매칭된 행(matched_file_count > 0): 매칭실패여부 = 빈칸 (건드리지 않음)
- 매칭 안 된 행(matched_file_count == 0): 열린재정_파일매칭_작업표.csv 의
  동일 No. 행에서 matched_filename / review_note 두 컬럼을 모두 확인하여
    · 사업없음       → "사업없음"
    · 사업자료없음   → "사업자료없음"
    · 동일사업추정   → "동일사업추정({Unnamed:23 의 연결번호})"
  작업표에도 없으면 빈칸 유지.

실행 방법:
    python add_match_failure_column.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# ── 파일 경로 설정 ────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
TARGET_CSV   = SCRIPT_DIR / "사업별결산세출지출현황_2024년도_파일매칭.csv"   # 매칭 결과 CSV
WORKFILE_CSV = SCRIPT_DIR / "열린재정_파일매칭_작업표.csv"                   # 수동 작업 기록표
OUTPUT_CSV   = SCRIPT_DIR / "사업별결산세출지출현황_2024년도_파일매칭.csv"   # 원본 덮어쓰기

# 작업표에서 찾을 키워드 (우선순위 순서 — 먼저 나오는 게 우선)
KEYWORDS = ["사업없음", "사업자료없음", "동일사업추정"]


def detect_keyword(fname: str, rnote: str) -> str:
    """
    matched_filename, review_note 두 컬럼 중 어디에 있든 키워드를 반환합니다.

    우선순위: 사업없음 > 사업자료없음 > 동일사업추정
    키워드가 없으면 빈 문자열 반환.
    """
    # 두 컬럼을 합쳐서 한 번에 검색 (어느 컬럼에 있든 탐지 가능)
    combined = f"{fname} {rnote}"
    for kw in KEYWORDS:
        if kw in combined:
            return kw
    return ""


def build_workfile_index(wf: pd.DataFrame) -> dict[int, dict]:
    """
    작업표(열린재정_파일매칭_작업표.csv)를 읽어 No. → {keyword, linked_no} 형태의 인덱스를 반환합니다.

    keyword : '사업없음' | '사업자료없음' | '동일사업추정'
    linked_no: 동일사업추정인 경우 연결 번호 (Unnamed:23 컬럼에서 추출)

    키워드가 없는 행은 인덱스에 포함하지 않습니다.
    """
    index: dict[int, dict] = {}
    for _, row in wf.iterrows():
        # No. 컬럼을 정수로 변환 (변환 실패 시 해당 행 건너뜀)
        try:
            no = int(row["No."])
        except (ValueError, TypeError):
            continue

        fname  = str(row.get("matched_filename", "") or "")
        rnote  = str(row.get("review_note",      "") or "")
        kw     = detect_keyword(fname, rnote)

        # 키워드가 없는 행은 인덱스 대상 제외
        if not kw:
            continue

        # 동일사업추정인 경우 연결번호 추출 (float → int → str 변환으로 소수점 제거)
        linked = ""
        if kw == "동일사업추정":
            raw_linked = row.get("Unnamed: 23", "")
            if pd.notna(raw_linked) and str(raw_linked).strip():
                linked = str(int(float(str(raw_linked).strip())))

        index[no] = {"keyword": kw, "linked_no": linked}
    return index


def main() -> None:
    """매칭실패여부 컬럼을 추가하여 파일매칭 CSV를 갱신합니다."""
    # ── 파일 읽기 ──────────────────────────────────────────────────────────────
    # 파일매칭 CSV는 탭 구분자(sep="\t") 사용
    df = pd.read_csv(TARGET_CSV, encoding="utf-8-sig", sep="\t")
    print(f"파일매칭 CSV: {len(df)}행")

    wf = pd.read_csv(WORKFILE_CSV, encoding="utf-8-sig")
    print(f"작업표 CSV:   {len(wf)}행")

    # 작업표에서 No. → 키워드 인덱스 구성
    wf_index = build_workfile_index(wf)
    print(f"작업표 키워드 인덱스: {len(wf_index)}개")

    # ── 매칭실패여부 컬럼 생성 ─────────────────────────────────────────────────
    failure_col: list[str] = []
    stats = {"사업없음": 0, "사업자료없음": 0, "동일사업추정": 0, "미기재": 0, "매칭성공": 0}

    for _, row in df.iterrows():
        # 파일이 1개 이상 매칭된 행은 실패여부 컬럼을 빈칸으로 유지
        if int(row.get("matched_file_count", 0)) > 0:
            failure_col.append("")
            stats["매칭성공"] += 1
            continue

        # No. 컬럼을 정수로 변환 (실패 시 빈칸 처리)
        try:
            no = int(row["No."])
        except (ValueError, TypeError):
            failure_col.append("")
            stats["미기재"] += 1
            continue

        # 작업표 인덱스에서 해당 번호 조회
        entry = wf_index.get(no)
        if entry is None:
            # 작업표에 해당 번호가 없으면 빈칸 유지
            failure_col.append("")
            stats["미기재"] += 1
            continue

        kw = entry["keyword"]
        # 동일사업추정인 경우 연결번호를 괄호 안에 표시
        if kw == "동일사업추정" and entry["linked_no"]:
            value = f"동일사업추정({entry['linked_no']})"
        else:
            value = kw

        failure_col.append(value)
        stats[kw] += 1

    df["매칭실패여부"] = failure_col

    # ── 결과 저장 (원본 덮어쓰기) ─────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig", sep="\t")
    print(f"\n결과 저장: {OUTPUT_CSV.name}")

    # ── 요약 통계 출력 ─────────────────────────────────────────────────────────
    unmatched_total = (df["matched_file_count"] == 0).sum()
    filled = unmatched_total - stats["미기재"]
    print("\n=== 결과 요약 ===")
    print(f"  매칭 성공 행:                  {stats['매칭성공']:>5}건")
    print(f"  매칭 실패 행 (총):             {unmatched_total:>5}건")
    print(f"    └ 사업없음:                  {stats['사업없음']:>5}건")
    print(f"    └ 사업자료없음:              {stats['사업자료없음']:>5}건")
    print(f"    └ 동일사업추정:              {stats['동일사업추정']:>5}건")
    print(f"    └ 작업표에도 미기재:         {stats['미기재']:>5}건")
    print(f"  매칭실패여부 채운 항목:        {filled:>5}건 / {unmatched_total}건")


if __name__ == "__main__":
    main()
