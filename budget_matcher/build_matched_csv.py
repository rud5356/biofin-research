"""
사업별결산세출지출현황_2024년도.csv 에 두 개의 컬럼을 추가하여 저장합니다.

추가하는 컬럼:
  matched_filename  : v2 폴더에서 No. 기준으로 찾은 파일명 (여러 개면 | 구분)
  matched_file_count: 매칭된 파일 수
  매칭실패여부       : 매칭 실패한 행에 한해 작업표에서 가져온 사유
                      (사업없음 / 사업자료없음 / 동일사업추정(연결번호))
                      매칭 성공 행은 빈칸 유지.

실행 방법:
    python build_matched_csv.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ── 경로 설정 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
SOURCE_CSV   = SCRIPT_DIR / "사업별결산세출지출현황_2024년도.csv"            # 원본 예산 CSV
WORKFILE_CSV = SCRIPT_DIR / "열린재정_파일매칭_작업표.csv"                   # 수동 작업 기록표
OUTPUT_CSV   = SCRIPT_DIR / "사업별결산세출지출현황_2024년도_파일매칭.csv"    # 저장될 결과 CSV
V2_ROOT      = Path(r"C:\Yuna\국가생물다양성_열린재정 데이터_v2")             # 파일이 담긴 폴더
EXTENSIONS   = {".hwp", ".pdf"}                                              # 대상 파일 확장자
KEYWORDS     = ["사업없음", "사업자료없음", "동일사업추정"]                    # 작업표 키워드


# ── v2 폴더 스캔: No. → 파일명 목록 ──────────────────────────────────────────
def scan_v2(root: Path) -> dict[int, list[str]]:
    """
    v2 폴더를 재귀 탐색하여 번호(No.) → 파일명 목록 인덱스를 반환합니다.

    파일명이 '{번호}_...' 형식으로 시작하는 HWP/PDF 파일만 수집합니다.
    """
    index: dict[int, list[str]] = defaultdict(list)
    if not root.exists():
        print(f"[ERROR] 폴더 없음: {root}", file=sys.stderr)
        return index

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        m = re.match(r"^(\d+)_", path.name)
        if m:
            index[int(m.group(1))].append(path.name)

    # 각 번호 내에서 파일명 알파벳 순 정렬
    return {k: sorted(v) for k, v in index.items()}


# ── 작업표에서 키워드 인덱스 구성: No. → (키워드, 연결번호) ──────────────────
def detect_keyword(fname: str, rnote: str) -> str:
    """
    matched_filename / review_note 두 컬럼을 합쳐 키워드를 탐지합니다.

    우선순위: 사업없음 > 사업자료없음 > 동일사업추정
    """
    combined = f"{fname} {rnote}"
    for kw in KEYWORDS:
        if kw in combined:
            return kw
    return ""


def build_workfile_index(wf: pd.DataFrame) -> dict[int, dict]:
    """
    수동 작업표(열린재정_파일매칭_작업표.csv)를 읽어
    No. → {keyword, linked_no} 형태의 인덱스를 반환합니다.

    keyword  : '사업없음' | '사업자료없음' | '동일사업추정'
    linked_no: 동일사업추정인 경우 연결 번호 (Unnamed:23 컬럼 값)
    """
    index: dict[int, dict] = {}
    for _, row in wf.iterrows():
        try:
            no = int(row["No."])
        except (ValueError, TypeError):
            continue

        fname  = str(row.get("matched_filename", "") or "")
        rnote  = str(row.get("review_note",      "") or "")
        kw     = detect_keyword(fname, rnote)
        if not kw:
            continue

        # 동일사업추정이면 연결번호 추출 (float → int → str 변환으로 소수점 제거)
        linked = ""
        if kw == "동일사업추정":
            raw = row.get("Unnamed: 23", "")
            if pd.notna(raw) and str(raw).strip():
                try:
                    linked = str(int(float(str(raw).strip())))
                except ValueError:
                    linked = str(raw).strip()

        index[no] = {"keyword": kw, "linked_no": linked}
    return index


# ── 메인 ────────────────────────────────────────────────────────────────────
def main() -> None:
    """원본 CSV에 매칭 컬럼을 추가하고 결과를 저장합니다."""
    # 원본 예산 CSV 읽기
    df = pd.read_csv(SOURCE_CSV, encoding="utf-8-sig")
    print(f"원본 CSV: {SOURCE_CSV.name}  ({len(df)}행)")

    if "No." not in df.columns:
        print("[ERROR] 'No.' 컬럼이 없습니다.", file=sys.stderr)
        sys.exit(1)

    # v2 폴더 스캔: 번호 → 파일명 목록
    print(f"v2 폴더 스캔: {V2_ROOT}")
    file_index  = scan_v2(V2_ROOT)
    total_files = sum(len(v) for v in file_index.values())
    print(f"  파일 수: {total_files}개 (번호 종류: {len(file_index)}개)")

    # matched_filename / matched_file_count 컬럼 생성
    matched_filenames: list[str] = []
    matched_counts: list[int] = []
    for no_val in df["No."]:
        try:
            no = int(no_val)
        except (ValueError, TypeError):
            matched_filenames.append("")
            matched_counts.append(0)
            continue
        files = file_index.get(no, [])
        # 여러 파일이 매칭된 경우 ' | '로 구분
        matched_filenames.append(" | ".join(files))
        matched_counts.append(len(files))

    df["matched_filename"]   = matched_filenames
    df["matched_file_count"] = matched_counts

    # 수동 작업표 읽기 → 매칭실패여부 컬럼 생성
    print(f"작업표 읽기: {WORKFILE_CSV.name}")
    wf       = pd.read_csv(WORKFILE_CSV, encoding="utf-8-sig")
    wf_index = build_workfile_index(wf)
    print(f"  키워드 인덱스: {len(wf_index)}개")

    failure_col: list[str] = []
    stats = {"사업없음": 0, "사업자료없음": 0, "동일사업추정": 0, "미기재": 0, "매칭성공": 0}

    for _, row in df.iterrows():
        # 파일이 매칭된 행은 빈칸 유지
        if int(row["matched_file_count"]) > 0:
            failure_col.append("")
            stats["매칭성공"] += 1
            continue

        try:
            no = int(row["No."])
        except (ValueError, TypeError):
            failure_col.append("")
            stats["미기재"] += 1
            continue

        entry = wf_index.get(no)
        if entry is None:
            failure_col.append("")
            stats["미기재"] += 1
            continue

        kw = entry["keyword"]
        if kw == "동일사업추정" and entry["linked_no"]:
            value = f"동일사업추정({entry['linked_no']})"
        else:
            value = kw

        failure_col.append(value)
        stats[kw] += 1

    df["매칭실패여부"] = failure_col

    # 결과 저장
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUTPUT_CSV.name}")

    # 요약 통계 출력
    unmatched = (df["matched_file_count"] == 0).sum()
    print("\n=== 결과 요약 ===")
    print(f"  전체 행:              {len(df):>6}건")
    print(f"  파일 매칭 성공:       {stats['매칭성공']:>6}건")
    print(f"  파일 매칭 실패:       {unmatched:>6}건")
    print(f"    └ 사업없음:         {stats['사업없음']:>6}건")
    print(f"    └ 사업자료없음:     {stats['사업자료없음']:>6}건")
    print(f"    └ 동일사업추정:     {stats['동일사업추정']:>6}건")
    print(f"    └ 작업표 미기재:    {stats['미기재']:>6}건")


if __name__ == "__main__":
    main()
