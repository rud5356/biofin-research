"""
사업별결산세출지출현황_2024년도.csv 의 No. 컬럼을 기준으로
국가생물다양성_열린재정 데이터_v2 폴더의 파일명을 찾아 새 컬럼으로 추가합니다.

파일명 규칙: {No}_{소관명}_{세부사업명}.hwp / .pdf
동일 번호에 여러 파일이 있으면 '|' 로 구분하여 기록합니다.

실행 방법:
    python add_filename_column.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BUDGET_CSV = Path(__file__).resolve().parent / "사업별결산세출지출현황_2024년도.csv"
V2_ROOT    = Path(r"C:\Yuna\국가생물다양성_열린재정 데이터_v2")  # HWP/PDF 파일이 저장된 폴더
EXTENSIONS = {".hwp", ".pdf"}                                    # 매칭 대상 확장자
OUTPUT_CSV = Path(__file__).resolve().parent / "사업별결산세출지출현황_2024년도_파일매칭.csv"


def scan_files(root: Path) -> dict[int, list[str]]:
    """
    v2 폴더를 재귀 탐색하여 번호(No.) → 파일명 목록 인덱스를 반환합니다.

    파일명이 '{번호}_...' 형식으로 시작하는 HWP/PDF 파일만 수집합니다.
    같은 번호에 여러 파일이 있으면 목록에 모두 포함합니다.

    예) 123_환경부_습지보호사업.hwp → {123: ["123_환경부_습지보호사업.hwp"]}
    """
    index: dict[int, list[str]] = defaultdict(list)
    if not root.exists():
        print(f"[ERROR] 폴더 없음: {root}", file=sys.stderr)
        return index

    for path in root.rglob("*"):
        # 파일이 아니거나 지정 확장자가 아니면 건너뜀
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        # 파일명이 숫자_... 형식인지 확인 (정규식 앞자리 숫자 매칭)
        m = re.match(r"^(\d+)_", path.name)
        if m:
            index[int(m.group(1))].append(path.name)

    # 각 번호 내에서 파일명을 알파벳 순 정렬
    return {k: sorted(v) for k, v in index.items()}


def main() -> None:
    """CSV를 읽고, v2 폴더를 스캔하여 matched_filename / matched_file_count 컬럼을 추가합니다."""
    # ── CSV 읽기 (인코딩 자동 감지) ────────────────────────────────────────────
    # BOM이 포함된 utf-8-sig, 한글 Windows 기본 cp949, 일반 utf-8 순서로 시도
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            df = pd.read_csv(BUDGET_CSV, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        print(f"[ERROR] CSV 인코딩 감지 실패: {BUDGET_CSV}", file=sys.stderr)
        sys.exit(1)

    print(f"CSV 읽기 완료: {BUDGET_CSV.name}  ({len(df)}행)")

    if "No." not in df.columns:
        print("[ERROR] 'No.' 컬럼이 없습니다.", file=sys.stderr)
        sys.exit(1)

    # ── v2 폴더 스캔: 번호 → 파일명 목록 인덱스 구성 ──────────────────────────
    print(f"v2 폴더 스캔 중: {V2_ROOT}")
    file_index = scan_files(V2_ROOT)
    print(f"  총 파일 수: {sum(len(v) for v in file_index.values())}개  (번호 종류: {len(file_index)}개)")

    # ── 번호별 파일명 매칭 ────────────────────────────────────────────────────
    matched_filenames: list[str] = []
    matched_counts: list[int] = []

    for no_val in df["No."]:
        try:
            no = int(no_val)
        except (ValueError, TypeError):
            # No. 값이 숫자가 아니면 빈 값으로 처리
            matched_filenames.append("")
            matched_counts.append(0)
            continue

        files = file_index.get(no, [])
        # 여러 파일이 매칭된 경우 ' | '로 구분하여 하나의 셀에 기록
        matched_filenames.append(" | ".join(files))
        matched_counts.append(len(files))

    df["matched_filename"]   = matched_filenames
    df["matched_file_count"] = matched_counts

    # ── 결과 저장 ──────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {OUTPUT_CSV.name}")

    # ── 요약 통계 ──────────────────────────────────────────────────────────────
    total       = len(df)
    found_1     = (df["matched_file_count"] == 1).sum()
    found_multi = (df["matched_file_count"] > 1).sum()
    not_found   = (df["matched_file_count"] == 0).sum()

    print("\n=== 결과 요약 ===")
    print(f"  전체 행:          {total:>6}건")
    print(f"  파일 1개 매칭:    {found_1:>6}건")
    print(f"  파일 2개 이상:    {found_multi:>6}건")
    print(f"  매칭 없음:        {not_found:>6}건")


if __name__ == "__main__":
    main()
