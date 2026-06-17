"""
내역사업포함여부=1인 HWP 파일을 분야별로 분류하여 복사합니다.

출력 구조:
    국가생물다양성_열린재정 데이터/
    └── 내역사업있음/
        ├── 공공질서및안전/
        ├── 환경/
        └── ...

사용법:
    python copy_naeyeok_files.py
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

DATA_DIR    = Path(__file__).parent / "국가생물다양성_열린재정 데이터"
CSV_PATH    = DATA_DIR / "사업별결산세출지출현황_2024년도_내역사업포함여부.csv"
OUT_DIR     = DATA_DIR / "내역사업있음"


def build_hwp_index(hwp_dir: Path) -> dict[str, Path]:
    return {p.name: p for p in hwp_dir.rglob("*.hwp")}


def main() -> None:
    # HWP 인덱스
    print(f"HWP 인덱스 구성 중...")
    hwp_index = build_hwp_index(DATA_DIR)
    print(f"  → {len(hwp_index):,}개 파일\n")

    # CSV 읽기
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # 내역사업=1 필터
    targets = [
        r for r in rows
        if str(r.get("내역사업포함여부", "")).strip() == "1"
        and r.get("matched_filename", "").strip()
    ]
    print(f"내역사업=1 대상: {len(targets):,}건\n")

    copied = skipped = 0
    분야_counts: dict[str, int] = {}

    for row in targets:
        분야 = row["분야명"].strip()
        fname = row["matched_filename"].strip()

        src = hwp_index.get(fname)
        if src is None:
            skipped += 1
            continue

        dst_dir = OUT_DIR / 분야
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / fname

        if not dst.exists():
            shutil.copy2(src, dst)

        copied += 1
        분야_counts[분야] = 분야_counts.get(분야, 0) + 1

    print(f"{'='*50}")
    print(f"복사 완료: {copied:,}건  (파일 없음 건너뜀: {skipped}건)")
    print(f"출력 폴더: {OUT_DIR}\n")
    print("분야별 현황:")
    for 분야, cnt in sorted(분야_counts.items()):
        print(f"  {분야}: {cnt:,}건")
    print("="*50)


if __name__ == "__main__":
    main()
