"""
label_v2 == 1 로 표시된 예산 문서 파일을 모아서 폴더별로 정리하는 스크립트.

입력: biodiv_document_text_dataset_labeled_v2.csv
      + 국가생물다양성_열린재정 데이터/ 폴더 (원본 HWP/PDF 파일들)

출력: --out-dir 에 지정한 폴더 (기본: data/biodiv_label1_files/)
      ├── 일반,지방행정/
      │   ├── 190_국무조정실 및 국무총리비서실_특별자치시·도지원단 운영.hwp
      │   └── ...
      ├── 환경/
      │   └── ...
      └── ...

사용법:
    python collect_label1_files.py
    python collect_label1_files.py --out-dir C:/output/biodiv_label1
    python collect_label1_files.py --csv data/my_labeled.csv --copy
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

# ─── 기본 경로 설정 ───────────────────────────────────────────────────────────
# 이 스크립트가 있는 budget_biodiv_cls/ 폴더를 기준으로 경로를 잡습니다.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data"

DEFAULT_CSV     = DATA_DIR / "biodiv_document_text_dataset_labeled_v2.csv"
DEFAULT_OUT_DIR = DATA_DIR / "biodiv_label1_files"

# CSV에서 사용하는 컬럼 이름
COL_LABEL         = "label_v2"          # 1이면 생물다양성 관련
COL_RESOLVED      = "resolved_paths"    # 실제 파일 절대경로
COL_RELATIVE      = "relative_paths"    # 폴더명\파일명 (출력 폴더 구조에 사용)
COL_FILENAME      = "matched_filename"  # 파일명만


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="label_v2==1 파일 수집 스크립트")
    p.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"입력 CSV 경로 (기본: {DEFAULT_CSV})",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"파일을 모을 출력 폴더 (기본: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        default=True,
        help="파일을 복사합니다 (기본값, 항상 활성화)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="실제로 복사하지 않고 어떤 파일이 수집될지만 출력합니다",
    )
    return p.parse_args()


def load_label1_rows(csv_path: Path) -> list[dict]:
    """CSV를 읽어 label_v2 == '1' 인 행만 반환합니다."""
    if not csv_path.exists():
        print(f"[오류] CSV 파일을 찾을 수 없습니다: {csv_path}")
        sys.exit(1)

    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get(COL_LABEL, "").strip() == "1":
                rows.append(row)

    return rows


def get_src_and_dest(row: dict, out_dir: Path) -> tuple[Path | None, Path | None]:
    """
    한 행에서 원본 파일 경로(src)와 복사 대상 경로(dest)를 계산합니다.

    relative_paths 컬럼 예시: '일반,지방행정\\190_국무조정실...hwp'
    → 폴더명: '일반,지방행정'
    → 파일명: '190_국무조정실...hwp'
    → dest:  out_dir / '일반,지방행정' / '190_국무조정실...hwp'

    resolved_paths 컬럼 예시: 'C:\\...\\일반,지방행정\\190_...hwp'
    → src: Path('C:\\...\\일반,지방행정\\190_...hwp')
    """
    resolved = row.get(COL_RESOLVED, "").strip()
    relative = row.get(COL_RELATIVE, "").strip()
    filename = row.get(COL_FILENAME, "").strip()

    # resolved_paths 가 없으면 파일을 찾을 수 없음
    if not resolved:
        return None, None

    # 여러 파일이 매칭된 경우 '|'로 구분되어 있을 수 있으므로 첫 번째 경로만 사용
    src_str = resolved.split("|")[0].strip()
    src = Path(src_str)

    # 출력 폴더 구조 결정: relative_paths에서 폴더명을 추출
    if relative:
        rel_path = Path(relative.split("|")[0].strip())
        # rel_path 가 '폴더명\파일명' 형태이면 parent가 폴더명
        if rel_path.parent.name:
            subfolder = rel_path.parent.name   # 예: '일반,지방행정'
            dest = out_dir / subfolder / rel_path.name
        else:
            # 폴더 구분이 없는 경우 파일명 그대로
            dest = out_dir / rel_path.name
    elif filename:
        # relative_paths 컬럼이 비어있으면 파일명만으로 처리
        dest = out_dir / filename
    else:
        return None, None

    return src, dest


def collect_files(rows: list[dict], out_dir: Path, dry_run: bool) -> None:
    """label_v2==1 행들의 파일을 출력 폴더로 복사합니다."""
    total     = len(rows)
    copied    = 0
    skipped   = 0   # 이미 출력 폴더에 있는 경우
    not_found = 0   # 원본 파일이 없는 경우

    print(f"\n수집 대상: {total}개 행 (label_v2 == 1)")
    print(f"출력 폴더: {out_dir}")
    if dry_run:
        print("[dry-run 모드] 실제 복사는 수행하지 않습니다.\n")
    else:
        print()

    for row in rows:
        src, dest = get_src_and_dest(row, out_dir)

        if src is None:
            # resolved_paths가 없어서 파일을 찾을 수 없는 경우
            fname = row.get(COL_FILENAME, "(파일명 없음)")
            print(f"  [경로없음] {fname}")
            not_found += 1
            continue

        if not src.exists():
            # 원본 파일이 실제로 존재하지 않는 경우
            print(f"  [파일없음] {src.name}")
            not_found += 1
            continue

        if dest.exists():
            # 이미 복사된 파일은 건너뜀
            skipped += 1
            continue

        if dry_run:
            # dry-run 이면 복사 경로만 출력
            print(f"  [예정] {src.relative_to(src.parent.parent)} → {dest.relative_to(out_dir)}")
            copied += 1
            continue

        # 대상 폴더가 없으면 생성
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(src, dest)   # copy2는 파일 메타데이터(수정일 등)도 보존
            copied += 1
        except Exception as e:
            print(f"  [복사오류] {src.name}: {e}")
            not_found += 1

    # 결과 요약
    print(f"\n{'─' * 50}")
    if dry_run:
        print(f"[dry-run] 복사 예정: {copied}개 | 이미 있음: {skipped}개 | 파일 없음: {not_found}개")
    else:
        print(f"완료  복사: {copied}개 | 이미 있음(건너뜀): {skipped}개 | 파일 없음: {not_found}개")

    # 생성된 하위 폴더 목록 출력
    if not dry_run and out_dir.exists():
        subfolders = sorted(p.name for p in out_dir.iterdir() if p.is_dir())
        if subfolders:
            print(f"\n생성된 폴더 ({len(subfolders)}개):")
            for sf in subfolders:
                count = len(list((out_dir / sf).iterdir()))
                print(f"  {sf}/  ({count}개 파일)")


def main() -> None:
    args = parse_args()

    # 1) CSV에서 label_v2==1 행 읽기
    rows = load_label1_rows(args.csv)
    print(f"CSV 로드 완료: {args.csv.name} → label_v2==1 행: {len(rows)}개")

    if not rows:
        print("수집할 파일이 없습니다.")
        return

    # 2) 파일 복사
    collect_files(rows, args.out_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
