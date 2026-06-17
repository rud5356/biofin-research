"""
국가생물다양성_열린재정 데이터의 HWP 사업설명자료를 읽어
내역사업 포함 여부(1/0)를 판별하고
사업별결산세출지출현황 CSV에 '내역사업포함여부' 컬럼을 추가합니다.

HWP 파일에서 '내역사업' 키워드가 본문에 등장하면 1, 없으면 0.
matched_filename 이 없는 행(매칭 실패)은 빈값으로 처리합니다.

사용법:
    python add_naeyeok_column.py
    python add_naeyeok_column.py --input "국가생물다양성_열린재정 데이터/사업별결산세출지출현황_2024년도_파일매칭_최종.csv"
    python add_naeyeok_column.py --output results.csv
"""
from __future__ import annotations

import argparse
import csv
import olefile
import struct
import zlib
from pathlib import Path

DATA_DIR     = Path(__file__).parent / "국가생물다양성_열린재정 데이터"
DEFAULT_CSV  = DATA_DIR / "사업별결산세출지출현황_2024년도_파일매칭_최종.csv"
KEYWORDS     = ["내역사업명", "내역사업별"]  # 표 컬럼 헤더 or 기능별(내역사업별) 표 제목
OUTPUT_COL   = "내역사업포함여부"
INPUT_ENC    = "euc-kr"
INPUT_DELIM  = "\t"
OUTPUT_ENC   = "utf-8-sig"
OUTPUT_DELIM = ","   # Excel 호환 콤마 구분자


# ── HWP 텍스트 추출 ──────────────────────────────────────────────────────────

def extract_hwp_text(hwp_path: Path) -> str:
    """HWP(OLE) 파일의 BodyText 섹션에서 본문 텍스트를 추출합니다."""
    ole = olefile.OleFileIO(str(hwp_path))
    texts: list[str] = []
    for entry in ole.listdir():
        if len(entry) == 2 and entry[0] == "BodyText" and entry[1].startswith("Section"):
            data = ole.openstream(entry).read()
            try:
                raw = zlib.decompress(data, -15)
            except Exception:
                continue
            pos = 0
            while pos < len(raw) - 4:
                header = struct.unpack_from("<I", raw, pos)[0]
                tag_id = header & 0x3FF
                size   = (header >> 20) & 0xFFF
                pos += 4
                if size == 0xFFF:
                    if pos + 4 <= len(raw):
                        size = struct.unpack_from("<I", raw, pos)[0]
                        pos += 4
                if pos + size > len(raw):
                    break
                chunk = raw[pos : pos + size]
                pos += size
                # HWPTAG_PARA_TEXT = 67
                if tag_id == 67 and size > 0:
                    try:
                        texts.append(chunk.decode("utf-16-le", errors="replace"))
                    except Exception:
                        pass
    ole.close()
    return "".join(texts)


def check_naeyeok(hwp_path: Path) -> int:
    """HWP 파일에 '내역사업' 키워드가 있으면 1, 없으면 0을 반환합니다."""
    text = extract_hwp_text(hwp_path)
    return 1 if any(kw in text for kw in KEYWORDS) else 0


# ── HWP 파일 인덱스 ──────────────────────────────────────────────────────────

def build_hwp_index(hwp_dir: Path) -> dict[str, Path]:
    """hwp_dir 하위 모든 .hwp 파일을 {파일명: 경로} 인덱스로 구성합니다."""
    index: dict[str, Path] = {}
    for p in hwp_dir.rglob("*.hwp"):
        index[p.name] = p
    return index


# ── CSV 처리 ─────────────────────────────────────────────────────────────────

def process_csv(
    input_path: Path,
    output_path: Path,
    hwp_index: dict[str, Path],
) -> None:
    with open(input_path, encoding=INPUT_ENC, newline="") as f:
        rows = list(csv.DictReader(f, delimiter=INPUT_DELIM))

    if not rows:
        print("CSV가 비어 있습니다.")
        return

    fieldnames = list(rows[0].keys())
    if OUTPUT_COL not in fieldnames:
        fieldnames.append(OUTPUT_COL)

    total = len(rows)
    cnt_1 = cnt_0 = cnt_skip = cnt_err = 0
    warn_files: list[str] = []

    print(f"처리 시작: {total}건")

    for i, row in enumerate(rows, 1):
        mf = row.get("matched_filename", "").strip()

        if not mf:
            row[OUTPUT_COL] = ""
            cnt_skip += 1
            continue

        hwp_path = hwp_index.get(mf)
        if hwp_path is None:
            row[OUTPUT_COL] = ""
            cnt_skip += 1
            if mf not in warn_files:
                warn_files.append(mf)
            continue

        try:
            result = check_naeyeok(hwp_path)
            row[OUTPUT_COL] = result
            if result == 1:
                cnt_1 += 1
            else:
                cnt_0 += 1
        except Exception as e:
            row[OUTPUT_COL] = ""
            cnt_err += 1
            print(f"  [ERROR] {mf}: {e}")

        if i % 500 == 0:
            print(f"  {i:,}/{total:,} 처리 완료 (내역사업=1: {cnt_1}건)")

    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding=OUTPUT_ENC, newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=fieldnames, delimiter=OUTPUT_DELIM,
            quoting=csv.QUOTE_MINIMAL, extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'='*50}")
    print(f"저장: {output_path}")
    print(f"  내역사업 있음 (1): {cnt_1:,}건")
    print(f"  내역사업 없음 (0): {cnt_0:,}건")
    print(f"  HWP 없음 / 매칭실패 (빈값): {cnt_skip:,}건")
    if cnt_err:
        print(f"  오류 (빈값): {cnt_err}건")
    if warn_files:
        print(f"\n  [WARN] 인덱스에 없는 파일 ({len(warn_files)}종):")
        for wf in warn_files[:10]:
            print(f"    {wf}")
        if len(warn_files) > 10:
            print(f"    ... 외 {len(warn_files)-10}종")
    print("="*50)


# ── 진입점 ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HWP 사업설명자료에서 내역사업 포함 여부를 판별해 CSV에 추가")
    p.add_argument("--input",   type=Path, default=DEFAULT_CSV,
                   help=f"입력 CSV (기본: {DEFAULT_CSV.name})")
    p.add_argument("--hwp-dir", type=Path, default=DATA_DIR,
                   help=f"HWP 파일 루트 디렉토리 (기본: {DATA_DIR.name})")
    p.add_argument("--output",  type=Path, default=None,
                   help="출력 CSV 경로 (기본: 입력 파일과 같은 경로에 덮어쓰기)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.input

    print(f"HWP 인덱스 구성 중: {args.hwp_dir}")
    hwp_index = build_hwp_index(args.hwp_dir)
    print(f"  → {len(hwp_index):,}개 HWP 파일 발견\n")

    process_csv(args.input, output_path, hwp_index)


if __name__ == "__main__":
    main()
