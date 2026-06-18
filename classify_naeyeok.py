"""
내역사업있음 폴더의 HWP 파일을 분석하여
내역사업이 1개인지 여러 개인지 분류합니다.

출력:
  국가생물다양성_열린재정 데이터/내역사업있음_분류결과.csv

사용법:
    python classify_naeyeok.py            # 분석 + CSV 저장
    python classify_naeyeok.py --copy     # 분석 + CSV + 파일 복사
"""
from __future__ import annotations

import argparse
import csv
import olefile
import re
import shutil
import struct
import zlib
from pathlib import Path

DATA_DIR    = Path(__file__).parent / "국가생물다양성_열린재정 데이터"
NAEYEOK_DIR = DATA_DIR / "내역사업있음"
OUTPUT_CSV  = DATA_DIR / "내역사업있음_분류결과.csv"
OUT_1개     = NAEYEOK_DIR / "1개"
OUT_여러개  = NAEYEOK_DIR / "여러개"

GUBN_VALUES = {'보조', '출연', '출자', '민간위탁'}

# Pattern A — 내역사업명 표 섹션 끝 마커
A_END_MARKERS = [
    "3) '", "3)'", "3) \"",
    "7. 사업", "7.사업",
    "집행절차", "사업 집행",
    "추진 경위", "산출 근거",
]

# Pattern B — 기능별 표 섹션 끝 마커
B_END_MARKERS = ["비목별 분류", "검토의견", "성과지표", "집행이력"]

# 기관명 판별 (사업명으로 오인하지 않도록)
KIGWAN_START = ('한국', '국립', '정부', '지자체', '지방자치', '농협', 'LH', '농금원',
                '각 기', '도로공', '항만공')
KIGWAN_END   = ('공사', '협회', '재단', '기금', '진흥원', '관리원', '평가원',
                '위원회', '연구원', '연구소', '사업단')

# Pattern B에서 내역사업 항목을 나타내는 점(·) 유사 문자들
DOT_CHARS = ('·', 'ㆍ', '•', '․')  # ․ = ONE DOT LEADER (산림청 등)


# ── HWP 텍스트 추출 ──────────────────────────────────────────────────────────

def extract_hwp_text(hwp_path: Path) -> str:
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
                if tag_id == 67 and size > 0:
                    try:
                        texts.append(chunk.decode("utf-16-le", errors="replace"))
                    except Exception:
                        pass
    ole.close()
    return "".join(texts)


# ── Pattern A: 내역사업명 표 ─────────────────────────────────────────────────

def _is_기관명(cell: str) -> bool:
    return (any(cell.startswith(p) for p in KIGWAN_START) or
            any(cell.endswith(p)   for p in KIGWAN_END))


def count_pattern_a(text: str) -> int:
    """내역사업명 표에서 내역사업 개수 추정 (셀 구분자=\\r 기준)"""
    kw_pos = text.find("내역사업명")
    if kw_pos == -1:
        return 0

    # 헤더 끝 위치 (컬럼 헤더 이후부터 데이터 시작)
    header_end = kw_pos
    for m in ["해당 조항", "법적근거"]:
        idx = text.find(m, kw_pos)
        if idx != -1:
            header_end = max(header_end, idx + len(m))

    # 데이터 섹션 끝 위치
    section_end = header_end + 2500
    for m in A_END_MARKERS:
        idx = text.find(m, header_end)
        if 0 < idx < section_end:
            section_end = idx

    cells = [c.strip() for c in text[header_end:section_end].split('\r') if c.strip()]

    # 방법 A: 사업명 → 구분값 or 금액 패턴 카운팅 (병합 셀 포함 대응)
    name_count = 0
    for i, c in enumerate(cells[:-1]):
        if len(c) < 4 or not re.search(r'[가-힣]{3,}', c):
            continue
        if c in GUBN_VALUES:
            continue
        if _is_기관명(c):
            continue
        if re.search(r'법 제|｢|｣|▪|기본법|관리법|진흥법|보호법|시행령|정률|정액', c):
            continue
        if re.match(r'^[\d,\.%\[\] ]+$', c):
            continue
        nx = cells[i + 1]
        if nx in GUBN_VALUES or re.match(r'^\d{1,3}(,\d{3})*(\.\d+)?$', nx):
            name_count += 1

    # 방법 B: 구분값 셀 직접 카운트 (사업명이 "-"인 경우 fallback)
    gubn_count = sum(1 for c in cells if c in GUBN_VALUES)

    return max(name_count, gubn_count)


# ── Pattern B: 기능별(내역사업별) 표 ─────────────────────────────────────────

def count_pattern_b(text: str) -> int:
    """기능별(내역사업별) 표에서 내역사업 개수 추정"""
    b_pos = text.find("내역사업별")
    if b_pos == -1:
        return 0

    # 섹션 끝 (최대 5000자, 비목별 분류 등 이전)
    section_end = b_pos + 5000
    for m in B_END_MARKERS:
        idx = text.find(m, b_pos + 50)
        if 0 < idx < section_end:
            section_end = idx

    # ① 기능별 분류(합계) 아래 · 항목 카운팅 (산림청 유형)
    기능별_pos = text.find("기능별 분류", b_pos)
    if 0 < 기능별_pos < section_end:
        end_기능별 = section_end
        idx = text.find("비목별 분류", 기능별_pos + 10)
        if 0 < idx < end_기능별:
            end_기능별 = idx
        sect = text[기능별_pos:end_기능별]
        cells = [c.strip() for c in sect.split('\r') if c.strip()]
        dot_items = [
            c for c in cells
            if c and c[0] in DOT_CHARS
            and len(c) <= 22
            and not re.search(r'\d{4}|\d{3}-\d{2}', c)  # 연도·예산코드 제외
            and re.search(r'[가-힣]{2,}', c)
        ]
        if dot_items:
            return len(dot_items)

    # ② ○ 항목 카운팅 (헌법재판소 등 — 중복 제거)
    sect = text[b_pos : min(b_pos + 2000, section_end)]
    cells = [c.strip() for c in sect.split('\r') if c.strip()]
    circle_items = {
        c for c in cells
        if c and c[0] == '○'
        and '합계' not in c
        and re.search(r'[가-힣]{2,}', c)
    }
    return len(circle_items)


# ── 파일 분류 ─────────────────────────────────────────────────────────────────

def classify_file(hwp_path: Path) -> dict:
    try:
        text = extract_hwp_text(hwp_path)
    except Exception as e:
        return {"패턴": "오류", "내역사업_수": 0, "분류": "오류", "비고": str(e)[:60]}

    has_a = "내역사업명" in text
    has_b = "내역사업별" in text

    if has_a and has_b:
        count = max(count_pattern_a(text), count_pattern_b(text))
        pattern = "AB"
    elif has_a:
        count = count_pattern_a(text)
        pattern = "A"
    elif has_b:
        count = count_pattern_b(text)
        pattern = "B"
    else:
        return {"패턴": "?", "내역사업_수": 0, "분류": "미확인", "비고": "키워드 없음"}

    분류 = "1개" if count <= 1 else "여러개"
    return {"패턴": pattern, "내역사업_수": count, "분류": 분류, "비고": ""}


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="내역사업 1개/여러개 분류")
    parser.add_argument("--copy", action="store_true", help="분류 결과대로 파일 복사")
    args = parser.parse_args()

    if not NAEYEOK_DIR.exists():
        print(f"폴더 없음: {NAEYEOK_DIR}")
        return

    # HWP 파일 수집 (1개/여러개 결과 폴더 제외)
    hwp_files = [
        (d.name, f)
        for d in sorted(NAEYEOK_DIR.iterdir())
        if d.is_dir() and d.name not in ("1개", "여러개")
        for f in sorted(d.glob("*.hwp"))
    ]

    total = len(hwp_files)
    print(f"분석 대상: {total:,}건")

    results = []
    cnt_1 = cnt_여 = cnt_미 = cnt_err = 0

    for i, (분야, hwp_path) in enumerate(hwp_files, 1):
        info = classify_file(hwp_path)
        results.append({
            "파일명":      hwp_path.name,
            "분야명":      분야,
            "패턴":        info["패턴"],
            "내역사업_수": info["내역사업_수"],
            "분류":        info["분류"],
            "비고":        info["비고"],
        })
        d = info["분류"]
        if d == "1개":      cnt_1  += 1
        elif d == "여러개": cnt_여  += 1
        elif d == "미확인": cnt_미  += 1
        else:               cnt_err += 1

        if i % 500 == 0:
            print(f"  {i:,}/{total:,} 처리 완료 (1개:{cnt_1}, 여러개:{cnt_여})")

    # CSV 저장
    fieldnames = ["파일명", "분야명", "패턴", "내역사업_수", "분류", "비고"]
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    print(f"\n{'='*55}")
    print(f"분석 완료: {total:,}건")
    print(f"  내역사업 1개:   {cnt_1:,}건")
    print(f"  내역사업 여러개: {cnt_여:,}건")
    print(f"  미확인:         {cnt_미:,}건")
    print(f"  오류:           {cnt_err:,}건")
    print(f"CSV 저장: {OUTPUT_CSV.name}")
    print("="*55)

    if not args.copy:
        return

    # 파일 복사: 내역사업있음/1개/{분야명}/ 및 여러개/{분야명}/
    print("\n파일 복사 중...")
    copied_1 = copied_여 = 0
    for row in results:
        if row["분류"] not in ("1개", "여러개"):
            continue
        src = NAEYEOK_DIR / row["분야명"] / row["파일명"]
        if not src.exists():
            continue
        dst_root = OUT_1개 if row["분류"] == "1개" else OUT_여러개
        dst_dir  = dst_root / row["분야명"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / row["파일명"]
        if not dst.exists():
            shutil.copy2(src, dst)
        if row["분류"] == "1개":
            copied_1  += 1
        else:
            copied_여 += 1

    print(f"  1개 폴더 복사:   {copied_1:,}건  → {OUT_1개}")
    print(f"  여러개 폴더 복사: {copied_여:,}건  → {OUT_여러개}")


if __name__ == "__main__":
    main()
