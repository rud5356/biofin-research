"""버그 수정 검증 — 3가지 이슈 처리:
1. Pattern A: 사업명이 "-" 인 경우 (count=0 → 구분셀 카운트로 fallback)
2. Pattern A: 섹션이 너무 길어 과대 계산
3. Pattern B: 기능별 분류 section 추출 버그 + ○ 중복 카운팅
"""
import olefile, re, struct, zlib
from pathlib import Path

def extract_hwp_text(hwp_path):
    ole = olefile.OleFileIO(str(hwp_path))
    texts = []
    for entry in ole.listdir():
        if len(entry) == 2 and entry[0] == "BodyText" and entry[1].startswith("Section"):
            data = ole.openstream(entry).read()
            try:
                raw = zlib.decompress(data, -15)
            except: continue
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
                if pos + size > len(raw): break
                chunk = raw[pos:pos+size]
                pos += size
                if tag_id == 67 and size > 0:
                    try: texts.append(chunk.decode("utf-16-le", errors="replace"))
                    except: pass
    ole.close()
    return "".join(texts)

GUBN_VALUES = {'보조', '출연', '출자', '민간위탁'}
A_END_MARKERS = ["3) '", "3)'", "3) \"", "7. 사업", "7.사업", "집행절차", "사업 집행", "추진 경위", "산출 근거"]
KIGWAN_START  = ('한국', '국립', '정부', '지자체', '지방자치', '농협', 'LH', '농금원', '각 기', '도로공')
KIGWAN_END    = ('공사', '협회', '재단', '기금', '진흥원', '관리원', '평가원', '위원회', '연구원', '연구소')

def _is_기관명(cell):
    return (any(cell.startswith(p) for p in KIGWAN_START) or
            any(cell.endswith(p)   for p in KIGWAN_END))

def count_pattern_a(text):
    kw_pos = text.find("내역사업명")
    if kw_pos == -1:
        return 0
    header_end = kw_pos
    for m in ["해당 조항", "법적근거"]:
        idx = text.find(m, kw_pos)
        if idx != -1:
            header_end = max(header_end, idx + len(m))
    section_end = header_end + 2500
    for m in A_END_MARKERS:
        idx = text.find(m, header_end)
        if 0 < idx < section_end:
            section_end = idx
    section = text[header_end:section_end]
    cells = [c.strip() for c in section.split('\r') if c.strip()]

    # 방법1: 사업명 앞에 구분값이 오는 패턴
    name_count = 0
    for i, c in enumerate(cells[:-1]):
        if len(c) < 4 or not re.search(r'[가-힣]{3,}', c): continue
        if c in GUBN_VALUES: continue
        if _is_기관명(c): continue
        if re.search(r'법 제|｢|｣|▪|기본법|관리법|진흥법|보호법|시행령|정률|정액', c): continue
        if re.match(r'^[\d,\.%\[\] ]+$', c): continue
        nx = cells[i + 1]
        if (nx in GUBN_VALUES or re.match(r'^\d{1,3}(,\d{3})*(\.\d+)?$', nx)):
            name_count += 1

    # 방법2: 구분값 셀 직접 카운트 (fallback: 사업명이 "-" 등인 경우)
    gubn_count = sum(1 for c in cells if c in GUBN_VALUES)

    return max(name_count, gubn_count)

def count_pattern_b(text):
    b_pos = text.find("내역사업별")
    if b_pos == -1:
        return 0
    # 섹션 끝 (최대 5000자, 비목별 분류 등 이전)
    section_end = b_pos + 5000
    for m in ["비목별 분류", "검토의견", "성과지표", "집행이력"]:
        idx = text.find(m, b_pos + 50)
        if 0 < idx < section_end:
            section_end = idx

    # 기능별 분류 표가 있는 경우 (산림청 등): "기능별 분류" 아래 · 항목
    기능별_pos = text.find("기능별 분류", b_pos)
    if 0 < 기능별_pos < section_end:
        end_기능별 = section_end
        for m in ["비목별 분류"]:
            idx = text.find(m, 기능별_pos + 10)
            if 0 < idx < end_기능별:
                end_기능별 = idx
        sect = text[기능별_pos:end_기능별]
        cells = [c.strip() for c in sect.split('\r') if c.strip()]
        dot_items = [c for c in cells
                     if c and c[0] in ('·', 'ㆍ', '•')
                     and len(c) <= 22
                     and not re.search(r'\d{4}|\d{3}-\d{2}', c)  # 연도·예산코드 제외
                     and re.search(r'[가-힣]{2,}', c)]
        if dot_items:
            return len(dot_items)

    # 기능별 분류 없음 (헌법재판소 등): ○ 항목 카운팅 (중복 제거)
    sect = text[b_pos:min(b_pos + 2000, section_end)]  # 앞 2000자만
    cells = [c.strip() for c in sect.split('\r') if c.strip()]
    circle_items = {c for c in cells
                    if c and c[0] == '○'
                    and '합계' not in c
                    and re.search(r'[가-힣]{2,}', c)}
    return len(circle_items)

base = Path(r"c:\repos\biofin-research\국가생물다양성_열린재정 데이터\내역사업있음")

print("=== Pattern A ===")
for path_str, expected in [
    ("환경/3474_해양수산부_갯벌생태계 복원사업.hwp",       2),
    ("교통및물류/7044_국토교통부_고속도로조사.hwp",          1),
    ("교통및물류/7045_국토교통부_함양-울산고속도로건설.hwp",  1),
    ("농림수산/5312_농림축산식품부_농업재해보험.hwp",        4),
    ("과학기술/8499_과학기술정보통신부_과학기술혁신정책 지원(R%26D).hwp", 3),
    ("산업·중소기업및에너지/6075_산업통상자원부_전략물자수출입통제기반구축.hwp", 5),
]:
    f = base / path_str
    if not f.exists(): print(f"  없음: {path_str}"); continue
    count = count_pattern_a(extract_hwp_text(f))
    ok = "✓" if count == expected else f"✗ (기대:{expected})"
    print(f"  {f.name[:48]:48s} count={count} {ok}")

print("\n=== Pattern B ===")
for fname, expected in [
    ("공공질서및안전/1184_헌법재판소_인건비.hwp",     1),
    ("공공질서및안전/1185_헌법재판소_본부기본경비.hwp", 1),
]:
    f = base / fname
    count = count_pattern_b(extract_hwp_text(f))
    ok = "✓" if count == expected else f"✗ (기대:{expected})"
    print(f"  {f.name[:48]:48s} count={count} {ok}")

묘목 = list((base / "농림수산").glob("*5762*")) or list(base.rglob("*5762*묘목*"))
if 묘목:
    f = 묘목[0]
    count = count_pattern_b(extract_hwp_text(f))
    ok = "✓" if count == 5 else f"✗ (기대:5)"
    print(f"  {f.name[:48]:48s} count={count} {ok}")
