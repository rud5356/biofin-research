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

base = Path(r"c:\repos\biofin-research\국가생물다양성_열린재정 데이터\내역사업있음")
묘목 = list((base / "농림수산").glob("*5762*")) or list(base.rglob("*5762*묘목*"))
f = 묘목[0]
text = extract_hwp_text(f)
print(f"파일: {f.name}, 텍스트 길이: {len(text)}")

b_pos = text.find("내역사업별")
print(f"b_pos={b_pos}")

section_end = b_pos + 5000
비목별_pos = text.find("비목별 분류", b_pos + 50)
print(f"비목별 분류 위치: {비목별_pos}")
if 0 < 비목별_pos < section_end:
    section_end = 비목별_pos
print(f"section_end={section_end}")

기능별_pos = text.find("기능별 분류", b_pos)
print(f"기능별 분류 위치: {기능별_pos} (section_end 이내? {0 < 기능별_pos < section_end})")

# 기능별 분류 주변 텍스트 확인
if 기능별_pos != -1:
    print(f"\n기능별 분류 앞뒤 repr(50자):")
    print(repr(text[기능별_pos:기능별_pos+100]))

    # end_기능별 설정
    end_기능별 = section_end
    idx = text.find("비목별 분류", 기능별_pos + 10)
    print(f"\n기능별 이후 비목별 분류 위치: {idx}")
    if 0 < idx < end_기능별:
        end_기능별 = idx
    print(f"end_기능별={end_기능별}")

    sect = text[기능별_pos:end_기능별]
    print(f"\nsect 길이: {len(sect)}")
    print(f"sect repr(200자): {repr(sect[:200])}")

    cells = [c.strip() for c in sect.split('\r') if c.strip()]
    print(f"\ncells 수: {len(cells)}")
    print("전체 cells (앞30개):")
    for i, c in enumerate(cells[:30]):
        is_dot = c and c[0] in ('·', 'ㆍ', '•')
        print(f"  [{i}] len={len(c)} dot={is_dot} repr={repr(c)}")
