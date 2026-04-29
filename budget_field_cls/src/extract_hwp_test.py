from pathlib import Path
from document_text import extract_hwp_text

path = Path(r"C:\Yuna\국가생물다양성_열린재정 데이터_v2\일반,지방행정\1_국회_입법활동지원.hwp")
text, method = extract_hwp_text(path)

print("method:", method)
print("chars:", len(text))
print()
print(text[:2000])