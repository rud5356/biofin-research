"""
HWP(한글) 및 PDF 문서에서 텍스트를 추출하고 정제하는 모듈.

HWP 파일은 한글과컴퓨터에서 만든 독점 바이너리 형식으로,
olefile 라이브러리로 내부 스트림을 읽고 struct로 이진 데이터를 파싱합니다.
PDF는 pypdf 라이브러리로 페이지별 텍스트를 추출합니다.
"""

from __future__ import annotations

import re
import struct
import unicodedata
import zlib
from pathlib import Path

import olefile          # HWP 파일을 OLE(복합 문서) 형식으로 읽는 라이브러리
from pypdf import PdfReader   # PDF 텍스트 추출 라이브러리


# ─── HWP 파싱 상수 ────────────────────────────────────────────────────────────
# HWP 내부 레코드 태그 ID 67 = 단락(문단) 텍스트 블록
_HWP_PARA_TEXT_TAG = 67

# ─── 텍스트 정제용 정규식 패턴 ───────────────────────────────────────────────
# 제어 문자(Control Character) 제거: \x00~\x08, \x0b~\x1f, \x7f (줄바꿈 \n은 유지)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

# 허용하는 문자만 남기고 나머지 제거:
# - 숫자, 영문, 한글 자모/완성형, 공백
# - 문장 부호 및 특수 기호 (괄호, 화살표, 체크박스 기호 등)
_DISALLOWED_CHAR_RE = re.compile(
    r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ\s\.,;:!?%()/\-\[\]{}<>&+*'\"""''·,~_=#@○△▲▽▼□■※ㆍ]"
)

# 여러 개의 공백/탭을 하나의 공백으로 통일 (줄바꿈 제외)
_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")

# 3줄 이상 연속된 빈 줄을 2줄로 압축
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# 단어 토큰 패턴: 숫자, 영문, 한글 연속 문자열
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def clean_extracted_text(text: str) -> str:
    """
    문서에서 추출한 원시 텍스트를 모델 학습에 적합한 형태로 정제합니다.

    정제 순서:
    1. 유니코드 정규화 (NFKC): 반각/전각 문자 통일, 합성 문자 분리 방지
    2. 줄바꿈 통일 (CRLF → LF)
    3. 제어 문자 제거
    4. 허용 문자 외 제거
    5. 줄 앞뒤 공백 제거
    6. 연속 공백 → 단일 공백
    7. 과도한 빈 줄 압축
    """
    # 빈 문자열이나 None이 들어와도 안전하게 처리
    normalized = unicodedata.normalize("NFKC", text or "")

    # Windows 줄바꿈(\r\n)과 구식 맥 줄바꿈(\r)을 모두 \n으로 통일
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # 제어 문자(null 바이트 등)를 공백으로 교체
    normalized = _CONTROL_CHAR_RE.sub(" ", normalized)

    # 허용 목록 외 특수문자 제거
    normalized = _DISALLOWED_CHAR_RE.sub(" ", normalized)

    # 줄 앞뒤 공백 제거 (각 줄별로)
    normalized = re.sub(r" *\n *", "\n", normalized)

    # 여러 공백을 하나로 압축
    normalized = _WHITESPACE_RE.sub(" ", normalized)

    # 3줄 이상 빈 줄을 2줄로 압축
    normalized = _MULTI_NEWLINE_RE.sub("\n\n", normalized)

    return normalized.strip()


def count_words(text: str) -> int:
    """한글·영문·숫자 단어 수를 반환합니다."""
    return len(_TOKEN_RE.findall(text or ""))


def _read_hwp_body_sections(path: Path) -> list[tuple[str, bytes]]:
    """
    HWP 파일에서 본문 섹션(BodyText/Section*)의 원시 바이트 데이터를 읽습니다.

    HWP 파일은 OLE(Object Linking and Embedding) 복합 문서 형식을 사용합니다.
    파일 내부에 여러 스트림(Stream)이 폴더 구조처럼 들어있습니다:
      - FileHeader: 파일 정보 및 플래그
      - BodyText/Section0, Section1, ...: 본문 텍스트 데이터

    반환값: [(스트림 이름, 바이트 데이터), ...]
    """
    with olefile.OleFileIO(str(path)) as ole:
        # FileHeader 스트림에서 압축 여부 플래그를 읽습니다.
        # header[36:40]의 4바이트를 리틀 엔디언 부호 없는 정수로 해석
        header = ole.openstream("FileHeader").read()
        flags = struct.unpack("<I", header[36:40])[0]
        # 플래그의 1번째 비트(LSB)가 1이면 본문이 zlib으로 압축되어 있습니다.
        is_compressed = bool(flags & 1)

        # OLE 내 모든 스트림 중 BodyText/SectionN 형태의 것만 선택
        sections = [
            entry
            for entry in ole.listdir()
            if len(entry) == 2 and entry[0] == "BodyText" and entry[1].startswith("Section")
        ]
        # Section0, Section1, ... 순서로 정렬 (숫자 기준)
        sections = sorted(sections, key=lambda entry: int(entry[1].replace("Section", "")))

        result: list[tuple[str, bytes]] = []
        for entry in sections:
            stream_name = "/".join(entry)   # "BodyText/Section0" 형태
            payload = ole.openstream(stream_name).read()
            if is_compressed:
                # zlib.decompress(data, -15): wbits=-15는 헤더 없는 deflate 형식
                payload = zlib.decompress(payload, -15)
            result.append((stream_name, payload))

        return result


def _extract_text_from_hwp_section(section_bytes: bytes) -> str:
    """
    HWP 섹션의 이진 데이터에서 텍스트 단락들을 파싱하여 문자열로 반환합니다.

    HWP 섹션은 레코드(Record)의 연속으로 구성됩니다.
    각 레코드는 4바이트 헤더로 시작하며, 헤더에 태그 ID와 크기가 인코딩되어 있습니다:
      - 하위 10비트 (& 0x3FF): 레코드 태그 ID
      - 상위 12비트 (>> 20 & 0xFFF): 페이로드 크기 (0xFFF이면 다음 4바이트가 실제 크기)
    태그 ID가 67(PARA_TEXT)인 레코드만 텍스트를 담고 있습니다.
    """
    fragments: list[str] = []
    offset = 0

    while offset < len(section_bytes):
        # 4바이트 레코드 헤더를 읽어 태그 ID와 크기를 추출
        header = struct.unpack_from("<I", section_bytes, offset)[0]
        tag_id = header & 0x3FF          # 하위 10비트 = 태그 ID
        size = (header >> 20) & 0xFFF    # 상위 12비트 = 페이로드 크기
        offset += 4

        # 크기가 0xFFF(4095)이면 다음 4바이트에 실제 크기가 별도로 저장됨
        if size == 0xFFF:
            size = struct.unpack_from("<I", section_bytes, offset)[0]
            offset += 4

        payload = section_bytes[offset : offset + size]

        # 태그 ID 67 = 단락 텍스트 (PARA_TEXT): UTF-16 LE 인코딩
        if tag_id == _HWP_PARA_TEXT_TAG and payload:
            fragments.append(payload.decode("utf-16le", errors="ignore"))

        offset += size

    # 빈 단락은 제외하고 줄바꿈으로 연결
    return "\n".join(fragment for fragment in fragments if fragment.strip())


def _extract_hwp_preview_text(path: Path) -> str:
    """
    HWP 파일의 미리보기 텍스트(PrvText)를 읽습니다.

    PrvText는 본문 파싱이 실패할 때의 대안으로,
    HWP가 자동 생성하는 간략한 텍스트 스트림입니다.
    """
    with olefile.OleFileIO(str(path)) as ole:
        if not ole.exists("PrvText"):
            return ""
        preview = ole.openstream("PrvText").read()
        # 인코딩을 순서대로 시도하여 처음 성공한 방식 사용
        for encoding in ("utf-16le", "utf-8", "cp949"):
            try:
                return preview.decode(encoding, errors="ignore")
            except UnicodeDecodeError:
                continue
    return ""


def extract_hwp_text(path: Path) -> tuple[str, str]:
    """
    HWP 파일에서 텍스트를 추출합니다.

    추출 전략:
    1. BodyText 섹션 파싱 (가장 완전한 본문)
    2. 실패 시 PrvText 미리보기로 대체
    3. 둘 다 실패하면 빈 문자열 반환

    반환값: (텍스트, 추출 방법 레이블)
    """
    sections = _read_hwp_body_sections(path)
    fragments = [_extract_text_from_hwp_section(payload) for _, payload in sections]
    body_text = clean_extracted_text("\n".join(f for f in fragments if f.strip()))

    if body_text:
        return body_text, "hwp_bodytext"

    # 본문 파싱 결과가 비어있으면 미리보기 텍스트로 대체
    preview_text = clean_extracted_text(_extract_hwp_preview_text(path))
    if preview_text:
        return preview_text, "hwp_preview"

    return "", "hwp_empty"


def extract_pdf_text(path: Path) -> tuple[str, str]:
    """
    PDF 파일의 모든 페이지에서 텍스트를 추출하고 정제합니다.

    반환값: (정제된 텍스트, "pdf_pypdf")
    """
    reader = PdfReader(str(path))
    # 페이지별 텍스트를 추출 (추출 실패 시 빈 문자열로 대체)
    pages = [page.extract_text() or "" for page in reader.pages]
    return clean_extracted_text("\n".join(pages)), "pdf_pypdf"


def extract_document_text(path: Path) -> tuple[str, str]:
    """
    파일 확장자에 따라 HWP 또는 PDF 텍스트 추출 함수를 호출합니다.

    반환값: (추출된 텍스트, 추출 방법 레이블)
    지원하지 않는 형식이면 ValueError를 발생시킵니다.
    """
    suffix = path.suffix.lower()
    if suffix == ".hwp":
        return extract_hwp_text(path)
    if suffix == ".pdf":
        return extract_pdf_text(path)
    raise ValueError(f"지원하지 않는 파일 형식입니다: {path.suffix}")
