"""TXT, PDF, HWPX, HWP 사업설명자료의 본문 추출기.

HWP는 설치 환경 편차가 크다. 기본 OLE 파서를 먼저 사용하고, pyhwp의
``hwp5txt`` 명령, 선택적으로 Windows 한글 COM 자동화를 차례로 시도한다.
어떤 방식도 성공하지 못하면 호출자가 개별 문서만 제외할 수 있도록
구조화된 ``DocumentParseError``를 발생시킨다.
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess
import tempfile
import unicodedata
import zipfile
import zlib
from pathlib import Path
from typing import Callable


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".hwpx", ".hwp"}


class DocumentParseError(RuntimeError):
    def __init__(self, reason: str, path: str | Path, detail: str = "") -> None:
        self.reason = reason
        self.path = str(path)
        self.detail = detail
        message = f"{reason}: {path}"
        if detail:
            message += f" ({detail})"
        super().__init__(message)


def clean_document_text(text: str, min_line_length: int = 2) -> str:
    """섹션 제목은 보존하면서 페이지 번호·표 선·제어문자 등만 제거한다."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 탭과 줄바꿈을 제외한 C0/C1 제어문자 제거
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    # HWP 글머리표가 Unicode 사설영역(PUA) 문자로 남는 경우도 노이즈로 본다.
    text = re.sub(r"[\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd]", " ", text)

    cleaned: list[str] = []
    previous = None
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t\u00a0]+", " ", raw_line).strip()
        if not line:
            continue
        # 단독 페이지 번호와 '3 / 20', '- 3 -' 형태만 제거한다.
        if re.fullmatch(r"(?:[-–—]\s*)?\d{1,4}(?:\s*/\s*\d{1,4})?(?:\s*[-–—])?", line):
            continue
        # 표에서 반복되는 선 또는 장식 문자만 있는 행 제거
        if re.fullmatch(r"[\-_=─━┄┅┈┉┌┐└┘├┤┬┴┼│|+·•.\s]{3,}", line):
            continue
        # 의미 있는 짧은 제목(예: 목적, 근거)은 남기고 한 글자 잡음만 버린다.
        if len(line) < min_line_length and not re.search(r"[가-힣A-Za-z0-9]", line):
            continue
        # 연속해서 완전히 같은 헤더/푸터가 나온 경우 하나만 남긴다.
        if line == previous:
            continue
        cleaned.append(line)
        previous = line
    return "\n".join(cleaned).strip()


def _read_txt(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "utf-16"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    errors: list[str] = []
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:  # 손상 PDF/암호 PDF 등 라이브러리별 예외가 다양하다.
        errors.append(f"pypdf={exc}")
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            return "\n\n".join((page.extract_text() or "") for page in pdf.pages)
    except Exception as exc:
        errors.append(f"pdfplumber={exc}")
    raise RuntimeError("; ".join(errors))


def _read_hwpx(path: Path) -> str:
    """HWPX ZIP 내부 section XML을 문서 순서대로 읽는다."""
    from lxml import etree

    with zipfile.ZipFile(path) as archive:
        section_names = sorted(
            (
                name
                for name in archive.namelist()
                if re.search(r"(?:^|/)section\d+\.xml$", name, re.IGNORECASE)
            ),
            key=lambda name: int(re.search(r"section(\d+)", name, re.IGNORECASE).group(1)),
        )
        if not section_names:
            raise RuntimeError("HWPX 안에서 section XML을 찾지 못했습니다")
        parts: list[str] = []
        parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True)
        for name in section_names:
            root = etree.fromstring(archive.read(name), parser=parser)
            # hp:t 요소가 실제 문자열을 담는다. namespace prefix와 무관하게 찾는다.
            texts = [node.text or "" for node in root.xpath("//*[local-name()='t']")]
            parts.append("\n".join(texts))
        return "\n".join(parts)


def _decode_hwp_para_text(payload: bytes) -> str:
    """PARA_TEXT 내부의 8-unit inline control 블록을 건너뛴다.

    이를 단순 UTF-16LE decode하면 ``dces`` 같은 control parameter 바이트가
    ``捤獥``처럼 가짜 한자로 나타난다. 줄/문단/공백 성격 control만 문자로
    남기고 나머지 제어 블록은 16바이트 단위로 제거한다.
    """
    usable_length = len(payload) - (len(payload) % 2)
    units = struct.unpack(f"<{usable_length // 2}H", payload[:usable_length])
    block_controls = set(range(0x01, 0x0A)) | {0x0B, 0x0C} | set(range(0x0E, 0x18))
    output: list[str] = []
    index = 0
    while index < len(units):
        code = units[index]
        if code in block_controls:
            if code == 0x09:
                output.append("\t")
            index += min(8, len(units) - index)
            continue
        if code in (0x0A, 0x0D):
            output.append("\n")
        elif code in (0x18, 0x1E):
            output.append("-")
        elif code == 0x1F:
            output.append(" ")
        elif 0xD800 <= code <= 0xDBFF and index + 1 < len(units):
            low = units[index + 1]
            if 0xDC00 <= low <= 0xDFFF:
                output.append(chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)))
                index += 2
                continue
        elif code >= 0x20 and not 0xDC00 <= code <= 0xDFFF:
            output.append(chr(code))
        index += 1
    return "".join(output)


def _hwp_stream_records(data: bytes) -> list[str]:
    """HWP 5 BodyText 레코드 중 PARA_TEXT(tag 67)를 UTF-16LE로 푼다."""
    result: list[str] = []
    offset = 0
    data_len = len(data)
    while offset + 4 <= data_len:
        header = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        tag_id = header & 0x3FF
        size = (header >> 20) & 0xFFF
        if size == 0xFFF:
            if offset + 4 > data_len:
                break
            size = struct.unpack_from("<I", data, offset)[0]
            offset += 4
        if size < 0 or offset + size > data_len:
            break
        payload = data[offset : offset + size]
        offset += size
        if tag_id == 67 and payload:
            result.append(_decode_hwp_para_text(payload))
    return result


def _read_hwp_ole(path: Path) -> str:
    try:
        import olefile
    except ImportError as exc:
        raise RuntimeError("olefile 패키지가 설치되지 않았습니다") from exc

    if not olefile.isOleFile(str(path)):
        raise RuntimeError("HWP 5 OLE 파일이 아닙니다")
    with olefile.OleFileIO(str(path)) as ole:
        if not ole.exists("FileHeader"):
            raise RuntimeError("HWP FileHeader 스트림이 없습니다")
        header = ole.openstream("FileHeader").read()
        compressed = len(header) > 36 and bool(header[36] & 0x01)
        sections: list[tuple[int, str]] = []
        for entry in ole.listdir(streams=True, storages=False):
            joined = "/".join(entry)
            match = re.fullmatch(r"BodyText/Section(\d+)", joined, re.IGNORECASE)
            if match:
                sections.append((int(match.group(1)), joined))
        if not sections:
            raise RuntimeError("HWP BodyText/Section 스트림이 없습니다")
        paragraphs: list[str] = []
        for _, stream_name in sorted(sections):
            stream = ole.openstream(stream_name).read()
            if compressed:
                try:
                    stream = zlib.decompress(stream, -15)
                except zlib.error as exc:
                    raise RuntimeError(f"HWP 압축 스트림 해제 실패: {stream_name}") from exc
            paragraphs.extend(_hwp_stream_records(stream))
        return "\n".join(paragraphs)


def _read_hwp_pyhwp(path: Path) -> str:
    executable = shutil.which("hwp5txt")
    if not executable:
        raise RuntimeError("pyhwp의 hwp5txt 명령을 찾지 못했습니다")
    process = subprocess.run(
        [executable, str(path)],
        capture_output=True,
        timeout=180,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"hwp5txt 종료 코드 {process.returncode}: {detail[:300]}")
    return process.stdout.decode("utf-8", errors="replace")


def _read_hwp_com(path: Path) -> str:
    """한글이 설치된 Windows에서만 명시적으로 선택하는 COM 폴백."""
    try:
        import win32com.client  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("pywin32가 설치되지 않았습니다") from exc

    hwp = None
    temp_path: Path | None = None
    try:
        hwp = win32com.client.gencache.EnsureDispatch("HWPFrame.HwpObject")
        hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
        if not hwp.Open(str(path), "HWP", "forceopen:true"):
            raise RuntimeError("한글 COM Open이 실패했습니다")
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        if not hwp.SaveAs(str(temp_path), "TEXT", "code:UTF8"):
            raise RuntimeError("한글 COM 텍스트 저장이 실패했습니다")
        return _read_txt(temp_path)
    finally:
        if hwp is not None:
            try:
                hwp.Quit()
            except Exception:
                pass
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _read_hwp(path: Path, use_hwp_com: bool) -> str:
    parsers: list[tuple[str, Callable[[Path], str]]] = [
        ("olefile", _read_hwp_ole),
        ("pyhwp", _read_hwp_pyhwp),
    ]
    if use_hwp_com:
        parsers.append(("com", _read_hwp_com))
    errors: list[str] = []
    for name, parser in parsers:
        try:
            text = parser(path)
            if text and text.strip():
                return text
            errors.append(f"{name}=empty")
        except Exception as exc:
            errors.append(f"{name}={exc}")
    raise RuntimeError("; ".join(errors))


def extract_document(path: str | Path, use_hwp_com: bool = False) -> str:
    """확장자에 맞춰 본문을 추출하고 공통 전처리를 적용한다."""
    path = Path(path)
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError("UNSUPPORTED_EXTENSION", path, extension)
    readers: dict[str, Callable[[Path], str]] = {
        ".txt": _read_txt,
        ".pdf": _read_pdf,
        ".hwpx": _read_hwpx,
    }
    try:
        raw_text = _read_hwp(path, use_hwp_com) if extension == ".hwp" else readers[extension](path)
    except DocumentParseError:
        raise
    except Exception as exc:
        reason = "HWP_PARSE_FAILED" if extension == ".hwp" else "DOCUMENT_PARSE_FAILED"
        raise DocumentParseError(reason, path, str(exc)[:1000]) from exc
    text = clean_document_text(raw_text)
    if not text:
        raise DocumentParseError("EMPTY_DOCUMENT", path, "본문 추출 결과가 비어 있습니다")
    return text
