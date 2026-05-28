"""
budget_field_cls 프로젝트 전반에서 공통으로 사용하는 유틸리티 함수 모음.

여러 스크립트에서 반복적으로 쓰이는 기능들을 한 곳에 모아두었습니다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def console_safe(value: object) -> str:
    """
    윈도우 콘솔처럼 UTF-8을 완전히 지원하지 않는 환경에서도
    한글·특수문자가 깨지지 않도록 안전하게 문자열로 변환합니다.

    표현할 수 없는 문자는 \\uXXXX 형태의 이스케이프로 대체됩니다.
    """
    # sys.stdout.encoding: 현재 터미널이 지원하는 문자 인코딩 (예: cp949, utf-8)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    # 지원하지 않는 문자는 backslashreplace 방식으로 이스케이프 처리
    return str(value).encode(encoding, errors="backslashreplace").decode(encoding)


def ensure_directory(path: Path) -> Path:
    """
    지정한 경로에 폴더가 없으면 자동으로 생성하고, 경로를 반환합니다.

    parents=True: 중간 단계의 폴더도 함께 생성 (예: a/b/c에서 a, b가 없어도 OK)
    exist_ok=True: 이미 폴더가 있어도 오류를 발생시키지 않음
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: object) -> None:
    """
    파이썬 객체(딕셔너리, 리스트 등)를 JSON 파일로 저장합니다.

    ensure_ascii=False: 한글 등 비ASCII 문자를 그대로 저장 (이스케이프하지 않음)
    indent=2: 들여쓰기 2칸으로 사람이 읽기 좋게 저장
    """
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
