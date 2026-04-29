from __future__ import annotations

import json
import sys
from pathlib import Path


def console_safe(value: object) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(value).encode(encoding, errors="backslashreplace").decode(encoding)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
