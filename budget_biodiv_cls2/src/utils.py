"""학습 파이프라인 전반에서 공유하는 작은 유틸리티."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("budget_document_classifier")


def configure_logging(debug: bool = False) -> logging.Logger:
    """CLI와 라이브러리 코드가 같은 형식으로 로그를 남기게 한다."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    return LOGGER


def set_seed(seed: int) -> None:
    """Python/NumPy/PyTorch 난수를 고정해 실험 재현성을 높인다."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # dry-run은 PyTorch가 없어도 데이터 점검 단계까지 갈 수 있게 둔다.
        LOGGER.debug("PyTorch가 없어 PyTorch seed 설정을 건너뜁니다.")


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def read_csv_flexible(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """한국 공공데이터에서 흔한 UTF-8/CP949 인코딩을 순서대로 시도한다."""
    path = Path(path)
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"CSV 인코딩을 판별할 수 없습니다: {path}") from last_error


def write_csv(rows: pd.DataFrame | Iterable[dict[str, Any]], path: str | Path) -> None:
    """Excel에서도 한글이 깨지지 않도록 UTF-8 BOM CSV로 저장한다."""
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(data: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=str)


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "cudnn" in text)


def safe_scalar(value: Any) -> Any:
    """pandas/NumPy scalar를 CSV와 JSON에 안전한 Python scalar로 바꾼다."""
    if pd.isna(value):
        return ""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value
