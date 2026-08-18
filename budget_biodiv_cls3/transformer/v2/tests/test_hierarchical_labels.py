from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from build_dataset import (  # noqa: E402
    build_hierarchical_label_codes,
    compose_subcategory_code,
)


def test_compose_subcategory_code_zero_pads_lower_number() -> None:
    assert compose_subcategory_code(6, 5) == "6.05"
    assert compose_subcategory_code(0, 0) == "0"


def test_standalone_lower_label_is_reconstructed_from_hierarchy() -> None:
    frame = pd.DataFrame(
        {
            "BIOFIN 1차 카테고리": [6],
            "하위": [5],
            "하위 카테고리": [5],
            "_source_row": [2],
        }
    )
    assert build_hierarchical_label_codes(frame).tolist() == ["6.05"]


def test_mismatched_combined_label_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "BIOFIN 1차 카테고리": [6],
            "하위": [5],
            "하위 카테고리": ["5.05"],
            "_source_row": [2],
        }
    )
    with pytest.raises(ValueError, match="일치하지 않습니다"):
        build_hierarchical_label_codes(frame)
