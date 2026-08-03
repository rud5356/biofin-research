from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from predict_attention_classifier import _match_by_filename_or_id  # noqa: E402


def test_year_prefix_is_not_treated_as_row_id() -> None:
    budget = pd.DataFrame({"_source_row": [2024], "_year": [2023]})
    path = Path("2023_국방부_함정장비_2023014110000422331304.hwp")

    row, match_type, _ = _match_by_filename_or_id(
        path,
        budget,
        filename_lookup={},
        id_lookup={"2023": [0]},
    )

    assert row is None
    assert match_type == ""


def test_legacy_numeric_prefix_can_still_use_row_id() -> None:
    budget = pd.DataFrame({"_source_row": [43], "_year": [2023]})
    path = Path("42_legacy_document.hwp")

    row, match_type, _ = _match_by_filename_or_id(
        path,
        budget,
        filename_lookup={},
        id_lookup={"42": [0]},
    )

    assert row is not None
    assert match_type == "ROW_ID"
    assert int(row["_source_row"]) == 43
