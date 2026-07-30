from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from predict_attention_classifier import save_classified_budget  # noqa: E402


def test_save_classified_budget_adds_only_result_column(tmp_path: Path) -> None:
    budget = pd.DataFrame(
        {
            "No.": [1, 2],
            "내역사업명": ["사업 A", "사업 B"],
            "_source_row": [2, 3],
            "_year": [2023, 2023],
        }
    )
    predictions = [{"source_row": 3, "pred_label": 6, "pred_subcategory": "2.04"}]
    output_path = tmp_path / "classified.csv"

    save_classified_budget(
        budget,
        predictions,
        output_path,
        "예측 하위 카테고리",
    )

    result = pd.read_csv(output_path, encoding="utf-8-sig")
    assert result.columns.tolist() == [
        "No.",
        "내역사업명",
        "예측 하위 카테고리",
    ]
    assert pd.isna(result.loc[0, "예측 하위 카테고리"])
    assert result.loc[1, "예측 하위 카테고리"] == 2.04
