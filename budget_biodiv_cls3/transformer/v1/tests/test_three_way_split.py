from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from train_attention_classifier import (  # noqa: E402
    build_business_group_key,
    split_records_three_way,
)


def test_three_way_split_has_no_business_group_overlap() -> None:
    records = []
    for label in range(10):
        for index in range(20):
            records.append(
                {
                    "label": label,
                    "ministry": f"부처-{label}",
                    "activity_name": f"사업-{label}-{index}",
                    "source_file": "labels.csv",
                    "source_row": label * 20 + index + 2,
                    "file_path": f"{label}-{index}.hwp",
                }
            )

    train, valid, test = split_records_three_way(records, 0.1, 0.1, 42)
    groups = [
        {build_business_group_key(record) for record in split}
        for split in (train, valid, test)
    ]

    assert not (groups[0] & groups[1])
    assert not (groups[0] & groups[2])
    assert not (groups[1] & groups[2])
    assert 0.75 <= len(train) / len(records) <= 0.85
    assert 0.05 <= len(valid) / len(records) <= 0.15
    assert 0.05 <= len(test) / len(records) <= 0.15
