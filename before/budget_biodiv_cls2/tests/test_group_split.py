from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from train_attention_classifier import (  # noqa: E402
    build_business_group_key,
    save_split_outputs,
    split_records,
)


def make_records() -> list[dict]:
    records: list[dict] = []
    for business_index in range(60):
        label = business_index % 10
        for year in (2019, 2020, 2021):
            records.append(
                {
                    "year": year,
                    "ministry": f"부처 {business_index % 6}",
                    "activity_name": f"동일 사업 {business_index}",
                    "label": label,
                    "source_type": "budget_metadata",
                    "file_path": f"metadata://{year}/{business_index}",
                    "source_file": f"{year}.csv",
                    "source_row": business_index + 2,
                    "text": "테스트 본문",
                }
            )
    return records


class GroupSplitTest(unittest.TestCase):
    def test_same_business_never_crosses_split(self) -> None:
        records = make_records()
        train, valid = split_records(records, valid_ratio=0.2, seed=42)

        train_keys = {build_business_group_key(record) for record in train}
        valid_keys = {build_business_group_key(record) for record in valid}
        self.assertFalse(train_keys & valid_keys)
        self.assertEqual(len(train) + len(valid), len(records))
        self.assertAlmostEqual(len(valid) / len(records), 0.2, delta=0.08)
        self.assertEqual(set(range(10)), {int(record["label"]) for record in train})
        self.assertEqual(set(range(10)), {int(record["label"]) for record in valid})

    def test_split_is_deterministic(self) -> None:
        records = make_records()
        first_train, first_valid = split_records(records, valid_ratio=0.2, seed=7)
        second_train, second_valid = split_records(records, valid_ratio=0.2, seed=7)
        self.assertEqual(
            [record["file_path"] for record in first_train],
            [record["file_path"] for record in second_train],
        )
        self.assertEqual(
            [record["file_path"] for record in first_valid],
            [record["file_path"] for record in second_valid],
        )

    def test_group_key_normalizes_spacing_and_width(self) -> None:
        first = {"ministry": "환경 부", "activity_name": "습지보호 사업(계속)"}
        second = {"ministry": "환경부", "activity_name": "습지보호사업（계속）"}
        self.assertEqual(
            build_business_group_key(first), build_business_group_key(second)
        )

    def test_split_audit_files_report_zero_overlap(self) -> None:
        records = make_records()
        # 동일 사업의 한 연도에 다른 label을 넣어 충돌 감사 로그도 검증한다.
        records[1]["label"] = 9
        train, valid = split_records(records, valid_ratio=0.2, seed=42)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            save_split_outputs(train, valid, output_dir)
            summary = json.loads(
                (output_dir / "split_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(0, summary["overlap_groups"])
            self.assertEqual(1, summary["conflicting_label_groups"])
            self.assertTrue((output_dir / "split_assignments.csv").exists())
            self.assertTrue((output_dir / "split_conflicting_groups.csv").exists())


if __name__ == "__main__":
    unittest.main()
