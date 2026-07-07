from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_dataset import discover_documents
from predict_attention_classifier_v2 import build_prediction_records, load_budget_data


class PredictV2InputTest(unittest.TestCase):
    def test_builds_document_only_prediction_records_with_optional_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            doc_dir = root / "docs"
            doc_dir.mkdir()
            (doc_dir / "2024_환경부_습지복원사업_1234567890.txt").write_text(
                "훼손된 습지의 생태기능을 복원한다.", encoding="utf-8"
            )
            (doc_dir / "2024_산림청_보호구역관리_1234567891.txt").write_text(
                "보호구역의 서식지와 야생생물을 관리한다.", encoding="utf-8"
            )

            budget_file = root / "test_budget.csv"
            fieldnames = [
                "회계연도",
                "소관명",
                "회계명",
                "분야명",
                "프로그램명",
                "단위사업명",
                "세부사업명",
                "정부안금액(천원)",
                "국회확정금액(천원)",
                "biofin_category",
            ]
            rows = [
                {
                    "회계연도": 2024,
                    "소관명": "환경부",
                    "회계명": "일반회계",
                    "분야명": "환경",
                    "프로그램명": "자연환경보전",
                    "단위사업명": "생태계복원",
                    "세부사업명": "습지복원사업",
                    "정부안금액(천원)": 100,
                    "국회확정금액(천원)": 120,
                    "biofin_category": 2,
                },
                {
                    "회계연도": 2024,
                    "소관명": "산림청",
                    "회계명": "일반회계",
                    "분야명": "농림수산",
                    "프로그램명": "산림보전",
                    "단위사업명": "보호구역관리",
                    "세부사업명": "보호구역관리",
                    "정부안금액(천원)": 80,
                    "국회확정금액(천원)": 90,
                    "biofin_category": "",
                },
            ]
            with budget_file.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            budget = load_budget_data(budget_file)
            documents = discover_documents(doc_dir)
            records, success, failed = build_prediction_records(documents, budget)

            self.assertEqual(len(records), 2)
            self.assertEqual(len(success), 2)
            self.assertTrue(failed.empty)
            by_activity = {record["activity_name"]: record for record in records}
            self.assertEqual(by_activity["습지복원사업"]["true_label"], 2)
            self.assertIsNone(by_activity["보호구역관리"]["true_label"])
            self.assertIn("[세부사업 예산편성 정보]", by_activity["습지복원사업"]["text"])
            self.assertIn("[사업설명자료 본문]", by_activity["습지복원사업"]["text"])

    def test_matches_open_fiscal_filename_by_matched_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            doc_dir = root / "환경"
            doc_dir.mkdir()
            document = doc_dir / "3847_환경부_생태계 훼손지 복원.txt"
            document.write_text("생태계 훼손지를 복원한다.", encoding="utf-8")

            budget_file = root / "open_fiscal.csv"
            fieldnames = [
                "No.",
                "회계연도",
                "소관명",
                "회계코드명",
                "분야명",
                "부문명",
                "프로그램명",
                "단위사업명",
                "세부사업명",
                "세출예산금액",
                "matched_filename",
            ]
            with budget_file.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "No.": 3847,
                        "회계연도": 2024,
                        "소관명": "환경부",
                        "회계코드명": "일반회계",
                        "분야명": "환경",
                        "부문명": "자연환경",
                        "프로그램명": "자연환경보전",
                        "단위사업명": "생태복원",
                        "세부사업명": "생태계 훼손지 복원",
                        "세출예산금액": "1,000,000",
                        "matched_filename": document.name,
                    }
                )

            budget = load_budget_data(budget_file)
            records, success, failed = build_prediction_records(
                discover_documents(doc_dir), budget
            )

            self.assertEqual(len(records), 1)
            self.assertTrue(failed.empty)
            self.assertEqual(records[0]["match_type"], "MATCHED_FILENAME")
            self.assertEqual(records[0]["year"], 2024)
            self.assertEqual(records[0]["activity_name"], "생태계 훼손지 복원")
            self.assertEqual(budget.iloc[0]["회계명"], "일반회계")
            self.assertEqual(budget.iloc[0]["국회확정금액(천원)"], 1000.0)


if __name__ == "__main__":
    unittest.main()
