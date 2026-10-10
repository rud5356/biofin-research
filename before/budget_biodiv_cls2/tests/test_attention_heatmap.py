from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate_attention_heatmap import generate_attention_heatmap


class AttentionHeatmapTest(unittest.TestCase):
    def test_generates_filtered_standalone_html(self) -> None:
        fieldnames = [
            "year",
            "ministry",
            "activity_name",
            "file_path",
            "source_type",
            "true_label",
            "pred_label",
            "probability",
            "chunk_index",
            "chunk_text_preview",
            "attention_weight",
        ]
        rows = [
            {
                "year": 2023,
                "ministry": "환경부",
                "activity_name": "습지복원사업",
                "file_path": "/tmp/wetland.hwp",
                "source_type": "document",
                "true_label": 2,
                "pred_label": 2,
                "probability": 0.8,
                "chunk_index": 0,
                "chunk_text_preview": "예산정보 <사업개요>",
                "attention_weight": 0.2,
            },
            {
                "year": 2023,
                "ministry": "환경부",
                "activity_name": "습지복원사업",
                "file_path": "/tmp/wetland.hwp",
                "source_type": "document",
                "true_label": 2,
                "pred_label": 2,
                "probability": 0.8,
                "chunk_index": 1,
                "chunk_text_preview": "훼손된 습지의 생태기능 복원",
                "attention_weight": 0.8,
            },
            {
                "year": 2023,
                "ministry": "산림청",
                "activity_name": "보호구역관리사업",
                "file_path": "/tmp/single.hwp",
                "source_type": "document",
                "true_label": 1,
                "pred_label": 0,
                "probability": 0.6,
                "chunk_index": 0,
                "chunk_text_preview": "짧은 문서",
                "attention_weight": 0.6,
            },
            {
                "year": 2023,
                "ministry": "산림청",
                "activity_name": "보호구역관리사업",
                "file_path": "/tmp/single.hwp",
                "source_type": "document",
                "true_label": 1,
                "pred_label": 0,
                "probability": 0.6,
                "chunk_index": 1,
                "chunk_text_preview": "보호구역 관리 본문",
                "attention_weight": 0.4,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_csv = temp_path / "attention_outputs.csv"
            output_html = temp_path / "attention_heatmap.html"
            with input_csv.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            summary = generate_attention_heatmap(
                input_csv,
                output_html,
                max_documents=2,
                min_chunks=2,
            )

            self.assertEqual(summary["input_documents"], 2)
            self.assertEqual(summary["multi_chunk_documents"], 2)
            self.assertEqual(summary["rendered_documents"], 2)
            page = output_html.read_text(encoding="utf-8")
            self.assertIn("문서 조각별 Attention 히트맵", page)
            self.assertIn("습지복원사업", page)
            self.assertIn("예산정보 &lt;사업개요&gt;", page)
            self.assertIn("보호구역관리사업", page)
            self.assertIn('data-result="correct"', page)
            self.assertIn('data-result="incorrect"', page)

    def test_accepts_unlabeled_prediction_output(self) -> None:
        fieldnames = [
            "year",
            "ministry",
            "activity_name",
            "file_path",
            "source_type",
            "pred_label",
            "probability",
            "chunk_index",
            "chunk_text_preview",
            "attention_weight",
        ]
        rows = [
            {
                "year": 2025,
                "ministry": "환경부",
                "activity_name": "신규예측사업",
                "file_path": "/tmp/new.hwp",
                "source_type": "document",
                "pred_label": 2,
                "probability": 0.7,
                "chunk_index": 0,
                "chunk_text_preview": "신규 사업설명",
                "attention_weight": 1.0,
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_csv = temp_path / "attention_outputs.csv"
            output_html = temp_path / "attention_heatmap.html"
            with input_csv.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            summary = generate_attention_heatmap(
                input_csv,
                output_html,
                max_documents=10,
                min_chunks=1,
            )

            self.assertEqual(summary["unlabeled_documents"], 1)
            page = output_html.read_text(encoding="utf-8")
            self.assertIn("신규예측사업", page)
            self.assertIn('data-result="unlabeled"', page)
            self.assertIn("기준 라벨 없음", page)


if __name__ == "__main__":
    unittest.main()
