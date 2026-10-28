import csv
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from apply_bar import (APPLIED, NO_CATEGORY, RATE_MISSING, UNKNOWN_CATEGORY, apply_bar,
                       find_category_column, load_rates, main)

RATES = {"0": Decimal(0), "2": Decimal(50), "6": Decimal("33.3"), "9": None}


class ApplyBarTests(unittest.TestCase):
    def test_rate_rounding_and_statuses(self):
        rows = [{"c": "2", "예산액": "1001", "지출액": "3"},
                {"c": "6.0", "예산액": "1000.0", "지출액": ""},
                {"c": "9", "예산액": "500", "지출액": "1"},
                {"c": "", "예산액": "500", "지출액": "1"},
                {"c": "11", "예산액": "500", "지출액": "1"},
                {"c": "0", "예산액": "1,000", "지출액": "1"}]
        result, summary = apply_bar(rows, "c", ["예산액", "지출액"], RATES, "v")
        self.assertEqual([r["BAR 귀속 예산액"] for r in result], ["501", "333", "", "", "", "0"])
        self.assertEqual([r["BAR 귀속 지출액"] for r in result], ["2", "", "", "", "", "0"])
        self.assertEqual([r["BAR 적용상태"] for r in result],
                         [APPLIED, APPLIED, RATE_MISSING, NO_CATEGORY, UNKNOWN_CATEGORY, APPLIED])
        self.assertEqual([r["BAR 귀속률(%)"] for r in result], ["50", "33.3", "", "", "", "0"])
        self.assertNotIn("BAR 버전", rows[0])
        self.assertEqual(summary["2"]["attributed"]["예산액"], Decimal(501))

    def test_decimals(self):
        result, _ = apply_bar([{"c": "6", "a": "1"}], "c", ["a"], RATES, "v", decimals=2)
        self.assertEqual(result[0]["BAR 귀속 a"], "0.33")
        # 백만원 단위: 원 단위(소수 6자리)까지 반올림하고 뒤쪽 0은 적지 않는다.
        rows = [{"c": "6", "a": "1"}, {"c": "2", "a": "1761"}, {"c": "2", "a": "1760"}]
        result, _ = apply_bar(rows, "c", ["a"], RATES, "v", decimals=6)
        self.assertEqual([r["BAR 귀속 a"] for r in result], ["0.333", "880.5", "880"])

    def test_bad_amount_reports_line(self):
        with self.assertRaisesRegex(ValueError, "3행"):
            apply_bar([{"c": "2", "a": "1"}, {"c": "2", "a": "abc"}], "c", ["a"], RATES, "v")

    def test_category_column_priority(self):
        self.assertEqual(find_category_column(["LLM BIOFIN 1차 카테고리", "pipeline_label"]), "pipeline_label")
        self.assertEqual(find_category_column(["예측 BIOFIN 1차 카테고리"]), "예측 BIOFIN 1차 카테고리")
        with self.assertRaises(ValueError):
            find_category_column(["x"])

    def test_invalid_rates_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "r.json"
            for rates in [{"1": 120}, {"1": "50"}, {"1": True}]:
                path.write_text(json.dumps({"version": "v", "rates_percent": rates}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_rates(path)

    def test_main_keeps_encoding_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "in.csv"
            rates = Path(folder) / "r.json"
            rates.write_text(json.dumps({"version": "v1", "rates_percent": {"0": 0, "2": 50}}), encoding="utf-8")
            with source.open("w", encoding="cp949", newline="") as stream:
                csv.writer(stream).writerows([["세부사업명", "예산액", "LLM BIOFIN 1차 카테고리"],
                                              ["가", "100", "2"], ["나", "7", "0"]])
            args = ["--input", str(source), "--rates", str(rates)]
            self.assertEqual(main(args), 0)
            with (Path(folder) / "in_bar.csv").open(encoding="cp949", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["BAR 귀속 예산액"] for r in rows], ["50", "0"])
            self.assertEqual(rows[0]["BAR 버전"], "v1")
            self.assertTrue((Path(folder) / "in_bar_summary.csv").is_file())
            with self.assertRaises(SystemExit):
                main(args)
            self.assertEqual(main(args + ["--overwrite"]), 0)


if __name__ == "__main__":
    unittest.main()
