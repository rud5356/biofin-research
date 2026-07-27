import tempfile
import unittest
from pathlib import Path

import pandas as pd

from crawlers.open_fiscal import config
from crawlers.open_fiscal.crawl_business_docs import (
    YEAR_OUTPUT_COLUMNS,
    _candidate_from_record,
    _write_year_outputs,
    _year_business_key,
    _year_business_row,
)


class OpenFiscalConfigTest(unittest.TestCase):
    def test_default_base_dir_points_to_existing_pipeline_output(self) -> None:
        expected = config.REPO_ROOT / "budget_biodiv_cls2" / "outputs"
        self.assertEqual(config.BASE_DIR, expected)

    def test_open_fiscal_urls_use_expected_host(self) -> None:
        urls = [
            config.SITE_URL,
            config.LIST_API_URL,
            config.DETAIL_URL,
            config.DOWNLOAD_BASE_URL,
        ]
        self.assertTrue(all("openfiscaldata.go.kr" in url for url in urls))
        self.assertIsInstance(config.SAVE_DIR, Path)

    def test_save_dir_is_isolated_under_open_fiscal(self) -> None:
        expected = config.REPO_ROOT / "crawlers" / "open_fiscal" / "outputs"
        self.assertEqual(config.SAVE_DIR, expected)

    def test_business_key_and_csv_row_use_site_codes(self) -> None:
        record = {
            "acntYr": "2024",
            "offcCd": "116",
            "offcNm": "위원회",
            "acntCd": "110",
            "acntNm": "일반회계",
            "acctCd": "00",
            "fldCd": "010",
            "fldNm": "일반행정",
            "sectCd": "016",
            "sectNm": "일반행정",
            "pgmCd": "1000",
            "pgmNm": "프로그램",
            "actvCd": "1031",
            "actvNm": "단위사업",
            "sayCd": "300",
            "sayNm": "세부사업",
            "sayBrkdFileNm": "document.hwp",
        }
        candidate = _candidate_from_record(record, config.LIST_API_URL, 2024)
        self.assertIsNotNone(candidate)
        key = _year_business_key(2024, record)
        self.assertEqual(key, "2024-116-110-010-016-1000-1031-300")
        row = _year_business_row(2024, candidate or {})
        self.assertEqual(row["business_key"], key)
        self.assertEqual(row["site_filename"], "document.hwp")
        self.assertEqual(row["download_status"], "pending")

    def test_business_without_document_remains_in_csv(self) -> None:
        record = {
            "acntYr": "2024",
            "offcCd": "001",
            "offcNm": "부처",
            "acntCd": "110",
            "acctCd": "00",
            "fldCd": "010",
            "sectCd": "016",
            "pgmCd": "1000",
            "actvCd": "1001",
            "actvNm": "단위사업",
            "sayCd": "100",
            "sayNm": "문서 없는 세부사업",
            "sayBrkdFileNm": None,
        }
        candidate = _candidate_from_record(record, config.LIST_API_URL, 2024)
        self.assertIsNotNone(candidate)
        row = _year_business_row(2024, candidate or {})
        self.assertEqual(row["download_status"], "no_document")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_year_outputs(root, 2024, [row], [], [])
            frame = pd.read_csv(root / "open_fiscal_2024.csv", encoding="utf-8-sig")
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.loc[0, "다운로드상태"], "no_document")
            self.assertTrue(pd.isna(frame.loc[0, "사업설명자료_파일명"]))

    def test_combined_csv_keeps_budget_columns_and_document_path(self) -> None:
        business = {
            "business_key": "2024-key",
            "year": 2024,
            "ministry": "환경부",
            "account_name": "일반회계",
            "field_name": "환경",
            "sector_name": "자연",
            "program_name": "프로그램",
            "unit_name": "단위사업",
            "activity_name": "세부사업",
            "budget_amount": 100,
            "download_status": "success",
            "error": "",
        }
        document = {
            "business_key": "2024-key",
            "saved_filename": "document.hwp",
            "relative_path": "사업설명자료/document.hwp",
            "download_status": "success",
            "error": "",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_year_outputs(root, 2024, [business], [document], [])
            path = root / "open_fiscal_2024.csv"
            frame = pd.read_csv(path, encoding="utf-8-sig")
            self.assertEqual(frame.columns.tolist(), YEAR_OUTPUT_COLUMNS)
            self.assertEqual(frame.loc[0, "예산액"], 100)
            self.assertEqual(
                frame.loc[0, "사업설명자료_상대경로"],
                "사업설명자료/document.hwp",
            )


if __name__ == "__main__":
    unittest.main()
