import tempfile
import unittest
from pathlib import Path

from crawlers.local_fiscal import config
from crawlers.local_fiscal.crawl_business_docs import (
    CrawlConfig,
    RegionSpec,
    detail_url,
    merge_manifest,
    normalize_business,
    read_manifest,
    safe_filename,
    write_manifest,
)


def sample_config(root: Path) -> CrawlConfig:
    return CrawlConfig(
        reference_date="2024-12-31",
        regions=(RegionSpec("부산", "26"),),
        limit=5,
        output_dir=root / "pdf",
        manifest_path=root / "manifest.csv",
        headed=False,
        browser_channel="chrome",
        timeout_ms=60_000,
        min_delay=0,
        pdf_scale=0.70,
        overwrite=False,
        retry_failed=False,
        list_only=False,
    )


class LocalFiscalConfigTest(unittest.TestCase):
    def test_target_regions_are_the_requested_seven_cities(self) -> None:
        self.assertEqual(
            config.TARGET_REGIONS,
            {
                "서울": "11",
                "부산": "26",
                "대구": "27",
                "대전": "30",
                "인천": "28",
                "광주": "29",
                "울산": "31",
            },
        )

    def test_local_fiscal_output_is_isolated(self) -> None:
        expected = config.REPO_ROOT / "crawlers" / "local_fiscal" / "outputs"
        self.assertEqual(config.BASE_DIR, expected)
        self.assertNotIn("open_fiscal", str(config.BASE_DIR))

    def test_detail_url_uses_observed_site_parameters(self) -> None:
        url = detail_url("626000020223007B", "2600000", "2024", "2024-12-31")
        self.assertIn("dbizCd=626000020223007B", url)
        self.assertIn("lafCd=2600000", url)
        self.assertIn("fyr=2024", url)
        self.assertIn("inqYmd=20241231", url)

    def test_safe_filename_replaces_windows_reserved_characters(self) -> None:
        self.assertEqual(safe_filename('사업:/이름*?"'), "사업__이름")

    def test_normalize_business_maps_list_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = sample_config(Path(temp_dir))
            row = normalize_business(
                {
                    "code": "626000020223007B",
                    "lafCd": "2600000",
                    "codeNm": "부산",
                    "codeNm2": "본청",
                    "codeNm3": "일반회계",
                    "codeNm4": "테스트 사업",
                    "codeNm5": "환경",
                    "codeNm6": "자연",
                    "rsltYr": "2024",
                    "amt1": 1000,
                    "amt6": 700,
                    "amt7": 300,
                },
                cfg,
                RegionSpec("부산", "26"),
            )
            self.assertEqual(row["business_name"], "테스트 사업")
            self.assertEqual(row["budget_total"], "1000")
            self.assertEqual(row["expenditure"], "700")
            self.assertEqual(row["local_code"], "2600000")
            self.assertEqual(row["local_government"], "부산본청")
            self.assertEqual(row["region_code"], "26")
            self.assertTrue(row["pdf_file"].endswith("626000020223007B.pdf"))

    def test_manifest_round_trip_and_status_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cfg = sample_config(root)
            row = normalize_business(
                {
                    "code": "ABC",
                    "lafCd": "2600000",
                    "codeNm3": "일반회계",
                    "codeNm4": "테스트",
                    "rsltYr": "2024",
                },
                cfg,
                RegionSpec("부산", "26"),
            )
            row["status"] = "success"
            write_manifest(cfg.manifest_path, [row])
            loaded = read_manifest(cfg.manifest_path)
            fresh = dict(row, status="pending")
            merged = merge_manifest([fresh], loaded)
            self.assertEqual(merged[0]["status"], "success")


if __name__ == "__main__":
    unittest.main()
