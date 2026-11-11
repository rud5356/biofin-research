import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import tempfile
import time
import unittest
import unittest.mock
from unittest.mock import patch

import biofin_menu as menu


class ConfigTests(unittest.TestCase):
    def test_default_config_is_independent_of_working_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                self.assertEqual(menu.load_config(), menu.CONFIG)
            finally:
                os.chdir(original)

    def test_load_custom_values_with_utf8_bom(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            values = dict(menu.CONFIG, WORKERS=4, PROJECT_DIR=None)
            path.write_text(json.dumps(values), encoding="utf-8-sig")
            self.assertEqual(menu.load_config(path), values)

    def test_missing_invalid_and_non_object_config(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            with self.assertRaises(ValueError):
                menu.load_config(path)
            for text in ['{"WORKERS":', '[]']:
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    menu.load_config(path)

    def test_invalid_defaults_fail_before_command(self):
        for values in ({"AUGMENTATION_TARGET": 51}, {"WORKERS": 0},
                       {"CRAWL_REGIONS": ["없는지역"]}, {"BAR_UNIT": "달러"}):
            with patch.dict(menu.DEFAULTS, values), self.assertRaises(RuntimeError):
                menu.validate_defaults()

    def test_comments_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text('{"_comment_VALUE": "설명", "VALUE": 3}', encoding="utf-8")
            self.assertEqual(menu.load_config(path), {"VALUE": 3})


class DocumentSelectionTests(unittest.TestCase):
    def select(self, root, choices):
        with patch.object(menu, "choose", side_effect=choices), patch.object(menu, "integer", return_value="2024"), contextlib.redirect_stdout(io.StringIO()):
            return menu.select_document_dir(root)

    def test_open_and_all_local_regions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for region in [None, *menu.REGIONS]:
                docs = root / menu.dataset_path("2024", region) / "사업설명자료"
                docs.mkdir(parents=True)
                (docs / "example.pdf").touch()
                choices = [1] if region is None else [2, menu.REGIONS.index(region) + 1]
                self.assertEqual(self.select(root, choices), docs.relative_to(root).as_posix())
            self.assertEqual(len(set(menu.CONFIG["REGION_CODES"].values())), 17)

    def test_requires_document_subfolder(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, "폴더가 없습니다"):
                self.select(root, [1])
            docs = root / menu.dataset_path("2024")
            docs.mkdir(parents=True)
            (docs / "list.csv").touch()
            with self.assertRaisesRegex(RuntimeError, "폴더가 없습니다"):
                self.select(root, [1])
            (docs / "example.hwpx").touch()
            with self.assertRaisesRegex(RuntimeError, "폴더가 없습니다"):
                self.select(root, [1])
            docs = docs / "사업설명자료"
            docs.mkdir()
            with self.assertRaisesRegex(RuntimeError, "사업설명자료가 없습니다"):
                self.select(root, [1])
            (docs / "example.pdf").touch()
            self.assertEqual(self.select(root, [1]), docs.relative_to(root).as_posix())

    def test_csv_single_multiple_and_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, "CSV 파일이 없습니다"):
                menu.select_csv(root, ".")
            (root / "one.CSV").touch()
            (root / "사업설명자료").mkdir()
            (root / "사업설명자료" / "nested.csv").touch()
            with patch.object(menu, "choose") as choose:
                self.assertEqual(menu.select_csv(root, "."), "one.CSV")
                choose.assert_not_called()
            (root / "two.csv").touch()
            with patch.object(menu, "choose", return_value=2) as choose:
                self.assertEqual(menu.select_csv(root, "."), "two.csv")
                choose.assert_called_once_with("입력 CSV 선택", ["one.CSV", "two.csv"])

    def test_custom_root_and_region_code(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(menu.CONFIG, {"DOCUMENT_DIR": "자료"}):
            with patch.dict(menu.CONFIG["REGION_CODES"], {"강원": "gangwon-do"}):
                root = Path(folder)
                docs = root / "자료/지방재정/2024/gangwon-do/사업설명자료"
                docs.mkdir(parents=True)
                (docs / "example.pdf").touch()
                self.assertEqual(self.select(root, [2, menu.REGIONS.index("강원") + 1]), docs.relative_to(root).as_posix())


class MenuTests(unittest.TestCase):
    def test_result_directory_auto_name_and_increment(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            first = menu.result_directory(root, "llm", "document/열린재정/2024/사업설명자료")
            self.assertRegex(first, r"llm/v1/outputs/\d{8}_2024_열린재정_1$")
            self.assertFalse((root / first).exists())
            (root / first).mkdir(parents=True)
            (root / (first[:-1] + "3")).touch()
            self.assertEqual(menu.result_directory(root, "llm", "document/열린재정/2024/사업설명자료"), first[:-1] + "4")
            local = menu.result_directory(root, "transformer", "document/지방재정/2023/gangwon/사업설명자료")
            self.assertRegex(local, r"transformer/v1/outputs/\d{8}_2023_지방재정_1$")

    def command(self, choices, csv_text="name\nexample\n", integers=None, defaults=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "input.csv").write_text(csv_text, encoding="utf-8")
            (root / menu.CONFIG["TRANSFORMER_MODEL_DIR"]).mkdir(parents=True)
            for script in ["transformer/v1/augmentation_h200/balanced_csv.py", "llm/v1/classify_biofin_category_with_vllm.py",
                           "llm/v1/classify_biofin_category_with_vllm_V2.py",
                           "llm/v1/classify_biofin_category_with_ollama.py",
                           "transformer/v1/src/train_attention_classifier.py",
                           "transformer/v1/src/predict_attention_classifier.py"]:
                path = root / script
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            output = io.StringIO()
            with patch.dict(menu.DEFAULTS, defaults or {}), contextlib.redirect_stdout(output), \
                    patch("sys.argv", ["biofin_menu.py", "--project-dir", folder, "--dry-run"]), \
                    patch.object(menu, "choose", side_effect=choices), \
                    patch.object(menu, "select_document_dir", return_value="document/열린재정/2024/사업설명자료"), \
                    patch.object(menu, "select_csv", return_value="input.csv"), \
                    patch.object(menu, "project_path", side_effect=["input.csv", "docs", "results"]) as browse, \
                    patch.object(menu, "ask", side_effect=lambda prompt, default=None: str(default)), \
                    patch.object(menu, "integer", side_effect=lambda prompt, default, minimum=1: str((integers or {}).get(default, default))), \
                    patch.object(menu.os, "getuid", return_value=1000, create=True), \
                    patch.object(menu.os, "getgid", return_value=1000, create=True), \
                    patch.object(menu.subprocess, "run") as run:
                self.assertEqual(menu.main(), 0)
                run.assert_not_called()
                if choices[0] != 4:
                    browse.assert_not_called()
            commands = [line for line in output.getvalue().splitlines() if line.startswith("docker ")]
            return shlex.split(commands[0])

    def test_real_task_menu_offers_augmentation(self):
        with tempfile.TemporaryDirectory() as folder:
            output = io.StringIO()
            with contextlib.redirect_stdout(output), patch("sys.argv", ["biofin_menu.py", "--project-dir", folder, "--dry-run"]), patch("builtins.input", return_value="4"), patch.object(menu, "augmentation_command", return_value=["docker", "run"]) as command:
                self.assertEqual(menu.main(), 0)
            command.assert_called_once()
            self.assertIn("4. 데이터 증강", output.getvalue())

    def test_prediction_uses_configured_model_without_folder_prompt(self):
        cmd = self.command([1, 1])
        self.assertEqual(cmd[cmd.index("--model_dir") + 1], menu.CONFIG["TRANSFORMER_MODEL_DIR"])
        with patch.dict(menu.CONFIG, {"TRANSFORMER_MODEL_DIR": "transformer/v1/outputs/other_model"}):
            cmd = self.command([1, 1])
            self.assertEqual(cmd[cmd.index("--model_dir") + 1], "transformer/v1/outputs/other_model")

    def test_augmentation_menu_builds_finite_generation_command(self):
        cmd = self.command([4, 1])
        self.assertIn("transformer/v1/augmentation_h200/balanced_csv.py", cmd)
        self.assertEqual(cmd[cmd.index("--target-per-category") + 1], "400")
        self.assertEqual(cmd[cmd.index("--base-url") + 1], menu.CONFIG["VLLM_URL"])
        self.assertIn("AUG_API_KEY_ENV=VLLM_API_KEY", cmd)
        self.assertNotIn("--until", cmd)
        self.assertTrue(cmd[cmd.index("--merged-output-dir") + 1].endswith("/merged_training"))

    def test_custom_defaults_are_inserted(self):
        cmd = self.command([1, 3], defaults={"MAX_TOKENS": 512, "WORKERS": 4, "RETRIES": 0})
        for option, value in [("--max-tokens", "512"), ("--workers", "4"), ("--retries", "0")]:
            self.assertEqual(cmd[cmd.index(option) + 1], value)
        cmd = self.command([4], defaults={"AUGMENTATION_TARGET": 200})
        self.assertEqual(cmd[cmd.index("--target-per-category") + 1], "200")
        cmd = self.command([2, 1, 1], defaults={"TRAIN_EPOCHS": 9})
        self.assertEqual(cmd[cmd.index("--epochs") + 1], "9")

    def test_vllm_api_options(self):
        cmd = self.command([1, 3, 1])
        self.assertIn("llm/v1/classify_biofin_category_with_vllm_V2.py", cmd)
        self.assertIn("VLLM_API_KEY", cmd)
        self.assertIn("--vllm-url", cmd)
        self.assertEqual(cmd[cmd.index("--max-tokens") + 1], "2048")
        self.assertIn("--workers", cmd)
        for option in ["--num-ctx", "--ollama-url", "--max-num-seqs", "--gpu-memory-utilization"]:
            self.assertNotIn(option, cmd)

    def test_vllm_v2_script(self):
        cmd = self.command([1, 3, 2])
        self.assertIn("llm/v1/classify_biofin_category_with_vllm_V2.py", cmd)
        self.assertNotIn("llm/v1/classify_biofin_category_with_vllm.py", cmd)
        self.assertIn("--vllm-url", cmd)

    def test_ollama_options_preserved(self):
        cmd = self.command([1, 2, 1])
        self.assertIn("--ollama-url", cmd)
        self.assertIn("--num-ctx", cmd)
        self.assertNotIn("--max-tokens", cmd)

    def test_prediction_chains_bar_after_classification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "results").mkdir()
            (root / "results" / "input_llm_classified.csv").touch()
            ok = unittest.mock.Mock(returncode=0)
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(menu.subprocess, "run", return_value=ok) as run:
                self.assertEqual(menu.run_bar(root, "results/input_llm_classified.csv", "예산액,지출액", "백만원", False), 0)
            cmd = run.call_args[0][0]
            self.assertEqual(cmd[1], str(root / "bar/apply_bar.py"))
            self.assertEqual(cmd[cmd.index("--input") + 1], str(root / "results/input_llm_classified.csv"))
            self.assertEqual(cmd[cmd.index("--amount-cols") + 1], "예산액,지출액")
            self.assertEqual(cmd[cmd.index("--unit") + 1], "백만원")
            with contextlib.redirect_stdout(io.StringIO()), patch.object(menu.subprocess, "run") as run:
                self.assertEqual(menu.run_bar(root, "results/missing.csv", "예산액", "원", False), 0)
                run.assert_not_called()

    def test_transformer_bar_runs_in_container(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "results").mkdir()
            (root / "results" / "classified.csv").touch()
            ok = unittest.mock.Mock(returncode=0)
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(menu.subprocess, "run", return_value=ok) as run:
                self.assertEqual(menu.run_bar(root, "results/classified.csv", "예산액", "백만원", False,
                                              ("biofin", "/work/biofin_cls3")), 0)
            cmd = run.call_args[0][0]
            self.assertEqual(cmd[:7], ["docker", "exec", "-w", "/work/biofin_cls3", "biofin", "python", "bar/apply_bar.py"])
            self.assertEqual(cmd[cmd.index("--input") + 1], "results/classified.csv")
            self.assertEqual(cmd[cmd.index("--rates") + 1], menu.CONFIG["BAR_RATES"])

    def test_keep_final_results_after_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            results = Path(folder)
            old = results / "old_note.txt"
            old.touch()
            os.utime(old, (1, 1))
            for name in ["classified.csv", "classified_bar.csv", "attention_heatmap.html", "attention_outputs.csv",
                         "prediction_summary.json", "classified_bar_summary.csv", "run_summary.json"]:
                (results / name).touch()
            with contextlib.redirect_stdout(io.StringIO()):
                menu.keep_final_results(results, time.time() - 10, False)
            self.assertEqual(sorted(p.name for p in results.iterdir()),
                             ["attention_heatmap.html", "classified_bar.csv", "old_note.txt"])

    def test_keep_final_results_skips_without_bar(self):
        with tempfile.TemporaryDirectory() as folder:
            results = Path(folder)
            (results / "category_label_cache.csv").touch()
            with contextlib.redirect_stdout(io.StringIO()):
                menu.keep_final_results(results, time.time() - 10, False)
            self.assertTrue((results / "category_label_cache.csv").exists())

    def test_restore_owner_after_docker_exec(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ok = unittest.mock.Mock(returncode=0)
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(menu.os, "getuid", return_value=1000, create=True), \
                    patch.object(menu.os, "getgid", return_value=1001, create=True), \
                    patch.object(menu.subprocess, "run", return_value=ok) as run:
                self.assertEqual(menu.restore_owner(root, "biofin", "/work/biofin_cls3", "results", False), 0)
                run.assert_not_called()
                (root / "results").mkdir()
                self.assertEqual(menu.restore_owner(root, "biofin", "/work/biofin_cls3", "results", False), 0)
            self.assertEqual(run.call_args[0][0],
                             ["docker", "exec", "biofin", "chown", "-R", "1000:1001", "/work/biofin_cls3/results"])

    def test_original_only(self):
        cmd = self.command([2, 1, 1, 2, 2])
        self.assertEqual(cmd[cmd.index("--augmentation_per_class") + 1], "0")

    def test_augmented_capped(self):
        cmd = self.command([2, 1, 2], "row_type,training_eligible\noriginal,1\n", defaults={"TRAIN_AUGMENTATION_PER_CLASS": 100})
        self.assertEqual(cmd[cmd.index("--augmentation_per_class") + 1], "100")

    def test_augmented_all(self):
        cmd = self.command([2, 1, 2, 1, 2, 2], "row_type,training_eligible\noriginal,1\n")
        self.assertNotIn("--augmentation_per_class", cmd)


class CrawlTests(unittest.TestCase):
    def command(self, choices, answers, existing=(), defaults=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "crawlers").mkdir()
            (root / "crawlers" / "openapi.py").touch()
            for name in existing:
                (root / "document" / name).mkdir(parents=True)
            answers = iter(answers)
            output = io.StringIO()
            with patch.dict(menu.DEFAULTS, dict(CRAWL_YEAR=2024, **(defaults or {}))), contextlib.redirect_stdout(output), \
                    patch("sys.argv", ["biofin_menu.py", "--project-dir", folder, "--dry-run"]), \
                    patch.object(menu, "choose", side_effect=choices), \
                    patch("builtins.input", side_effect=lambda prompt: next(answers, "")), \
                    patch.object(menu.subprocess, "run") as run:
                self.assertEqual(menu.main(), 0)
                run.assert_not_called()
            lines = output.getvalue().splitlines()
            cmd = shlex.split(lines[lines.index("실행할 명령어:") + 1])
            return cmd, output.getvalue()

    def test_local_fiscal_region_and_folder(self):
        cmd, _ = self.command([3, 1, 1], ["2024"], defaults={"CRAWL_REGIONS": ["강원", "세종"], "CRAWL_LIMIT": 0})
        self.assertEqual(cmd[cmd.index("--regions") + 1:cmd.index("--date")], ["강원"])
        self.assertEqual(cmd[cmd.index("--date") + 1], "2024-12-31")
        self.assertEqual(cmd[cmd.index("--limit") + 1], "0")
        self.assertIn("--list-only", cmd)
        self.assertIn("document/지방재정/2024/gangwon/2024_gangwon.csv", cmd)

    def test_local_fiscal_default_folder_and_pdf(self):
        cmd, _ = self.command([3, 1, 1], [], defaults={"CRAWL_REGIONS": ["강원"], "CRAWL_LIST_ONLY": False})
        self.assertIn("document/지방재정/2024/gangwon/2024_gangwon.csv", cmd)
        self.assertNotIn("--list-only", cmd)
        self.assertEqual(cmd[cmd.index("--browser-channel") + 1], "")

    def test_existing_folder_reuses_standard_path(self):
        cmd, text = self.command([3, 1, 1], [""], existing=["지방재정/2024/seoul"])
        self.assertIn("document/지방재정/2024/seoul/2024_seoul.csv", cmd)
        self.assertIn("기존 수집 폴더", text)

    def test_open_fiscal_saves_under_year(self):
        cmd, _ = self.command([3, 2], ["2023"], defaults={"CRAWL_LIST_ONLY": False})
        self.assertIn("crawlers/open_fiscal/crawl_business_docs.py", cmd)
        self.assertEqual(cmd[cmd.index("--save-dir") + 1], "document/열린재정")
        self.assertEqual(cmd[cmd.index("--year") + 1], "2023")

    def test_open_fiscal_api_list(self):
        cmd, _ = self.command([3, 3], ["2024"])
        self.assertIn("document/열린재정/2024/open_fiscal_api_2024.csv", cmd)
        self.assertEqual(cmd[cmd.index("--limit") + 1], "0")

    def test_container_setup(self):
        ok, fail = unittest.mock.Mock(returncode=0), unittest.mock.Mock(returncode=1)
        with contextlib.redirect_stdout(io.StringIO()):
            with patch.object(menu.subprocess, "run") as run:
                self.assertTrue(menu.prepare_container("biofin", []))
                run.assert_not_called()
            with patch.object(menu.subprocess, "run", return_value=ok) as run:
                self.assertTrue(menu.prepare_container("biofin", ["playwright"]))
                self.assertEqual(run.call_count, 1)
            with patch.object(menu.subprocess, "run", return_value=fail) as run, \
                    patch.object(menu, "ask", return_value="n"):
                self.assertFalse(menu.prepare_container("biofin", ["playwright"]))
                self.assertEqual(run.call_count, 1)
            with patch.object(menu.subprocess, "run", side_effect=[fail, ok, ok]) as run, \
                    patch.object(menu, "ask", return_value="y"):
                self.assertTrue(menu.prepare_container("biofin", ["pandas", "playwright"]))
                self.assertEqual(run.call_args_list[1][0][0],
                                 ["docker", "exec", "biofin", "python", "-m", "pip", "install", "pandas", "playwright"])
                self.assertEqual(run.call_args_list[2][0][0][3:],
                                 ["python", "-m", "playwright", "install", "--with-deps", "chromium"])


if __name__ == "__main__":
    unittest.main()
