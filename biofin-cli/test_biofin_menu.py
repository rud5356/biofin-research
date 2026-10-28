import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import tempfile
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


class MenuTests(unittest.TestCase):
    def command(self, choices, csv_text="name\nexample\n", integers=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "input.csv").write_text(csv_text, encoding="utf-8")
            for script in ["transformer/v1/augmentation_h200/balanced_csv.py", "llm/v1/classify_biofin_category_with_vllm.py",
                           "llm/v1/classify_biofin_category_with_vllm_V2.py",
                           "llm/v1/classify_biofin_category_with_ollama.py",
                           "transformer/v1/src/train_attention_classifier.py"]:
                path = root / script
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            output = io.StringIO()
            with contextlib.redirect_stdout(output), \
                    patch("sys.argv", ["biofin_menu.py", "--project-dir", folder, "--dry-run"]), \
                    patch.object(menu, "choose", side_effect=choices), \
                    patch.object(menu, "project_path", side_effect=["input.csv", "docs", "results"]), \
                    patch.object(menu, "ask", side_effect=lambda prompt, default=None: str(default)), \
                    patch.object(menu, "integer", side_effect=lambda prompt, default, minimum=1: str((integers or {}).get(default, default))), \
                    patch.object(menu.os, "getuid", return_value=1000, create=True), \
                    patch.object(menu.os, "getgid", return_value=1000, create=True), \
                    patch.object(menu.subprocess, "run") as run:
                self.assertEqual(menu.main(), 0)
                run.assert_not_called()
            commands = [line for line in output.getvalue().splitlines() if line.startswith("docker ")]
            return shlex.split(commands[0])

    def test_real_task_menu_offers_augmentation(self):
        with tempfile.TemporaryDirectory() as folder:
            output = io.StringIO()
            with contextlib.redirect_stdout(output), patch("sys.argv", ["biofin_menu.py", "--project-dir", folder, "--dry-run"]), patch("builtins.input", return_value="4"), patch.object(menu, "augmentation_command", return_value=["docker", "run"]) as command:
                self.assertEqual(menu.main(), 0)
            command.assert_called_once()
            self.assertIn("4. 데이터 증강", output.getvalue())

    def test_augmentation_menu_builds_finite_generation_command(self):
        cmd = self.command([4, 1])
        self.assertIn("transformer/v1/augmentation_h200/balanced_csv.py", cmd)
        self.assertEqual(cmd[cmd.index("--target-per-category") + 1], "400")
        self.assertEqual(cmd[cmd.index("--base-url") + 1], menu.CONFIG["VLLM_URL"])
        self.assertIn("AUG_API_KEY_ENV=VLLM_API_KEY", cmd)
        self.assertNotIn("--until", cmd)

    def test_vllm_api_options(self):
        cmd = self.command([2, 3, 1, 1])
        self.assertIn("llm/v1/classify_biofin_category_with_vllm.py", cmd)
        self.assertIn("VLLM_API_KEY", cmd)
        self.assertIn("--vllm-url", cmd)
        self.assertEqual(cmd[cmd.index("--max-tokens") + 1], "2048")
        self.assertIn("--workers", cmd)
        for option in ["--num-ctx", "--ollama-url", "--max-num-seqs", "--gpu-memory-utilization"]:
            self.assertNotIn(option, cmd)

    def test_vllm_v2_script(self):
        cmd = self.command([2, 3, 2, 2])
        self.assertIn("llm/v1/classify_biofin_category_with_vllm_V2.py", cmd)
        self.assertIn("--vllm-url", cmd)

    def test_ollama_options_preserved(self):
        cmd = self.command([2, 2, 1])
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

    def test_original_only(self):
        cmd = self.command([1, 1, 1, 2, 2])
        self.assertEqual(cmd[cmd.index("--augmentation_per_class") + 1], "0")

    def test_augmented_capped(self):
        cmd = self.command([1, 1, 2, 2, 2, 2], "row_type,training_eligible\noriginal,1\n")
        self.assertEqual(cmd[cmd.index("--augmentation_per_class") + 1], "100")

    def test_augmented_all(self):
        cmd = self.command([1, 1, 2, 1, 2, 2], "row_type,training_eligible\noriginal,1\n")
        self.assertNotIn("--augmentation_per_class", cmd)


class CrawlTests(unittest.TestCase):
    def command(self, choices, answers, existing=()):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "crawlers").mkdir()
            (root / "crawlers" / "openapi.py").touch()
            for name in existing:
                (root / "document" / name).mkdir(parents=True)
            answers = iter(answers)
            output = io.StringIO()
            with contextlib.redirect_stdout(output), \
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
        cmd, text = self.command([3, 1, 2, 1], ["2024", "제주도", "강원 세종", "", "0", "내자료"])
        self.assertEqual(cmd[:9], ["docker", "exec", "-it", "-w", "/work/biofin_cls3", "biofin", "python", "-u",
                                   "crawlers/local_fiscal/crawl_business_docs.py"])
        self.assertEqual(cmd[cmd.index("--regions") + 1:cmd.index("--date")], ["강원", "세종"])
        self.assertEqual(cmd[cmd.index("--date") + 1], "2024-12-31")
        self.assertEqual(cmd[cmd.index("--limit") + 1], "0")
        self.assertEqual(cmd[cmd.index("--manifest") + 1], "document/내자료/내자료.csv")
        self.assertEqual(cmd[cmd.index("--output-dir") + 1], "document/내자료/사업설명자료")
        self.assertIn("--list-only", cmd)
        self.assertNotIn("--browser-channel", cmd)
        self.assertIn("LOFIN365_API_KEY", text)

    def test_local_fiscal_default_folder_and_pdf(self):
        cmd, _ = self.command([3, 1, 2, 2], ["2024", "강원"])
        self.assertEqual(cmd[cmd.index("--manifest") + 1], "document/강원_2024/강원_2024.csv")
        self.assertEqual(cmd[cmd.index("--limit") + 1], "5")
        self.assertNotIn("--list-only", cmd)
        self.assertEqual(cmd[cmd.index("--browser-channel") + 1], "")

    def test_existing_folder_needs_confirmation(self):
        cmd, _ = self.command([3, 1, 1, 1], ["2024", "", "", "", "", "새폴더"], existing=["지방재정_2024"])
        self.assertEqual(cmd[cmd.index("--manifest") + 1], "document/새폴더/새폴더.csv")
        self.assertEqual(len(cmd[cmd.index("--regions") + 1:cmd.index("--date")]), 17)

    def test_open_fiscal_saves_under_year(self):
        cmd, _ = self.command([3, 2, 2], ["2024", "", "-"])
        self.assertIn("crawlers/open_fiscal/crawl_business_docs.py", cmd)
        self.assertEqual(cmd[cmd.index("--save-dir") + 1], "document")
        self.assertEqual(cmd[cmd.index("--browser-channel") + 1], "")

    def test_open_fiscal_api_list(self):
        cmd, _ = self.command([3, 3], ["2023", "", "a/b", "예산목록"])
        self.assertIn("crawlers/open_fiscal/api_budget_list.py", cmd)
        self.assertEqual(cmd[cmd.index("--output") + 1], "document/예산목록/예산목록.csv")

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
