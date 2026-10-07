import contextlib
import io
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

import biofin_menu as menu


class MenuTests(unittest.TestCase):
    def command(self, choices, csv_text="name\nexample\n", integers=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "input.csv").write_text(csv_text, encoding="utf-8")
            for script in ["llm/v1/classify_biofin_category_with_vllm.py",
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

    def test_vllm_api_options(self):
        cmd = self.command([2, 3])
        self.assertIn("llm/v1/classify_biofin_category_with_vllm.py", cmd)
        self.assertIn("VLLM_API_KEY", cmd)
        self.assertIn("--vllm-url", cmd)
        self.assertEqual(cmd[cmd.index("--max-tokens") + 1], "2048")
        self.assertIn("--workers", cmd)
        for option in ["--num-ctx", "--ollama-url", "--max-num-seqs", "--gpu-memory-utilization"]:
            self.assertNotIn(option, cmd)

    def test_ollama_options_preserved(self):
        cmd = self.command([2, 2])
        self.assertIn("--ollama-url", cmd)
        self.assertIn("--num-ctx", cmd)
        self.assertNotIn("--max-tokens", cmd)

    def test_original_only(self):
        cmd = self.command([1, 1, 1, 2, 2])
        self.assertEqual(cmd[cmd.index("--augmentation_per_class") + 1], "0")

    def test_augmented_capped(self):
        cmd = self.command([1, 1, 2, 2, 2, 2], "row_type,training_eligible\noriginal,1\n")
        self.assertEqual(cmd[cmd.index("--augmentation_per_class") + 1], "100")

    def test_augmented_all(self):
        cmd = self.command([1, 1, 2, 1, 2, 2], "row_type,training_eligible\noriginal,1\n")
        self.assertNotIn("--augmentation_per_class", cmd)


if __name__ == "__main__":
    unittest.main()
