from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import augment
import balanced_csv as b


def fixture(path):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["1차 카테고리", "하위 카테고리", "세부사업명", "business_key", "판단근거"])
        writer.writeheader()
        for first, lower, count in [("0", "0", 100), ("1", "4", 1), ("3", "4", 2)]:
            for i in range(count):
                writer.writerow({"1차 카테고리": first, "하위 카테고리": lower,
                                 "세부사업명": f"사업 {first} {i}", "business_key": f"{first}-{i}",
                                 "판단근거": "DO NOT INCLUDE IN TEXT"})


def arguments(source, output, *extra):
    return b.parser().parse_args(["--input", str(source), "--output", str(output), "--mock",
                                 "--until", "2099-01-01T00:00:00+09:00", "--interval", "0", *extra])


class BalancedTests(unittest.TestCase):
    def test_primary_only_csv_and_merge(self):
        from build_merged_training_csv import build
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.csv"
            fields = ["business_key", "세부사업명", "BIOFIN 1차 카테고리", "예산액"]
            with source.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows([
                    dict(zip(fields, ["a", "보전 사업", "2", "100"])),
                    dict(zip(fields, ["b", "미분류 사업", "", "200"]))])
            groups, fingerprint, excluded = b.load_csv(source, "primary")
            self.assertEqual(set(groups), {"2"})
            self.assertEqual(excluded, 1)
            self.assertEqual(groups["2"][0]["row"]["하위 카테고리"], "")
            self.assertIn("예산액: 100", groups["2"][0]["text"])
            args = arguments(source, Path(tmp) / "out", "--max-batches", "1")
            b.run(args)
            shard = next(Path(args.output).rglob("aug-*.jsonl"))
            rows = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                row["backend"] = "vllm"
            shard.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            report = build(source, source, shard.parent, Path(tmp) / "merged")
            self.assertEqual(report["counts"]["augmentation_kept_rows"], 50)
            self.assertEqual(report["counts"]["unlabeled_original_rows"], 1)
            with self.assertRaises(ValueError):
                b.load_csv(source, "subcategory")

    def test_alias_conflicts_and_invalid_labels_rejected(self):
        self.assertEqual(b.normalize_labels({"biofin_category": "3.0"})["1차 카테고리"], "3")
        for row in [{"biofin_category": "invalid"}, {"biofin_category": "10"},
                    {"biofin_category": "1", "BIOFIN 1차 카테고리": "2"}]:
            with self.assertRaises(ValueError):
                b.normalize_labels(row)

    def test_progress_heartbeat_and_final_elapsed(self):
        import threading
        progress = b.Progress(0.01)
        emitted = threading.Event()
        with patch.object(progress, "report", side_effect=emitted.set):
            progress.thread.start()
            try:
                self.assertTrue(emitted.wait(2))
            finally:
                progress.stopped.set()
                progress.thread.join()
        self.assertFalse(progress.thread.is_alive())
        self.assertEqual(b.duration(3661), "01:01:01")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.csv"
            fixture(source)
            args = arguments(source, Path(tmp) / "out", "--target-per-category", "50")
            with self.assertLogs("balanced", level="INFO") as captured:
                b.run(args)
            messages = "\\n".join(captured.output)
            self.assertIn("목표 달성", messages)
            self.assertIn("원본+증강=150/150 (100.0%)", messages)
            self.assertIn("총 실행 시간=", messages)
            self.assertFalse(hasattr(args, "progress"))

    def test_categories_do_not_merge_lower_numbers(self):
        self.assertEqual(b.category({"1차 카테고리": "1.0", "하위 카테고리": "4"}, "subcategory"), "1.04")
        self.assertEqual(b.category({"1차 카테고리": "3", "하위 카테고리": "4"}, "subcategory"), "3.04")
        with self.assertRaises(ValueError):
            b.category({"1차 카테고리": "", "하위 카테고리": "4"}, "primary")

    def test_restart_chooses_least_generated_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "source.csv", Path(tmp) / "out"
            fixture(source)
            original_bytes = source.read_bytes()
            args = arguments(source, out, "--max-batches", "1")
            b.run(args)
            b.run(args)
            files = list((out / "balanced_h200_v1").glob("aug-*.jsonl"))
            self.assertEqual(len(files), 2)
            counts = Counter()
            for path in files:
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(rows), 50)
                self.assertEqual(len(set(row["text"] for row in rows)), 50)
                self.assertTrue(all("DO NOT INCLUDE" not in row["text"] for row in rows))
                counts.update(row["label"] for row in rows)
            # Smallest total first: 1 (1 row), then 3 (2 rows); 0 (100 rows) waits.
            self.assertEqual(counts, {"1": 50, "3": 50})
            self.assertEqual(source.read_bytes(), original_bytes)

    def test_failed_generation_saves_no_partial_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.csv"
            fixture(source)
            args = arguments(source, Path(tmp) / "out", "--max-failures", "1")
            with patch("augment.generate", side_effect=[["첫 생성문"], RuntimeError("offline")]):
                with self.assertRaises(RuntimeError):
                    b.run(args)
            self.assertEqual(list(Path(args.output).rglob("aug-*.jsonl")), [])

    def test_insufficient_space_preserves_generated_files_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.csv"
            fixture(source)
            original_bytes = source.read_bytes()
            args = arguments(source, tmp, "--cap-bytes", "100000", "--delete-bytes", "30000", "--shard-bytes", "50000", "--min-free-bytes", "0")
            store = b.BatchStore(args, hashlib.sha256(original_bytes).hexdigest())
            try:
                with self.assertRaises(RuntimeError):
                    for i in range(20):
                        store.write_batch([{"label": str(i % 2), "text": "x" * 400} for _ in range(50)])
                files = list(store.directory.glob("aug-*.jsonl"))
                self.assertGreater(len(files), 0)
                before = {p: p.read_bytes() for p in files}
                with self.assertRaises(RuntimeError):
                    store.ensure_space(100000)
                self.assertEqual(before, {p: p.read_bytes() for p in files})
                self.assertEqual(source.read_bytes(), original_bytes)
            finally:
                store.close()

    def test_different_source_rejected_before_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.csv"
            fixture(source)
            args = arguments(source, Path(tmp) / "out", "--max-batches", "1")
            b.run(args)
            files = list(Path(args.output).rglob("aug-*.jsonl"))
            with source.open("a", encoding="utf-8") as file:
                file.write("\n")
            with self.assertRaises(RuntimeError):
                b.run(args)
            self.assertTrue(all(path.exists() for path in files))

    def test_stop_at_400_total_per_category_and_restart_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.csv"
            fixture(source)
            args = arguments(source, Path(tmp) / "out")
            args.until = None
            b.run(args)
            counts = Counter()
            for path in Path(args.output).rglob("aug-*.jsonl"):
                counts.update(json.loads(line)["label"] for line in path.read_text(encoding="utf-8").splitlines())
            # Originals 100/1/2 are topped up to 400 in total; last batch is smaller than 50.
            self.assertEqual(counts, {"0": 300, "1": 399, "3": 398})
            with patch("augment.generate") as generate:
                b.run(args)
                generate.assert_not_called()

    def test_completed_generation_builds_merged_csv_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, merged = Path(tmp) / "source.csv", Path(tmp) / "merged"
            fixture(source)
            args = arguments(source, Path(tmp) / "out", "--target-per-category", "50",
                             "--merged-output-dir", str(merged))
            b.run(args)
            report = json.loads((merged / "merge_report.json").read_text(encoding="utf-8"))
            self.assertTrue((merged / "open_fiscal_2023_augmented.csv").exists())
            self.assertEqual(report["counts"]["original_rows"], 103)
            b.run(args)  # Rerun after completion keeps the existing merged folder.

    def test_batch_larger_than_remaining_does_not_overshoot(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.csv"
            fixture(source)
            args = arguments(source, Path(tmp) / "out", "--target-per-category", "100",
                             "--balance-batch-size", "150")
            b.run(args)
            counts = Counter()
            for path in Path(args.output).rglob("aug-*.jsonl"):
                counts.update(json.loads(line)["label"] for line in path.read_text(encoding="utf-8").splitlines())
            self.assertEqual(counts, {"1": 99, "3": 98})


if __name__ == "__main__":
    unittest.main()
