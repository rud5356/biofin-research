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
    def test_categories_do_not_merge_lower_numbers(self):
        self.assertEqual(b.category({"1차 카테고리": "1.0", "하위 카테고리": "4"}, "subcategory"), "1.04")
        self.assertEqual(b.category({"1차 카테고리": "3", "하위 카테고리": "4"}, "subcategory"), "3.04")
        with self.assertRaises(ValueError):
            b.category({"1차 카테고리": "", "하위 카테고리": "4"}, "primary")

    def test_restart_chooses_next_minority_and_preserves_source(self):
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

    def test_cleanup_recounts_retained_rows_and_protects_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.csv"
            fixture(source)
            original_bytes = source.read_bytes()
            args = arguments(source, tmp, "--cap-bytes", "100000", "--delete-bytes", "30000", "--shard-bytes", "50000", "--min-free-bytes", "0")
            store = b.BatchStore(args, hashlib.sha256(original_bytes).hexdigest())
            try:
                for i in range(20):
                    store.write_batch([{"label": str(i % 2), "text": "x" * 400} for _ in range(50)])
                actual = Counter()
                for path in store.directory.glob("aug-*.jsonl"):
                    actual.update(json.loads(line)["label"] for line in path.read_text().splitlines())
                self.assertEqual(store.counts(), actual)
                self.assertLess(sum(actual.values()), 1000)
                self.assertLessEqual(augment.tree_bytes(Path(tmp)), 100000)
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

    def test_stop_at_balance(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.csv"
            fixture(source)
            args = arguments(source, Path(tmp) / "out", "--stop-when-balanced")
            b.run(args)
            self.assertEqual(len(list(Path(args.output).rglob("aug-*.jsonl"))), 4)


if __name__ == "__main__":
    unittest.main()
