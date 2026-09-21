#!/usr/bin/env python3
"""Read-only BIOFIN CSV input; least-populated category first in complete 50-row batches."""
from collections import Counter, defaultdict
import csv
import datetime as dt
from decimal import Decimal, InvalidOperation
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import signal
import time
import uuid

import augment

LOG = logging.getLogger("balanced")
METADATA = (
    "회계연도", "소관명", "회계명", "계정명", "분야명", "부문명",
    "프로그램명", "단위사업명", "세부사업명", "경비구분", "지출구분",
    "정부안금액(천원)", "국회확정금액(천원)",
)


def code(value):
    try:
        number = Decimal(str(value).strip())
        if not number.is_finite() or number != number.to_integral_value() or number < 0:
            raise ValueError("Category must be a nonnegative integer")
        return str(int(number))
    except (InvalidOperation, ValueError):
        raise ValueError(f"Invalid category: {value!r}")


def category(row, level):
    first = code(row["1차 카테고리"])
    lower = code(row["하위 카테고리"])
    if int(first) > 9 or (first == "0" and lower != "0"):
        raise ValueError(f"Invalid category pair: {first}/{lower}")
    if level == "primary" or first == "0":
        return first
    if not 1 <= int(lower) <= 99:
        raise ValueError(f"Invalid subcategory: {first}/{lower}")
    return f"{first}.{int(lower):02d}"


def load_csv(path, level, encoding="utf-8-sig"):
    path = Path(path)
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    groups = defaultdict(list)
    excluded = 0
    keys = set()
    with path.open(encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)
        required = {"1차 카테고리", "하위 카테고리", "세부사업명", "business_key"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns: {sorted(required - set(reader.fieldnames or []))}")
        for row_number, row in enumerate(reader, 2):
            if "split" in row and row["split"] != "train":
                excluded += 1
                continue
            key = row["business_key"].strip()
            if not key or key in keys:
                raise ValueError(f"Empty/duplicate business_key at CSV record {row_number}")
            keys.add(key)
            label = category(row, level)
            text = "\n".join(f"{name}: {row[name].strip()}" for name in METADATA if row.get(name, "").strip())
            if not row["세부사업명"].strip():
                raise ValueError(f"Missing business name at CSV record {row_number}")
            groups[label].append({"row": row, "text": text, "source_row": row_number})
    if not groups:
        raise ValueError("No eligible source rows")
    return dict(groups), fingerprint, excluded


def choose_category(original, retained):
    return min(original, key=lambda label: (original[label] + retained[label], Decimal(label)))


def make_plan(original, retained, batch_size):
    totals = {label: original[label] + retained[label] for label in original}
    target = max(totals.values())
    return [{"category": label, "original": original[label], "retained_generated": retained[label],
             "total": totals[label], "initial_target": target,
             "generate_to_target": ((max(0, target - totals[label]) + batch_size - 1) // batch_size) * batch_size}
            for label in sorted(original, key=lambda label: (totals[label], Decimal(label)))]


class BatchStore(augment.Store):
    """One immutable, atomically published shard per completed category batch."""
    def __init__(self, args, fingerprint):
        super().__init__(args.output, args.cap_bytes, args.delete_bytes,
                         args.shard_bytes, args.min_free_bytes,
                         directory_name="balanced_h200_v1")
        self.index = {}
        self.identity = {"source_sha256": fingerprint, "level": args.category_level,
                         "mode": "mock" if args.mock else "real", "text_mode": "csv_metadata_v1"}
        try:
            metadata = self.directory / ".dataset.json"
            if metadata.is_symlink():
                raise RuntimeError("Dataset identity must not be a symlink")
            if metadata.exists():
                if json.loads(metadata.read_text(encoding="utf-8")) != self.identity:
                    raise RuntimeError("Different input/category/mode: use a separate output directory")
            else:
                if any(self.directory.glob("aug-*.jsonl")):
                    raise RuntimeError("Shards exist without dataset identity")
                with metadata.open("x", encoding="utf-8") as file:
                    json.dump(self.identity, file)
            # Only our own uncommitted temporary outputs can be discarded after a crash.
            for path in self.directory.glob("aug-*.pending"):
                if augment.SHARD.fullmatch(path.with_suffix(".jsonl").name) and not path.is_symlink():
                    path.unlink()
        except BaseException:
            self.close()
            raise

    def counts(self):
        current = set()
        counts = Counter()
        for path in self.directory.glob("aug-*.jsonl"):
            if path.is_symlink() or not augment.SHARD.fullmatch(path.name):
                raise RuntimeError(f"Unexpected output file: {path}")
            current.add(path)
            if path not in self.index:
                with path.open(encoding="utf-8") as file:
                    first = json.loads(file.readline())
                meta = first["balance_batch"]
                if meta["identity"] != self.identity or meta["size"] < 50 or meta["size"] % 50:
                    raise RuntimeError(f"Invalid completed batch: {path}")
                self.index[path] = (first["label"], meta["size"])
            label, count = self.index[path]
            counts[label] += count
        self.index = {path: value for path, value in self.index.items() if path in current}
        return counts

    def write_batch(self, records):
        if len(records) < 50 or len(records) % 50 or len({row["label"] for row in records}) != 1:
            raise ValueError("A batch must contain one category and a multiple of 50 rows")
        records[0]["balance_batch"] = {"size": len(records), "identity": self.identity}
        payload = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in records)
        if len(payload) > self.shard_bytes:
            raise ValueError("Completed batch exceeds --shard-bytes")
        self.ensure_space(len(payload))
        target = self.directory / f"aug-{uuid.uuid4().hex}.jsonl"
        temporary = target.with_suffix(".pending")
        try:
            with temporary.open("xb") as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, target)
            if os.name == "posix":
                fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.index[target] = (records[0]["label"], len(records))


def collect_batch(args, label, seeds, fingerprint, deadline):
    accepted = []
    seen = {" ".join(seed["text"].split()) for seed in seeds}
    shuffled = list(seeds)
    random.shuffle(shuffled)
    attempts = failures = 0
    while len(accepted) < args.balance_batch_size:
        if augment.STOP or time.time() >= deadline:
            raise augment.Deadline()
        if attempts >= args.max_requests_per_batch:
            raise RuntimeError("Too few distinct valid results to finish a batch; no partial batch saved")
        seed = shuffled[attempts % len(shuffled)]
        attempts += 1
        try:
            variants = augment.generate(args, seed["text"], label, deadline - time.time())
            before = len(accepted)
            for text in variants:
                if not isinstance(text, str):
                    continue
                clean = " ".join(text.split())
                if not clean or clean in seen or len(text) > args.max_chars:
                    continue
                seen.add(clean)
                row = seed["row"]
                accepted.append({"text": text.strip(), "label": label,
                    "primary_category": code(row["1차 카테고리"]),
                    "subcategory": code(row["하위 카테고리"]),
                    "source_business_key": row["business_key"],
                    "source_row": seed["source_row"], "source_csv_sha256": fingerprint,
                    "source_text_sha256": hashlib.sha256(seed["text"].encode()).hexdigest(),
                    "source_document_relative_path": row.get("사업설명자료_상대경로", ""),
                    "text_basis": "csv_metadata", "needs_review": True,
                    "source_split": row.get("split", "unassigned"),
                    "backend": "mock" if args.mock else args.backend, "model": args.model,
                    "created_at": dt.datetime.now(dt.timezone.utc).isoformat()})
                if len(accepted) == args.balance_batch_size:
                    break
            if len(accepted) == before:
                raise ValueError("No usable distinct text")
            failures = 0
            LOG.info("category=%s pending=%d/%d", label, len(accepted), args.balance_batch_size)
        except augment.Deadline:
            raise
        except Exception as exc:
            failures += 1
            LOG.warning("category=%s request_failed=%s consecutive=%d", label, type(exc).__name__, failures)
            if failures >= args.max_failures:
                raise RuntimeError("Repeated API/validation failure; no partial batch saved") from exc
            time.sleep(max(0, min(2 ** min(failures, 6), deadline - time.time())))
        time.sleep(max(0, min(args.interval, deadline - time.time())))
    return accepted


def run(args):
    groups, fingerprint, excluded = load_csv(args.input, args.category_level, args.encoding)
    original = Counter({label: len(seeds) for label, seeds in groups.items()})
    if any(len(seed["text"]) > args.max_chars for seeds in groups.values() for seed in seeds):
        raise ValueError("Source metadata exceeds --max-chars; increase the limit")
    if args.plan:
        print(json.dumps({"source_sha256": fingerprint, "category_level": args.category_level,
                          "excluded_nontrain": excluded, "rows": make_plan(original, Counter(), args.balance_batch_size)},
                         ensure_ascii=False, indent=2))
        return
    deadline = augment.parse_deadline(args.until)
    if time.time() >= deadline:
        LOG.info("Deadline reached; no output changed")
        return
    source = Path(args.input).resolve()
    if source.is_relative_to(Path(args.output).resolve() / "balanced_h200_v1"):
        raise ValueError("Source CSV cannot reside in the generated output directory")
    store = BatchStore(args, fingerprint)
    batches = 0
    try:
        store.ensure_space(0)
        while not augment.STOP and time.time() < deadline:
            retained = store.counts()
            unknown = set(retained) - set(original)
            if unknown:
                raise RuntimeError(f"Unexpected categories in saved data: {unknown}")
            totals = [original[label] + retained[label] for label in original]
            if args.stop_when_balanced and min(totals) >= max(original.values()) and max(totals) - min(totals) < args.balance_batch_size:
                LOG.info("Balanced within one batch: min=%d max=%d", min(totals), max(totals))
                return
            label = choose_category(original, retained)
            LOG.info("selected=%s original=%d retained=%d", label, original[label], retained[label])
            try:
                records = collect_batch(args, label, groups[label], fingerprint, deadline)
            except augment.Deadline:
                LOG.info("Deadline/stop: discard incomplete batch")
                return
            if augment.STOP or time.time() >= deadline:
                return
            store.write_batch(records)
            batches += 1
            LOG.info("saved_batch=%d category=%s saved=%d", batches, label, len(records))
            if args.max_batches and batches >= args.max_batches:
                return
    finally:
        store.close()


def parser():
    p = augment.parser()
    p.description = __doc__
    p.add_argument("--category-level", choices=("primary", "subcategory"), default="primary")
    p.add_argument("--encoding", default="utf-8-sig")
    p.add_argument("--balance-batch-size", type=int, default=50)
    p.add_argument("--max-requests-per-batch", type=int, default=500)
    p.add_argument("--plan", action="store_true", help="Original CSV distribution only; no writes/API/deletion")
    p.add_argument("--stop-when-balanced", action="store_true")
    return p


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = parser()
    args = p.parse_args()
    args.base_url = (args.base_url or augment.URLS[args.backend]).rstrip("/")
    if not args.input or (not args.plan and not args.mock and not args.model):
        p.error("--input CSV required; real generation also requires --model")
    if args.balance_batch_size < 50 or args.balance_batch_size % 50:
        p.error("--balance-batch-size must be a positive multiple of 50")
    if any(getattr(args, name) <= 0 for name in ("cap_bytes", "delete_bytes", "shard_bytes", "variants", "max_tokens", "max_chars", "timeout", "max_failures", "max_requests_per_batch")):
        p.error("Size/request limits must be positive")
    if args.delete_bytes >= args.cap_bytes or args.shard_bytes > args.cap_bytes:
        p.error("Require delete-bytes < cap-bytes and shard-bytes <= cap-bytes")
    if min(args.min_free_bytes, args.interval, args.max_batches) < 0:
        p.error("Free space, interval and batch limit cannot be negative")
    def stop(signum, frame):
        augment.STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    run(args)


if __name__ == "__main__":
    main()
