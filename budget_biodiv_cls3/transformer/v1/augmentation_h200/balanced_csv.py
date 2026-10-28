#!/usr/bin/env python3
"""Augment the smallest category (original + generated) 50 rows at a time until every
category reaches 400 rows in total, then stop; original CSV is read-only."""
from collections import Counter, defaultdict
import argparse
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
import shutil
import time
import threading
import uuid

import augment

LOG = logging.getLogger("balanced")
METADATA = (
    "회계연도", "소관명", "회계명", "계정명", "분야명", "부문명",
    "프로그램명", "단위사업명", "세부사업명", "경비구분", "지출구분",
    "정부안금액(천원)", "국회확정금액(천원)", "예산액",
)


def code(value):
    try:
        number = Decimal(str(value).strip())
        if not number.is_finite() or number != number.to_integral_value() or number < 0:
            raise ValueError("Category must be a nonnegative integer")
        return str(int(number))
    except (InvalidOperation, ValueError):
        raise ValueError(f"Invalid category: {value!r}")


PRIMARY_COLUMNS = ("1차 카테고리", "BIOFIN 1차 카테고리", "biofin_category")
LOWER_COLUMNS = ("하위 카테고리", "BIOFIN 하위 카테고리", "biofin_subcategory")


def normalize_labels(row, level="primary"):
    """Accept known label headers; never turn missing gold labels into class 0."""
    def read(names):
        values = {code(row[name]) for name in names if str(row.get(name) or "").strip()}
        if len(values) > 1:
            raise ValueError(f"Conflicting label columns: {names}")
        return next(iter(values), "")
    first, lower = read(PRIMARY_COLUMNS), read(LOWER_COLUMNS)
    if not first:
        return None
    normalized = dict(row, **{"1차 카테고리": first, "하위 카테고리": lower})
    category(normalized, level)
    return normalized


def category(row, level):
    first = code(row["1차 카테고리"])
    lower = code(row["하위 카테고리"]) if row.get("하위 카테고리", "") != "" else ""
    if int(first) > 9 or (first == "0" and lower not in ("", "0")):
        raise ValueError(f"Invalid category pair: {first}/{lower}")
    if level == "primary" or first == "0":
        return first
    if not lower or not 1 <= int(lower) <= 99:
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
        required = {"세부사업명", "business_key"}
        if not set(PRIMARY_COLUMNS).intersection(reader.fieldnames or []):
            raise ValueError(f"Label column required: one of {PRIMARY_COLUMNS}; found {reader.fieldnames}")
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns: {sorted(required - set(reader.fieldnames or []))}")
        for row_number, row in enumerate(reader, 2):
            if "split" in row and row["split"] != "train":
                excluded += 1
                continue
            row = normalize_labels(row, level)
            if row is None:
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


def choose_category(pending, totals):
    return min(pending, key=lambda label: (totals[label], Decimal(label)))


def make_plan(original, retained, target):
    return [{"category": label, "original": original[label],
             "retained_generated": retained[label], "target_total": target,
             "generate_to_target": max(0, target - original[label] - retained[label])}
            for label in sorted(original, key=Decimal)]


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

    def ensure_space(self, incoming):
        # A finite dataset must never recycle previously completed batches.
        if (augment.tree_bytes(self.root) + incoming > self.cap or
                shutil.disk_usage(self.root).free - incoming < self.min_free):
            raise RuntimeError("Insufficient storage; existing augmentation files preserved")

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
                if meta["identity"] != self.identity or meta["size"] < 1:
                    raise RuntimeError(f"Invalid completed batch: {path}")
                self.index[path] = (first["label"], meta["size"])
            label, count = self.index[path]
            counts[label] += count
        self.index = {path: value for path, value in self.index.items() if path in current}
        return counts

    def write_batch(self, records):
        if not records or len({row["label"] for row in records}) != 1:
            raise ValueError("A batch must contain rows of exactly one category")
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


def collect_batch(args, label, seeds, fingerprint, deadline, previous_texts=()):
    accepted = []
    seen = {" ".join(seed["text"].split()) for seed in seeds}
    seen.update(previous_texts)
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
            progress = getattr(args, "progress", None)
            if progress:
                progress.update(phase=f"API 응답 대기 (요청 {attempts})")
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
                    "subcategory": row["하위 카테고리"],
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
            if progress:
                progress.update(phase="배치 생성 중", pending=len(accepted))
            LOG.info("category=%s pending=%d/%d", label, len(accepted), args.balance_batch_size)
        except augment.Deadline:
            raise
        except Exception as exc:
            failures += 1
            if getattr(args, "progress", None):
                args.progress.update(phase=f"API 재시도 대기 (연속 실패 {failures})")
            LOG.warning("category=%s request_failed=%s(%s) consecutive=%d",
                        label, type(exc).__name__, str(exc)[:200], failures)
            if failures >= args.max_failures:
                raise RuntimeError("Repeated API/validation failure; no partial batch saved") from exc
            time.sleep(max(0, min(2 ** min(failures, 6), deadline - time.time())))
        time.sleep(max(0, min(args.interval, deadline - time.time())))
    return accepted



def duration(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class Progress:
    def __init__(self, interval):
        self.started = time.monotonic()
        self.interval = interval
        self.state = {"phase": "입력 확인", "counts": {}, "target": 0,
                      "category": "-", "pending": 0, "batch_size": 0}
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def report(self):
        with self.lock:
            state = dict(self.state)
        counts, target = state["counts"], state["target"]
        total = len(counts) * target
        saved = sum(min(count, target) for count in counts.values())
        per_category = ", ".join(f"{label}:{count}/{target}" for label, count in sorted(counts.items(), key=lambda pair: Decimal(pair[0])))
        LOG.info("경과=%s | 상태=%s | 원본+증강=%d/%d (%.1f%%) | 현재 카테고리=%s | 미저장 배치=%d/%d | 카테고리별=[%s]",
                 duration(time.monotonic() - self.started), state["phase"],
                 saved, total, 100 * saved / total if total else 0,
                 state["category"], state["pending"], state["batch_size"], per_category)

    def watch(self):
        while not self.stopped.wait(self.interval):
            self.report()


def run(args):
    if args.plan:
        return _run(args)
    progress = Progress(getattr(args, "log_interval", 10))
    args.progress = progress
    LOG.info("증강 시작 | 입력=%s | 저장 위치=%s | 카테고리별 원본+증강 최대=%d",
             Path(args.input).resolve(), Path(args.output).resolve() / "balanced_h200_v1",
             args.target_per_category)
    progress.thread.start()
    try:
        _run(args)
    except BaseException:
        progress.update(phase="오류 또는 중단")
        raise
    finally:
        progress.stopped.set()
        progress.thread.join()
        progress.report()
        LOG.info("실행 종료 | 총 실행 시간=%s", duration(time.monotonic() - progress.started))
        del args.progress


def _run(args):

    groups, fingerprint, excluded = load_csv(args.input, args.category_level, args.encoding)
    LOG.info("Excluded unlabeled/nontrain rows: %d", excluded)
    original = Counter({label: len(seeds) for label, seeds in groups.items()})
    if any(len(seed["text"]) > args.max_chars for seeds in groups.values() for seed in seeds):
        raise ValueError("Source metadata exceeds --max-chars; increase the limit")
    if args.plan:
        print(json.dumps({"source_sha256": fingerprint, "category_level": args.category_level,
                          "excluded_unlabeled_or_nontrain": excluded, "rows": make_plan(original, Counter(), args.target_per_category)},
                         ensure_ascii=False, indent=2))
        return
    progress = getattr(args, "progress", None)
    def status(**values):
        if progress:
            progress.update(**values)
    status(counts=dict(original), target=args.target_per_category)
    deadline = augment.parse_deadline(args.until) if args.until else float("inf")
    if time.time() >= deadline:
        status(phase="종료 시각 도달")
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
            totals = original + retained
            status(counts={label: totals[label] for label in original}, phase="진행 확인")
            # Categories whose originals already reach the cap are never augmented.
            pending = [label for label in original if totals[label] < args.target_per_category]
            if not pending:
                status(phase="목표 달성", category="-", pending=0, batch_size=0)
                LOG.info("Target reached: totals=%s generated=%s", dict(totals), dict(retained))
                return
            label = choose_category(pending, totals)
            LOG.info("selected=%s original=%d retained=%d", label, original[label], retained[label])
            try:
                batch_args = argparse.Namespace(**vars(args))
                batch_args.balance_batch_size = min(args.balance_batch_size,
                                                    args.target_per_category - totals[label])
                status(phase="기존 문장 확인", category=label, pending=0, batch_size=batch_args.balance_batch_size)
                previous_texts = set()
                for path in store.directory.glob("aug-*.jsonl"):
                    with path.open(encoding="utf-8") as stream:
                        for line in stream:
                            row = json.loads(line)
                            if row["label"] == label:
                                previous_texts.add(" ".join(row["text"].split()))
                records = collect_batch(batch_args, label, groups[label], fingerprint, deadline, previous_texts)
            except augment.Deadline:
                status(phase="종료 시각 도달 또는 중단 요청")
                LOG.info("Deadline/stop: discard incomplete batch")
                return
            if augment.STOP or time.time() >= deadline:
                status(phase="종료 시각 도달 또는 중단 요청")
                return
            store.write_batch(records)
            batches += 1
            retained[label] += len(records)
            status(phase="배치 저장 완료", counts={key: original[key] + retained[key] for key in original}, pending=0)
            if progress:
                progress.report()
            LOG.info("saved_batch=%d category=%s saved=%d", batches, label, len(records))
            if args.max_batches and batches >= args.max_batches:
                status(phase="배치 제한 도달")
                return
        status(phase="종료 시각 도달 또는 중단 요청")
    finally:
        store.close()


def parser():
    p = augment.parser()
    p.description = __doc__
    p.set_defaults(until=None, output="transformer/v1/outputs/augmentation")
    p.add_argument("--target-per-category", type=int, default=400,
                   help="Maximum rows per category including originals; augmentation stops there")
    p.add_argument("--category-level", choices=("primary", "subcategory"), default="primary")
    p.add_argument("--log-interval", type=int, default=10, help="Progress heartbeat interval in seconds")
    p.add_argument("--encoding", default="utf-8-sig")
    p.add_argument("--balance-batch-size", type=int, default=50)
    p.add_argument("--max-requests-per-batch", type=int, default=500)
    p.add_argument("--plan", action="store_true", help="Original CSV distribution only; no writes/API/deletion")
    return p


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = parser()
    args = p.parse_args()
    args.base_url = (args.base_url or augment.URLS[args.backend]).rstrip("/")
    if not args.input or (not args.plan and not args.mock and not args.model):
        p.error("--input CSV required; real generation also requires --model")
    if args.log_interval <= 0:
        p.error("--log-interval must be positive")
    if args.target_per_category <= 0:
        p.error("--target-per-category must be positive")
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
