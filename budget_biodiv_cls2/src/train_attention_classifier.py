"""사업설명자료 Attention Pooling 분류 학습 CLI.

설치 예시::

    pip install pandas numpy scikit-learn torch transformers accelerate tqdm \
        pypdf pdfplumber olefile beautifulsoup4 lxml

데이터 점검은 pretrained model을 내려받지 않는다::

    python src/train_attention_classifier.py --dry_run
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_dataset import (
    BudgetDocumentDataset,
    build_metadata_fallback_records,
    discover_documents,
    document_collate_fn,
    extract_matched_documents,
    load_label_data,
    match_documents_to_labels,
)
from document_parser import DocumentParseError, extract_document
from utils import configure_logging, ensure_dir, is_cuda_oom, set_seed, write_csv, write_json


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = PROJECT_DIR / "outputs"
DEFAULT_DOC_DIR = DEFAULT_BASE_DIR / "사업설명자료"
DEFAULT_LABEL_FILE = DEFAULT_BASE_DIR / "세부사업 예산편성현황(총액)_years_category.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_DIR / "model_results"
LOGGER = logging.getLogger("budget_document_classifier")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="사업설명자료와 예산정보 기반 BIOFIN Attention Pooling 문서분류",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--doc_dir", default=str(DEFAULT_DOC_DIR))
    parser.add_argument("--label_file", default=str(DEFAULT_LABEL_FILE))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model_name", default="klue/roberta-base")
    parser.add_argument("--label_column", default="biofin_category")
    parser.add_argument("--num_labels", type=int, default=10, help="category 0~9이므로 기본 10")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128, help="인접 chunk 사이에 겹칠 token 수")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--encoder_chunk_batch_size",
        type=int,
        default=16,
        help="한 번의 transformer 호출에 넣을 chunk 수(문서 batch와 별도)",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--class_weight", action="store_true", help="train 분포 역비례 class weight 적용")
    parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hwp_com", action="store_true", help="OLE/pyhwp 실패 시 한글 COM 자동화 시도")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("epochs는 1 이상이어야 합니다")
    if args.batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다")
    if args.max_length < 8:
        raise ValueError("max_length가 너무 작습니다")
    if args.stride < 0 or args.stride >= args.max_length - 2:
        raise ValueError("stride는 0 이상 max_length-2 미만이어야 합니다")
    if not 0 < args.valid_ratio < 1:
        raise ValueError("valid_ratio는 0과 1 사이여야 합니다")
    if args.encoder_chunk_batch_size < 1:
        raise ValueError("encoder_chunk_batch_size는 1 이상이어야 합니다")
    if args.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience는 1 이상이어야 합니다")
    if args.num_labels < 2:
        raise ValueError("num_labels는 2 이상이어야 합니다")


def save_match_outputs(
    success: pd.DataFrame,
    failed: pd.DataFrame,
    output_dir: Path,
) -> None:
    # 모델 입력용 긴 metadata_text는 감사 CSV에서 제외한다.
    public_success = success.drop(columns=["metadata_text"], errors="ignore")
    write_csv(public_success, output_dir / "dataset_match_success.csv")
    write_csv(failed, output_dir / "dataset_match_failed.csv")
    if success.empty:
        distribution = pd.DataFrame(columns=["label", "count"])
    else:
        distribution = (
            success["label"].value_counts().sort_index().rename_axis("label").reset_index(name="count")
        )
    write_csv(distribution, output_dir / "label_distribution.csv")


def print_distribution(success: pd.DataFrame) -> None:
    if success.empty:
        LOGGER.warning("매칭 성공 데이터가 없습니다.")
        return
    counts = success["label"].value_counts().sort_index()
    LOGGER.info("매칭 데이터 label 분포:\n%s", counts.to_string())
    if len(counts) < 2:
        LOGGER.warning(
            "매칭 문서에 label이 한 종류(%s)뿐입니다. 이 상태로는 분류 경계를 학습할 수 없습니다.",
            counts.index[0],
        )
    for label, count in counts.items():
        if count < 5:
            LOGGER.warning("label %s sample이 %d개뿐이라 안정적인 stratified split이 어렵습니다.", label, count)


def run_dry_run(
    documents: list[Path],
    labels: pd.DataFrame,
    label_files: list[Path],
    matched: pd.DataFrame,
    training_candidates: pd.DataFrame,
    failed: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> int:
    LOGGER.info("[dry-run] 문서 파일: %d개", len(documents))
    LOGGER.info("[dry-run] 레이블 CSV: %d개 / 행: %d개", len(label_files), len(labels))
    fallback_count = int((training_candidates["match_type"] == "METADATA_FALLBACK").sum())
    LOGGER.info(
        "[dry-run] 문서 매칭: %d개 / 예산정보 fallback: %d개 / 실패 로그: %d건",
        len(matched),
        fallback_count,
        len(failed),
    )
    if not failed.empty:
        LOGGER.info("[dry-run] 실패 사유:\n%s", failed["reason"].value_counts().to_string())
    print_distribution(training_candidates)

    # 앞의 파일이 손상됐더라도 성공 본문 3개를 보여주기 위해 최대 30개까지 시도한다.
    samples: list[tuple[dict[str, Any], str]] = []
    sample_failures: list[dict[str, Any]] = []
    for row in matched.head(30).to_dict("records"):
        if len(samples) >= 3:
            break
        try:
            text = extract_document(row["file_path"], use_hwp_com=args.hwp_com)
            samples.append((row, text))
        except DocumentParseError as exc:
            sample_failures.append(
                {
                    "reason": exc.reason,
                    "detail": exc.detail,
                    "year": row["year"],
                    "ministry": row["ministry"],
                    "activity_name": row["activity_name"],
                    "file_path": row["file_path"],
                    "source_file": row["source_file"],
                    "source_row": row["source_row"],
                }
            )
    if sample_failures:
        failed = pd.concat([failed, pd.DataFrame(sample_failures)], ignore_index=True)
        save_match_outputs(training_candidates, failed, output_dir)
    for index, (row, text) in enumerate(samples, start=1):
        preview = text[:700].replace("\n", " | ")
        LOGGER.info(
            "[dry-run] 본문 샘플 %d: %s / %s자\n%s",
            index,
            Path(row["file_path"]).name,
            len(text),
            preview,
        )
    if len(samples) < 3:
        LOGGER.warning("본문 추출 성공 샘플이 %d개뿐입니다. 실패 로그를 확인하세요.", len(samples))
    LOGGER.info("dry-run 완료: %s", output_dir)
    return 0


def split_records(
    records: list[dict[str, Any]], valid_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from sklearn.model_selection import train_test_split

    if len(records) < 2:
        raise ValueError("학습/검증 분리에 최소 2개 문서가 필요합니다")
    labels = np.asarray([int(record["label"]) for record in records])
    unique, counts = np.unique(labels, return_counts=True)
    valid_count = max(1, int(math.ceil(len(records) * valid_ratio)))
    train_count = len(records) - valid_count
    can_stratify = (
        len(unique) > 1
        and counts.min() >= 2
        and valid_count >= len(unique)
        and train_count >= len(unique)
    )
    if not can_stratify:
        LOGGER.warning(
            "클래스별 표본 수 또는 split 크기 때문에 stratified split 대신 고정 seed random split을 사용합니다."
        )
    train_records, valid_records = train_test_split(
        records,
        test_size=valid_count,
        random_state=seed,
        shuffle=True,
        stratify=labels if can_stratify else None,
    )
    return train_records, valid_records


def make_class_weights(records: list[dict[str, Any]], device: Any, num_labels: int):
    import torch

    counts = np.bincount([int(record["label"]) for record in records], minlength=num_labels)
    weights = np.zeros(num_labels, dtype=np.float32)
    present = counts > 0
    weights[present] = len(records) / (present.sum() * counts[present])
    LOGGER.info("class weights: %s", weights.tolist())
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _autocast(device: Any, enabled: bool):
    import torch

    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train(args: argparse.Namespace, records: list[dict[str, Any]], output_dir: Path) -> int:
    import torch
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from evaluate import evaluate_model, save_evaluation_outputs
    from model import DocumentAttentionClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.mixed_precision and device.type == "cuda")
    if args.mixed_precision and not amp_enabled:
        LOGGER.warning("CUDA가 없어 mixed precision을 비활성화합니다.")
    LOGGER.info("device=%s, mixed_precision=%s", device, amp_enabled)

    train_records, valid_records = split_records(records, args.valid_ratio, args.seed)
    LOGGER.info("train=%d, valid=%d", len(train_records), len(valid_records))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_dataset = BudgetDocumentDataset(
        train_records, tokenizer, max_length=args.max_length, stride=args.stride
    )
    valid_dataset = BudgetDocumentDataset(
        valid_records, tokenizer, max_length=args.max_length, stride=args.stride
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=document_collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=document_collate_fn,
    )

    try:
        model = DocumentAttentionClassifier(
            model_name=args.model_name,
            num_labels=args.num_labels,
            attention_size=args.attention_size,
            dropout=args.dropout,
            encoder_chunk_batch_size=args.encoder_chunk_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
        ).to(device)
    except RuntimeError as exc:
        if is_cuda_oom(exc):
            raise RuntimeError(
                "모델을 GPU에 올리는 중 메모리가 부족합니다. 더 작은 pretrained model을 선택하세요."
            ) from exc
        raise

    class_weights = (
        make_class_weights(train_records, device, args.num_labels)
        if args.class_weight
        else None
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_score = -float("inf")
    best_path = output_dir / "best_model.pt"
    epochs_without_improvement = 0
    train_log: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            try:
                with _autocast(device, amp_enabled):
                    outputs = model(
                        batch["input_ids"].to(device, non_blocking=True),
                        batch["attention_mask"].to(device, non_blocking=True),
                        batch["chunk_mask"].to(device, non_blocking=True),
                    )
                    labels = batch["labels"].to(device, non_blocking=True)
                    loss = criterion(outputs["logits"], labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"NaN/Inf loss 발생(epoch={epoch}, step={step}, loss={loss.item()})"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                epoch_losses.append(float(loss.item()))
            except RuntimeError as exc:
                if is_cuda_oom(exc):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        "GPU 메모리가 부족합니다. --batch_size 1 또는 더 작은 "
                        "--encoder_chunk_batch_size 값을 사용하세요."
                    ) from exc
                raise
            if step % 20 == 0 or step == len(train_loader):
                LOGGER.info(
                    "epoch %d/%d step %d/%d loss=%.5f",
                    epoch,
                    args.epochs,
                    step,
                    len(train_loader),
                    float(np.mean(epoch_losses[-20:])),
                )

        valid_metrics, _, _ = evaluate_model(
            model,
            valid_loader,
            device,
            criterion=criterion,
            mixed_precision=amp_enabled,
            collect_details=False,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "valid_loss": valid_metrics["loss"],
            "accuracy": valid_metrics["accuracy"],
            "macro_f1": valid_metrics["macro_f1"],
            "weighted_f1": valid_metrics["weighted_f1"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        train_log.append(row)
        write_csv(train_log, output_dir / "train_log.csv")
        LOGGER.info("epoch %d metrics=%s", epoch, row)

        score = valid_metrics["macro_f1"]
        if score > best_score + 1e-8:
            best_score = score
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_name": args.model_name,
                "num_labels": args.num_labels,
                "epoch": epoch,
                "metrics": valid_metrics,
                "args": vars(args),
            }
            temporary_path = output_dir / "best_model.pt.tmp"
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(best_path)
            LOGGER.info("best model 저장: %s", best_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                LOGGER.info("early stopping: %d epoch 연속 개선 없음", epochs_without_improvement)
                break

    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch 2.5 이전 호환
        checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics, prediction_rows, attention_rows = evaluate_model(
        model,
        valid_loader,
        device,
        criterion=criterion,
        mixed_precision=amp_enabled,
        collect_details=True,
    )
    reported_metrics = save_evaluation_outputs(
        prediction_rows, attention_rows, output_dir, num_labels=args.num_labels
    )
    final_metrics.update(reported_metrics)
    write_json(final_metrics, output_dir / "metrics.json")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    LOGGER.info("최종 validation metrics=%s", final_metrics)
    LOGGER.info("학습 완료: %s", output_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    configure_logging(args.debug)
    try:
        validate_arguments(args)
        set_seed(args.seed)
        output_dir = ensure_dir(args.output_dir)
        documents = discover_documents(args.doc_dir)
        labels, label_files = load_label_data(
            args.label_file,
            label_column=args.label_column,
            num_labels=args.num_labels,
        )
        LOGGER.info("문서 %d개, label 행 %d개를 확인했습니다.", len(documents), len(labels))
        matched, failed = match_documents_to_labels(documents, labels)
        _, initial_fallback_success = build_metadata_fallback_records(
            labels, failed
        )
        training_candidates = pd.concat(
            [matched, initial_fallback_success], ignore_index=True, sort=False
        )
        save_match_outputs(training_candidates, failed, output_dir)

        if args.dry_run:
            return run_dry_run(
                documents,
                labels,
                label_files,
                matched,
                training_candidates,
                failed,
                args,
                output_dir,
            )

        records: list[dict[str, Any]] = []
        parsed_success = pd.DataFrame()
        parse_failed = pd.DataFrame()
        if not matched.empty:
            LOGGER.info("매칭 성공 문서의 전체 본문 추출을 시작합니다.")
            records, parsed_success, parse_failed = extract_matched_documents(
                matched, use_hwp_com=args.hwp_com
            )
        if not parse_failed.empty:
            failed = pd.concat([failed, parse_failed], ignore_index=True)
        fallback_records, fallback_success = build_metadata_fallback_records(labels, failed)
        records.extend(fallback_records)
        final_success = pd.concat(
            [parsed_success, fallback_success], ignore_index=True, sort=False
        )
        save_match_outputs(final_success, failed, output_dir)
        LOGGER.info(
            "본문 사용=%d, 예산정보 fallback=%d, 문서 파싱 실패=%d",
            len(records) - len(fallback_records),
            len(fallback_records),
            len(parse_failed),
        )
        if len(records) < 2:
            raise ValueError("본문 추출 후 학습 가능한 문서가 2개 미만입니다")
        unique_labels = sorted({int(record["label"]) for record in records})
        if len(unique_labels) < 2:
            raise ValueError(
                f"학습 문서의 label이 {unique_labels} 한 종류뿐입니다. "
                f"서로 다른 {args.label_column} 문서가 최소 한 개씩 필요합니다."
            )
        return train(args, records, output_dir)
    except KeyboardInterrupt:
        LOGGER.warning("사용자가 작업을 중단했습니다. 지금까지 생성된 로그는 보존됩니다.")
        return 130
    except Exception as exc:
        LOGGER.exception("파이프라인 실패: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
