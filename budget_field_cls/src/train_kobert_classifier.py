"""
KoBERT 계열 모델을 파인튜닝해 예산 분야 분류 모델을 학습하는 스크립트.

입력: 학습/검증/테스트 셋 CSV (split_workfile_text_dataset.py로 생성)
출력: best_model/ (최고 검증 macro-F1 모델), training_history.csv, test_metrics.json 등

파인튜닝(fine-tuning):
    KoBERT 같은 사전학습 모델(pre-trained model)을 우리 데이터에 맞게 추가 학습합니다.
    모든 레이어의 가중치를 업데이트하는 full fine-tuning 방식을 사용합니다.

라벨 매핑:
    학습 셋에서 발견된 라벨을 알파벳/숫자 순으로 정렬해 0~N-1로 번호를 부여합니다.
    이 매핑은 label_mapping.json으로 저장되므로 추론 시에도 동일하게 복원할 수 있습니다.

사용 예:
    python src/train_kobert_classifier.py
    python src/train_kobert_classifier.py --epochs 5 --batch-size 8
    python src/train_kobert_classifier.py --device cuda  # GPU 강제 지정
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLASSIFICATION_TEXT_COLUMN,
    DEFAULT_KOBERT_MODEL,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_WEIGHT_DECAY,
    WORKFILE_KOBERT_MODEL_DIR,
    WORKFILE_TEST_SPLIT_PATH,
    WORKFILE_TRAIN_SPLIT_PATH,
    WORKFILE_VAL_SPLIT_PATH,
)
from utils import save_json

# torch/transformers는 선택적 의존성이므로 ImportError를 저장해 나중에 유용한 메시지를 출력합니다.
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
except ImportError as exc:
    torch   = None
    Dataset = object
    DataLoader = object
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="BIOFIN 워크파일 텍스트 분리 셋에서 KoBERT 계열 분류기를 파인튜닝합니다."
    )
    parser.add_argument("--train-csv",     type=Path,  default=WORKFILE_TRAIN_SPLIT_PATH,
                        help="학습 셋 CSV 경로")
    parser.add_argument("--val-csv",       type=Path,  default=WORKFILE_VAL_SPLIT_PATH,
                        help="검증 셋 CSV 경로")
    parser.add_argument("--test-csv",      type=Path,  default=WORKFILE_TEST_SPLIT_PATH,
                        help="테스트 셋 CSV 경로")
    parser.add_argument("--output-dir",    type=Path,  default=WORKFILE_KOBERT_MODEL_DIR,
                        help="모델 체크포인트와 결과 파일 저장 폴더")
    parser.add_argument("--model-name",                default=DEFAULT_KOBERT_MODEL,
                        help="HuggingFace 모델 이름 또는 로컬 체크포인트 경로")
    parser.add_argument("--text-column",               default=DEFAULT_CLASSIFICATION_TEXT_COLUMN,
                        help="문서 텍스트 컬럼명")
    parser.add_argument("--label-column",              default="label",
                        help="라벨 이름 컬럼명")
    parser.add_argument("--label-id-column",           default="label_id",
                        help="원본 라벨 ID 컬럼명")
    parser.add_argument("--max-length",    type=int,   default=DEFAULT_MAX_LENGTH,
                        help="토크나이저 최대 토큰 길이")
    parser.add_argument("--max-chars",     type=int,   default=0,
                        help="토크나이징 전 문자 수 제한 (0=없음)")
    parser.add_argument("--epochs",        type=int,   default=DEFAULT_NUM_EPOCHS,
                        help="학습 에폭 수")
    parser.add_argument("--batch-size",    type=int,   default=DEFAULT_BATCH_SIZE,
                        help="배치 크기")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
                        help="AdamW 학습률")
    parser.add_argument("--weight-decay",  type=float, default=DEFAULT_WEIGHT_DECAY,
                        help="AdamW 가중치 감쇠 (L2 정규화)")
    # warmup-ratio: 전체 스텝의 이 비율만큼 학습률을 선형으로 높인 뒤 다시 낮춥니다.
    # 학습 초반에 학습률이 너무 크면 사전학습 가중치가 망가질 수 있습니다.
    parser.add_argument("--warmup-ratio",  type=float, default=0.1,
                        help="선형 스케줄러의 웜업 비율")
    # 그래디언트 누적: 작은 배치로 큰 배치 효과를 냅니다.
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="그래디언트 누적 스텝 수")
    # 그래디언트 클리핑: 너무 큰 그래디언트를 max_grad_norm으로 잘라 학습 안정화
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="그래디언트 클리핑 최대 norm")
    parser.add_argument("--num-workers",   type=int,   default=0,
                        help="DataLoader 워커 수 (Windows에서는 0 권장)")
    parser.add_argument("--seed",          type=int,   default=42,
                        help="난수 시드")
    parser.add_argument("--cpu",           action="store_true",
                        help="CUDA가 있어도 CPU로 강제 실행")
    parser.add_argument("--device",
                        choices=("auto", "cuda", "cpu"),
                        default="auto",
                        help="학습 장치. 'auto'=CUDA 자동 선택, 'cuda'=CUDA 없으면 오류")
    return parser.parse_args()


def ensure_runtime_dependencies() -> None:
    """torch/transformers가 설치되어 있지 않으면 유용한 오류 메시지를 출력합니다."""
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "KoBERT 학습에 필요한 패키지가 없습니다. "
            "최소 설치: torch, transformers, sentencepiece"
        ) from _IMPORT_ERROR


def set_seed(seed: int) -> None:
    """재현성을 위해 모든 난수 생성기의 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(args: argparse.Namespace) -> "torch.device":
    """
    --cpu와 --device 옵션을 해석해 학습에 사용할 장치를 반환합니다.

    --cpu와 --device를 동시에 지정하면 오류를 발생시킵니다.
    --device cuda를 지정했는데 CUDA가 없으면 오류를 발생시킵니다.
    'auto'는 CUDA가 있으면 CUDA, 없으면 CPU를 사용합니다.
    """
    if args.cpu and args.device != "auto":
        raise ValueError("--cpu와 --device를 동시에 사용할 수 없습니다.")

    requested_device = "cpu" if args.cpu else args.device

    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda를 요청했지만 CUDA GPU를 찾을 수 없습니다. "
                "PyTorch CUDA 빌드, NVIDIA 드라이버, Docker --gpus 옵션을 확인하세요."
            )
        return torch.device("cuda")

    # auto 모드: CUDA가 없으면 CPU로 자동 fallback
    if torch.cuda.is_available():
        return torch.device("cuda")

    print("CUDA를 찾을 수 없어 CPU로 실행합니다. GPU 강제 사용은 --device cuda를 사용하세요.")
    return torch.device("cpu")


def print_device_summary(device: "torch.device") -> None:
    """사용 장치 정보를 출력합니다."""
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")


def read_split(path: Path, text_column: str, label_column: str) -> pd.DataFrame:
    """
    CSV를 읽어 텍스트/라벨 컬럼이 있고 비어있지 않은 행만 반환합니다.

    fillna("") + strip() != "": 빈 문자열과 공백만 있는 셀을 모두 제거합니다.
    """
    dataframe       = pd.read_csv(path, encoding="utf-8-sig")
    required_columns = {text_column, label_column}
    missing_columns  = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"{path}에 필수 컬럼이 없습니다: {sorted(missing_columns)}")

    dataframe = dataframe.copy()
    dataframe[text_column]  = dataframe[text_column].fillna("").astype(str)
    dataframe[label_column] = dataframe[label_column].fillna("").astype(str)
    dataframe = dataframe[dataframe[text_column].str.strip()  != ""]
    dataframe = dataframe[dataframe[label_column].str.strip() != ""]
    return dataframe.reset_index(drop=True)


def build_label_mapping(
    train_df: pd.DataFrame,
    label_column: str,
    label_id_column: str,
) -> tuple[dict[str, int], dict[int, dict[str, object]]]:
    """
    학습 셋에서 라벨 이름 → 모델 ID 매핑을 만듭니다.

    HuggingFace 모델은 0~N-1의 정수 라벨을 사용합니다.
    original_label_id(원래 숫자 ID)가 있으면 그 순서를 따르고,
    없으면 라벨 이름의 알파벳 순으로 정렬합니다.

    반환값:
        label_to_model_id: 라벨 이름 → 모델 라벨 ID (0~N-1)
        model_id_to_meta: 모델 라벨 ID → {model_label_id, original_label_id, label}
    """
    records: list[tuple[object, str]] = []
    seen_keys: set[tuple[str, str]] = set()

    for row in train_df.to_dict(orient="records"):
        label             = str(row[label_column])
        original_label_id = str(row.get(label_id_column, ""))
        key               = (label, original_label_id)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        records.append((row.get(label_id_column, ""), label))

    def sort_key(item: tuple[object, str]) -> tuple[int, object]:
        """original_label_id가 정수이면 숫자 순, 아니면 라벨 이름 순으로 정렬합니다."""
        raw_id, label = item
        try:
            return (0, int(raw_id))
        except (TypeError, ValueError):
            return (1, label)

    records.sort(key=sort_key)

    label_to_model_id: dict[str, int]             = {}
    model_id_to_meta:  dict[int, dict[str, object]] = {}
    for model_label_id, (original_label_id, label) in enumerate(records):
        label_to_model_id[label] = model_label_id
        model_id_to_meta[model_label_id] = {
            "model_label_id":    model_label_id,
            "original_label_id": "" if pd.isna(original_label_id) else original_label_id,
            "label":             label,
        }

    return label_to_model_id, model_id_to_meta


def attach_model_labels(
    dataframe: pd.DataFrame,
    label_column: str,
    label_to_model_id: dict[str, int],
) -> pd.DataFrame:
    """
    데이터프레임에 model_label_id 컬럼을 추가합니다.

    검증/테스트 셋에 학습 셋에 없는 라벨이 있으면 ValueError를 발생시킵니다.
    (학습되지 않은 클래스를 예측할 수 없기 때문)
    """
    dataframe                 = dataframe.copy()
    dataframe["model_label_id"] = dataframe[label_column].map(label_to_model_id)
    missing_labels            = dataframe[dataframe["model_label_id"].isna()][label_column].unique().tolist()
    if missing_labels:
        raise ValueError(f"학습 셋에 없는 라벨이 발견되었습니다: {missing_labels}")
    dataframe["model_label_id"] = dataframe["model_label_id"].astype(int)
    return dataframe


def maybe_truncate_text(text: str, max_chars: int) -> str:
    """max_chars > 0이면 텍스트를 해당 문자 수로 자릅니다."""
    if max_chars and max_chars > 0:
        return text[:max_chars]
    return text


@dataclass
class TextRecord:
    """데이터셋의 단일 샘플을 나타냅니다."""
    record_index:    int
    text:            str
    model_label_id:  int
    raw_row:         dict[str, object]  # 원본 행 데이터 (예측 결과 파일에 포함)


class TextClassificationDataset(Dataset):
    """
    PyTorch Dataset: 텍스트 분류용 데이터셋.

    __getitem__은 TextRecord를 반환하고,
    collate_fn(make_collate_fn 참조)에서 토크나이징이 이루어집니다.
    Dataset에서 토크나이징하지 않는 이유: DataLoader의 배치 패딩을
    배치 단위로 처리하면 메모리를 더 효율적으로 사용할 수 있습니다.
    """
    def __init__(
        self,
        dataframe:   pd.DataFrame,
        text_column: str,
        max_chars:   int,
    ) -> None:
        self.records: list[TextRecord] = []
        for index, row in enumerate(dataframe.to_dict(orient="records")):
            self.records.append(TextRecord(
                record_index=index,
                text=maybe_truncate_text(str(row[text_column]), max_chars),
                model_label_id=int(row["model_label_id"]),
                raw_row=row,
            ))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TextRecord:
        return self.records[index]


def make_collate_fn(tokenizer, max_length: int):
    """
    DataLoader의 collate_fn을 반환합니다.

    클로저(closure)로 tokenizer와 max_length를 캡처합니다.
    배치 단위로 토크나이징하면 배치 내 최장 시퀀스에 맞춰 패딩해
    불필요한 메모리 낭비를 줄입니다.

    labels 타입은 long(정수): CrossEntropyLoss 요구사항
    records: 원본 TextRecord를 함께 전달해 예측 결과 파일 작성에 활용
    """
    def collate_fn(batch: list[TextRecord]) -> dict[str, object]:
        texts  = [item.text for item in batch]
        labels = [item.model_label_id for item in batch]
        encoded = tokenizer(
            texts,
            padding=True,    # 배치 내 최장 길이로 패딩
            truncation=True, # max_length 초과 시 잘라냄
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"]  = torch.tensor(labels, dtype=torch.long)
        encoded["records"] = batch
        return encoded

    return collate_fn


def move_batch_to_device(
    batch: dict[str, object],
    device: "torch.device",
) -> tuple[dict[str, "torch.Tensor"], list[TextRecord]]:
    """
    배치에서 TextRecord를 꺼내고 나머지 텐서를 지정 장치로 이동합니다.

    batch.pop("records"): 텐서가 아닌 Python 객체는 .to(device)를 지원하지 않으므로 먼저 꺼냅니다.
    """
    records      = batch.pop("records")
    tensor_batch = {key: value.to(device) for key, value in batch.items()}
    return tensor_batch, records


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_meta: dict[int, dict[str, object]],
) -> dict[str, object]:
    """
    예측 결과에서 분류 성능 지표를 계산합니다.

    라벨별 지표:
        precision(정밀도) = TP / (TP + FP): 양성으로 예측한 것 중 실제 양성 비율
        recall(재현율)    = TP / (TP + FN): 실제 양성 중 맞게 예측한 비율
        F1 = 2 * precision * recall / (precision + recall): 조화평균

    집계 지표:
        macro F1: 각 클래스 F1의 단순 평균 (클래스 불균형에도 동등하게 평가)
        weighted F1: 각 클래스 샘플 수로 가중 평균
    """
    if len(y_true) == 0:
        raise ValueError("빈 평가 셋에서는 지표를 계산할 수 없습니다.")

    supports:         list[int]   = []
    per_label_metrics: list[dict[str, object]] = []
    f1_scores:        list[float] = []
    weighted_f1_sum   = 0.0

    for model_label_id, meta in sorted(label_meta.items()):
        # 이진 분류를 클래스별로 반복 (one-vs-rest)
        true_positive  = int(np.sum((y_true == model_label_id) & (y_pred == model_label_id)))
        false_positive = int(np.sum((y_true != model_label_id) & (y_pred == model_label_id)))
        false_negative = int(np.sum((y_true == model_label_id) & (y_pred != model_label_id)))
        support        = int(np.sum(y_true == model_label_id))

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall    = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        supports.append(support)
        f1_scores.append(f1)
        weighted_f1_sum += f1 * support
        per_label_metrics.append({
            "model_label_id":    model_label_id,
            "original_label_id": meta["original_label_id"],
            "label":             meta["label"],
            "support":           support,
            "precision":         precision,
            "recall":            recall,
            "f1":                f1,
        })

    accuracy      = float(np.mean(y_true == y_pred))
    total_support = int(sum(supports))
    macro_f1      = float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0
    weighted_f1   = float(weighted_f1_sum / total_support) if total_support else 0.0

    return {
        "accuracy":    accuracy,
        "macro_f1":    macro_f1,
        "weighted_f1": weighted_f1,
        "support":     total_support,
        "per_label":   per_label_metrics,
    }


def evaluate(
    model,
    data_loader: "DataLoader",
    device: "torch.device",
    label_meta: dict[int, dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """
    모델을 평가하고 (성능 지표, 예측 행 목록)을 반환합니다.

    torch.no_grad(): 평가 시 그래디언트 계산 비활성화 (메모리 절약)
    softmax: 로짓을 확률로 변환
    argmax: 가장 높은 확률의 클래스 인덱스 = 예측값

    prediction_rows: 각 샘플에 대해 실제 라벨/예측 라벨/신뢰도를 기록합니다.
    """
    model.eval()
    total_loss    = 0.0
    total_batches = 0
    y_true:           list[int]             = []
    y_pred:           list[int]             = []
    prediction_rows:  list[dict[str, object]] = []

    with torch.no_grad():
        for batch in data_loader:
            tensor_batch, records = move_batch_to_device(batch, device)
            outputs  = model(**tensor_batch)
            logits   = outputs.logits
            total_loss    += float(outputs.loss.item())
            total_batches += 1

            # 클래스별 확률 계산 및 최대 확률 클래스 선택
            probabilities    = torch.softmax(logits, dim=-1)
            predicted_ids    = torch.argmax(probabilities, dim=-1)
            predicted_scores = torch.max(probabilities, dim=-1).values
            true_ids         = tensor_batch["labels"]

            predicted_ids_np    = predicted_ids.detach().cpu().numpy()
            predicted_scores_np = predicted_scores.detach().cpu().numpy()
            true_ids_np         = true_ids.detach().cpu().numpy()

            y_true.extend(true_ids_np.tolist())
            y_pred.extend(predicted_ids_np.tolist())

            # 각 샘플에 대한 예측 결과 기록
            for record, true_id, pred_id, confidence in zip(
                records,
                true_ids_np.tolist(),
                predicted_ids_np.tolist(),
                predicted_scores_np.tolist(),
            ):
                row = dict(record.raw_row)
                row.update({
                    "true_model_label_id": true_id,
                    "true_label":          label_meta[true_id]["label"],
                    "pred_model_label_id": pred_id,
                    "pred_label":          label_meta[pred_id]["label"],
                    "confidence":          float(confidence),
                    "correct":             int(true_id == pred_id),
                })
                prediction_rows.append(row)

    metrics         = compute_metrics(np.array(y_true), np.array(y_pred), label_meta)
    metrics["loss"] = total_loss / total_batches if total_batches else 0.0
    return metrics, prediction_rows


def train_one_epoch(
    model,
    data_loader: "DataLoader",
    optimizer,
    scheduler,
    device: "torch.device",
    gradient_accumulation_steps: int,
    max_grad_norm: float,
) -> float:
    """
    한 에폭 학습을 수행하고 평균 손실을 반환합니다.

    그래디언트 누적:
        gradient_accumulation_steps 배치마다 한 번씩 파라미터를 업데이트합니다.
        배치가 작아도 더 큰 배치 효과를 낼 수 있습니다.

    그래디언트 클리핑:
        clip_grad_norm_으로 그래디언트 크기를 제한합니다.
        Transformer 파인튜닝에서 폭발적 그래디언트를 방지합니다.

    선형 스케줄러:
        warmup → 선형 감소로 학습률을 조절합니다.
        optimizer.step() 후 scheduler.step()을 호출해 학습률을 업데이트합니다.
    """
    model.train()
    optimizer.zero_grad()
    total_loss    = 0.0
    total_batches = 0

    for step, batch in enumerate(data_loader, start=1):
        tensor_batch, _ = move_batch_to_device(batch, device)
        outputs = model(**tensor_batch)
        # 누적 스텝 수로 손실을 나눠서 실제 배치 크기로 스케일링
        loss    = outputs.loss / gradient_accumulation_steps
        loss.backward()

        # 누적 완료 또는 마지막 배치일 때만 파라미터 업데이트
        if step % gradient_accumulation_steps == 0 or step == len(data_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss    += float(outputs.loss.item())
        total_batches += 1

    return total_loss / total_batches if total_batches else 0.0


def build_dataloader(
    dataframe:    pd.DataFrame,
    tokenizer,
    text_column:  str,
    max_chars:    int,
    max_length:   int,
    batch_size:   int,
    shuffle:      bool,
    num_workers:  int,
) -> tuple["TextClassificationDataset", "DataLoader"]:
    """Dataset과 DataLoader를 생성합니다."""
    dataset     = TextClassificationDataset(dataframe, text_column=text_column, max_chars=max_chars)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collate_fn(tokenizer, max_length=max_length),
    )
    return dataset, data_loader


def run(args: argparse.Namespace) -> int:
    """KoBERT 분류기 학습 파이프라인 실행."""
    ensure_runtime_dependencies()
    set_seed(args.seed)

    train_csv  = args.train_csv.resolve()
    val_csv    = args.val_csv.resolve()
    test_csv   = args.test_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_split(train_csv, args.text_column, args.label_column)
    val_df   = read_split(val_csv,   args.text_column, args.label_column)
    test_df  = read_split(test_csv,  args.text_column, args.label_column)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("학습/검증/테스트 셋이 모두 비어있지 않아야 합니다.")

    # 라벨 매핑 구성 및 각 셋에 model_label_id 컬럼 추가
    label_to_model_id, label_meta = build_label_mapping(train_df, args.label_column, args.label_id_column)
    train_df = attach_model_labels(train_df, args.label_column, label_to_model_id)
    val_df   = attach_model_labels(val_df,   args.label_column, label_to_model_id)
    test_df  = attach_model_labels(test_df,  args.label_column, label_to_model_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_meta),
        # HuggingFace 모델이 자동으로 라벨 이름을 기억하게 설정
        id2label={index: meta["label"] for index, meta in label_meta.items()},
        label2id={meta["label"]: index for index, meta in label_meta.items()},
        problem_type="single_label_classification",
    )

    device = resolve_device(args)
    print_device_summary(device)
    model.to(device)

    # 학습/검증/테스트 DataLoader 생성
    _, train_loader = build_dataloader(
        train_df, tokenizer=tokenizer, text_column=args.text_column,
        max_chars=args.max_chars, max_length=args.max_length,
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    _, val_loader = build_dataloader(
        val_df, tokenizer=tokenizer, text_column=args.text_column,
        max_chars=args.max_chars, max_length=args.max_length,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    _, test_loader = build_dataloader(
        test_df, tokenizer=tokenizer, text_column=args.text_column,
        max_chars=args.max_chars, max_length=args.max_length,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # 선형 학습률 스케줄러: warmup 후 선형 감소
    # updates_per_epoch: 그래디언트 누적을 고려한 실제 파라미터 업데이트 횟수
    updates_per_epoch    = math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps))
    total_training_steps = max(1, updates_per_epoch * args.epochs)
    warmup_steps         = int(total_training_steps * args.warmup_ratio)
    scheduler            = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    history_rows:      list[dict[str, object]] = []
    best_val_macro_f1  = -1.0
    best_model_dir     = output_dir / "best_model"
    best_metrics_path  = output_dir / "best_val_metrics.json"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            gradient_accumulation_steps=max(1, args.gradient_accumulation_steps),
            max_grad_norm=args.max_grad_norm,
        )
        val_metrics, _ = evaluate(model, val_loader, device, label_meta)
        history_row = {
            "epoch":            epoch,
            "train_loss":       train_loss,
            "val_loss":         val_metrics["loss"],
            "val_accuracy":     val_metrics["accuracy"],
            "val_macro_f1":     val_metrics["macro_f1"],
            "val_weighted_f1":  val_metrics["weighted_f1"],
        }
        history_rows.append(history_row)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f}"
        )

        # 검증 macro-F1이 개선되면 모델 저장
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            save_json(best_metrics_path, val_metrics)

    # 최고 검증 모델로 테스트 셋 평가
    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    best_model.to(device)
    test_metrics, test_predictions = evaluate(best_model, test_loader, device, label_meta)

    # 결과 파일 저장
    training_history_path = output_dir / "training_history.csv"
    label_mapping_path    = output_dir / "label_mapping.json"
    test_metrics_path     = output_dir / "test_metrics.json"
    predictions_path      = output_dir / "test_predictions.csv"
    run_config_path       = output_dir / "run_config.json"

    pd.DataFrame(history_rows).to_csv(training_history_path, index=False, encoding="utf-8-sig")
    save_json(label_mapping_path, list(label_meta.values()))
    save_json(test_metrics_path, test_metrics)
    pd.DataFrame(test_predictions).to_csv(predictions_path, index=False, encoding="utf-8-sig")
    save_json(run_config_path, {
        "train_csv":                   str(train_csv),
        "val_csv":                     str(val_csv),
        "test_csv":                    str(test_csv),
        "output_dir":                  str(output_dir),
        "model_name":                  args.model_name,
        "text_column":                 args.text_column,
        "max_length":                  args.max_length,
        "max_chars":                   args.max_chars,
        "epochs":                      args.epochs,
        "batch_size":                  args.batch_size,
        "learning_rate":               args.learning_rate,
        "weight_decay":                args.weight_decay,
        "warmup_ratio":                args.warmup_ratio,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed":                        args.seed,
        "device":                      str(device),
        "num_labels":                  len(label_meta),
    })

    print(f"최고 모델 저장: {best_model_dir}")
    print(f"학습 히스토리: {training_history_path}")
    print(f"테스트 지표: {test_metrics_path}")
    print(f"테스트 예측: {predictions_path}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
