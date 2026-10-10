"""문서-레이블 매칭, 본문 추출, tokenizer chunk Dataset 구성."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # --dry_run은 모델 패키지 설치 전에도 실행 가능해야 한다.
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass

from document_parser import DocumentParseError, SUPPORTED_EXTENSIONS, extract_document
from utils import read_csv_flexible, safe_scalar


LOGGER = logging.getLogger("budget_document_classifier")
REQUIRED_COLUMNS = {"회계연도", "소관명", "세부사업명"}
ACCOUNT_WORDS = (
    "일반회계",
    "특별회계",
    "기금",
    "손익계정",
    "자본계정",
    "책임운영기관특별회계",
)
METADATA_INPUT_COLUMNS = (
    "회계연도",
    "소관명",
    "회계명",
    "계정명",
    "분야명",
    "부문명",
    "프로그램명",
    "단위사업명",
    "세부사업명",
    "경비구분",
    "지출구분",
    "정부안금액(천원)",
    "국회확정금액(천원)",
)


def normalize_name(value: Any, compensate_account: bool = False) -> str:
    """NFKC, 확장자/긴 ID 제거, 특수문자 제거를 한곳에서 적용한다.

    회계명 괄호는 완전일치에는 보존한다. 첫 매칭이 실패했을 때만
    ``compensate_account=True`` 변형을 써서 R&D 같은 의미 괄호의 손실을 막는다.
    """
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"\.(?:hwp|hwpx|pdf|txt)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:[_\-\s])?\d{10,}$", "", text)
    if compensate_account:
        account_pattern = "|".join(re.escape(word.casefold()) for word in ACCOUNT_WORDS)
        text = re.sub(rf"[\(\[{{]\s*(?:{account_pattern})\s*[\)\]}}]", "", text)
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def parse_document_filename(path: str | Path) -> dict[str, Any] | None:
    """``연도_소관_세부사업명_긴ID.ext``를 오른쪽 ID까지 안전하게 분리한다."""
    path = Path(path)
    stem = re.sub(r"(?:[_\-\s])?\d{10,}$", "", path.stem)
    match = re.match(r"^(?P<year>\d{4})_(?P<ministry>[^_]+)_(?P<activity>.+)$", stem)
    if not match:
        return None
    return {
        "year": int(match.group("year")),
        "ministry": match.group("ministry").strip(),
        "activity_name": match.group("activity").strip(),
        "file_path": str(path.resolve()),
    }


def build_budget_metadata_text(row: pd.Series | dict[str, Any]) -> str:
    """예산편성 원천 컬럼을 자연어형 구조 텍스트로 만든다.

    ``biodiv_label``, ``confidence``, ``reason``, ``evidence``,
    ``biofin_category``는 정답 또는 정답 생성 과정의 정보이므로 입력에서
    의도적으로 제외한다.
    """
    lines = ["[세부사업 예산편성 정보]"]
    for column in METADATA_INPUT_COLUMNS:
        value = row.get(column, "")
        if value is None or pd.isna(value) or str(value).strip() == "":
            continue
        lines.append(f"{column}: {str(value).strip()}")
    return "\n".join(lines)


def discover_documents(doc_dir: str | Path) -> list[Path]:
    doc_dir = Path(doc_dir)
    if not doc_dir.is_dir():
        LOGGER.warning(
            "DOC_DIR이 없어 사업설명자료 없이 예산편성 정보만 사용합니다: %s",
            doc_dir,
        )
        return []
    return sorted(
        path
        for path in doc_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _fallback_label_files(label_file: Path) -> list[Path]:
    """통합 파일이 없으면 outputs 하위의 연도별 category 파일을 재귀 탐색한다."""
    if label_file.name != "세부사업 예산편성현황(총액)_years_category.csv":
        return []
    return sorted(
        label_file.parent.rglob(
            "세부사업 예산편성현황(총액)_[0-9][0-9][0-9][0-9]_category.csv"
        )
    )


def load_label_data(
    label_file: str | Path,
    label_column: str = "biofin_category",
    num_labels: int = 10,
) -> tuple[pd.DataFrame, list[Path]]:
    label_file = Path(label_file)
    files = [label_file] if label_file.is_file() else _fallback_label_files(label_file)
    if not files:
        raise FileNotFoundError(
            f"LABEL_FILE이 없습니다: {label_file}. "
            "통합 파일 또는 같은 폴더의 연도별 *_category.csv를 준비하세요."
        )
    if label_file not in files:
        LOGGER.warning("통합 LABEL_FILE이 없어 연도별 category CSV %d개를 합쳐 읽습니다.", len(files))

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = read_csv_flexible(path)
        frame["_source_file"] = str(path.resolve())
        frame["_source_row"] = range(2, len(frame) + 2)  # 헤더를 고려한 실제 CSV 행 번호
        frames.append(frame)
    labels = pd.concat(frames, ignore_index=True)
    missing = sorted((REQUIRED_COLUMNS | {label_column}) - set(labels.columns))
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(missing)}")

    labels["_year"] = pd.to_numeric(labels["회계연도"], errors="coerce").astype("Int64")
    numeric_label = pd.to_numeric(labels[label_column], errors="coerce")
    is_integer = numeric_label.notna() & (numeric_label % 1 == 0)
    in_range = numeric_label.between(0, num_labels - 1, inclusive="both")
    labels["_label"] = numeric_label.astype("Int64")
    labels["_label_status"] = "OK"
    labels.loc[numeric_label.isna(), "_label_status"] = "MISSING_LABEL"
    labels.loc[numeric_label.notna() & ~(is_integer & in_range), "_label_status"] = "INVALID_LABEL"
    invalid_count = int((labels["_label_status"] == "INVALID_LABEL").sum())
    missing_count = int((labels["_label_status"] == "MISSING_LABEL").sum())
    if invalid_count:
        LOGGER.warning(
            "0~%d 정수가 아닌 %s %d개를 학습에서 제외합니다.",
            num_labels - 1,
            label_column,
            invalid_count,
        )
    if missing_count:
        LOGGER.warning("%s이 비어 있는 행 %d개를 학습에서 제외합니다.", label_column, missing_count)
    labels["_ministry_norm"] = labels["소관명"].map(normalize_name)
    labels["_activity_norm"] = labels["세부사업명"].map(normalize_name)
    labels["_activity_account_norm"] = labels["세부사업명"].map(
        lambda value: normalize_name(value, compensate_account=True)
    )
    return labels, files


def _failure_row(
    reason: str,
    detail: str = "",
    document: dict[str, Any] | None = None,
    label_row: pd.Series | None = None,
) -> dict[str, Any]:
    return {
        "reason": reason,
        "detail": detail,
        "year": (document or {}).get("year", safe_scalar(label_row.get("회계연도")) if label_row is not None else ""),
        "ministry": (document or {}).get("ministry", safe_scalar(label_row.get("소관명")) if label_row is not None else ""),
        "activity_name": (document or {}).get(
            "activity_name", safe_scalar(label_row.get("세부사업명")) if label_row is not None else ""
        ),
        "file_path": (document or {}).get("file_path", ""),
        "source_file": (
            safe_scalar(label_row.get("_source_file"))
            if label_row is not None
            else (document or {}).get("source_file", "")
        ),
        "source_row": (
            safe_scalar(label_row.get("_source_row"))
            if label_row is not None
            else (document or {}).get("source_row", "")
        ),
    }


def match_documents_to_labels(
    documents: Sequence[Path], labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """연도/소관을 먼저 고정한 뒤 사업명을 3단계 우선순위로 매칭한다."""
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, row in labels.iterrows():
        if pd.notna(row["_year"]) and row["_ministry_norm"]:
            groups[(int(row["_year"]), row["_ministry_norm"])].append(index)

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    label_rows_with_document: set[int] = set()

    for path in documents:
        identity = parse_document_filename(path)
        if identity is None:
            failures.append(_failure_row("LABEL_NOT_FOUND", "파일명 형식을 해석할 수 없습니다", {"file_path": str(path.resolve())}))
            continue
        ministry_norm = normalize_name(identity["ministry"])
        activity_norm = normalize_name(identity["activity_name"])
        account_norm = normalize_name(identity["activity_name"], compensate_account=True)
        candidate_indices = groups.get((identity["year"], ministry_norm), [])

        exact = [idx for idx in candidate_indices if labels.at[idx, "_activity_norm"] == activity_norm]
        if exact:
            matched, match_type = exact, "EXACT"
        else:
            account_exact = [
                idx
                for idx in candidate_indices
                if account_norm and labels.at[idx, "_activity_account_norm"] == account_norm
            ]
            if account_exact:
                matched, match_type = account_exact, "ACCOUNT_COMPENSATED"
            else:
                contained = []
                for idx in candidate_indices:
                    target = labels.at[idx, "_activity_norm"]
                    if min(len(activity_norm), len(target)) >= 4 and (
                        activity_norm in target or target in activity_norm
                    ):
                        contained.append(idx)
                matched, match_type = contained, "CONTAINS"

        label_rows_with_document.update(matched)
        if not matched:
            failures.append(_failure_row("LABEL_NOT_FOUND", "연도/소관/세부사업명 일치 행 없음", identity))
            continue
        if len(matched) > 1:
            rows = ",".join(str(int(labels.at[idx, "_source_row"])) for idx in matched[:20])
            failures.append(
                _failure_row("MULTIPLE_LABEL_ROWS", f"{len(matched)}개 행 일치(source rows: {rows})", identity)
            )
            continue

        idx = matched[0]
        row = labels.loc[idx]
        if row["_label_status"] == "MISSING_LABEL":
            failures.append(_failure_row("LABEL_NOT_FOUND", "정답 label 값이 비어 있습니다", identity, row))
            continue
        if row["_label_status"] == "INVALID_LABEL":
            failures.append(
                _failure_row("INVALID_LABEL", "정답 label이 허용 범위를 벗어남", identity, row)
            )
            continue
        successes.append(
            {
                **identity,
                "label": int(row["_label"]),
                "match_type": match_type,
                "source_file": row["_source_file"],
                "source_row": int(row["_source_row"]),
                "source_type": "document",
                "metadata_text": build_budget_metadata_text(row),
            }
        )

    # 문서가 전혀 걸리지 않은 CSV 행도 감사 로그에 남긴다.
    for idx, row in labels.iterrows():
        if idx not in label_rows_with_document:
            detail = "매칭 문서 없음"
            if row["_label_status"] == "INVALID_LABEL":
                detail += "; 잘못된 정답 label"
            failures.append(_failure_row("DOC_NOT_FOUND", detail, label_row=row))

    success_frame = pd.DataFrame(
        successes,
        columns=[
            "year", "ministry", "activity_name", "file_path", "label",
            "match_type", "source_file", "source_row", "source_type", "metadata_text",
        ],
    )
    failure_frame = pd.DataFrame(
        failures,
        columns=[
            "reason", "detail", "year", "ministry", "activity_name",
            "file_path", "source_file", "source_row",
        ],
    )
    return success_frame, failure_frame


def extract_matched_documents(
    matched: pd.DataFrame,
    use_hwp_com: bool = False,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """매칭 성공 문서를 개별적으로 추출해 실패 문서만 격리한다."""
    records: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    selected = matched if limit is None else matched.head(limit)
    total = len(selected)
    for index, row in enumerate(selected.to_dict("records"), start=1):
        try:
            text = extract_document(row["file_path"], use_hwp_com=use_hwp_com)
        except DocumentParseError as exc:
            failures.append(_failure_row(exc.reason, exc.detail, row))
            continue
        metadata_text = row.get("metadata_text", "")
        combined_text = (
            f"{metadata_text}\n\n[사업설명자료 본문]\n{text}" if metadata_text else text
        )
        records.append({**row, "text": combined_text, "source_type": "document"})
        public_row = {key: value for key, value in row.items() if key != "metadata_text"}
        parsed_rows.append(
            {**public_row, "source_type": "document", "text_length": len(combined_text)}
        )
        if index % 50 == 0 or index == total:
            LOGGER.info(
                "본문 추출 %d/%d (성공=%d, 실패=%d)",
                index,
                total,
                len(records),
                len(failures),
            )
    return records, pd.DataFrame(parsed_rows), pd.DataFrame(failures)


def build_metadata_fallback_records(
    labels: pd.DataFrame,
    failures: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """문서가 없거나 파싱에 실패한 유효 label 행을 예산정보 텍스트로 대체한다."""
    fallback_reasons = {
        "DOC_NOT_FOUND",
        "HWP_PARSE_FAILED",
        "DOCUMENT_PARSE_FAILED",
        "EMPTY_DOCUMENT",
    }
    if failures.empty:
        return [], pd.DataFrame()

    label_lookup: dict[tuple[str, int], pd.Series] = {}
    for _, row in labels.iterrows():
        try:
            key = (str(row["_source_file"]), int(row["_source_row"]))
        except (TypeError, ValueError):
            continue
        label_lookup[key] = row

    records: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []
    used: set[tuple[str, int]] = set()
    for failure in failures.to_dict("records"):
        if failure.get("reason") not in fallback_reasons:
            continue
        try:
            key = (str(failure.get("source_file", "")), int(failure.get("source_row", 0)))
        except (TypeError, ValueError):
            continue
        if key in used or key not in label_lookup:
            continue
        row = label_lookup[key]
        if row["_label_status"] != "OK" or pd.isna(row["_year"]):
            continue
        text = build_budget_metadata_text(row)
        if len(text.splitlines()) <= 1:
            continue
        used.add(key)
        pseudo_path = f"metadata://{Path(key[0]).name}#row={key[1]}"
        record = {
            "year": int(row["_year"]),
            "ministry": safe_scalar(row["소관명"]),
            "activity_name": safe_scalar(row["세부사업명"]),
            "file_path": pseudo_path,
            "label": int(row["_label"]),
            "match_type": "METADATA_FALLBACK",
            "source_file": key[0],
            "source_row": key[1],
            "source_type": "budget_metadata",
            "text": text,
        }
        records.append(record)
        log_rows.append(
            {
                key_name: value
                for key_name, value in record.items()
                if key_name != "text"
            }
            | {"text_length": len(text)}
        )
    return records, pd.DataFrame(log_rows)


class BudgetDocumentDataset(Dataset):
    """문서 전체를 tokenizer token ID 기준의 겹치는 chunk로 반환한다."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        tokenizer: Any,
        max_length: int = 512,
        stride: int = 128,
    ) -> None:
        if torch is None:
            raise ImportError("학습에는 PyTorch가 필요합니다: pip install torch")
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
        self.chunk_body_length = max_length - self.special_tokens
        if self.chunk_body_length <= 0:
            raise ValueError("max_length가 special token 수보다 커야 합니다")
        if stride < 0 or stride >= self.chunk_body_length:
            raise ValueError(
                f"stride는 0 이상 본문 chunk 길이({self.chunk_body_length}) 미만이어야 합니다"
            )
        self._cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _encode(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        # 최신 transformers의 TokenizersBackend에는 prepare_for_model이 없다.
        # 공개 tokenizer 호출 API로 overflow chunk와 stride를 한 번에 생성한다.
        encoded = self.tokenizer(
            record["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            padding="max_length",
            return_attention_mask=True,
            return_overflowing_tokens=True,
        )
        input_ids = encoded["input_ids"]
        attention_masks = encoded["attention_mask"]
        # 일부 tokenizer는 단일 chunk를 1차원으로 반환하므로 형태를 통일한다.
        if input_ids and isinstance(input_ids[0], int):
            input_ids = [input_ids]
            attention_masks = [attention_masks]
        if not input_ids:
            raise ValueError(f"tokenizer 결과가 비어 있습니다: {record['file_path']}")

        previews: list[str] = []
        for chunk_ids, chunk_attention in zip(input_ids, attention_masks):
            valid_ids = [
                token_id
                for token_id, is_valid in zip(chunk_ids, chunk_attention)
                if is_valid
            ]
            previews.append(
                self.tokenizer.decode(valid_ids[:100], skip_special_tokens=True)
                .replace("\n", " ")[:300]
            )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "label": torch.tensor(int(record["label"]), dtype=torch.long),
            "meta": {
                "year": record["year"],
                "ministry": record["ministry"],
                "activity_name": record["activity_name"],
                "file_path": record["file_path"],
                "source_type": record.get("source_type", "document"),
                "chunk_text_previews": previews,
            },
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        # num_workers=0 기본값에서는 epoch마다 재토큰화하지 않도록 메모리 캐시한다.
        if index not in self._cache:
            self._cache[index] = self._encode(index)
        return self._cache[index]


def document_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """서로 다른 chunk 개수를 배치 최대치까지 2차원 padding한다."""
    if torch is None:
        raise ImportError("학습에는 PyTorch가 필요합니다: pip install torch")
    if not batch:
        raise ValueError("빈 배치는 collate할 수 없습니다")
    batch_size = len(batch)
    max_chunks = max(item["input_ids"].shape[0] for item in batch)
    max_length = batch[0]["input_ids"].shape[1]
    pad_token_id = 0
    input_ids = torch.full((batch_size, max_chunks, max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_chunks, max_length), dtype=torch.long)
    chunk_mask = torch.zeros((batch_size, max_chunks), dtype=torch.bool)
    for batch_index, item in enumerate(batch):
        chunk_count = item["input_ids"].shape[0]
        input_ids[batch_index, :chunk_count] = item["input_ids"]
        attention_mask[batch_index, :chunk_count] = item["attention_mask"]
        chunk_mask[batch_index, :chunk_count] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "chunk_mask": chunk_mask,
        "labels": torch.stack([item["label"] for item in batch]),
        "meta": [item["meta"] for item in batch],
    }
