"""Rule -> transformer -> LLM routing, independent of model implementations."""
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Prediction:
    label: int
    confidence: float | None = None


# Adapters return predictions keyed by pipeline_row_id, never by row order.
Predictor = Callable[[list[dict]], dict[str, Prediction]]


@dataclass
class Config:
    zero_rules: list[dict] = field(default_factory=list)
    trusted_categories: list[int] = field(default_factory=list)
    min_confidence: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for label in self.trusted_categories:
            validate_prediction(Prediction(label))
        for label, threshold in self.min_confidence.items():
            if label not in [str(i) for i in range(10)]:
                raise ValueError(f"Invalid threshold category: {label}")
            if not 0 <= threshold <= 1:
                raise ValueError("Confidence thresholds must be between 0 and 1")
        for rule in self.zero_rules:
            if not rule.get("column") or not rule.get("keyword"):
                raise ValueError("Each rule needs a column and nonempty keyword")
            if rule.get("match", "contains") not in ("contains", "exact"):
                raise ValueError("Rule match must be contains or exact")


def validate_prediction(prediction):
    if not isinstance(prediction, Prediction):
        raise ValueError("Adapters must return Prediction objects")
    if type(prediction.label) is not int or prediction.label not in range(10):
        raise ValueError("Category must be an integer from 0 to 9")
    if prediction.confidence is not None and not 0 <= prediction.confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1")


OUTPUT_COLUMNS = [
    "pipeline_row_id", "pipeline_label", "pipeline_source", "pipeline_status",
    "pipeline_reason", "pipeline_transformer_label", "pipeline_transformer_confidence",
]


def run_pipeline(rows: list[dict], config: Config,
                 transformer: Predictor | None = None,
                 llm: Predictor | None = None) -> list[dict]:
    """Keep every input row in original order; unconnected stages stay pending.

    An omitted transformer prediction (e.g. missing document) routes to LLM.
    Adapter exceptions propagate, so operational failures cannot look successful.
    """
    result = []
    for index, original in enumerate(rows):
        if set(original) & set(OUTPUT_COLUMNS):
            raise ValueError("Input contains reserved pipeline output columns")
        row = {**original, **dict.fromkeys(OUTPUT_COLUMNS, "")}
        row.update(pipeline_row_id=str(index), pipeline_status="pending")
        for rule in config.zero_rules:
            if rule["column"] not in original:
                raise ValueError(f"Missing rule column: {rule['column']}")
            value = str(original[rule["column"]] or "")
            keyword = rule["keyword"]
            matches = value == keyword if rule.get("match", "contains") == "exact" else keyword in value
            if matches:
                row.update(pipeline_label=0, pipeline_source="rule",
                           pipeline_status="classified",
                           pipeline_reason=f"{rule['column']}:{keyword}")
                break
        result.append(row)

    def predict(adapter, candidates):
        if adapter is None or not candidates:
            return {}
        predictions = adapter([dict(row) for row in candidates])
        allowed = {row["pipeline_row_id"] for row in candidates}
        if not isinstance(predictions, dict) or not set(predictions) <= allowed:
            raise ValueError("Adapter returned invalid or unknown row IDs")
        for prediction in predictions.values():
            validate_prediction(prediction)
        return predictions

    remaining = [row for row in result if row["pipeline_status"] == "pending"]
    predictions = predict(transformer, remaining)
    for row in remaining:
        prediction = predictions.get(row["pipeline_row_id"])
        if prediction is None:
            continue
        row["pipeline_transformer_label"] = prediction.label
        row["pipeline_transformer_confidence"] = prediction.confidence if prediction.confidence is not None else ""
        threshold = config.min_confidence.get(str(prediction.label))
        accepted = prediction.label in config.trusted_categories and (
            threshold is None or (prediction.confidence is not None and prediction.confidence >= threshold))
        if accepted:
            row.update(pipeline_label=prediction.label, pipeline_source="transformer",
                       pipeline_status="classified", pipeline_reason="trusted_category")

    remaining = [row for row in result if row["pipeline_status"] == "pending"]
    predictions = predict(llm, remaining)
    for row in remaining:
        prediction = predictions.get(row["pipeline_row_id"])
        if prediction is not None:
            row.update(pipeline_label=prediction.label, pipeline_source="llm",
                       pipeline_status="classified", pipeline_reason="llm_classification")
        else:
            row["pipeline_reason"] = "llm_not_connected" if llm is None else "llm_no_prediction"
    return result
