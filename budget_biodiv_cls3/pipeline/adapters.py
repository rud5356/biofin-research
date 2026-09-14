"""Model integration points. None means the model is not connected yet.

Replace each None with a callable taking list[dict] and returning
dict[str, Prediction]. Load a model once, then reuse it across batches.
Use pipeline_row_id to map predictions back to the exact input rows.
Only the remaining rows are passed to each adapter.

Example result: {row['pipeline_row_id']: Prediction(label=2, confidence=0.96)}
Existing model entry points:
  transformer/v1/src/predict_attention_classifier.py (categories 0-9)
  llm/v1/classify_biofin_category_with_ollama.py (categories 0-9)
Document lookup and model-specific text preparation belong in the adapter.
"""
from core import Prediction, Predictor

transformer_predictor: Predictor | None = None
llm_predictor: Predictor | None = None
