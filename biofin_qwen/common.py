import json
from pathlib import Path

import torch

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_FILE = Path(__file__).with_name("sentiment.jsonl")
ADAPTER_DIR = Path(__file__).with_name("outputs") / "qwen-sentiment-lora"
SYSTEM = "한국어 문장의 감정을 분류하세요. 반드시 '긍정' 또는 '부정' 중 하나만 답하세요."


def load_split():
    """각 클래스의 다섯 번째 샘플을 평가용으로 빼서 16/4로 고정 분할한다."""
    rows = [json.loads(line) for line in DATA_FILE.read_text(encoding="utf-8").splitlines()]
    counts, train, test = {}, [], []
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
        (test if counts[row["label"]] % 5 == 0 else train).append(row)
    return train, test


def messages(text, label=None):
    result = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    if label is not None:
        result.append({"role": "assistant", "content": label})
    return result


def device_dtype():
    return ({"device_map": "auto", "dtype": torch.float16} if torch.cuda.is_available()
            else {"dtype": torch.float32})
