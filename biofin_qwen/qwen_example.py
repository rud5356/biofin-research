"""Qwen 모델과 토크나이저를 로컬 폴더에 다운로드한다."""
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SAVE_DIR = Path(__file__).with_name("models") / "qwen2.5-0.5b-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)
print(f"다운로드 완료: {SAVE_DIR}")
