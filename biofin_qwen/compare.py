import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ADAPTER_DIR, BASE_MODEL, device_dtype, load_split, messages


def predict(model, tokenizer, text):
    prompt = tokenizer.apply_chat_template(messages(text), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    answer = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return "긍정" if "긍정" in answer else "부정" if "부정" in answer else answer


def evaluate(model, tokenizer, rows, name):
    predictions = [predict(model, tokenizer, row["text"]) for row in rows]
    correct = sum(pred == row["label"] for pred, row in zip(predictions, rows))
    print(f"\n{name}: {correct}/{len(rows)} = {correct / len(rows):.1%}")
    for row, pred in zip(rows, predictions):
        print(f"  정답={row['label']} 예측={pred!r} | {row['text']}")


adapter_config = ADAPTER_DIR / "adapter_config.json"
if not adapter_config.exists():
    raise FileNotFoundError(
        f"LoRA 학습 결과가 없습니다: {adapter_config}\n"
        "먼저 `python train_lora.py`를 오류 없이 끝까지 실행하세요."
    )

_, test_rows = load_split()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **device_dtype())
evaluate(base, tokenizer, test_rows, "학습 전")
lora_model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
evaluate(lora_model, tokenizer, test_rows, "학습 후")
