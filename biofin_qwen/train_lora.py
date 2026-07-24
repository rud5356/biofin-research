from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from common import ADAPTER_DIR, BASE_MODEL, device_dtype, load_split, messages

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **device_dtype())
train_rows, _ = load_split()
dataset = Dataset.from_list([{"text": tokenizer.apply_chat_template(
    messages(row["text"], row["label"]), tokenize=False
)} for row in train_rows])

lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
args = SFTConfig(
    output_dir=str(ADAPTER_DIR), num_train_epochs=3,
    per_device_train_batch_size=2, gradient_accumulation_steps=2,
    learning_rate=2e-4, logging_steps=1, save_strategy="no",
    max_length=256, report_to="none", fp16=False, bf16=False,
)
trainer = SFTTrainer(
    model=model, args=args, train_dataset=dataset,
    processing_class=tokenizer, peft_config=lora,
)
trainer.train()
trainer.model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))
print(f"LoRA 저장 완료: {ADAPTER_DIR}")
