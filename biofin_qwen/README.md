# Qwen LoRA 감정 분류 실습

20개의 한국어 문장을 16개 학습/4개 평가 데이터로 고정 분할합니다. 평가는 예시용 소규모 측정이므로 숫자를 일반화된 성능으로 해석하면 안 됩니다.

## 실행 순서

```powershell
cd C:\repos\biofin-research\biofin_qwen
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python convert_csv.py
python qwen_example.py       # 선택: 모델을 미리 로컬에 다운로드
python train_lora.py
python compare.py
```

`train_lora.py`와 `compare.py`는 Hugging Face에서 모델을 자동으로 내려받으므로 `qwen_example.py`는 생략해도 됩니다. LoRA 어댑터는 `outputs/qwen-sentiment-lora`에 저장됩니다. GPU가 없으면 CPU로도 실행되지만 오래 걸릴 수 있습니다.
