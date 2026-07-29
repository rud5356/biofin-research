# Transformer 분류기

사업설명자료 본문과 예산 메타데이터를 Attention Pooling Transformer로
학습하고 BIOFIN 1차 카테고리 0~9를 예측합니다.

프로젝트 루트에서 실행합니다.

```powershell
python transformer/src/train_attention_classifier.py --dry_run
python -m pytest transformer/tests
```

데이터는 프로젝트 루트의 `document/`, 결과는 `outputs/model_results/`를
기본 경로로 사용합니다.
