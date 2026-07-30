# BIOFIN 1차 카테고리 Transformer (v1)

사업설명자료 본문을 입력으로 사용하여 `BIOFIN 1차 카테고리` 0~9를
분류하는 기존 10개 분류 모델입니다.

`BIOFIN 1차 카테고리`가 비어 있는 행은 카테고리 `0`으로 학습합니다.

## Docker에서 학습

```bash
docker exec -it -w /work/biofin_cls3 biofin python -u \
  transformer/v1/src/train_attention_classifier.py \
  --label_file document/2023biofin_label_matched.csv \
  --doc_dir document/2023/사업설명자료 \
  --document_only \
  --undersample_majority \
  --majority_cap_multiplier 10 \
  --majority_cap_min 1000 \
  --class_weight \
  --output_dir transformer/v1/outputs/2023_transformer_weighted
```

학습 전에 `--dry_run`을 추가하면 문서 매칭과 레이블 분포만 점검할 수
있습니다.

언더샘플링은 8:1:1 분할 후 학습 세트의 카테고리 `0`에만 적용됩니다.
상한은 `max(0 이외 클래스 건수의 중앙값 × 10, 1000)`으로 계산하며,
검증·테스트 세트는 원래 분포를 유지합니다.
