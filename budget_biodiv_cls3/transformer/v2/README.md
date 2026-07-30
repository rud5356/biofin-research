# 하위 카테고리 Transformer (v2)

사업설명자료 본문을 입력으로 사용해 CSV의 `하위 카테고리`를 예측합니다.
기존 1차 카테고리 모델과 입력 및 학습 구조는 같고, 출력 레이블만 하위
카테고리로 변경했습니다.

## 레이블 처리

- `하위 카테고리`가 비어 있는 행은 하위 카테고리 코드 `0`으로 학습합니다.
- `0.0`, `2.040` 같은 값은 각각 `0`, `2.04`로 정규화합니다.
- 실제 코드 문자열을 모델 내부 정수 클래스 ID로 자동 변환합니다.
- 변환표는 모델 출력 폴더의 `label_map.json`에 저장합니다.
- 예측 CSV에는 내부 ID가 아니라 원래 하위 카테고리 코드가 기록됩니다.

## Docker에서 사전 점검

```bash
docker exec -it -w /work/biofin_cls3 biofin python -u \
  transformer/v2/src/train_attention_classifier.py \
  --label_file document/2023biofin_label_matched.csv \
  --doc_dir document/2023/사업설명자료 \
  --document_only \
  --dry_run \
  --output_dir transformer/v2/outputs/2023_dry_run
```

## 학습

```bash
docker exec -it -w /work/biofin_cls3 biofin python -u \
  transformer/v2/src/train_attention_classifier.py \
  --label_file document/2023biofin_label_matched.csv \
  --doc_dir document/2023/사업설명자료 \
  --document_only \
  --undersample_majority \
  --majority_cap_multiplier 10 \
  --majority_cap_min 1000 \
  --class_weight \
  --output_dir transformer/v2/outputs/2023_subcategory_transformer
```

분할 비율의 기본값은 학습/검증/테스트 `8:1:1`입니다. 문서가 같은 자료가
서로 다른 세트로 섞이지 않도록 그룹 단위로 분할합니다. 표본이 매우 적은
하위 카테고리는 검증·테스트 평가가 불안정할 수 있습니다.

언더샘플링은 분할 후 학습 세트의 하위 카테고리 `0`에만 적용됩니다.
상한은 `max(0 이외 클래스 건수의 중앙값 × 10, 1000)`으로 계산하며,
검증·테스트 세트는 원래 분포를 유지합니다.
