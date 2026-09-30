# 원본 기준선과 제한 증강 비교

서버 `/work/biofin_cls3/transformer/v1/src/`에 아래 4개 파일을 반영한다.
`build_dataset.py`, `augmentation_csv.py`, `train_attention_classifier.py`, `evaluate.py`.
증강 CSV는 기존 파일을 그대로 사용한다.

## 1. 원본만 학습

```bash
cd /work/biofin_cls3
python transformer/v1/src/train_attention_classifier.py \
  --label_file document/2023/open_fiscal_2023_augmented.csv \
  --doc_dir document/2023/사업설명자료 \
  --augmentation_per_class 0 \
  --learning_rate 5e-6 --batch_size 1 --epochs 5 --seed 42 \
  --eval_steps 500 \
  --output_dir transformer/v1/outputs/20260930_original_only_lr5e6
```

## 2. 원본 + 클래스당 최대 300개 증강

```bash
cd /work/biofin_cls3
python transformer/v1/src/train_attention_classifier.py \
  --label_file document/2023/open_fiscal_2023_augmented.csv \
  --doc_dir document/2023/사업설명자료 \
  --augmentation_per_class 300 \
  --learning_rate 5e-6 --batch_size 1 --epochs 5 --seed 42 \
  --eval_steps 500 \
  --output_dir transformer/v1/outputs/20260930_aug300_lr5e6
```

두 실험 모두 `--class_weight`, `--balanced_sampling`, `--undersample_majority`를
붙이지 않는다. 원본은 유지하며 증강 상한은 원본 수에 적용하지 않는다.
같은 입력, 파싱 환경, 라이브러리 버전과 seed에서 원본 분할을 동일하게 사용한다.
기존 best_model.pt를 이어서 학습하지 않고 사전학습 모델에서 새로 시작한다.

`--augmentation_per_class 0`은 증강을 사용하지 않는다.
양수는 클래스당 증강 상한으로, 고정 seed로 원본 사업별 순회 선택을 하여
증강량이 많은 일부 사업에 편중되지 않도록 한다. 선택된 증강 세트는 epoch 간 고정된다.
옵션을 생략하면 이전처럼 전체 증강을 사용하므로 두 명령에는 반드시 명시한다.

`--eval_steps 500`은 500 학습 batch마다 검증한다. 검증 이후 학습 모드로 복귀하며,
중간 검증과 epoch 종료 검증 모두 검증 Macro F1이 개선되면 best_model.pt를 저장한다.
early stopping은 중간 검증을 포함해 개선이 없는 epoch가 2회 연속일 때 적용한다.
`validation_step_log.csv`의 prediction_counts로 한 클래스 편중 여부를 확인한다.
테스트 결과로 두 실험의 설정을 선택하지 말고 검증 Macro F1을 비교한다.

로컬 검증: 샘플링·기존 기능·CLI 회귀 테스트 11개 통과. 실제 GPU 학습은 미실행.
