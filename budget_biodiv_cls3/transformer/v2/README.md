# Transformer v2: BIOFIN 하위 카테고리

예산정보와 사업설명자료로 `6.05` 같은 계층형 하위 코드를 학습·예측합니다.
명령은 `budget_biodiv_cls3`에서 실행합니다. 설치는 [공통 안내](../README.md)를 참고합니다.

## 입력과 계층 코드

기본 정답 컬럼은 `하위 카테고리`입니다. 클래스 수는 고정 35개가 아니라
학습 CSV에서 관측한 코드로 결정합니다. `--num_labels` 기본값 0은 자동 설정입니다.

- 비해당은 `0`, 상위 6·하위 5는 `6.05`입니다.
- `0.0`, `2.040`은 각각 `0`, `2.04`로 정규화합니다.
- 빈 정답은 0으로 처리합니다.
- `BIOFIN 1차 카테고리`와 `하위`가 함께 있으면 결합 코드와 일치하는지 검증합니다.
- 결합 정답이 빈칸 또는 0 이외 단독 정수이면 해당 원본 계층 컬럼으로 복원할 수 있습니다.
- 클래스 ID 매핑은 `label_map.json`에 저장하고, 예측 결과에는 코드 문자열을 기록합니다.

최근 `document/open/BIOFIN_2023_취합_2026.09.14.csv`의 컬럼은 `1차`, `하위`입니다.
현재 코드의 상위 원본 컬럼 상수는 `BIOFIN 1차 카테고리`이므로 `1차`를 자동 인식하지 않습니다.
이 파일의 `하위`만 정답으로 지정하면 서로 다른 상위 카테고리의 같은 하위 번호가
하나의 클래스로 합쳐질 수 있습니다.

사용 전 별도 학습 CSV에 `하위 카테고리` 결합 코드를 만들고,
상위 컬럼도 `BIOFIN 1차 카테고리`로 맞춰 계층 검증을 받도록 준비합니다.
이 문서 업데이트은 데이터 변환이나 코드 상수 변경을 수행하지 않습니다.
관련 함수는 `src/build_dataset.py`의 `compose_subcategory_code()`와
`build_hierarchical_label_codes()`입니다. LLM v2 평가에도 같은 결합 코드를 사용합니다.

## 점검과 학습

아래는 기존 결합 정답이 있는 매칭 CSV를 사용하는 예시입니다.

```powershell
python transformer/v2/src/train_attention_classifier.py `
  --label_file "document/2023biofin_label_matched.csv" `
  --label_column "하위 카테고리" `
  --doc_dir "document/2023/사업설명자료" `
  --document_only --dry_run `
  --output_dir "transformer/v2/outputs/2023_check"
```

```powershell
python transformer/v2/src/train_attention_classifier.py `
  --label_file "document/2023biofin_label_matched.csv" `
  --label_column "하위 카테고리" `
  --doc_dir "document/2023/사업설명자료" `
  --document_only --undersample_majority --class_weight `
  --majority_cap_multiplier 10 --majority_cap_min 1000 `
  --output_dir "transformer/v2/outputs/2023_subcategory_model"
```

사업 그룹 단위로 기본 8:1:1 분할합니다. 위 옵션의 언더샘플링은 학습 세트의
비해당 0에 적용하고 검증·테스트 분포는 유지합니다. `--document_only`를 빼면
문서가 없거나 파싱에 실패한 유효 정답 행을 예산정보로 대체할 수 있습니다.

기본 출력 폴더는 `transformer/v2/outputs/subcategory_model`입니다.
`best_model.pt`, `tokenizer/`, `label_map.json`, 분할 내역과 평가 결과를 함께 보관합니다.

## 예측

```powershell
python transformer/v2/src/predict_attention_classifier.py `
  --model_dir "transformer/v2/outputs/2023_subcategory_model" `
  --doc_dir "document/2023/사업설명자료" `
  --budget_file "document/2023biofin_label_matched.csv" `
  --label_column "하위 카테고리" `
  --output_dir "transformer/v2/outputs/2023_predictions" `
  --no-heatmap
```

해당 v2 모델의 매핑으로 예측 코드를 복원합니다. v1 체크포인트를 대신 사용할 수 없습니다.
표본이 적은 하위 코드의 평가는 전체 정확도와 별도로 확인합니다.
