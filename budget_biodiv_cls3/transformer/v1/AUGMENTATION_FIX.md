# 증강 연결 수정 및 재학습

문서 파일명에서 가져온 사업명에는 예산 코드가 붙어 있었지만 증강문 부모의
사업명은 원본 CSV의 이름이었다. `build_dataset.py`에서 문서를 정답 행과
매칭한 직후 소관명과 사업명을 원본 CSV 값으로 통일한다. 명시적 문서 경로
매칭과 일반 이름 매칭 모두 적용하며, 이 처리는 데이터 분할 전에 수행한다.

`augmentation_csv.py`는 원본 CSV 행과 그룹 키가 다르면 오류로 중단한다.
클래스별 전체 증강량, 부모 미연결 제외량, 실제 추가량도 기록한다.
`--dry_run`도 학습과 동일하게 원본 언더샘플링 후 증강 연결을 검사한다.

## 기존 결과의 처리

`20260930_005914`의 체크포인트와 결과는 수정하지 않았다. 그룹 이름을
바로잡으면 기존 분할에서 서로 다른 세트에 걸친 같은 사업 그룹 24개가
확인된다. 기존 분할이나 체크포인트로 수정 후 성능을 주장하면 안 된다.
새 출력 폴더에서 원본을 다시 그룹 분할하고 처음부터 학습해야 한다.

## 이전 파싱 결과로 연결만 검증

`budget_biodiv_cls3`에서 실행한다. 모델, 토크나이저, 문서 본문을 다시
로드하지 않고 이전 성공 문서 목록과 원본 CSV를 재매칭한다. 매칭되는
원본 행 번호와 정답이 이전과 같은지 확인한 뒤 실제 분할/언더샘플링/
증강 연결 함수를 실행한다. 학습 옵션은 seed 42, 원본 0 상한 300이다.

```bash
python transformer/v1/src/audit_augmentation_linkage.py --label_file outputs/merged_training_20260930/open_fiscal_2023_augmented.csv --previous_output transformer/v1/outputs/20260930_005914 --output_dir transformer/v1/outputs/20260930_augmentation_linkage_fixed
```

`linkage_audit.json`이 최종 학습 후보 건수와 증강 연결 결과다.
이 감사 폴더의 `split_summary.json`과 `split_assignments.csv`는
언더샘플링/증강 전 원본 분할이다. 본문 파싱 성공 여부나 scikit-learn 버전이
다르면 서버의 실제 재학습 건수는 달라질 수 있다.
`--reuse_previous_split`은 기존 분할의 진단용이며 사업 중복이 있으면 중단한다.

2026-09-30 로컬 검증(Python 3.11, scikit-learn 1.6.1): 원본 3,972건,
증강 추가 234,144건, 최종 학습 후보 234,880건, 그룹 중복 0건.
클래스 1은 22,786건, 클래스 3은 25,894건이며 모든 클래스에 증강이 연결됐다.
평가 사업 증강 66,701건은 제외했고, 원본 0 언더샘플링으로 부모가 빠진
증강 20,680건도 제외했다. 회귀 테스트 9개가 통과했다.
이는 데이터 연결 검증 결과이며 새 모델 학습/성능 평가 결과가 아니다.

## 서버 재학습

수정된 `build_dataset.py`, `augmentation_csv.py`,
`train_attention_classifier.py`를 서버의 `transformer/v1/src/`에도 반영한다.
아래는 이전 로그의 서버 경로(`/work/biofin_cls3`) 기준 예시다.
기존 실험의 다른 하이퍼파라미터는 사용자가 실행했던 옵션과 맞춘다.

```bash
cd /work/biofin_cls3
python transformer/v1/src/train_attention_classifier.py --label_file document/2023/open_fiscal_2023_augmented.csv --doc_dir document/2023/사업설명자료 --undersample_majority --majority_cap_multiplier 1 --majority_cap_min 300 --epochs 5 --seed 42 --output_dir transformer/v1/outputs/20260930_augmented_fixed_retrain
```

먼저 같은 명령에 `--dry_run`을 붙여 서버 입력 경로와 증강 요약을 확인할 수 있다.
실제 학습 시에는 최종 `augmentation_summary.json`과 `split_summary.json`을 확인한다.
`--balanced_sampling`을 사용하면 연결된 모든 행을 매 epoch 사용하는 것이 아니라
`--samples_per_class`만큼 클래스별 추출한다. 기본값은 클래스당 100건이다.

증강 문장이 수만 건이어도 독립적인 원본 사업 수가 늘어난 것은 아니다.
희소 클래스는 원본 사업 단위의 검증/테스트 표본이 여전히 작다.
