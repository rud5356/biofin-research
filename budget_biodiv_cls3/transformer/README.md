# BIOFIN Transformer

예산 메타데이터와 사업설명자료를 결합하고, 긴 문서를 token chunk로 나눠
Attention Pooling으로 문서 단위 분류를 수행합니다.
모든 명령은 `budget_biodiv_cls3`에서 실행합니다.

| 버전 | 정답 | 클래스 수 | 안내 |
| --- | --- | --- | --- |
| v1 | 상위 카테고리 0~9 | 기본 10개 | [v1 README](v1/README.md) |
| v2 | `6.05` 같은 계층형 하위 코드 | 입력에서 관측한 코드로 자동 결정 | [v2 README](v2/README.md) |

실행 파일은 `transformer/v1/src/`와 `transformer/v2/src/`에 있습니다.
이전 문서의 `transformer/src/` 경로는 사용하지 않습니다.

## 설치와 데이터 규칙

```powershell
python -m pip install -r requirements.txt
```

두 버전 모두 학습 CSV에 `회계연도`, `소관명`, `세부사업명` 및 지정한 정답 컬럼이 필요합니다.
기본 학습 입력은 `document/2023biofin_label.csv`, 문서는 `document/2023/사업설명자료`입니다.
최근 취합 CSV의 `1차`는 v1에서 `--label_column "1차"`로 지정합니다.
v2의 `하위` 단독 번호는 결합된 하위 코드와 다르므로 v2 입력 규칙을 따릅니다.

기본 분할은 사업 그룹 기준 학습/검증/테스트 8:1:1입니다.
`--document_only`가 없으면 매칭 문서가 없거나 파싱에 실패한 유효 정답 행을
예산 메타데이터 입력으로 대체할 수 있습니다.
`--dry_run`은 학습 전 로컬 데이터 점검용입니다.

## 결과와 실행 환경

- v1 기본 학습 결과: `transformer/v1/outputs/model_results/`
- v2 기본 학습 결과: `transformer/v2/outputs/subcategory_model/`
- 예측은 해당 버전에서 학습한 모델 폴더를 `--model_dir`로 지정합니다.
- v2는 `label_map.json`을 모델과 함께 보관합니다.

Transformer는 Python 프로세스의 PyTorch 실행 환경을 사용합니다.
LLM의 `--ollama-url` 옵션으로 Transformer 학습 GPU를 변경할 수는 없습니다.
Docker에서는 저장소를 컨테이너에 마운트하고 해당 폴더를 작업 디렉터리로 설정한 뒤
동일한 Python 명령을 실행합니다. 컨테이너 이름과 마운트 경로는 환경에 맞춰 지정합니다.
