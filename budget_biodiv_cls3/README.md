# BIOFIN 예산사업 분류

예산 CSV의 사업정보와 사업설명자료를 사용해 BIOFIN 카테고리를 분류합니다.
Transformer 학습·예측, Ollama LLM 분류, 단계별 라우팅 파이프라인을 포함합니다.

## 실행 위치와 구성

이 문서와 하위 README의 명령은 별도 설명이 없으면 다음 폴더에서 실행합니다.

```powershell
Set-Location C:\repos\biofin-research\budget_biodiv_cls3
```

실제 폴더명은 `budget_biodiv_cls3`입니다. `budget/_biodiv/_cls3`로 분리된 경로가 아닙니다.

| 구성 | 역할 | 안내 |
| --- | --- | --- |
| Transformer v1 | 상위 카테고리 0~9 학습·예측 | [README](transformer/v1/README.md) |
| Transformer v2 | 계층형 하위 코드 학습·예측 | [README](transformer/v2/README.md) |
| LLM v1 | Ollama로 상위 카테고리 0~9 분류 | [README](llm/v1/README.md) |
| LLM v2 | Ollama로 비해당 0 및 39개 하위 코드 분류 | [README](llm/v2/README.md) |
| 단계별 파이프라인 | 규칙 → Transformer → LLM 라우팅, 모델 연결은 미구현 | [README](pipeline/README.md) |

Transformer 설치는 `python -m pip install -r requirements.txt`, LLM 문서 파싱 설치는
`python -m pip install -r llm/requirements.txt`를 사용합니다.

## 입력 CSV와 정답 컬럼

최근 정리한 파일은 `document/open/BIOFIN_2023_취합_2026.09.14.csv`입니다.
UTF-8 BOM으로 저장했으며 `BIOFIN분류`는 삭제하고 `1차`, `하위`의 빈칸을 0으로 채웠습니다.

| 용도 | 코드의 기본 정답 컬럼 | 최근 CSV 사용 시 |
| --- | --- | --- |
| Transformer v1 학습 | `BIOFIN 1차 카테고리` | `--label_column "1차"` 지정 |
| LLM v1 평가 | `BIOFIN 1차 카테고리` | `--gold-label-col "1차"` 지정 |
| Transformer v2 학습 | `하위 카테고리` | 상위·하위를 결합한 코드 준비 필요 |
| LLM v2 평가 | `하위 카테고리` | `6.05` 같은 결합 코드 준비 필요 |

`1차`와 `BIOFIN 1차 카테고리`는 자동으로 같은 컬럼으로 인식되지 않습니다.
Transformer v1의 필수 컬럼 오류는 `--label_column`으로 실제 헤더를 지정해 해결합니다.
`BIOFIN분류`를 복원할 필요는 없습니다. 코드 기본값 자체는 변경하지 않았습니다.

`하위`의 단독 번호 5와 계층 코드 `6.05`는 다릅니다. v2에 단순히
`--label_column "하위"`만 지정하면 상위 정보가 사라질 수 있으므로
[v2 입력 규칙](transformer/v2/README.md)을 먼저 확인합니다.

## 문서 매칭

기존 기본 입력 `document/2023biofin_label.csv`를 열린재정 목록 및 문서와 매칭합니다.

```powershell
python match_2023_biofin_documents.py
```

기본 문서 폴더는 `document/2023/사업설명자료`, 목록은
`document/2023/open_fiscal_2023.csv`입니다. 결과는
`document/2023biofin_label_matched.csv`, 실패 내역은
`document/2023biofin_label_match_failed.csv`입니다.
다른 입력은 `--label-csv`, 출력은 `--output-csv`, `--failure-csv`로 지정합니다.
이 스크립트는 정답 컬럼명을 바꾸거나 하위 코드를 만드는 용도가 아닙니다.

## 최근 CSV로 Transformer v1 점검

```powershell
python transformer/v1/src/train_attention_classifier.py `
  --label_file "document/open/BIOFIN_2023_취합_2026.09.14.csv" `
  --label_column "1차" `
  --doc_dir "document/2023/사업설명자료" `
  --output_dir "transformer/v1/outputs/20260914_check" `
  --dry_run
```

`--dry_run`은 모델 학습 없이 데이터 매칭·분포를 점검하고 로컬 점검 결과를 저장합니다.
`--document_only`를 추가하면 문서가 없는 행의 예산정보 대체 입력을 제외합니다.
학습·예측 명령과 결과 파일은 각 Transformer README에 정리했습니다.

## Ollama 연결

기본 연결은 `http://localhost:11434`, 모델은 `gemma3:12b`입니다.
회사 공용 서버는 `--ollama-url`, 설치된 모델은 `--model`로 별도 지정합니다.
원격 접속정보, 조회 방법, 소량 테스트와 캐시 분리는 [LLM 공통 안내](llm/README.md)를 참고합니다.

Transformer의 옵션은 `--label_file`, `--dry_run`처럼 밑줄을,
LLM은 `--input-file`, `--dry-run`처럼 하이픈을 사용합니다.

## 전문가 검증 UI 프로토타입 (데모)

`web/` 폴더에 분류 결과·전문가 검증 화면의 클릭 가능한 프론트엔드 프로토타입이 있습니다.
실제 모델·서버·DB와 연결되지 않은 가상 데이터 데모이며, 위 Transformer/LLM/파이프라인 코드와는 독립적입니다.
실행 방법은 [web/README.md](web/README.md)를 참고합니다.
