# BIOFIN 연구 프로젝트

2026-10-10 폴더 정리 기준입니다. 현재 개발 코드와 원본 자료는 기존 경로를 유지하고, 초기 버전과 과거 실험 결과는 before에 보관합니다.

## 현재 작업 위치

| 경로 | 용도 |
| --- | --- |
| [budget_biodiv_cls3](budget_biodiv_cls3/README.md) | 현재 분류 프로젝트 |
| [LLM v1](budget_biodiv_cls3/llm/v1/README.md) | 상위 카테고리 0~9, Ollama·vLLM 분류 |
| [LLM v2](budget_biodiv_cls3/llm/v2/README.md) | 하위 카테고리 분류 |
| [Transformer](budget_biodiv_cls3/transformer/README.md) | 모델 학습·예측 |
| [데이터 증강](budget_biodiv_cls3/augmentation_h200/README.md) | 증강 및 학습 데이터 구성 |
| [웹 데모](budget_biodiv_cls3/web/README.md) | 전문가 검증 화면 프로토타입 |
| [파이프라인](budget_biodiv_cls3/pipeline/README.md) | 단계별 분류 구조 |
| [실행 도우미](biofin-cli/README.md) | 서버에서 사용할 CLI |
| [크롤러](crawlers/README.md) | 열린재정·지방재정 자료 수집 |
| presentation | 발표자료 |
| example | 예제 자료 |
| [before](before/README.md) | 초기 버전·9월 LLM 결과·점검 결과·백업 |

## 입력자료 및 실행

- 현재 입력자료: budget_biodiv_cls3/document/open 및 document/local
- 2023년 평가용 입력: budget_biodiv_cls3/document/open/BIOFIN_2023_취합_2026.09.14_document_matched.csv
- 해당 CSV 정답 컬럼: 1차 카테고리
- 중앙재정 원문: crawlers/open_fiscal/outputs/{연도}/사업설명자료
- 지방재정 원문: crawlers/local_fiscal/outputs
- 모델·증강 학습 데이터와 원본 문서는 보관 필요성이 있어 유지했습니다.

프로젝트 루트에서 다음 도움말로 실제 실행 옵션을 확인할 수 있습니다. 실제 분류 실행에는 모델 서버 연결 설정이 필요합니다.

~~~powershell
cd C:\repos\biofin-research\budget_biodiv_cls3
python llm/v1/classify_biofin_category_with_vllm.py --help
python llm/v1/classify_biofin_category_with_ollama.py --help
~~~

## 최근 결과

결과의 기준 위치는 [LLM 결과 안내](budget_biodiv_cls3/llm/v1/outputs/README.md)를 참고합니다.

- 2023년 최종 비교·통합 엑셀: budget_biodiv_cls3/llm/v1/outputs/20261007_2023_3회비교
- 2024년 중앙·울산·강원 비교: budget_biodiv_cls3/llm/v1/outputs/20261006_2024_3회비교
- 10월 개별 실행 결과 및 일부 사업 재실행 기록도 유지했습니다.
- 2023년 정확도 92.90%는 기존 3,950건과 v9 재분류 22건을 합친 결과입니다. 전체를 v9로 재실행한 결과는 아닙니다.

## 보관 및 복원

before 안에는 이동 전의 상대 경로를 유지했습니다. 삭제한 파일은 없습니다. [이동 목록](before/이동목록_20261010.json)에 원래 경로와 보관 경로를 기록했습니다.
열린재정 크롤러의 과거 label 입력 기본 경로와 레이블 복원 코드의 구버전 참조는 before 위치로 수정했습니다.
과거 산출물 내부에 기록된 절대경로는 실행 이력 보존을 위해 변경하지 않았습니다. 보관 코드 재실행 시 경로를 확인하십시오.
