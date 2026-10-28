# 회사 서버에서 카테고리별 최대 400개까지 증강

## 메뉴 실행

biofin_menu에서 4. 데이터 증강을 선택합니다.
--project-dir에는 transformer, document가 들어 있는 서버의 프로젝트 폴더를 지정합니다.
원본 CSV, 결과 폴더, vLLM/Ollama, API 기본 URL(/v1 제외), 모델, 목표 건수를 선택합니다.
기본 API와 모델은 biofin-cli/config.json에서 가져옵니다.
Linux, Docker, Python 3.10 이상과 접속 가능한 모델 API가 필요합니다.
기본 컨테이너 이미지는 python:3.11-slim이며 추가 Python 패키지는 필요 없습니다.
vLLM 인증은 호스트의 VLLM_API_KEY(또는 설정된 VLLM_API_KEY_ENV)를 전달합니다.
AUG_API_KEY가 있으면 우선 사용합니다. 서버 네트워크·인증·모델 응답은 별도 확인이 필요합니다.

## 저장 위치

메뉴에서 고른 결과 폴더 아래 balanced_h200_v1/aug-*.jsonl로 저장합니다.
기본 제안 위치는 <프로젝트>/transformer/v1/outputs/augmentation/balanced_h200_v1/입니다.
메뉴가 실행 전에 호스트 서버의 전체 저장 경로를 출력합니다.
Docker 내부 /workspace는 호스트 프로젝트 폴더와 연결되어 결과가 서버에 바로 저장됩니다.
모델 API 서버에 결과를 저장하는 방식이 아닙니다.

## 동작

- 원본+증강 합계가 가장 적은 1차 카테고리를 골라 50개씩 생성하고, 모든 카테고리 합계가 400개가 되면 종료합니다.
- 원본이 이미 400개 이상인 카테고리는 증강하지 않습니다. 400을 넘기지 않도록 마지막 배치는 50개보다 작을 수 있습니다.
- 예: 원본 30개인 카테고리는 370개, 원본 380개인 카테고리는 20개, 원본 500개인 카테고리는 0개를 생성합니다.
- --target-per-category로 최대 건수(원본 포함)를 변경할 수 있습니다.
- 완성된 배치만 저장합니다.
- 동일 원본과 같은 출력 경로로 재실행하면 이어갑니다. 이미 목표 이상이면 건너뜁니다.
- 기본 날짜 제한은 없습니다. --until로 지정할 수 있습니다.
- 공간 부족 시 기존 파일을 보존하고 오류로 멈춥니다. 기본 여유 공간은 1GB입니다.
- 같은 카테고리에 이미 저장된 문장은 새 배치에서 제외합니다.
- 원본 CSV는 변경하지 않습니다. 생성문은 의미 보존 검토가 필요합니다.

필수 CSV 열: 세부사업명, business_key와 1차 라벨 열.
1차 라벨은 1차 카테고리 / BIOFIN 1차 카테고리 / biofin_category 중 하나를 자동 인식합니다.
1차 증강에서는 하위 카테고리 열이 없어도 됩니다. 정답이 빈 행은 제외합니다.
하위 카테고리 증강에는 하위 라벨이 필요하며, 서로 충돌하는 라벨 열은 오류로 처리합니다.
UTF-8 BOM CSV를 기본으로 읽으며 split 열이 있으면 train 행만 사용합니다.
사업명·예산정보만 사용합니다. 연결된 HWP/PDF 본문은 읽지 않습니다.

## 직접 실행 및 학습

    python transformer/v1/augmentation_h200/balanced_csv.py --input document/labeled.csv --output transformer/v1/outputs/augmentation --backend vllm --base-url http://localhost:8000 --model MODEL_ID --target-per-category 400

--plan은 원본 분포와 계획만 출력합니다. --mock은 저장 시험용이며 학습에 사용하지 않습니다.

    python transformer/v1/augmentation_h200/build_merged_training_csv.py --original-csv document/original.csv --label-csv document/labeled.csv --augmentation-dir transformer/v1/outputs/augmentation/balanced_h200_v1 --output-dir transformer/v1/outputs/merged_training

학습 메뉴에서 원본 + 증강 통합 CSV를 선택합니다.
검증/시험 원본에 연결된 증강문은 학습에서 제외되므로 실제 학습 수는 목표보다 적을 수 있습니다.
이미 병합 CSV가 있으면 다시 생성하지 않고 학습 메뉴의 클래스별 최대 개수를 400으로 지정할 수 있습니다.


## 실행 로그

시작 시 입력·저장 경로를 출력합니다. 기본 10초마다 경과 시간(HH:MM:SS), 상태, 저장 진행률, 카테고리별 건수, 현재 미저장 배치 건수를 출력합니다. API 응답 대기 중에도 표시됩니다. --log-interval 5로 주기를 변경할 수 있습니다. 배치 저장 시에도 즉시 출력하며 종료·오류 시 총 실행 시간을 표시합니다. 경과 시간은 재실행할 때 새로 시작하고 저장 건수는 이전 결과를 포함합니다. 로그는 터미널에 출력하며 별도 로그 파일은 자동 생성하지 않습니다.
