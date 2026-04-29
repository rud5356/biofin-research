# LLM NER Biodiversity

LLM을 활용해 생물다양성 문헌에서 개체명(종명, 지명, 날짜)을 자동 추출하는 NER 파이프라인입니다.  
PubMed API로 논문 초록을 수집하고, Ollama 기반 LLM으로 개체명을 추출합니다.

## 추출 대상 개체명

| 타입 | 설명 | 예시 |
|------|------|------|
| `SPECIES` | 종명 — 속명+종소명 형태의 학명만 추출 | Ailuropoda melanoleuca |
| `LOCATION` | 실제 지명, 국가명, 행정구역명 (서식지 일반명 제외) | Sichuan, China |
| `DATE` | 관찰 날짜, 기간 (처치 시점·발달 단계 제외) | April 2023, 2018–2021 |

> **추출 제외 항목**  
> - 목·과·속 단독 (Caudata, Anura 등)  
> - 유전자·단백질명 (SOX10, PLP1 등)  
> - 화합물·약물명 (cisplatin, Morusin 등)  
> - 세포주명 (HepG2, HeLa 등)  
> - 일반 생태 표현 ("multiple experimental systems" 등)  
> 약어 종명(D. suweonensis)은 문맥에서 전체 속명 확인 시 복원해서 출력합니다.

## 프로젝트 구조

```
llm_ner_biodiversity/
├── src/
│   ├── fetch_abstracts.py        # 1단계: PubMed에서 초록 수집 → abstracts.csv
│   ├── run_llm_ner_pipeline.py   # 2단계: 저장된 초록으로 LLM NER 실행
│   ├── visualize_locations.py    # 3단계: LOCATION 엔티티를 인터랙티브 지도로 시각화
│   ├── pubmed_client.py          # PubMed API 클라이언트 + 생물다양성 필터
│   ├── ner_pipeline.py           # NER 처리 로직 (체크포인트 포함)
│   ├── prompting.py              # LLM 프롬프트 관리 (few-shot 예시 포함)
│   ├── result_analysis.py        # 결과 분석 및 통계
│   └── config.py                 # 설정 및 상수
├── data/
│   └── abstracts.csv             # 수집된 초록 원본
├── results/
│   ├── ner_results_{model}_{mode}.csv     # 최종 NER 결과
│   ├── ner_results_checkpoint_{n}.csv     # 중간 저장 파일
│   ├── location_map.html                  # 지명 시각화 지도 (브라우저로 열기)
│   └── geocode_cache.json                 # 지오코딩 캐시 (재실행 시 재사용)
├── prompts/                      # 저장된 프롬프트 템플릿
└── environment.yml               # Conda 환경 설정
```

## 설치 및 환경 설정

### 1. Conda 환경 생성

```bash
conda env create -f environment.yml
conda activate llm_ner_biodiversity
```

### 2. 추가 패키지 설치

```bash
pip install folium geopy
```

### 3. Ollama 설치 및 모델 준비

[Ollama 공식 사이트](https://ollama.com)에서 설치 후:

```bash
ollama pull llama3.1:8b
ollama serve   # 백그라운드 실행 필요
```

### 4. config.py 설정

```python
NCBI_EMAIL = "your@email.com"       # NCBI API 사용 시 필요 (가입 불필요)
DEFAULT_KEYWORD = "Korean mammal species"
DEFAULT_LIMIT = 100                  # PubMed에서 가져올 논문 수 (필터 적용 시 목표 수)
DEFAULT_SAMPLE_SIZE = 30             # 실제 LLM으로 처리할 초록 수
DEFAULT_MODEL = "llama3.1:8b"        # Ollama 모델명
```

## 실행 방법

수집과 NER이 분리되어 있어 **키워드·프롬프트를 바꿔가며 반복 실험**할 수 있습니다.

### 1단계: 초록 수집

```bash
cd src

# 기본 실행 (키워드: "Korean mammal species", 100편, 필터 없음)
python fetch_abstracts.py

# 생물다양성 필터 적용 — 목표 편수를 채울 때까지 반복 수집
python fetch_abstracts.py --limit 200 --filter-biodiversity
```

`--filter-biodiversity` 옵션을 사용하면 의생명·임상 논문을 제외하고 **생태·분류·보전 관련 논문만** `data/abstracts.csv`에 저장합니다.  
목표 편수(`--limit`)를 채울 때까지 PubMed를 배치로 반복 조회하므로, 필터 후에도 지정한 수만큼 수집됩니다.

### 2단계: LLM NER 실행

```bash
# 기본 실행 (llama3.1:8b, few_shot)
python run_llm_ner_pipeline.py

# 모드 비교
python run_llm_ner_pipeline.py --mode zero_shot
python run_llm_ner_pipeline.py --mode few_shot --sample-size 50
```

### 3단계: 지명 지도 시각화

```bash
python visualize_locations.py

# 다른 결과 파일 지정
python visualize_locations.py --results-file ../results/ner_results_llama3.1:8b_few_shot.csv
```

Nominatim(OpenStreetMap)으로 지명을 위경도로 변환한 뒤 `results/location_map.html`을 생성합니다.  
브라우저로 열면 인터랙티브 지도에서 마커 클릭 시 출현 횟수와 관련 논문을 확인할 수 있습니다.  
지오코딩 결과는 `results/geocode_cache.json`에 캐싱되어 재실행 시 재사용됩니다.

### 주요 옵션

#### fetch_abstracts.py

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--keyword` | `"Korean mammal species"` | PubMed 검색 키워드 |
| `--limit` | `100` | 수집할 논문 수 (필터 적용 시 목표 수) |
| `--filter-biodiversity` | `False` | 생물다양성 논문만 수집 (의생명·임상 제외) |
| `--output-dir` | `data/` | abstracts.csv 저장 경로 |

#### run_llm_ner_pipeline.py

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--abstracts-file` | `data/abstracts.csv` | 수집된 초록 파일 경로 |
| `--sample-size` | `30` | 처리할 초록 수 |
| `--model` | `llama3.1:8b` | Ollama 모델명 |
| `--mode` | `few_shot` | 프롬프트 방식 (`few_shot` / `zero_shot`) |
| `--checkpoint-every` | `10` | N건마다 중간 저장 |
| `--output-dir` | `results/` | CSV 저장 경로 |

#### visualize_locations.py

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--results-file` | `results/ner_results_llama3.2_few_shot.csv` | NER 결과 CSV 파일 경로 |
| `--output` | `results/location_map.html` | 출력 HTML 지도 파일 경로 |
| `--cache` | `results/geocode_cache.json` | 지오코딩 캐시 파일 경로 |

## 출력 결과

### data/abstracts.csv

| 컬럼 | 내용 |
|------|------|
| `id` | PubMed ID (PMID) |
| `title` | 논문 제목 |
| `abstract` | 초록 전문 |

### results/ner_results_{model}_{mode}.csv

| 컬럼 | 내용 |
|------|------|
| `id` | PubMed ID (abstracts.csv와 조인 가능) |
| `title` | 논문 제목 |
| `abstract` | 초록 전문 |
| `entities` | 추출된 개체명 JSON 배열 |
| `parse_error` | JSON 파싱 실패 여부 |
| `error` | 오류 메시지 |
| `elapsed_sec` | 처리 시간(초) |

**entities 예시**:
```json
[
  {"type": "SPECIES", "text": "Ailuropoda melanoleuca"},
  {"type": "LOCATION", "text": "Sichuan"},
  {"type": "DATE", "text": "2018-2021"}
]
```

> 체크포인트 파일(`ner_results_checkpoint_{n}.csv`)은 처리 중단 시 복구용 중간 저장 파일입니다.

## 결과 분석

NER 파이프라인 실행 후 터미널에 자동으로 요약이 출력됩니다:
- 추출된 종명 / 지명 수
- 가장 많이 추출된 종명 Top 10
- JSON 파싱 실패율
- 평균 처리 시간

지명 시각화(`visualize_locations.py`) 실행 시:
- 고유 지명 수 및 총 출현 횟수
- 지오코딩 성공/실패 현황
- `results/location_map.html` — 브라우저에서 인터랙티브 지도 확인

## 의존성

- Python 3.10+
- `ollama` — 로컬 LLM 추론
- `biopython` — PubMed API 클라이언트
- `pandas` — 데이터 처리
- `folium` — 인터랙티브 지도 생성
- `geopy` — 지명 → 위경도 변환 (Nominatim)
