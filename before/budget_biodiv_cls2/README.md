# budget_biodiv_cls2

정부 예산·결산 사업을 BIOFIN 분류 기준에 따라 생물다양성 관련 여부(0/1)로 라벨링하는 파이프라인입니다.
Ollama 로컬 LLM을 사용하며, SHA256 해시 기반 캐시로 중복 호출을 방지합니다.

## 파이프라인 개요

```
[입력 CSV]
사업별결산세출지출현황_20XX.csv
세부사업 예산편성현황(총액)_20XX.csv
        │
        ▼
label_biodiv_with_ollama.py
  → outputs/label_cache.csv       (고유 사업 조합별 라벨 캐시)
  → outputs/label_audit.csv       (전체 검수 결과)
  → outputs/review_needed.csv     (confidence 낮은 항목)
  → outputs/{원본파일명}_labeled.csv (라벨 적용 결과)
        │
        ▼ (캐시를 다른 CSV에 재적용할 때)
apply_cache_labels.py
  → outputs/{원본파일명}_labeled.csv
```

## 폴더 구조

```
budget_biodiv_cls2/
├── label_biodiv_with_ollama.py   # 라벨링 메인 스크립트
├── apply_cache_labels.py         # 기존 캐시를 다른 CSV에 재적용
├── BIOFIN PROMPT.txt             # 프롬프트 v1 (초기 엄격 기준)
├── BIOFIN PROMPT_V2.txt          # 프롬프트 v2 (BIOFIN 9대 범주 기준)
├── BIOFIN PROMPT_V3_KEI버전.txt  # 프롬프트 v3 (UNDP·KEI 기준 + 키워드 확장, 포용적)
├── BIOFIN_PROMPT_V4_KEI버전.txt  # 프롬프트 v4 (현재 적용, 애매하면 0 원칙)
├── 사업별결산세출지출현황_20XX.csv  # 입력: 결산 데이터 (2019~2023)
├── 세부사업 예산편성현황(총액)_20XX.csv  # 입력: 예산 데이터 (2019~2023)
└── outputs/
    ├── label_cache.csv           # 사업 조합별 라벨 캐시 (재실행 시 재사용)
    ├── label_audit.csv           # 전체 고유 사업 검수 결과
    ├── review_needed.csv         # confidence < 0.7 항목 목록
    ├── run_summary.json          # 실행 요약
    └── *_labeled.csv             # 라벨 적용된 최종 결과 파일
```

## 핵심 설계

**캐시 기반 중복 방지**
`소관명 / 분야명 / 부문명 / 프로그램명 / 단위사업명 / 세부사업명` 6개 컬럼의 조합을 SHA256(24자)으로 해시해 캐시 키로 사용합니다. 같은 사업 조합은 연도가 달라도 캐시를 재사용하므로 Ollama 호출 횟수를 최소화합니다. 중단 후 재실행해도 처리된 항목은 건너뜁니다.

**판단 기준 (현재: v4)**
- BIOFIN 9대 범주 → UNDP(2018) 원문 → KEI 세부 기준 → 키워드(보조) 순으로 판단
- 사업명에서 실질 내용이 확인되는 경우에만 1, 애매하면 0
- 소관명·부처명은 판단 근거로 사용 금지
- 키워드 단독으로 1 부여 금지

## 사용법

### 기본 실행 (현재 디렉토리의 CSV 전체 라벨링)

```bash
python label_biodiv_with_ollama.py
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--input-dir` | 스크립트 위치 | 입력 CSV가 있는 폴더 |
| `--input-glob` | `세부사업 예산편성현황(총액)_*.csv` | 처리할 파일 패턴 |
| `--output-dir` | `outputs/` | 결과 저장 폴더 |
| `--cache-csv` | `outputs/label_cache.csv` | 캐시 파일 경로 |
| `--model` | `llama3.1:8b` | Ollama 모델명 |
| `--workers` | `3` | 동시 Ollama 호출 스레드 수 |
| `--limit-keys` | `0` (전체) | 테스트용: N개 사업만 처리 |
| `--dry-run` | — | Ollama 호출 없이 구조 확인만 |
| `--overwrite` | — | 기존 캐시 무시하고 재라벨링 |

```bash
# 테스트 (50개 사업만)
python label_biodiv_with_ollama.py --limit-keys 50 --dry-run

# 다른 파일 패턴 지정
python label_biodiv_with_ollama.py --input-glob "사업별결산세출지출현황_*.csv"

# 캐시 이름 지정 (프롬프트 버전별 관리)
python label_biodiv_with_ollama.py --cache-csv outputs/label_cache_v4.csv
```

### 기존 캐시를 다른 CSV에 재적용

```bash
python apply_cache_labels.py
python apply_cache_labels.py --input-dir ../other_dir --cache outputs/label_cache_v4.csv
```

## 출력 컬럼

`_labeled.csv`에 원본 컬럼 외에 다음 4개가 추가됩니다.

| 컬럼 | 설명 |
|------|------|
| `biodiv_label` | 0 (비관련) 또는 1 (관련) |
| `confidence` | LLM 신뢰도 (0.0~1.0) |
| `reason` | 판단 근거 (2~3문장) |
| `evidence` | 판단에 사용된 사업명 내 실제 단어/어구 |

## 라벨링 결과 (v4 기준, 세부사업 예산편성현황 2019~2023)

| 연도 | 전체 | 1 (관련) | 0 (비관련) |
|------|-----:|--------:|----------:|
| 2019 | 7,891 | 3,213 (40.7%) | 4,678 (59.3%) |
| 2020 | 8,235 | 3,419 (41.5%) | 4,816 (58.5%) |
| 2021 | 8,590 | 3,522 (41.0%) | 5,068 (59.0%) |
| 2022 | 8,959 | 3,782 (42.2%) | 5,177 (57.8%) |
| 2023 | 9,074 | 3,841 (42.3%) | 5,233 (57.7%) |
| **합계** | **42,749** | **17,777 (41.6%)** | **24,972 (58.4%)** |

## 프롬프트 버전 이력

| 버전 | 특징 | 관련 건수 (2019~2023 합계) |
|------|------|--------------------------|
| v1 | 엄격 기준 (사업명에 직접 근거 필요) | 142건 (0.3%) |
| v2 | BIOFIN 9대 범주 도입 | 5,859건 (13.7%) |
| v3 | UNDP·KEI 기준 + 키워드 확장, 포용적 판단 | 17,928건 (41.9%) |
| v4 | 판단 기준 정교화, 애매하면 0, 키워드 단독 금지 | 17,777건 (41.6%) |

## 환경 요구사항

- Python 3.11+
- Ollama 로컬 서버 실행 중 (`http://localhost:11434`)
- 권장 모델: `llama3.1:8b`
