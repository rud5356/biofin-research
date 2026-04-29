# budget_matcher

예산 CSV와 열린재정 파일 폴더를 대조하여 매칭 결과를 CSV로 출력하는 스크립트입니다.

## 실행 방법

```bash
python match_budget_files.py
```

기본값으로 실행 시:
- **CSV**: 스크립트 폴더 내 첫 번째 `.csv` 파일
- **파일 루트**: `C:\Yuna\국가생물다양성_열린재정 데이터`
- **출력 폴더**: `output\` (실행마다 덮어씀)
- **스캔 확장자**: `.hwp`, `.pdf`

## 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--csv` | 예산 CSV 경로 | 스크립트 폴더 내 첫 번째 CSV |
| `--budget-root` | 열린재정 파일 루트 폴더 | `C:\Yuna\국가생물다양성_열린재정 데이터` |
| `--output-dir` | 결과 출력 폴더 | `output\` |
| `--extensions` | 스캔할 파일 확장자 | `.hwp .pdf` |
| `--top-candidates` | 미매칭 행당 후보 최대 수 | `5` |

## CSV 필수 컬럼

| 컬럼 | 설명 |
|---|---|
| `분야명` | 예산 분야 (ex. 환경, 과학기술) |
| `소관명` | 부처/기관명 |
| `세부사업명` | 세부 사업명 |

## 출력 파일 (`output\`)

| 파일 | 설명 |
|---|---|
| `single_result.csv` | 전체 결과 (모든 행 포함) |
| `matched_exact.csv` | 완전 일치 매칭 |
| `matched_normalized.csv` | 정규화 후 일치 매칭 |
| `review_candidates.csv` | 자동 매칭 실패 — 유사 후보 있음 |
| `unmatched_no_candidates.csv` | 자동 매칭 실패 — 후보 없음 |
| `csv_out_of_scope_fields.csv` | 분야 폴더 미존재 행 |
| `file_only_unmatched.csv` | CSV에 없는 파일 목록 |
| `field_inventory.csv` | 분야별 폴더·파일 현황 |
| `summary.json` | 실행 요약 통계 |
| `unparsed_files.txt` | 파싱 실패 파일 목록 |

## 매칭 로직

1. **exact** — 분야·소관·사업명 완전 일치
2. **normalized** — 공백·특수문자 제거 후 일치
3. **review_candidate** — 유사도 점수 기반 후보 추천 (자동 매칭 실패 시)
4. **out_of_scope** — 해당 분야 폴더가 아직 없음

## 파일명 규칙

열린재정 파일은 `소관명_세부사업명.hwp` 또는 `.pdf` 형식이어야 합니다.
