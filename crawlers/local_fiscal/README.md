# 지방재정365 크롤러

[세부사업별 세출현황](https://www.lofin365.go.kr/portal/LF3120202.do)에서
전국 17개 시도의 사업 목록을 수집하고 각 사업의 상세 화면을 PDF로 저장합니다.

기본 조회 조건:

- 지역: 전국 17개 시도
- 자치단체: 기본값(해당 지역의 본청·자치구·군 전체)
- 회계: 전체
- 분야: 전체
- 기준일: 2024-12-31
- 처리 건수: 지역별 앞의 5건(총 35건)

## 시험 실행

```bash
python crawl_business_docs.py
```

목록만 확인하려면:

```bash
python crawl_business_docs.py --list-only
```

결과는 지역별 하위 폴더를 만들지 않고 한 폴더에 저장합니다.

```text
crawlers/local_fiscal/outputs/
├─ metropolitan_2024_manifest.csv
└─ 사업설명자료/
   ├─ 2024_서울_서울본청_일반회계_사업명_1100000_사업코드.pdf
   ├─ 2024_서울_서울종로구_일반회계_사업명_1111000_사업코드.pdf
   └─ 2024_부산_부산본청_일반회계_사업명_2600000_사업코드.pdf
```

## 전체 수집

```bash
python crawl_business_docs.py --limit 0
```

특정 지역만 수집할 수도 있습니다. 강원 2024년 전체 수집은 다음과 같습니다.

```bash
python crawl_business_docs.py --regions 강원 --date 2024-12-31 --limit 0
```

세종·제주·강원·전북과 6개 도를 이어서 수집하려면:

```bash
python crawl_business_docs.py --regions 세종 제주 강원 전북 경기 충북 충남 전남 경북 경남 --date 2024-12-31 --limit 0
```

기존 정상 PDF는 자동으로 건너뜁니다. 실패 항목만 다시 실행하려면:

```bash
python crawl_business_docs.py --limit 0 --retry-failed
```

출력 설정을 바꾼 후 기존 PDF도 다시 만들려면:

```bash
python crawl_business_docs.py --limit 0 --overwrite
```

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---:|---|
| `--regions` | 전국 17개 시도 | 수집할 지역 목록 |
| `--date` | `2024-12-31` | 조회 기준일 |
| `--limit` | `5` | 지역별 최대 사업 수, `0`은 전체 |
| `--list-only` | 꺼짐 | 목록 CSV만 저장 |
| `--retry-failed` | 꺼짐 | 실패 항목만 재처리 |
| `--overwrite` | 꺼짐 | 정상 PDF도 다시 저장 |
| `--headed` | 꺼짐 | 브라우저 창 표시 |
| `--browser-channel` | `chrome` | Playwright가 사용할 브라우저 |
| `--min-delay` | `0.8` | PDF 사이 최소 대기 시간(초) |
| `--pdf-scale` | `0.70` | A4 가로 PDF 출력 배율 |

PDF는 표가 잘리지 않도록 A4 가로 방향, 70% 배율로 저장합니다. 더 넓은
표가 잘리는 경우 `--pdf-scale 0.60`으로 실행할 수 있습니다.

통합 매니페스트에는 지역 코드, 지역, 자치단체 코드, 자치단체명과 각 PDF의
`pending`, `success`, `failed` 상태가 기록됩니다.
