# 지방재정365 크롤러

지방재정365의 세부사업별 세출현황에서 전국 17개 시도의 사업 목록을 수집하고
각 사업의 상세 화면을 PDF로 저장합니다.

- **사업 목록**: 정식 Open API(`https://www.lofin365.go.kr/lf/hub/QWGJK`)로 받습니다.
  브라우저가 필요 없습니다.
- **상세 PDF**: Open API가 제공하지 않아 Playwright로
  [상세 화면](https://www.lofin365.go.kr/portal/LF3120204.do)을 열어 저장합니다.
  사이트 화면이 개편되면 이 부분은 수정이 필요할 수 있습니다.

## 인증키 설정

1. [재정 데이터셋](https://www.lofin365.go.kr/portal/LF5100000.do)에서
   "세부사업별 세출현황"을 선택하고 "인증키 신청"으로 키를 발급받습니다.
2. 저장소 루트의 `.env.example`을 `.env`로 복사합니다.
3. `.env`의 `LOFIN365_API_KEY=` 뒤에 키를 붙여 넣습니다.

인증키가 없으면 견본 키가 적용되어 지역별 5건까지만 받을 수 있습니다.
아래 시험 실행은 인증키 없이도 동작합니다.

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

목록만 받으려면(Playwright 불필요):

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

인증키가 필요합니다.

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
| `--limit` | `5` | 지역별 최대 사업 수, `0`은 전체(인증키 필요) |
| `--list-only` | 꺼짐 | Open API로 목록 CSV만 저장 |
| `--retry-failed` | 꺼짐 | 실패 항목만 재처리 |
| `--overwrite` | 꺼짐 | 정상 PDF도 다시 저장 |
| `--headed` | 꺼짐 | 브라우저 창 표시 |
| `--browser-channel` | `chrome` | Playwright가 사용할 브라우저 |
| `--min-delay` | `0.8` | PDF 사이 최소 대기 시간(초) |
| `--pdf-scale` | `0.70` | A4 가로 PDF 출력 배율 |

PDF는 표가 잘리지 않도록 A4 가로 방향, 70% 배율로 저장합니다. 더 넓은
표가 잘리는 경우 `--pdf-scale 0.60`으로 실행할 수 있습니다.

## 목록 CSV 컬럼

목록 CSV(매니페스트)는 예측·LLM 분류 코드가 그대로 읽을 수 있도록 열린재정
CSV와 같은 한글 컬럼명으로 저장합니다.

```text
회계연도, 소관명, 회계명, 분야명, 부문명, 세부사업명, 예산액, business_key,
사업설명자료_파일명, 다운로드상태, 오류내용, 시도코드, 시도명, 지방자치단체코드,
기준일자, 국비, 시도비, 시군구비, 기타재원, 지출액, 잔액, 세부사업코드,
상세페이지_URL, 갱신일시
```

- `소관명`은 자치단체명(예: `강원본청`), `예산액`은 예산현액입니다.
- `다운로드상태`는 PDF별 `pending`, `success`, `failed`입니다.
- 지방재정에는 프로그램명·단위사업명이 없어 해당 컬럼은 만들지 않습니다.
- 영문 컬럼으로 저장된 예전 매니페스트도 그대로 이어받을 수 있고, 다시
  저장할 때 한글 컬럼으로 바뀝니다.

`balance`(집행잔액)는 Open API가 제공하지 않아 예산현액에서 지출액을 뺀 값입니다.
