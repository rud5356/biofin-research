# 열린재정 크롤러

열린재정에서 지정한 회계연도의 세부사업 목록을 CSV로 저장하고 각 사업의
사업설명자료를 내려받습니다. 예산 컬럼을 유지하면서 `business_key`와 상대
경로로 CSV와 문서를 정확히 연결합니다.

## 2024년 시험 수집

```bash
python crawl_business_docs.py --year 2024 --limit 5
```

목록 CSV만 만들려면:

```bash
python crawl_business_docs.py --year 2024 --limit 5 --list-only
```

## 2024년 전체 수집

`--limit 0`은 전체를 의미합니다.

```bash
python crawl_business_docs.py --year 2024 --limit 0
```

기본 결과:

```text
crawlers/open_fiscal/outputs/
└─ 2024/
   ├─ open_fiscal_2024.csv
   ├─ open_fiscal_2024_failed.csv
   └─ 사업설명자료/
      └─ 2024_소관명_세부사업명_business_key_원본파일명.hwp
```

`open_fiscal_2024.csv` 컬럼:

```text
회계연도
소관명
회계명
분야명
부문명
프로그램명
단위사업명
세부사업명
예산액
business_key
사업설명자료_파일명
사업설명자료_상대경로
다운로드상태
오류내용
```

`business_key`가 사업과 파일명에 함께 들어가며,
`사업설명자료_상대경로`로 실제 파일을 바로 찾을 수 있습니다.

사업설명자료가 제공되지 않은 사업도 CSV에서 제외하지 않습니다. 이 경우
파일명과 상대경로는 비워두고 `다운로드상태`를 `no_document`로 기록합니다.

## 이어받기

정상 파일이 이미 있으면 자동으로 건너뜁니다.

```bash
python crawl_business_docs.py --year 2024 --limit 0
```

실패 사업만 재시도:

```bash
python crawl_business_docs.py --year 2024 --limit 0 --retry-failed
```

기존 파일도 다시 다운로드:

```bash
python crawl_business_docs.py --year 2024 --limit 0 --overwrite
```

## 기존 label CSV 방식

`--year`를 생략하면 기존처럼 `budget_biodiv_cls2/outputs`의 label=1 입력 CSV를
읽어 해당 사업만 매칭하는 호환 모드로 실행됩니다.
