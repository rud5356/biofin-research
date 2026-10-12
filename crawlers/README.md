# 재정 데이터 크롤러

재정 데이터의 출처별로 크롤러를 분리합니다.

- `open_fiscal`: 열린재정(중앙재정)
- `local_fiscal`: 지방재정

각 크롤러의 실행법과 데이터 출처는 해당 폴더의 `README.md`를 참고하세요.

## Open API 인증키

두 출처 모두 정식 Open API를 사용하는 부분이 있습니다. 저장소 루트의
`.env.example`을 `.env`로 복사하고 발급받은 인증키를 넣으세요. `.env`는
저장소에 올라가지 않습니다. 공통 호출 코드는 `openapi.py`에 있습니다.

| 출처 | Open API로 받는 것 | 화면에서 받는 것(Playwright) |
|---|---|---|
| `local_fiscal` | 세부사업 목록·예산·지출액 | 상세 화면 PDF |
| `open_fiscal` | 예산 목록(`api_budget_list.py`, 보조 도구) | 세부사업 목록, 사업설명자료 |
