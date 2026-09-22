# BIOFIN 예산사업 분류·전문가 검증 시스템 (프론트엔드 프로토타입)

`budget_biodiv_cls3`의 모델·학습 코드와는 분리된 클릭 가능한 UI 데모입니다.
실제 모델·서버·DB·파일 시스템과 연결되지 않으며, 모든 데이터는 가상입니다.

## 실행 방법

```powershell
cd C:\repos\biofin-research\budget_biodiv_cls3\web
npm install
npm run dev
```

터미널에 표시되는 주소(기본 http://localhost:5173 )를 브라우저로 열면 됩니다.

## 구성

- React + TypeScript + Vite + Tailwind CSS
- 상태 관리: Zustand (`localStorage`에 검토 상태를 저장 — 새로고침해도 유지됩니다)
- 데모 데이터: `src/data/seed.ts` (가상 예산사업 46건), `src/data/categories.ts` (상위 0~9, 하위 39개 코드)

## 화면

| 경로 | 화면 |
| --- | --- |
| `/results` | 분류 결과·전문가 검증 (기본 진입 화면) |
| `/dashboard` | 현황 대시보드 |
| `/predict` | 예측 실행 (데모 시뮬레이션) |
| `/data` | 데이터·문서 관리 |
| `/compare` | 모델 비교·평가 |
| `/admin` | 연구·관리 (학습/증강/파이프라인/분류기준/실행이력/담당자) |

상단 배너의 "데모 상태 초기화" 버튼으로 검토 이력을 포함한 모든 데모 상태를 초기값으로 되돌릴 수 있습니다.

## 참고

이 폴더는 프로토타입 전용이며, 상위 폴더의 Transformer/LLM/파이프라인 코드는 수정하지 않았습니다.
