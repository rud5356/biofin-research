import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router-dom'
import './index.css'
import App from './App.tsx'

// HashRouter(#/results 형식)를 사용합니다 — 아티팩트·정적 호스팅·더블클릭 실행(file://) 등
// 서버의 경로 재작성 규칙이 없는 환경에서도 새로고침·직접 진입 시 깨지지 않도록 하기 위함입니다.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </StrictMode>,
)
