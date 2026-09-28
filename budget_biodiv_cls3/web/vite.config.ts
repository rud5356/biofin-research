import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  // 상대 경로로 빌드 — dist/index.html을 어떤 폴더 구조에 두거나
  // 더블클릭(file://)으로 열어도 자산 경로가 깨지지 않게 합니다.
  base: './',
  plugins: [react(), tailwindcss()],
})
