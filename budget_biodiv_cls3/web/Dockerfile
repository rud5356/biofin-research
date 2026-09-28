# BIOFIN 분류·전문가 검증 프로토타입 — 정적 프런트엔드 (백엔드 없음)
# 빌드: docker compose up -d --build
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
