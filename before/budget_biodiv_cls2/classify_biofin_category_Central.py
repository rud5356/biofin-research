"""
생물다양성 라벨(biodiv_label=1) 세부사업을 BIOFIN 9대 범주(1~9)로 세분류하는 스크립트.
biodiv_label=0인 행은 LLM 호출 없이 category=0으로 자동 지정한다.

사용 예:
    python classify_biofin_category.py
    python classify_biofin_category.py --dry-run          # LLM 호출 없이 구조 확인
    python classify_biofin_category.py --overwrite        # 기존 캐시 무시하고 재분류
    python classify_biofin_category.py --limit-keys 20   # 20개 사업만 테스트
    python classify_biofin_category.py --workers 4        # 병렬 4스레드
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import URLError


# ─── 기본 설정 ──────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "gemma3:12b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
INPUT_DIR          = Path(__file__).resolve().parent / "outputs"
INPUT_GLOB         = "세부사업 예산편성현황(총액)_*_labeled.csv"
OUTPUT_DIR         = Path(__file__).resolve().parent / "outputs"

ENCODINGS = ("utf-8-sig", "cp949", "utf-8")

KEY_COLUMNS = (
    "소관명",
    "분야명",
    "부문명",
    "프로그램명",
    "단위사업명",
    "세부사업명",
)

CACHE_COLUMNS = (
    "hash",
    "category",       # 정수 0~9
    "confidence",
    "reason",
    "raw_response",
)

# ─── 시스템 프롬프트 ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
너는 대한민국 재정사업을 BIOFIN(생물다양성 재정 이니셔티브) 9대 범주로 분류하는 전문 분류자이다.

입력 사업이 생물다양성과 관련 없으면 category=0, 관련 있으면 아래 9개 범주 중 가장 적합한 번호를 출력한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[BIOFIN 9대 범주]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 보호구역 및 기타 보전 조치
   - 보호구역 지정·확대·관리, 국립공원, 생태경관보전지역, 자연보호구역
   - 생물권보전지역, 람사르습지, 야생생물 보호, 생태축 및 생태통로, 종은행, 서식지 보호

2. 생태계 복원
   - 산림 복원, 습지 복원, 하천 복원, 연안 복원, 갯벌 복원, 자연성 회복
   - 생태통로 조성, 멸종위기종 복원, 종 복원, 훼손지 복원

3. 유전자원 접근 및 이익공유(ABS)
   - 나고야의정서, 유전자원 등록, 유전자원 관리, 이익공유 체계 구축

4. 지속가능한 이용 및 생물안전
   - 지속가능 농업·임업·어업·양식업, 생물자원 관리
   - 외래종 방제·예찰·퇴치, GMO 관리, LMO 관리, 생물안전

5. 오염관리
   - 수질오염 저감(수생태계 건강성 회복 목적), 토양오염 저감(생태계 보호 목적)
   - 해양오염 저감, 폐수관리, 폐기물관리, 비점오염 관리, 수생태계 건강성 회복, 지하수 오염관리
   ※ 단순 환경위생·공중보건 목적만이면 category=0

6. 생물다양성 인식 제고 및 지식
   - 생물다양성 조사·모니터링, 생태계 조사, 생물종 모니터링
   - 생태계 DB 구축, 생물다양성 정보시스템 구축, 생물다양성 연구·교육·홍보
   - 시민참여 생태조사, 생태계서비스 가치평가
   ※ 일반 정보화·DB는 생물다양성 정보를 직접 다루는 경우에만 해당

7. 녹색경제
   - 친환경 농업(생물다양성 보전 연계 명시), 유기농업, 생태관광
   - 지속가능 관광(자연자원 보전 연계), 지속가능 산림경영, 생물자원 기반 산업
   ※ 단순 친환경 에너지·녹색 인프라는 해당 없음

8. 생물다양성 및 개발계획
   - 국가생물다양성전략, 생물다양성 정책·법률·계획
   - 전략환경영향평가(생물다양성 항목 포함), 다자간 환경협약

9. 기타 생물다양성 관련 활동
   - 전통 생태지식, 지역 생태공동체 보전, 기타 생물다양성 보전 기여 사업
   - 1~8에 명확히 해당하지 않지만 생물다양성 관련성이 있는 경우

0. 생물다양성 관련 없음
   - 기본경비·인건비·일반행정·위원회운영·정보보안·국방·복지·의료 등
   - 사업 내용이 생태계·생물종·자연환경 보전과 무관한 경우

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[분류 절차]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1. 세부사업명 → 단위사업명 → 프로그램명 순으로 사업 내용을 파악한다.
STEP 2. 생물다양성과 명백히 무관하면 category=0.
STEP 3. 관련이 있으면 1~9 중 사업의 핵심 활동에 가장 가까운 범주를 선택한다.
STEP 4. 여러 범주에 걸치면 가장 비중이 큰 활동을 기준으로 하나만 선택한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[출력 형식 — JSON만 출력, 다른 텍스트 없음]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
  "category": 0~9 정수,
  "confidence": 0.0~1.0,
  "reason": "분류 근거 (2문장 이상, 사업명을 직접 인용하여 어떤 범주인지 설명)"
}
"""

PROMPT_TEMPLATE = """\
아래 사업을 BIOFIN 9대 범주(0~9)로 분류하라.

소관명: {소관명}
분야명: {분야명}
부문명: {부문명}
프로그램명: {프로그램명}
단위사업명: {단위사업명}
세부사업명: {세부사업명}
"""


# ─── 유틸 ────────────────────────────────────────────────────────────────────
def clean_surrogates(value: Any) -> str:
    text = str(value or "")
    cleaned: list[str] = []
    i = 0
    while i < len(text):
        c = ord(text[i])
        if 0xD800 <= c <= 0xDBFF and i + 1 < len(text):
            low = ord(text[i + 1])
            if 0xDC00 <= low <= 0xDFFF:
                cleaned.append(chr(0x10000 + ((c - 0xD800) << 10) + (low - 0xDC00)))
                i += 2
                continue
        if 0xD800 <= c <= 0xDFFF:
            i += 1
            continue
        cleaned.append(text[i])
        i += 1
    return "".join(cleaned)


def clean_cell(value: Any) -> str:
    return re.sub(r"\s+", " ", clean_surrogates(value).strip())


def build_key(row: dict) -> str:
    return "␟".join(clean_cell(row.get(col, "")) for col in KEY_COLUMNS)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def build_prompt(row: dict) -> str:
    cols = ("소관명", "분야명", "부문명", "프로그램명", "단위사업명", "세부사업명")
    values = {col: clean_cell(row.get(col, "")) for col in cols}
    return PROMPT_TEMPLATE.format(**values)


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    for enc in ENCODINGS:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                rows = [dict(r) for r in reader]
            return list(reader.fieldnames or []), rows
        except Exception:
            continue
    raise RuntimeError(f"CSV 읽기 실패: {path}")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {k: clean_surrogates(v) for k, v in row.items()} for row in rows
        )


# ─── 캐시 ────────────────────────────────────────────────────────────────────
def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    _, rows = read_csv(path)
    return {r["hash"]: r for r in rows if r.get("hash")}


def save_cache(path: Path, cache: dict[str, dict], lock: threading.Lock) -> None:
    with lock:
        rows = list(cache.values())
        write_csv(path, list(CACHE_COLUMNS), rows)


# ─── Ollama 호출 ──────────────────────────────────────────────────────────────
def parse_response(text: str) -> dict:
    # 마크다운 코드블록 제거
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # JSON 블록 추출
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # category 숫자만 추출 (폴백)
    m = re.search(r'"category"\s*:\s*(\d+)', text)
    if m:
        return {"category": int(m.group(1)), "confidence": 0.5, "reason": text[:200]}
    return {"category": -1, "confidence": 0.0, "reason": f"파싱 실패: {text[:200]}"}


def call_ollama(
    prompt: str,
    model: str,
    ollama_url: str,
    timeout: int,
    use_json_format: bool = True,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0, "top_p": 0.1},
    }
    if use_json_format:
        payload["format"] = "json"

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{ollama_url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return body["message"]["content"]


def classify_row(
    row: dict,
    model: str,
    ollama_url: str,
    timeout: int,
    retries: int,
    retry_delay: float,
    use_json_format: bool,
) -> dict:
    prompt = build_prompt(row)
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            raw = call_ollama(prompt, model, ollama_url, timeout, use_json_format)
            parsed = parse_response(raw)
            return {
                "category": int(parsed.get("category", -1)),
                "confidence": float(parsed.get("confidence", 0.0)),
                "reason": str(parsed.get("reason", "")),
                "raw_response": raw[:500],
            }
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(retry_delay)
    return {
        "category": -1,
        "confidence": 0.0,
        "reason": f"오류: {type(last_exc).__name__}: {last_exc}",
        "raw_response": "",
    }


# ─── 메인 ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BIOFIN 9대 범주 세분류 스크립트")
    p.add_argument("--input-dir",       type=Path, default=INPUT_DIR)
    p.add_argument("--input-glob",      default=INPUT_GLOB)
    p.add_argument("--output-dir",      type=Path, default=OUTPUT_DIR)
    p.add_argument("--model",           default=DEFAULT_MODEL)
    p.add_argument("--ollama-url",      default=DEFAULT_OLLAMA_URL)
    p.add_argument("--biodiv-col",      default="biodiv_label",
                   help="기존 이진 라벨 컬럼명 (기본: biodiv_label)")
    p.add_argument("--category-col",    default="biofin_category",
                   help="출력 범주 컬럼명 (기본: biofin_category)")
    p.add_argument("--cache-csv",       type=Path, default=None)
    p.add_argument("--timeout",         type=int,   default=60)
    p.add_argument("--retries",         type=int,   default=1)
    p.add_argument("--retry-delay",     type=float, default=1.0)
    p.add_argument("--delay",           type=float, default=0.05,
                   help="LLM 호출 간 대기(초)")
    p.add_argument("--workers",         type=int,   default=1)
    p.add_argument("--save-every",      type=int,   default=50)
    p.add_argument("--limit-keys",      type=int,   default=0,
                   help="테스트용: 앞 N개만 LLM 분류")
    p.add_argument("--dry-run",         action="store_true")
    p.add_argument("--overwrite",       action="store_true",
                   help="기존 캐시 무시하고 재분류")
    p.add_argument("--no-json-format",  action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cache_path = args.cache_csv or (args.output_dir / "category_cache.csv")
    cache: dict[str, dict] = {} if args.overwrite else load_cache(cache_path)
    cache_lock = threading.Lock()

    input_files = sorted(args.input_dir.glob(args.input_glob))
    if not input_files:
        print(f"[오류] 입력 파일 없음: {args.input_dir / args.input_glob}", file=sys.stderr)
        return 1

    print(f"입력 파일 {len(input_files)}개:")
    for f in input_files:
        print(f"  {f.name}")

    # 분류가 필요한 고유 사업(biodiv_label=1)만 수집
    all_file_data: list[tuple[Path, list[str], list[dict]]] = []
    unique_rows: dict[str, dict] = {}   # hash → 대표 row

    for path in input_files:
        headers, rows = read_csv(path)
        all_file_data.append((path, headers, rows))
        for row in rows:
            label_val = str(row.get(args.biodiv_col, "0")).strip()
            if label_val != "1":
                continue
            key = build_key(row)
            h = hash_key(key)
            if h not in unique_rows:
                unique_rows[h] = row

    total_unique = len(unique_rows)
    print(f"\nbidodiv_label=1 고유 사업: {total_unique}개")
    print(f"캐시 히트: {sum(1 for h in unique_rows if h in cache)}개")

    # LLM 분류 대상
    to_classify: list[tuple[str, dict]] = [
        (h, row)
        for h, row in unique_rows.items()
        if h not in cache
    ]
    if args.limit_keys > 0:
        to_classify = to_classify[: args.limit_keys]

    print(f"LLM 분류 대상: {len(to_classify)}개")

    if args.dry_run:
        print("[DRY-RUN] LLM 호출 없이 종료")
        return 0

    # LLM 분류 실행
    done = 0
    errors = 0

    def classify_one(item: tuple[str, dict]) -> tuple[str, dict]:
        h, row = item
        result = classify_row(
            row,
            args.model,
            args.ollama_url,
            args.timeout,
            args.retries,
            args.retry_delay,
            not args.no_json_format,
        )
        return h, result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(classify_one, item): item[0] for item in to_classify}
        for future in as_completed(futures):
            h, result = future.result()
            cat = result["category"]
            conf = result["confidence"]

            if cat == -1:
                errors += 1
                status = "오류"
            else:
                status = f"category={cat}"

            with cache_lock:
                cache[h] = {"hash": h, **result}

            done += 1
            row_repr = clean_cell(unique_rows[h].get("세부사업명", ""))[:40]
            print(f"[{done}/{len(to_classify)}] {status} (conf={conf:.2f}) | {row_repr}")

            if done % args.save_every == 0:
                save_cache(cache_path, cache, cache_lock)
                print(f"  → 캐시 중간저장 ({done}개)")

            time.sleep(args.delay)

    save_cache(cache_path, cache, cache_lock)
    print(f"\n분류 완료: {done}개 처리, 오류 {errors}개")

    # 각 파일에 biofin_category 컬럼 추가하여 저장
    cat_col = args.category_col
    for path, headers, rows in all_file_data:
        if cat_col not in headers:
            headers = headers + [cat_col]

        for row in rows:
            label_val = str(row.get(args.biodiv_col, "0")).strip()
            if label_val != "1":
                row[cat_col] = 0
                continue
            h = hash_key(build_key(row))
            cached = cache.get(h)
            if cached:
                row[cat_col] = cached.get("category", "")
            else:
                row[cat_col] = ""   # 미분류(limit-keys 등으로 건너뜀)

        # 출력 파일명: 원본에 _category 접미사
        stem = path.stem.replace("_labeled", "")
        out_path = args.output_dir / f"{stem}_category.csv"
        write_csv(out_path, headers, rows)
        print(f"저장: {out_path.name}")

    # 분류 결과 요약
    cat_counts: dict[int, int] = {}
    for h in unique_rows:
        cached = cache.get(h)
        if cached:
            cat = int(cached.get("category", -1))
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    cat_names = {
        0: "생물다양성 관련 없음",
        1: "보호구역 및 기타 보전 조치",
        2: "생태계 복원",
        3: "유전자원 접근 및 이익공유(ABS)",
        4: "지속가능한 이용 및 생물안전",
        5: "오염관리",
        6: "생물다양성 인식 제고 및 지식",
        7: "녹색경제",
        8: "생물다양성 및 개발계획",
        9: "기타 생물다양성 관련 활동",
        -1: "오류/미분류",
    }
    print("\n[범주별 사업 수]")
    for cat in sorted(cat_counts):
        print(f"  {cat}. {cat_names.get(cat, '?')}: {cat_counts[cat]}개")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
