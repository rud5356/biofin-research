"""
열린재정_파일매칭_작업표.csv 의 빈 matched_filename 을 채우는 스크립트.

우선순위:
  1. 국가생물다양성_열린재정 데이터_v2 폴더에서 분야명 폴더 > 소관명+세부사업명 기준 매칭
  2. (1)이 애매하거나 없으면 workfile.xlsx 에서 source_no 기준으로 가져오기
  3. 둘 다 없으면 빈칸 유지
"""
from __future__ import annotations

import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import unquote

import pandas as pd


# ── 경로 설정 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "열린재정_파일매칭_작업표.csv"
V2_ROOT = Path(r"C:\Yuna\국가생물다양성_열린재정 데이터_v2")
WORKFILE_XLSX = V2_ROOT / "열린재정_파일매칭_작업표.xlsx"
EXTENSIONS = {".hwp", ".pdf"}


# ── 텍스트 정규화 (match_budget_files.py 와 동일 로직) ─────────────────────────
_RND_ALIASES = {"r&d": "rnd", "r %26 d": "rnd", "r %26d": "rnd", "r and d": "rnd"}


def normalize(value: str | None) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = unquote(text)
    text = re.sub(r"\s+\((\d+)\)$", "", text)
    lowered = text.lower()
    for alias, canonical in _RND_ALIASES.items():
        lowered = lowered.replace(alias, canonical)
    return re.sub(r"[^0-9a-z가-힣]+", "", lowered)


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# ── 파일명 파싱 (소관명_세부사업명 추출) ──────────────────────────────────────────
def parse_stem(stem: str) -> tuple[str, str] | None:
    cleaned = unicodedata.normalize("NFKC", unquote(stem)).strip()
    cleaned = re.sub(r"\s+\((\d+)\)$", "", cleaned)
    prefixed = re.match(r"^(\d+)_(.+)$", cleaned)
    if prefixed and "_" in prefixed.group(2):
        cleaned = prefixed.group(2)
    if "_" not in cleaned:
        return None
    ministry, project = cleaned.split("_", 1)
    ministry, project = ministry.strip(), project.strip()
    if not ministry or not project:
        return None
    return ministry, project


@dataclass
class FileInfo:
    filename: str
    path: Path
    ministry_norm: str
    project_norm: str


# ── v2 폴더 스캔 ───────────────────────────────────────────────────────────────
def scan_v2(root: Path) -> dict[str, list[FileInfo]]:
    """field_norm → FileInfo 목록"""
    index: dict[str, list[FileInfo]] = defaultdict(list)
    if not root.exists():
        print(f"[ERROR] v2 폴더 없음: {root}", file=sys.stderr)
        return index
    for field_dir in sorted(root.iterdir()):
        if not field_dir.is_dir():
            continue
        field_norm = normalize(field_dir.name)
        for fpath in sorted(field_dir.iterdir()):
            if not fpath.is_file() or fpath.suffix.lower() not in EXTENSIONS:
                continue
            parsed = parse_stem(fpath.stem)
            if parsed is None:
                continue
            ministry_raw, project_raw = parsed
            index[field_norm].append(
                FileInfo(
                    filename=fpath.name,
                    path=fpath,
                    ministry_norm=normalize(ministry_raw),
                    project_norm=normalize(project_raw),
                )
            )
    return index


# ── 매칭 로직 ──────────────────────────────────────────────────────────────────
EXACT_THRESHOLD = 1.0
FUZZY_THRESHOLD = 0.75  # 프로젝트명 유사도 기준


def find_best_match(
    ministry_norm: str,
    project_norm: str,
    candidates: list[FileInfo],
) -> tuple[FileInfo | None, str]:
    """
    Returns (matched, reason).
    reason: 'exact' | 'fuzzy_unique' | 'fuzzy_ambiguous' | 'not_found'
    """
    # 1) exact match
    exact = [
        fi for fi in candidates
        if fi.ministry_norm == ministry_norm and fi.project_norm == project_norm
    ]
    if len(exact) == 1:
        return exact[0], "exact"
    if len(exact) > 1:
        return None, "exact_ambiguous"

    # 2) ministry exact + project fuzzy
    same_ministry = [fi for fi in candidates if fi.ministry_norm == ministry_norm]
    pool = same_ministry if same_ministry else candidates

    scored = [
        (fi, similarity(project_norm, fi.project_norm))
        for fi in pool
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    if not scored or scored[0][1] < FUZZY_THRESHOLD:
        return None, "not_found"

    top_score = scored[0][1]
    top_matches = [fi for fi, s in scored if s >= top_score - 0.01]
    if len(top_matches) == 1:
        return top_matches[0], "fuzzy_unique"
    return None, "fuzzy_ambiguous"


# ── workfile.xlsx 인덱스 ────────────────────────────────────────────────────────
def load_workfile_index(xlsx_path: Path) -> dict[str, str]:
    """source_no → matched_filename (비어있지 않은 것만)"""
    if not xlsx_path.exists():
        print(f"[WARN] workfile.xlsx 없음: {xlsx_path}", file=sys.stderr)
        return {}
    df = pd.read_excel(xlsx_path)
    if "matched_filename" not in df.columns or "source_no" not in df.columns:
        print("[WARN] workfile.xlsx 에 source_no / matched_filename 컬럼 없음", file=sys.stderr)
        return {}
    index: dict[str, str] = {}
    for _, row in df.iterrows():
        sno = str(row.get("source_no", "") or "").strip()
        fname = str(row.get("matched_filename", "") or "").strip()
        if sno and fname:
            index[sno] = fname
    return index


# ── 메인 ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"CSV 읽는 중: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    total = len(df)

    empty_mask = df["matched_filename"].isna() | (df["matched_filename"].astype(str).str.strip() == "")
    empty_idx = df.index[empty_mask].tolist()
    print(f"전체 {total}행 중 matched_filename 비어있는 행: {len(empty_idx)}개")

    if not empty_idx:
        print("채울 항목 없음. 종료.")
        return

    print(f"\nv2 폴더 스캔 중: {V2_ROOT}")
    file_index = scan_v2(V2_ROOT)
    total_files = sum(len(v) for v in file_index.values())
    print(f"  분야 폴더 수: {len(file_index)}, 총 파일 수: {total_files}")

    print(f"\nworkfile.xlsx 읽는 중: {WORKFILE_XLSX}")
    wb_index = load_workfile_index(WORKFILE_XLSX)
    print(f"  workfile 매칭 가능 항목: {len(wb_index)}개")

    # 결과 집계
    stats = {"v2_exact": 0, "v2_fuzzy": 0, "workfile": 0, "unfilled": 0, "ambiguous": 0}
    fill_log: list[dict] = []

    for idx in empty_idx:
        row = df.loc[idx]
        source_no = str(row.get("source_no", "") or "").strip()
        field_raw = str(row.get("분야명", "") or "").strip()
        ministry_raw = str(row.get("소관명", "") or "").strip()
        project_raw = str(row.get("세부사업명", "") or "").strip()

        field_norm = normalize(field_raw)
        ministry_norm = normalize(ministry_raw)
        project_norm = normalize(project_raw)

        candidates = file_index.get(field_norm, [])
        matched_fi, reason = find_best_match(ministry_norm, project_norm, candidates)

        if matched_fi is not None:
            df.at[idx, "matched_filename"] = matched_fi.filename
            stat_key = "v2_exact" if reason == "exact" else "v2_fuzzy"
            stats[stat_key] += 1
            fill_log.append({
                "source_no": source_no, "분야명": field_raw,
                "소관명": ministry_raw, "세부사업명": project_raw,
                "filled_filename": matched_fi.filename, "source": f"v2_{reason}",
            })
            continue

        # ambiguous 는 workfile 시도 전에 기록만
        if reason in ("exact_ambiguous", "fuzzy_ambiguous"):
            stats["ambiguous"] += 1

        # workfile 에서 시도
        if source_no in wb_index:
            fname = wb_index[source_no]
            df.at[idx, "matched_filename"] = fname
            stats["workfile"] += 1
            fill_log.append({
                "source_no": source_no, "분야명": field_raw,
                "소관명": ministry_raw, "세부사업명": project_raw,
                "filled_filename": fname, "source": f"workfile({reason})",
            })
            continue

        stats["unfilled"] += 1
        fill_log.append({
            "source_no": source_no, "분야명": field_raw,
            "소관명": ministry_raw, "세부사업명": project_raw,
            "filled_filename": "", "source": f"unfilled({reason})",
        })

    # CSV 저장
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\nCSV 저장 완료: {CSV_PATH}")

    # 로그 저장
    log_path = SCRIPT_DIR / "fill_missing_log.csv"
    pd.DataFrame(fill_log).to_csv(log_path, index=False, encoding="utf-8-sig")
    print(f"로그 저장 완료: {log_path}")

    print("\n=== 결과 요약 ===")
    print(f"  v2 폴더 exact 매칭: {stats['v2_exact']}건")
    print(f"  v2 폴더 fuzzy 매칭: {stats['v2_fuzzy']}건")
    print(f"  workfile.xlsx 참조: {stats['workfile']}건")
    print(f"  모호한 후보 존재(workfile에서 채워짐 포함): {stats['ambiguous']}건")
    print(f"  채우지 못한 항목:    {stats['unfilled']}건")
    filled = stats["v2_exact"] + stats["v2_fuzzy"] + stats["workfile"]
    print(f"  총 채운 항목:        {filled}건 / {len(empty_idx)}건")


if __name__ == "__main__":
    main()
