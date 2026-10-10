"""문서 조각별 Attention 가중치를 단일 HTML 히트맵으로 변환한다.

현재 분류 모델의 Attention은 토큰이 아니라 512-token 문서 조각 단위로
계산된다. 따라서 이 보고서는 모델이 어느 문서 조각을 상대적으로 크게
반영했는지 보여주며, 판단의 인과적 근거를 증명하지는 않는다.
"""

from __future__ import annotations

import argparse
import csv
import html as html_lib
from collections import OrderedDict
from pathlib import Path
from typing import Any


LABEL_NAMES = {
    0: "관련 없음",
    1: "보호구역 및 기타 보전 조치",
    2: "생태계 복원",
    3: "유전자원 접근 및 이익공유",
    4: "지속가능한 이용 및 생물안전",
    5: "오염관리",
    6: "인식 제고 및 지식",
    7: "녹색경제",
    8: "생물다양성 및 개발계획",
    9: "기타 생물다양성 관련 활동",
}

REQUIRED_COLUMNS = {
    "year",
    "ministry",
    "activity_name",
    "file_path",
    "pred_label",
    "probability",
    "chunk_index",
    "chunk_text_preview",
    "attention_weight",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                reader = csv.DictReader(file)
                fieldnames = set(reader.fieldnames or [])
                missing = sorted(REQUIRED_COLUMNS - fieldnames)
                if missing:
                    raise ValueError(
                        "Attention CSV 필수 컬럼이 없습니다: " + ", ".join(missing)
                    )
                return [dict(row) for row in reader]
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise RuntimeError(f"CSV 인코딩을 해석하지 못했습니다: {path}") from last_error
    raise RuntimeError(f"CSV를 읽지 못했습니다: {path}")


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _to_optional_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _group_documents(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row_number, row in enumerate(rows, start=1):
        file_path = str(row.get("file_path", "")).strip()
        key = file_path or "::".join(
            [
                str(row.get("year", "")),
                str(row.get("ministry", "")),
                str(row.get("activity_name", "")),
                str(row_number),
            ]
        )
        if key not in grouped:
            true_label = _to_optional_int(row.get("true_label"))
            pred_label = _to_int(row.get("pred_label"))
            grouped[key] = {
                "year": str(row.get("year", "")),
                "ministry": str(row.get("ministry", "")),
                "activity_name": str(row.get("activity_name", "")),
                "file_path": file_path,
                "source_type": str(row.get("source_type", "document")),
                "true_label": true_label,
                "pred_label": pred_label,
                "probability": _to_float(row.get("probability")),
                "chunks": [],
            }
        grouped[key]["chunks"].append(
            {
                "chunk_index": _to_int(row.get("chunk_index")),
                "preview": str(row.get("chunk_text_preview", "")),
                "weight": max(0.0, _to_float(row.get("attention_weight"))),
            }
        )

    documents = list(grouped.values())
    for document in documents:
        document["chunks"].sort(key=lambda item: item["chunk_index"])
        if document["true_label"] is None:
            document["result_kind"] = "unlabeled"
            document["is_correct"] = None
        elif document["true_label"] == document["pred_label"]:
            document["result_kind"] = "correct"
            document["is_correct"] = True
        else:
            document["result_kind"] = "incorrect"
            document["is_correct"] = False
        document["display_label"] = (
            document["true_label"]
            if document["true_label"] is not None
            else document["pred_label"]
        )
        document["max_weight"] = max(
            (chunk["weight"] for chunk in document["chunks"]), default=0.0
        )
    return documents


def _heat_style(weight: float, max_weight: float) -> tuple[str, str]:
    normalized = weight / max_weight if max_weight > 0 else 0.0
    normalized = min(1.0, max(0.0, normalized))
    alpha = 0.10 + 0.82 * normalized
    background = f"rgba(220, 38, 38, {alpha:.3f})"
    foreground = "#ffffff" if normalized >= 0.62 else "#111827"
    return background, foreground


def _label_text(label: int) -> str:
    return f"{label}. {LABEL_NAMES.get(label, '알 수 없음')}"


def _document_sort_key(document: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -len(document["chunks"]),
        -float(document["max_weight"]),
        str(document["activity_name"]),
    )


def _select_documents(
    candidates: list[dict[str, Any]], max_documents: int, errors_only: bool
) -> list[dict[str, Any]]:
    """기본 화면에 정답 일치와 불일치 문서를 번갈아 포함한다."""

    incorrect = sorted(
        (document for document in candidates if document["result_kind"] == "incorrect"),
        key=_document_sort_key,
    )
    if errors_only:
        return incorrect[:max_documents]

    correct = sorted(
        (document for document in candidates if document["result_kind"] == "correct"),
        key=_document_sort_key,
    )
    unlabeled = sorted(
        (document for document in candidates if document["result_kind"] == "unlabeled"),
        key=_document_sort_key,
    )
    selected: list[dict[str, Any]] = []
    index = 0
    while len(selected) < max_documents and (
        index < len(incorrect) or index < len(correct) or index < len(unlabeled)
    ):
        if index < len(incorrect) and len(selected) < max_documents:
            selected.append(incorrect[index])
        if index < len(correct) and len(selected) < max_documents:
            selected.append(correct[index])
        if index < len(unlabeled) and len(selected) < max_documents:
            selected.append(unlabeled[index])
        index += 1
    return selected


def _render_document(document: dict[str, Any], index: int) -> str:
    true_label = document["true_label"]
    pred_label = int(document["pred_label"])
    result_class = str(document["result_kind"])
    if result_class == "correct":
        result = "기준 라벨 일치"
    elif result_class == "incorrect":
        result = "기준 라벨 불일치"
    else:
        result = "기준 라벨 없음"
    max_weight = float(document["max_weight"])

    strip_cells: list[str] = []
    detail_cards: list[str] = []
    for chunk in document["chunks"]:
        weight = float(chunk["weight"])
        background, foreground = _heat_style(weight, max_weight)
        chunk_index = int(chunk["chunk_index"])
        preview = html_lib.escape(str(chunk["preview"])) or "(미리보기 없음)"
        tooltip = html_lib.escape(str(chunk["preview"]), quote=True)
        strip_cells.append(
            f'<div class="heat-cell" title="{tooltip}" '
            f'style="background:{background};color:{foreground}">'
            f'<span>조각 {chunk_index}</span><strong>{weight:.4f}</strong></div>'
        )
        detail_cards.append(
            f'<article class="chunk-card" style="border-left-color:{background}">'
            f'<div class="chunk-title"><strong>조각 {chunk_index}</strong>'
            f'<span>Attention {weight:.4f}</span></div>'
            f'<p>{preview}</p></article>'
        )

    activity = html_lib.escape(str(document["activity_name"]))
    ministry = html_lib.escape(str(document["ministry"]))
    year = html_lib.escape(str(document["year"]))
    file_path = html_lib.escape(str(document["file_path"]))
    search_text = html_lib.escape(
        f"{document['activity_name']} {document['ministry']} {document['year']}",
        quote=True,
    )
    true_badge = (
        f'<span class="badge true">기준 {_label_text(int(true_label))}</span>'
        if true_label is not None
        else ""
    )
    return f"""
    <section class="document" data-result="{result_class}"
             data-label="{int(document['display_label'])}" data-search="{search_text.lower()}">
      <header class="document-header">
        <div>
          <div class="eyebrow">문서 {index:02d} · {year} · {ministry}</div>
          <h2>{activity}</h2>
        </div>
        <div class="badges">
          {true_badge}
          <span class="badge pred">예측 {_label_text(pred_label)}</span>
          <span class="badge {result_class}">{result}</span>
          <span class="badge score">모델 점수 {float(document['probability']):.4f}</span>
        </div>
      </header>
      <div class="heat-strip">{''.join(strip_cells)}</div>
      <div class="legend-note">색이 진할수록 같은 문서 안에서 상대적으로 크게 반영된 조각</div>
      <details>
        <summary>조각별 내용 보기</summary>
        <div class="chunk-list">{''.join(detail_cards)}</div>
        <div class="file-path">{file_path}</div>
      </details>
    </section>
    """


def generate_attention_heatmap(
    input_csv: str | Path,
    output_html: str | Path,
    max_documents: int = 50,
    min_chunks: int = 2,
    errors_only: bool = False,
) -> dict[str, int]:
    """Attention CSV를 읽어 문서 조각 단위의 독립 실행형 HTML을 생성한다."""

    input_path = Path(input_csv)
    output_path = Path(output_html)
    if not input_path.is_file():
        raise FileNotFoundError(f"Attention CSV가 없습니다: {input_path}")
    if max_documents < 1:
        raise ValueError("max_documents는 1 이상이어야 합니다")
    if min_chunks < 1:
        raise ValueError("min_chunks는 1 이상이어야 합니다")

    rows = _read_csv(input_path)
    documents = _group_documents(rows)
    candidates = [
        document
        for document in documents
        if len(document["chunks"]) >= min_chunks
        and (not errors_only or document["result_kind"] == "incorrect")
    ]
    selected = _select_documents(candidates, max_documents, errors_only)
    if not selected:
        raise ValueError(
            "히트맵 조건에 맞는 문서가 없습니다. --heatmap_min_chunks 값을 낮추거나 "
            "--heatmap_errors_only 옵션을 해제하세요."
        )

    document_html = "".join(
        _render_document(document, index)
        for index, document in enumerate(selected, start=1)
    )
    input_document_count = len(documents)
    multi_chunk_count = sum(len(document["chunks"]) >= 2 for document in documents)
    error_count = sum(document["result_kind"] == "incorrect" for document in documents)
    unlabeled_count = sum(document["result_kind"] == "unlabeled" for document in documents)

    page = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>문서 조각별 Attention 히트맵</title>
  <style>
    :root {{ color-scheme: light; font-family: "Pretendard", "Noto Sans KR", Arial, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: #172033; background: #f4f7fb; }}
    .page {{ width: min(1280px, calc(100% - 40px)); margin: 0 auto; padding: 44px 0 80px; }}
    h1 {{ margin: 0 0 10px; font-size: 34px; }}
    .subtitle {{ margin: 0; color: #596579; line-height: 1.7; }}
    .notice {{ margin-top: 18px; padding: 14px 18px; border-left: 4px solid #f59e0b;
               background: #fffbeb; color: #7c4a03; line-height: 1.6; }}
    .stats {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 24px 0; }}
    .stat {{ padding: 18px; border: 1px solid #dbe3ef; border-radius: 14px; background: white; }}
    .stat strong {{ display: block; color: #245edb; font-size: 28px; }}
    .stat span {{ color: #667085; font-size: 13px; }}
    .controls {{ position: sticky; top: 0; z-index: 10; display: flex; gap: 10px; flex-wrap: wrap;
                 padding: 14px; margin: 0 0 20px; border: 1px solid #dbe3ef;
                 border-radius: 14px; background: rgba(255,255,255,.96); backdrop-filter: blur(8px); }}
    input, select {{ min-height: 40px; padding: 8px 12px; border: 1px solid #cbd5e1;
                     border-radius: 9px; background: white; font: inherit; }}
    input {{ flex: 1 1 320px; }}
    .document {{ margin: 18px 0; padding: 22px; border: 1px solid #dbe3ef;
                 border-radius: 18px; background: white; box-shadow: 0 8px 24px rgba(15,23,42,.05); }}
    .document-header {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 18px; }}
    .eyebrow {{ color: #667085; font-size: 13px; }}
    h2 {{ margin: 5px 0 0; font-size: 21px; }}
    .badges {{ display: flex; justify-content: flex-end; gap: 6px; flex-wrap: wrap; }}
    .badge {{ padding: 6px 9px; border-radius: 999px; font-size: 12px; background: #eef2f7; }}
    .badge.correct {{ color: #0f6b3c; background: #dcfce7; }}
    .badge.incorrect {{ color: #a61b1b; background: #fee2e2; }}
    .badge.unlabeled {{ color: #475569; background: #e2e8f0; }}
    .heat-strip {{ display: grid; grid-auto-flow: column; grid-auto-columns: minmax(92px, 1fr);
                   gap: 5px; margin-top: 18px; overflow-x: auto; padding-bottom: 5px; }}
    .heat-cell {{ min-height: 68px; padding: 10px; border-radius: 9px;
                  display: flex; flex-direction: column; justify-content: space-between; }}
    .heat-cell span {{ font-size: 12px; }} .heat-cell strong {{ font-size: 17px; }}
    .legend-note {{ margin: 8px 0 12px; color: #667085; font-size: 12px; }}
    details {{ border-top: 1px solid #e5e7eb; padding-top: 12px; }}
    summary {{ cursor: pointer; color: #245edb; font-weight: 700; }}
    .chunk-list {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 14px; }}
    .chunk-card {{ padding: 14px; border: 1px solid #e5e7eb; border-left: 8px solid;
                   border-radius: 10px; background: #fbfcfe; }}
    .chunk-title {{ display: flex; justify-content: space-between; gap: 12px; color: #334155; }}
    .chunk-card p {{ margin: 9px 0 0; line-height: 1.65; color: #475569; }}
    .file-path {{ margin-top: 12px; color: #94a3b8; font-size: 11px; overflow-wrap: anywhere; }}
    .hidden {{ display: none; }}
    @media (max-width: 800px) {{
      .stats {{ grid-template-columns: repeat(2, 1fr); }}
      .document-header {{ display: block; }} .badges {{ justify-content: flex-start; margin-top: 12px; }}
      .chunk-list {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <h1>문서 조각별 Attention 히트맵</h1>
    <p class="subtitle">문서별 조각 가중치를 색의 진하기로 표시함. 기본 화면은 불일치·일치 문서를 번갈아 표시함.</p>
    <div class="notice">Attention은 모델이 상대적으로 크게 반영한 조각을 보여주는 참고값이며,
      해당 조각이 판단의 원인임을 증명하지 않음. 현재 모델은 단어가 아닌 문서 조각 단위로 계산함.</div>
    <section class="stats">
      <div class="stat"><strong>{input_document_count:,}</strong><span>전체 검증 문서</span></div>
      <div class="stat"><strong>{multi_chunk_count:,}</strong><span>2개 이상 조각 문서</span></div>
      <div class="stat"><strong>{error_count:,}</strong><span>예측 불일치 문서</span></div>
      <div class="stat"><strong>{unlabeled_count:,}</strong><span>기준 라벨 없는 문서</span></div>
      <div class="stat"><strong>{len(selected):,}</strong><span>현재 표시 문서</span></div>
    </section>
    <section class="controls">
      <input id="search" type="search" placeholder="사업명·소관명·연도 검색">
      <select id="resultFilter">
        <option value="all">일치 여부 전체</option>
        <option value="incorrect">불일치만</option>
        <option value="correct">일치만</option>
        <option value="unlabeled">기준 라벨 없음</option>
      </select>
      <select id="labelFilter">
        <option value="all">표시 범주 전체</option>
        {''.join(f'<option value="{label}">{html_lib.escape(_label_text(label))}</option>' for label in range(10))}
      </select>
    </section>
    <div id="documents">{document_html}</div>
  </main>
  <script>
    const documents = [...document.querySelectorAll('.document')];
    const search = document.getElementById('search');
    const resultFilter = document.getElementById('resultFilter');
    const labelFilter = document.getElementById('labelFilter');
    function applyFilters() {{
      const query = search.value.trim().toLowerCase();
      const result = resultFilter.value;
      const label = labelFilter.value;
      for (const item of documents) {{
        const queryOk = !query || item.dataset.search.includes(query);
        const resultOk = result === 'all' || item.dataset.result === result;
        const labelOk = label === 'all' || item.dataset.label === label;
        item.classList.toggle('hidden', !(queryOk && resultOk && labelOk));
      }}
    }}
    search.addEventListener('input', applyFilters);
    resultFilter.addEventListener('change', applyFilters);
    labelFilter.addEventListener('change', applyFilters);
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return {
        "input_rows": len(rows),
        "input_documents": input_document_count,
        "multi_chunk_documents": multi_chunk_count,
        "error_documents": error_count,
        "unlabeled_documents": unlabeled_count,
        "rendered_documents": len(selected),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="attention_outputs.csv를 문서 조각별 Attention 히트맵 HTML로 변환"
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_html", required=True)
    parser.add_argument("--max_documents", type=int, default=50)
    parser.add_argument("--min_chunks", type=int, default=2)
    parser.add_argument("--errors_only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = generate_attention_heatmap(
        args.input_csv,
        args.output_html,
        max_documents=args.max_documents,
        min_chunks=args.min_chunks,
        errors_only=args.errors_only,
    )
    print(f"Attention 히트맵 저장: {Path(args.output_html).resolve()}")
    print(
        "입력 문서={input_documents}, 다중 조각={multi_chunk_documents}, "
        "오분류={error_documents}, 표시={rendered_documents}".format(**summary)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
