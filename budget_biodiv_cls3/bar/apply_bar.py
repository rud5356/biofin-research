"""분류 결과 CSV에 카테고리별 귀속률(BAR)을 적용해 귀속 예산 열을 추가한다 (Python 3.8+, 표준 라이브러리만 사용)."""
import argparse
import csv
import json
from collections import OrderedDict
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

ENCODINGS = ("utf-8-sig", "cp949")
# 앞에 있을수록 우선한다. pipeline 결과에는 원래 모델 열도 함께 있을 수 있다.
CATEGORY_COLUMNS = ("pipeline_label", "LLM BIOFIN 1차 카테고리", "예측 BIOFIN 1차 카테고리")
RATE_COLUMN = "BAR 귀속률(%)"
STATUS_COLUMN = "BAR 적용상태"
VERSION_COLUMN = "BAR 버전"
# 단위별로 원 단위까지 남기는 소수 자릿수 (지방재정365는 원, 열린재정은 백만원).
UNIT_DECIMALS = {"원": 0, "천원": 3, "백만원": 6}
APPLIED, RATE_MISSING, NO_CATEGORY, UNKNOWN_CATEGORY ="적용", "귀속률 미정", "카테고리 없음", "알 수 없는 카테고리"


def amount_column_name(column):
    return f"BAR 귀속 {column}"


def load_rates(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    rates = config.get("rates_percent")
    if not isinstance(rates, dict) or not config.get("version"):
        raise ValueError(f"귀속률 파일에 version과 rates_percent가 필요합니다: {path}")
    parsed = {}
    for key, value in rates.items():
        if value is None:
            parsed[str(key)] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 100:
            raise ValueError(f"카테고리 {key}의 귀속률은 0~100 사이 숫자 또는 null이어야 합니다: {value!r}")
        parsed[str(key)] = Decimal(str(value))
    return str(config["version"]), parsed


def read_csv(path):
    last_error = None
    for encoding in ENCODINGS:
        try:
            with Path(path).open(encoding=encoding, newline="") as stream:
                reader = csv.DictReader(stream)
                fields = list(reader.fieldnames or [])
                rows = list(reader)
        except UnicodeError as exc:
            last_error = exc
            continue
        if not fields or len(set(fields)) != len(fields):
            raise ValueError(f"CSV 헤더가 없거나 중복된 열 이름이 있습니다: {path}")
        if any(None in row or None in row.values() for row in rows):
            raise ValueError(f"CSV 컬럼 수가 헤더와 다릅니다: {path}")
        return fields, rows, encoding
    raise ValueError(f"CSV를 읽을 수 없습니다(utf-8-sig, cp949 모두 실패): {path}") from last_error


def find_category_column(fields, requested=None):
    if requested:
        if requested not in fields:
            raise ValueError(f"카테고리 열이 없습니다: {requested}")
        return requested
    for column in CATEGORY_COLUMNS:
        if column in fields:
            return column
    raise ValueError(f"카테고리 열을 찾지 못했습니다. --category-col로 지정해주세요. 찾은 후보: {', '.join(CATEGORY_COLUMNS)}")


def parse_category(value):
    """'6', '6.0' 같은 값을 '6'으로 맞춘다. 빈칸이면 None."""
    text = (value or "").strip()
    if not text:
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    return str(int(number)) if number == number.to_integral_value() else text


def parse_amount(value, column, line):
    text = (value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        amount = Decimal(text)
    except InvalidOperation:
        raise ValueError(f"{line}행 '{column}' 값이 숫자가 아닙니다: {value!r}") from None
    if not amount.is_finite():
        raise ValueError(f"{line}행 '{column}' 값이 숫자가 아닙니다: {value!r}")
    return amount


def format_amount(amount, decimals):
    value = amount.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)
    # 소수 자릿수를 쓰는 경우(백만원 단위 등) 880.500000 대신 880.5로 적는다.
    return format(value.normalize(), "f") if decimals > 0 else str(value)


def apply_bar(rows, category_column, amount_columns, rates, version, decimals=0):
    """원본 행을 바꾸지 않고 BAR 열을 붙인 새 행 목록과 카테고리별 요약을 반환한다."""
    result, summary = [], OrderedDict()
    for line, row in enumerate(rows, start=2):
        category = parse_category(row[category_column])
        amounts = {column: parse_amount(row[column], column, line) for column in amount_columns}
        if category is None:
            status, rate = NO_CATEGORY, None
        elif category not in rates:
            status, rate = UNKNOWN_CATEGORY, None
        else:
            rate = rates[category]
            status = APPLIED if rate is not None else RATE_MISSING
        output = dict(row)
        output[RATE_COLUMN] = "" if rate is None else format(rate.normalize(), "f")
        output[STATUS_COLUMN] = status
        output[VERSION_COLUMN] = version
        key = category if category is not None else "(빈칸)"
        item = summary.setdefault(key, {"status": status, "rate": output[RATE_COLUMN], "count": 0,
                                        "amounts": {column: Decimal(0) for column in amount_columns},
                                        "attributed": {column: Decimal(0) for column in amount_columns}})
        item["count"] += 1
        for column, amount in amounts.items():
            if amount is not None:
                item["amounts"][column] += amount
            if rate is None or amount is None:
                output[amount_column_name(column)] = ""
                continue
            attributed = (amount * rate / 100).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)
            output[amount_column_name(column)] = format_amount(attributed, decimals)
            item["attributed"][column] += attributed
        result.append(output)
    return result, summary


def summary_rows(summary, amount_columns, version, decimals):
    def order(key):
        return (0, int(key)) if key.isdigit() else (1, key)
    rows = []
    for key in sorted(summary, key=order):
        item = summary[key]
        row = OrderedDict([("BIOFIN 1차 카테고리", key), (STATUS_COLUMN, item["status"]),
                           (RATE_COLUMN, item["rate"]), ("건수", item["count"])])
        for column in amount_columns:
            row[f"{column} 합계"] = format_amount(item["amounts"][column], decimals)
            row[f"{amount_column_name(column)} 합계"] = (
                format_amount(item["attributed"][column], decimals) if item["status"] == APPLIED else "")
        row[VERSION_COLUMN] = version
        rows.append(row)
    return rows


def write_csv(path, fields, rows, encoding, overwrite):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding=encoding, newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="LLM·Transformer·pipeline 분류 결과 CSV")
    parser.add_argument("--output", type=Path, help="결과 CSV (기본: 입력 파일명_bar.csv)")
    parser.add_argument("--rates", type=Path, default=Path(__file__).with_name("bar_rates.json"))
    parser.add_argument("--amount-cols", default="예산액", help="귀속률을 곱할 금액 열, 쉼표로 여러 개 지정 (예: 예산액,지출액)")
    parser.add_argument("--category-col", help="카테고리 열 이름 (기본: 자동 탐색)")
    rounding = parser.add_mutually_exclusive_group()
    rounding.add_argument("--unit", choices=list(UNIT_DECIMALS),
                          help="입력 금액 단위. 지정하면 원 단위로 반올림 (예: 백만원 → 소수 6자리)")
    rounding.add_argument("--decimals", type=int, help="반올림 소수 자릿수를 직접 지정 (기본 0: 입력 금액 단위의 정수)")
    parser.add_argument("--overwrite", action="store_true", help="결과 파일이 이미 있으면 덮어쓰기")
    args = parser.parse_args(argv)
    decimals = UNIT_DECIMALS[args.unit] if args.unit else (args.decimals or 0)
    if decimals < 0:
        parser.error("--decimals는 0 이상이어야 합니다.")
    try:
        version, rates = load_rates(args.rates)
        fields, rows, encoding = read_csv(args.input)
        category_column = find_category_column(fields, args.category_col)
        amount_columns = [column.strip() for column in args.amount_cols.split(",") if column.strip()]
        missing = [column for column in amount_columns if column not in fields]
        if not amount_columns or missing:
            raise ValueError(f"금액 열이 없습니다: {', '.join(missing) or '(빈 값)'}")
        new_fields = [RATE_COLUMN] + [amount_column_name(column) for column in amount_columns] + [STATUS_COLUMN, VERSION_COLUMN]
        clashes = set(fields) & set(new_fields)
        if clashes:
            raise ValueError(f"이미 BAR 열이 있습니다(BAR 적용된 파일인지 확인): {', '.join(sorted(clashes))}")
        result, summary = apply_bar(rows, category_column, amount_columns, rates, version, decimals)
        output = args.output or args.input.with_name(f"{args.input.stem}_bar.csv")
        summary_path = output.with_name(f"{output.stem}_summary.csv")
        write_csv(output, fields + new_fields, result, encoding, args.overwrite)
        summary_table = summary_rows(summary, amount_columns, version, decimals)
        write_csv(summary_path, list(summary_table[0]) if summary_table else ["BIOFIN 1차 카테고리"],
                  summary_table, encoding, args.overwrite)
    except FileExistsError as error:
        parser.exit(1, f"오류: 결과 파일이 이미 있습니다. --overwrite 또는 --output을 사용하세요: {error.filename}\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"오류: {error}\n")
    counts = {}
    for row in result:
        counts[row[STATUS_COLUMN]] = counts.get(row[STATUS_COLUMN], 0) + 1
    print(f"카테고리 열: {category_column} / 금액 열: {', '.join(amount_columns)} / BAR 버전: {version}")
    print(f"처리 {len(result):,}행: " + ", ".join(f"{status} {count:,}" for status, count in counts.items()))
    if counts.get(RATE_MISSING):
        print(f"안내: 귀속률이 정해지지 않은 카테고리는 금액을 빈칸으로 두었습니다. {args.rates}를 채운 뒤 다시 실행하세요.")
    print(f"결과: {output}\n요약: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
