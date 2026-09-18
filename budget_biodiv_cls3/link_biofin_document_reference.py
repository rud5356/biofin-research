"""기존 문서 매칭 CSV에서 연결 정보를 가져온다. 원본 데이터와 행 순서는 유지한다."""
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEY_COLUMNS = (
    '회계연도', '소관명', '회계명', '계정명', '분야명', '부문명',
    '프로그램명', '단위사업명', '세부사업명',
)
LINK_COLUMNS = (
    'business_key', '사업설명자료_파일명', '사업설명자료_상대경로',
    '사업설명자료_절대경로', '문서매칭상태', '문서매칭방식', '문서매칭후보수',
)


def read_csv(path):
    for encoding in ('utf-8-sig', 'cp949'):
        try:
            with path.open(encoding=encoding, newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                headers = reader.fieldnames or []
            return headers, rows
        except UnicodeDecodeError:
            continue
    raise ValueError(f'CSV 인코딩을 읽을 수 없습니다: {path}')


def business_key(row):
    # 문장부호는 보존하여 서로 다른 사업을 과도하게 합치지 않는다.
    return tuple(re.sub(r'\s+', '', unicodedata.normalize('NFKC', row[c])) for c in KEY_COLUMNS)


def link_rows(headers, rows, ref_headers, reference):
    for name, columns in [('입력', headers), ('참고', ref_headers)]:
        required = KEY_COLUMNS + (LINK_COLUMNS if name == '참고' else ())
        missing = [c for c in required if c not in columns]
        if missing:
            raise ValueError(f'{name} CSV 필수 컬럼 없음: {missing}')
    overlap = set(headers).intersection(LINK_COLUMNS)
    if overlap:
        raise ValueError(f'원본 연결 컬럼 덮어쓰기를 방지합니다: {sorted(overlap)}')
    index = defaultdict(list)
    for row in reference:
        index[business_key(row)].append(row)
    output = []
    for row in rows:
        candidates = index.get(business_key(row), [])
        links = {tuple(r[c] for c in LINK_COLUMNS) for r in candidates}
        result = dict(row)
        if len(links) == 1:
            result.update(zip(LINK_COLUMNS, next(iter(links))))
        else:
            result.update(dict.fromkeys(LINK_COLUMNS, ''))
            result['문서매칭상태'] = 'REFERENCE_NOT_FOUND' if not candidates else 'REFERENCE_AMBIGUOUS'
            result['문서매칭후보수'] = str(len(candidates))
        output.append(result)
    return list(headers) + list(LINK_COLUMNS), output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-file', type=Path, default=ROOT/'document/open/BIOFIN_2023_취합_2026.09.14.csv')
    parser.add_argument('--reference-file', type=Path, default=ROOT/'document/open/2023biofin_label_matched.csv')
    parser.add_argument('--output-file', type=Path)
    args = parser.parse_args()
    output = args.output_file or args.input_file.with_name(args.input_file.stem + '_document_matched.csv')
    if output.resolve() in {args.input_file.resolve(), args.reference_file.resolve()}:
        parser.error('출력 경로는 원본 및 참고 CSV와 달라야 합니다.')
    headers, rows = read_csv(args.input_file)
    ref_headers, reference = read_csv(args.reference_file)
    out_headers, out_rows = link_rows(headers, rows, ref_headers, reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=out_headers)
        writer.writeheader()
        writer.writerows(out_rows)
    saved_headers, saved = read_csv(output)
    assert saved_headers == out_headers and len(saved) == len(rows)
    assert all(all(before[c] == after[c] for c in headers) for before, after in zip(rows, saved))
    print(f'저장: {output}')
    print(f'원본 {len(rows):,}행 / {len(headers)}개 컬럼 값·순서 보존 확인')
    print(f'참고 CSV의 문서 상태: {dict(Counter(r["문서매칭상태"] for r in saved))}')
    print('문서 상태는 참고 CSV에서 가져온 값입니다. 실행 환경의 실제 파일 존재 여부는 분류기 --dry-run으로 확인하세요.')


if __name__ == '__main__':
    main()
