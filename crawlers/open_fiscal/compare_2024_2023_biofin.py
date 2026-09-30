"""Partition 2024 LLM results against 2023, retaining both original and LLM labels."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from build_biofin_continuity import clean, identity, similarity, SequenceMatcher, IDENTITY_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'budget_biodiv_cls3/llm/v1/outputs'
CURRENT = BASE / '20260929_2024/open_fiscal_2024_11ministries_llm_classified.csv'
if not CURRENT.exists():
    CURRENT = BASE / '20260929_2024/열린재정2024_11개 대상 부처_llm_classified.csv'
PREVIOUS = BASE / '20260922_/BIOFIN_2023_취합_2026.09.14_llm_classified.csv'
OUTPUT = BASE / '20260930_2024_vs_2023'
THRESHOLD = 0.72
MARGIN = 0.05
LLM = 'LLM BIOFIN 1차 카테고리'


def read(path, encoding):
    with path.open(encoding=encoding, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert all(None not in row and all(v is not None for v in row.values()) for row in rows)
        return rows, reader.fieldnames


@lru_cache(maxsize=250000)
def ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()


def score(a, b):
    # Identical to build_biofin_continuity.similarity, with cached string ratios.
    if a[0] != b[0]:
        return 0.0
    return .60 * ratio(a[6], b[6]) + .25 * ratio(a[5], b[5]) + .15 * ratio(a[4], b[4])


def joined(group, column):
    return ' | '.join(dict.fromkeys(r[column] for _, r in group))


def agreement(a, b):
    if not a or not b:
        return '미확인'
    if ' | ' in a or ' | ' in b:
        return '복수값 검토'
    return '일치' if a == b else '불일치'


def compact_rows(rows, previous, is_duplicate):
    docs24, _ = read(ROOT / 'crawlers/open_fiscal/outputs/2024/open_fiscal_2024_11ministries.csv', 'utf-8-sig')
    docs23, _ = read(ROOT / 'budget_biodiv_cls3/document/open/BIOFIN_2023_취합_2026.09.14_document_matched.csv', 'utf-8-sig')
    lookup24 = {r['business_key']: r for r in docs24}
    lookup23 = {(r['No.'], identity(r)): r for r in docs23}
    assert len(lookup24) == len(docs24)
    assert len(lookup23) == len(docs23)
    result = []
    for row in rows:
        doc24 = lookup24[row['2024_business_key']]
        assert doc24['소관명'] == row['2024_소관명']
        assert doc24['세부사업명'] == row['2024_세부사업명']
        out = {'2024_' + field: row['2024_' + field] for field in IDENTITY_COLUMNS}
        out['2024_LLM카테고리'] = row['2024_LLM카테고리']
        if is_duplicate:
            out.update({'2023_' + field: row['2023_' + field] for field in IDENTITY_COLUMNS[:-1]})
            out['2023_세부사업명'] = row['2023_비교사업명']
            out.update({k: row[k] for k in [
                '2023_원래카테고리', '2023_LLM카테고리',
                '매칭방식', '유사도', '검토필요', '검토사유']})
        else:
            out['2023_참고사업명(매칭미확정)'] = row['2023_비교사업명']
            out['참고유사도'] = row['유사도']
        for field in ['business_key', '사업설명자료_파일명', '사업설명자료_상대경로']:
            out['2024_' + field] = doc24[field]
        if is_duplicate:
            selected = []
            for n in row['2023_원본행번호'].split(' | '):
                source = previous[int(n) - 2]
                doc = lookup23[(source['No.'], identity(source))]
                assert doc['1차 카테고리'] == source['1차 카테고리']
                selected.append(doc)
            for field in ['business_key', '사업설명자료_파일명', '사업설명자료_상대경로']:
                out['2023_' + field] = ' | '.join(dict.fromkeys(r[field] for r in selected))
        result.append(out)
    return result


def main():
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (CURRENT, PREVIOUS)}
    current, current_columns = read(CURRENT, 'utf-8-sig')
    previous, _ = read(PREVIOUS, 'cp949')
    assert all(r['회계연도'] == '2024' for r in current)
    assert all(r['회계연도'] == '2023' for r in previous)
    groups = defaultdict(list)
    by_ministry = defaultdict(list)
    by_name = defaultdict(list)
    for number, row in enumerate(previous, 2):
        groups[identity(row)].append((number, row))
    for key in groups:
        by_ministry[key[0]].append(key)
        by_name[(key[0], key[6])].append(key)
    duplicate, new = [], []
    for number, row in enumerate(current, 2):
        key = identity(row)
        if key in groups:
            ranked = [(1.0, key)]
            method = '7개 사업정보 정확일치'
            accepted = True
        else:
            names = by_name.get((key[0], key[6]), [])
            pool = names or by_ministry.get(key[0], [])
            ranked = sorted(((score(key, candidate), candidate) for candidate in pool),
                            key=lambda pair: (-pair[0], pair[1]))
            method = '같은 부처·세부사업명 일치' if names else '가중 유사도'
            accepted = bool(names) or bool(ranked and ranked[0][0] >= THRESHOLD)
        best_score, best = ranked[0] if ranked else (0.0, None)
        second_score, second = ranked[1] if len(ranked) > 1 else (None, None)
        if best:
            assert abs(best_score - similarity(key, best)) < 1e-12
        ambiguous = second is not None and best_score - second_score < MARGIN
        selected = groups[best] if best else []
        original_category = joined(selected, '1차 카테고리')
        previous_llm = joined(selected, LLM)
        current_llm = row[LLM]
        reasons = []
        if not accepted:
            reasons.append('유사도 기준 미달: 명칭 변경·통합·분리 여부 확인 후 신규 판단')
        elif method == '가중 유사도':
            reasons.append('유사명칭 사업의 실제 연속성 확인')
        elif method != '7개 사업정보 정확일치':
            reasons.append('사업명 동일하나 회계·상위사업 등 변경')
        if ambiguous:
            reasons.append('1·2순위 후보 점수 차 0.05 미만')
        if ' | ' in original_category or ' | ' in previous_llm:
            reasons.append('2023 동일 사업정보의 카테고리 충돌')
        out = {
            '구분': '중복·유사사업' if accepted else '신규사업 후보',
            '매칭방식': method if accepted else '기준 이상 매칭 없음',
            '검토필요': '예' if reasons else '아니오',
            '검토사유': '; '.join(reasons),
            '2024_소관명': row['소관명'],
            '2024_세부사업명': row['세부사업명'],
            '2024_LLM카테고리': current_llm,
            '2023_비교사업명': best[6] if best else '',
            '2023_원래카테고리': original_category,
            '2023_LLM카테고리': previous_llm,
            '2023_원래대비2023LLM': agreement(original_category, previous_llm),
            '2023원래대비2024LLM': agreement(original_category, current_llm) if accepted else '매칭미확정',
            '2023LLM대비2024LLM': agreement(previous_llm, current_llm) if accepted else '매칭미확정',
            '유사도': f'{best_score:.6f}',
            '2023_비교사업의미': '선정된 매칭 후보' if accepted else '참고용 최근접 후보(매칭 아님)',
            '2023_소관명': best[0] if best else '',
            '2023_회계명': best[1] if best else '',
            '2023_분야명': best[2] if best else '',
            '2023_부문명': best[3] if best else '',
            '2023_프로그램명': best[4] if best else '',
            '2023_단위사업명': best[5] if best else '',
            '2023_원본행번호': ' | '.join(str(n) for n, _ in selected),
            '2023_No.': joined(selected, 'No.'),
            '2023_동일사업정보_원본행수': len(selected),
            '2023_원래하위카테고리': joined(selected, '하위 카테고리'),
            '2023_원래판단근거': joined(selected, '판단근거'),
            '2023_LLM_confidence': joined(selected, 'confidence'),
            '2023_LLM_reason': joined(selected, 'reason'),
            '2023_LLM_evidence': joined(selected, 'evidence'),
            '차순위_2023사업명': second[6] if second else '',
            '차순위_2023회계명': second[1] if second else '',
            '차순위_2023프로그램명': second[4] if second else '',
            '차순위_2023단위사업명': second[5] if second else '',
            '차순위_유사도': f'{second_score:.6f}' if second is not None else '',
            '차순위_2023원본행번호': ' | '.join(str(n) for n, _ in groups[second]) if second else '',
            '차순위_2023원래카테고리': joined(groups[second], '1차 카테고리') if second else '',
            '차순위_2023LLM카테고리': joined(groups[second], LLM) if second else '',
            '상위2후보_모호여부': '예' if ambiguous else '아니오',
            '2024_원본행번호': number,
        }
        for column in current_columns:
            if column == LLM:
                continue
            out.setdefault('2024_' + column, row[column])
        (duplicate if accepted else new).append(out)
        if number % 500 == 0:
            print(f'Compared {number-1}/{len(current)}', flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, rows in [('2024_2023_중복_유사사업.csv', duplicate), ('2024_신규사업_후보.csv', new)]:
        compact = compact_rows(rows, previous, rows is duplicate)
        columns = list(compact[0])
        path = OUTPUT / filename
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(compact)
        loaded, headers = read(path, 'utf-8-sig')
        assert len(loaded) == len(rows) and headers == columns
        assert loaded == compact
        paths.append(path)
    all_rows = duplicate + new
    assert len(all_rows) == len(current)
    assert {r['2024_원본행번호'] for r in duplicate}.isdisjoint(r['2024_원본행번호'] for r in new)
    assert sorted(r['2024_원본행번호'] for r in all_rows) == list(range(2, len(current) + 2))
    for out in all_rows:
        original = current[out['2024_원본행번호'] - 2]
        for c in current_columns:
            target = '2024_LLM카테고리' if c == LLM else '2024_' + c
            assert out[target] == original[c]
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    summary = {
        'input_2024': len(current), 'input_2023': len(previous),
        'duplicate_similar': len(duplicate), 'new_candidates': len(new),
        'methods': dict(Counter(r['매칭방식'] for r in duplicate)),
        'duplicate_review_needed': sum(r['검토필요'] == '예' for r in duplicate),
        'ambiguous_duplicate': sum(r['상위2후보_모호여부'] == '예' for r in duplicate),
        'category_agreement': dict(Counter(r['2023원래대비2024LLM'] for r in duplicate)),
        'threshold': THRESHOLD, 'ambiguity_margin': MARGIN,
        'source_sha256': hashes,
        'new_by_ministry': dict(Counter(r['2024_소관명'] for r in new)),
    }
    (OUTPUT / 'matching_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUTPUT / '매칭기준.txt').write_text(
        '2024년 1행당 결과 1행. 원본 2024년 행은 누락·중복 배정 없이 두 CSV에 분할.\n'
        '2023 원래 카테고리 = 1차 카테고리, 2023/2024 LLM = LLM BIOFIN 1차 카테고리.\n'
        '같은 부처 내 7개 사업정보 정확일치 우선, 다음은 세부사업명 일치, 다음은 기존 가중 유사도 0.72 이상.\n'
        '기존 build_biofin_continuity.py와 동일한 정규화(NFKC·공백정리) 및 가중치: 세부사업명 60%, 단위사업명 25%, 프로그램명 15%.\n'
        '동일 사업명이 여러 개이면 가중 점수 순으로 선정. 동점은 사업정보 정렬 순이며 검토 필요로 표시.\n'
        '차순위는 서로 다른 7개 사업정보 기준. 정확일치가 있으면 다른 후보 검색을 생략. 동일 사업명 일치 시 차순위도 동일명 후보 중 선정.\n'
        '2023에 사업정보가 같은 여러 행은 후보 하나로 묶고 사업키·문서 경로를 |로 병기. 카테고리 값이 여러 개면 |로 병기.\n'
        '유사매칭과 사업명만 같은 건은 검토 필요. 1·2순위 점수 차 0.05 미만도 표시.\n'
        '신규 후보 파일의 2023 참고사업명은 최근접 후보이며 매칭 확정이 아님. 신규 후보 파일에는 2023 카테고리를 넣지 않음.\n'
        '간소화: 중복·유사 파일 27열, 신규 후보 파일 13열. 두 파일에 2024년 소관명·회계명·분야명·부문명·프로그램명·단위사업명·세부사업명 유지. 중복 파일은 2023년 7개 항목도 유지. 긴 LLM 설명·신뢰도·차순위 상세·중복 비교 열은 삭제.\n'
        '사업설명자료 파일명·상대경로는 원본 자료에서 사업키(2024), No.와 7개 사업정보(2023)로 정확히 연결. 빈 경로는 원본에 경로가 없는 경우.\n'
        '2024 상대경로 기준 폴더: crawlers/open_fiscal/outputs/2024. 2023은 해당 연도 문서 폴더를 기준으로 연결.\n'
        '행정경비·카테고리 0을 포함한 입력 전체를 비교. 기존 연속사업 분석의 BIOFIN 양성 필터/종료사업 제외 규칙은 적용하지 않음.\n'
        '파일은 UTF-8 BOM CSV. 부처 간 이관, 사업 통폐합, 큰 명칭 변경은 별도 검토 필요.\n',
        encoding='utf-8-sig')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
