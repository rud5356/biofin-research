import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import predict_attention_classifier as prediction
from document_parser import DocumentParseError


def make_budget(tmp_path):
    path = tmp_path / 'budget.csv'
    pd.DataFrame([
        {'회계연도': 2024, '소관명': '환경부', '세부사업명': name,
         '사업설명자료_파일명': filename, '1차 카테고리': label,
         '판단근거': 'SECRET_GOLD_REASON'}
        for name, filename, label in [('정상사업', 'ok.txt', 2),
                                      ('손상문서사업', 'bad.hwp', 6),
                                      ('문서없는사업', '', 9)]
    ]).to_csv(path, index=False, encoding='utf-8-sig')
    return prediction.load_budget_data(path, '1차 카테고리')


def test_mixed_document_and_metadata_inputs(tmp_path, monkeypatch):
    budget = make_budget(tmp_path)
    def extract(path, **kwargs):
        if path.name == 'bad.hwp':
            raise DocumentParseError('HWP_PARSE_FAILED', 'broken')
        return '정상 본문'
    monkeypatch.setattr(prediction, 'extract_document', extract)
    records, success, failed = prediction.build_prediction_records(
        [tmp_path / 'ok.txt', tmp_path / 'bad.hwp'], budget,
        filename_column='사업설명자료_파일명')
    assert [r['source_row'] for r in records] == [2, 3, 4]
    assert [r['source_type'] for r in records] == ['document', 'budget_metadata', 'budget_metadata']
    assert '정상 본문' in records[0]['text']
    assert records[1]['fallback_reason'] == 'HWP_PARSE_FAILED'
    assert records[2]['true_label'] == 9
    assert len({r['file_path'] for r in records}) == 3
    assert all('SECRET_GOLD_REASON' not in r['text'] and '1차 카테고리' not in r['text'] for r in records)
    assert len(success) == 3 and len(failed) == 1
    result_path = tmp_path / 'classified.csv'
    prediction.save_classified_budget(budget, [dict(r, pred_label=1) for r in records],
                                      result_path, 'prediction')
    assert pd.read_csv(result_path)['prediction'].tolist() == [1, 1, 1]


def test_all_metadata_and_limit(tmp_path):
    budget = make_budget(tmp_path)
    records, _, _ = prediction.build_prediction_records([], budget)
    assert len(records) == 3
    assert all(r['source_type'] == 'budget_metadata' for r in records)
    limited, _, _ = prediction.build_prediction_records([], budget, limit=1)
    assert len(limited) == 1


def test_successful_alternative_document_prevents_duplicate_fallback(tmp_path, monkeypatch):
    budget = make_budget(tmp_path).iloc[:1].copy()
    monkeypatch.setattr(prediction, '_match_by_filename_or_id',
                        lambda *args: (budget.iloc[0], 'MATCHED_FILENAME', ''))
    def extract(path, **kwargs):
        if path.name == 'bad.hwp':
            raise DocumentParseError('HWP_PARSE_FAILED', 'broken')
        return '본문'
    monkeypatch.setattr(prediction, 'extract_document', extract)
    records, _, failed = prediction.build_prediction_records(
        [tmp_path / 'bad.hwp', tmp_path / 'ok.txt', tmp_path / 'duplicate.txt'], budget)
    assert len(records) == 1 and records[0]['source_type'] == 'document'
    assert set(failed['reason']) == {'HWP_PARSE_FAILED', 'DUPLICATE_SOURCE_ROW'}
