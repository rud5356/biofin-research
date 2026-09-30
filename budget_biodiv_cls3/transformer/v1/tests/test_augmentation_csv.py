import csv
import hashlib
import json
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC.parents[2] / 'augmentation_h200'))
from augmentation_csv import append_train_augmentations, read_original_frame
from build_merged_training_csv import build
from build_dataset import load_label_data
from train_attention_classifier import build_business_group_key
from build_dataset import match_documents_to_labels
import pytest


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_merge_keeps_originals_and_only_adds_training_parent(tmp_path):
    original = tmp_path / 'original.csv'
    labels = tmp_path / 'labels.csv'
    rows = [{'business_key': str(i), '회계연도': '2023', '소관명': '부처',
             '세부사업명': f'사업{i}', '예산액': '00100'} for i in range(4)]
    write_csv(original, rows)
    write_csv(labels, [{**r, '1차 카테고리': '4', '하위 카테고리': '7'} for r in rows[:3]])
    before = original.read_bytes()
    source_hash = hashlib.sha256(labels.read_bytes()).hexdigest()
    aug = tmp_path / 'aug'
    aug.mkdir()
    records = [{'text': f'생성문 {i}', 'label': '4', 'primary_category': '4',
                'subcategory': '7', 'source_business_key': str(i),
                'source_csv_sha256': source_hash, 'backend': 'ollama'} for i in range(3)]
    records += [records[0], {**records[0], 'label': '2'}, {**records[0], 'source_business_key': 'missing'}]
    (aug / 'aug-example.jsonl').write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
    report = build(original, labels, aug, tmp_path / 'out')
    output = Path(report['csv'])
    merged = list(csv.DictReader(output.open(encoding='utf-8-sig', newline='')))
    assert len(merged) == 7
    for source, copy in zip(rows, merged):
        assert all(copy[key] == value for key, value in source.items())
    assert original.read_bytes() == before
    assert report['counts']['unlabeled_original_rows'] == 1
    assert report['counts']['excluded_duplicate_text'] == 1
    assert report['counts']['excluded_label_mismatch'] == 1
    assert report['counts']['excluded_unmatched_source_business'] == 1
    frame = read_original_frame(output)
    assert len(frame) == 3
    loaded, _ = load_label_data(output, 'BIOFIN 1차 카테고리')
    assert len(loaded) == 3 and set(loaded['_label']) == {4}
    originals = [{'ministry': '부처', 'activity_name': f'사업{i}', 'label': 4} for i in range(3)]
    train, valid, test = [originals[0]], [originals[1]], [originals[2]]
    result, stats = append_train_augmentations(output, train, valid, test, build_business_group_key)
    assert len(result) == 2 and result[1]['text'] == '생성문 0'
    assert result[1]['source_type'] == 'augmentation'
    assert stats['excluded_heldout'] == 2 and stats['added_to_train'] == 1
    assert train == [originals[0]] and valid == [originals[1]] and test == [originals[2]]


def test_legacy_csv_unchanged(tmp_path):
    path = tmp_path / 'legacy.csv'
    write_csv(path, [{'회計': '2023', 'label': '1'}])
    assert len(read_original_frame(path)) == 1
    train = [{'text': 'original'}]
    result, stats = append_train_augmentations(path, train, [], [], lambda row: '')
    assert result is train and stats == {}


@pytest.mark.parametrize('explicit', [False, True])
def test_filename_suffix_uses_original_identity_before_split(tmp_path, explicit):
    path = tmp_path / 'merged.csv'
    docs = [tmp_path / f'2023_부처_희소사업{i}_2023-001-100_123456789012.hwp'
            for i in range(3)]
    rows = []
    for i, doc in enumerate(docs):
        row = {'business_key': str(i), '회계연도': '2023', '소관명': '부처',
               '세부사업명': f'희소사업{i}', 'BIOFIN 1차 카테고리': '1',
               'row_type': 'original', 'training_eligible': '1',
               'augmented_text': '', 'augmentation_id': '',
               'source_business_key': '',
               '사업설명자료_파일명': doc.name if explicit else ''}
        rows.append(row)
    rows += [{**row, 'row_type': 'augmentation', 'source_business_key': str(i),
              'augmentation_id': str(i), 'augmented_text': f'희소 증강문 {i}'}
             for i, row in enumerate(list(rows))]
    write_csv(path, rows)
    labels, _ = load_label_data(path, 'BIOFIN 1차 카테고리')
    matched, _ = match_documents_to_labels(docs, labels)
    originals = matched.to_dict('records')
    assert len(originals) == 3
    assert [r['activity_name'] for r in originals] == [f'희소사업{i}' for i in range(3)]
    train, stats = append_train_augmentations(
        path, originals[:1], originals[1:2], originals[2:], build_business_group_key)
    assert len(train) == 2
    assert stats['added_by_category'] == {'1': 1}
    assert stats['excluded_heldout'] == 2
    assert build_business_group_key(train[0]) == build_business_group_key(train[1])
    broken = {**originals[0], 'activity_name': '희소사업0_2023-001-100'}
    with pytest.raises(ValueError, match='identity mismatch'):
        append_train_augmentations(path, [broken], originals[1:2], originals[2:],
                                   build_business_group_key)


def test_augmentation_rejects_shared_train_and_heldout_group(tmp_path):
    path = tmp_path / 'merged.csv'
    write_csv(path, [{'row_type': 'original', 'training_eligible': '1',
                      'augmented_text': '', 'augmentation_id': ''}])
    record = {'ministry': '부처', 'activity_name': '같은사업'}
    with pytest.raises(ValueError, match='overlap'):
        append_train_augmentations(path, [record], [record], [], build_business_group_key)


def test_class_cap_keeps_originals_balances_parents_and_excludes_heldout(tmp_path):
    from collections import Counter
    path = tmp_path / 'merged.csv'
    originals = []
    train, valid = [], []
    for label in (1, 3):
        for i in range(3):
            key = f'{label}-{i}'
            originals.append({'business_key': key, '회계연도': '2023', '소관명': '부처',
                              '세부사업명': key, 'BIOFIN 1차 카테고리': str(label),
                              'row_type': 'original', 'training_eligible': '1',
                              'augmented_text': '', 'augmentation_id': '', 'source_business_key': ''})
            (valid if i == 2 else train).append({'ministry': '부처', 'activity_name': key, 'label': label})
    rows = list(originals)
    for parent in originals:
        for j in range(10):
            key = parent['business_key']
            rows.append({**parent, 'row_type': 'augmentation', 'source_business_key': key,
                         'augmentation_id': f'{key}-{j}', 'augmented_text': f'text {key} {j}'})
    write_csv(path, rows)
    args = (path, train, valid, [], build_business_group_key)
    result, summary = append_train_augmentations(*args, per_class=5, seed=42)
    again, _ = append_train_augmentations(*args, per_class=5, seed=42)
    assert result == again
    assert result[:len(train)] == train
    assert len(result) == len(train) + 10
    assert summary['added_by_category'] == {'1': 5, '3': 5}
    assert summary['excluded_heldout'] == 20
    assert summary['excluded_by_limit'] == 30
    counts = Counter(r['source_business_key'] for r in result[len(train):])
    assert sorted(counts.values()) == [2, 2, 3, 3]
    assert not any(k.endswith('-2') for k in counts)
    only_original, summary = append_train_augmentations(*args, per_class=0)
    assert only_original == train and summary['added_to_train'] == 0
    full, _ = append_train_augmentations(*args)
    assert len(full) == len(train) + 40
    with pytest.raises(ValueError):
        append_train_augmentations(*args, per_class=-1)


def test_cli_accepts_controlled_augmentation_and_step_validation():
    from train_attention_classifier import build_argument_parser, validate_arguments
    for cap in ('0', '300'):
        args = build_argument_parser().parse_args([
            '--augmentation_per_class', cap, '--eval_steps', '500', '--learning_rate', '5e-6'])
        validate_arguments(args)
        assert not args.class_weight and not args.balanced_sampling and not args.undersample_majority
    args.eval_steps = -1
    with pytest.raises(ValueError):
        validate_arguments(args)
