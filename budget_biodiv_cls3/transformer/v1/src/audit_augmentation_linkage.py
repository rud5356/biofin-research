"""Recheck augmentation linkage using a previous run's parsed-original manifest.

Does not load a model or reparse documents. The new split uses canonical CSV
identities; do not reuse the previous checkpoint as a newly evaluated model.
"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from augmentation_csv import append_train_augmentations
from build_dataset import load_label_data, match_documents_to_labels
from train_attention_classifier import (
    build_business_group_key, split_records_three_way,
    undersample_majority_records, save_split_outputs,
)
from utils import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label_file', type=Path, required=True)
    parser.add_argument('--previous_output', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reuse_previous_split', action='store_true',
                        help='Audit old heldout membership; fail if canonical groups overlap')
    args = parser.parse_args()
    if args.output_dir.resolve() == args.previous_output.resolve():
        parser.error('Use a new output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels, _ = load_label_data(args.label_file, 'BIOFIN 1차 카테고리')
    previous = pd.read_csv(args.previous_output / 'dataset_match_success.csv')
    docs = [Path(p) for p in previous.loc[previous.source_type == 'document', 'file_path']]
    matched, _ = match_documents_to_labels(docs, labels)
    if len(matched) != len(docs):
        raise ValueError('Re-matching changed document count; inspect input CSV/version')
    prior_docs = previous.loc[previous.source_type == 'document']
    expected = {(Path(r['file_path']).name, int(r['source_row']), int(r['label']))
                for r in prior_docs.to_dict('records')}
    actual = {(Path(r['file_path']).name, int(r['source_row']), int(r['label']))
              for r in matched.to_dict('records')}
    if actual != expected:
        raise ValueError('Re-matching changed source rows or labels')
    records = matched.to_dict('records')
    lookup = labels.set_index('_source_row')
    for record in previous.loc[previous.source_type != 'document'].to_dict('records'):
        parent = lookup.loc[int(record['source_row'])]
        if int(parent['_label']) != int(record['label']):
            raise ValueError('Original label changed')
        record.update(ministry=str(parent['소관명']), activity_name=str(parent['세부사업명']))
        records.append(record)
    if args.reuse_previous_split:
        assignments = pd.read_csv(args.previous_output / 'split_assignments.csv')
        heldout_files = {
            Path(r['file_path']).name: r['split']
            for r in assignments.to_dict('records') if r['split'] != 'train'
        }
        splits = {'train': [], 'valid': [], 'test': []}
        for record in records:
            splits[heldout_files.get(Path(record['file_path']).name, 'train')].append(record)
        train, valid, test = (splits[name] for name in ('train', 'valid', 'test'))
    else:
        train, valid, test = split_records_three_way(records, .1, .1, args.seed)
    # Save originals only: augmentation text is already in the input CSV.
    save_split_outputs(train, valid, test, args.output_dir)
    before = Counter(str(r['label']) for r in train)
    train, undersampling = undersample_majority_records(train, 0, 1.0, 300, args.seed)
    originals = Counter(str(r['label']) for r in train)
    final, augmentation = append_train_augmentations(
        args.label_file, train, valid, test, build_business_group_key)
    groups = [{build_business_group_key(r) for r in rows}
              for rows in (final, valid, test)]
    overlap = (groups[0] & groups[1]) | (groups[0] & groups[2]) | (groups[1] & groups[2])
    assert not overlap
    missing = sorted(set(originals) - set(augmentation['added_by_category']))
    report = {
        'mode': 'linkage_audit_not_model_training',
        'source_manifest': str(args.previous_output),
        'reuse_previous_split': args.reuse_previous_split,
        'original_rows': len(records), 'seed': args.seed,
        'train_before_undersampling': dict(sorted(before.items())),
        'train_original_by_category': dict(sorted(originals.items())),
        'train_after_augmentation_by_category': dict(sorted(Counter(str(r['label']) for r in final).items())),
        'valid_by_category': dict(sorted(Counter(str(r['label']) for r in valid).items())),
        'test_by_category': dict(sorted(Counter(str(r['label']) for r in test).items())),
        'overlap_groups': len(overlap), 'classes_without_augmentation': missing,
        'augmentation': augmentation, 'undersampling': undersampling,
        'note': 'Reuses previous successful parsing manifest. Full training may differ if document parsing or library versions differ. split_summary.json describes originals before undersampling and augmentation.',
    }
    write_json(report, args.output_dir / 'linkage_audit.json')
    print(__import__('json').dumps(report, ensure_ascii=True, indent=2))
    if missing:
        raise ValueError(f'Classes still missing augmentations: {missing}')


if __name__ == '__main__':
    main()
