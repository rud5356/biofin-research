"""Optional merged-CSV support; legacy CSV behavior is unchanged."""
from collections import Counter, defaultdict
import csv
import random
from pathlib import Path

import pandas as pd
from utils import read_csv_flexible

MARKERS = {'row_type', 'training_eligible', 'augmented_text', 'augmentation_id'}


def is_merged(path):
    try:
        with Path(path).open(encoding='utf-8-sig', newline='') as file:
            return MARKERS.issubset(next(csv.reader(file), []))
    except UnicodeDecodeError:
        return False


def read_original_frame(path):
    if not is_merged(path):
        return read_csv_flexible(path)
    originals = []
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        for number, row in enumerate(csv.DictReader(file), 2):
            if row['row_type'] == 'original' and row['training_eligible'] == '1':
                if row.get('BIOFIN 1차 카테고리') not in {str(i) for i in range(10)}:
                    raise ValueError(f'Missing/invalid gold label in original row {number}')
                row['_source_row'] = number
                originals.append(row)
    if not originals:
        raise ValueError('Merged CSV has no labeled originals')
    return pd.DataFrame(originals)


def append_train_augmentations(path, train, valid, test, group_key, *, per_class=None, seed=42):
    if per_class is not None and per_class < 0:
        raise ValueError('augmentation_per_class must be >= 0')
    stats = Counter()
    if not is_merged(path):
        return train, dict(stats)
    train_groups = {group_key(r) for r in train}
    heldout = {group_key(r) for r in valid + test}
    if train_groups & heldout:
        raise ValueError('Original train/heldout business group overlap')
    if per_class == 0:
        return list(train), {'mode': 'original_only', 'added_to_train': 0,
                             'added_by_category': {}, 'per_class_limit': 0}
    # Source-specific label checks prevent unrelated or edited rows entering train.
    parents = {}
    parents_by_row = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        for number, row in enumerate(csv.DictReader(file), 2):
            if row['row_type'] == 'original' and row['training_eligible'] == '1':
                parents[row['business_key']] = row
                parents_by_row[number] = row
    # Catch filename-derived identities before silently discarding augmentations.
    for record in train + valid + test:
        if record.get('source_row') is None:
            continue
        parent = parents_by_row.get(int(record['source_row']))
        if parent is None:
            raise ValueError(f"Missing original CSV row: {record['source_row']}")
        canonical = {'ministry': parent['소관명'], 'activity_name': parent['세부사업명']}
        if group_key(record) != group_key(canonical):
            raise ValueError(
                f"Original business identity mismatch at CSV row {record['source_row']}; "
                'use matched CSV ministry/activity_name before splitting'
            )
    result = list(train)
    seen = set()
    source_file = str(Path(path).resolve())
    accepted_by_label = Counter()
    available_by_label = Counter()
    excluded_by_label = Counter()
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        for number, row in enumerate(csv.DictReader(file), 2):
            if row['row_type'] != 'augmentation':
                continue
            stats['available'] += 1
            if row['training_eligible'] != '1':
                stats['ineligible'] += 1
                continue
            parent = parents.get(row['source_business_key'])
            if parent is None:
                raise ValueError(f'Augmentation row {number} has no labeled source')
            label = parent['BIOFIN 1차 카테고리']
            available_by_label[label] += 1
            if row['BIOFIN 1차 카테고리'] != label:
                raise ValueError(f'Augmentation label mismatch at row {number}')
            record = {'year': int(parent['회계연도']), 'ministry': parent['소관명'],
                      'activity_name': parent['세부사업명'], 'label': int(label),
                      'source_file': source_file, 'source_row': number,
                      'source_type': 'augmentation', 'file_path': f"augmentation://{row['augmentation_id']}",
                      'text': row['augmented_text'], 'source_business_key': row['source_business_key']}
            group = group_key(record)
            if group in heldout:
                stats['excluded_heldout'] += 1
                continue
            if group not in train_groups:
                stats['excluded_no_train_parent'] += 1
                excluded_by_label[label] += 1
                continue
            normalized = ' '.join(record['text'].split())
            if not normalized:
                raise ValueError(f'Empty augmentation at row {number}')
            if normalized in seen:
                stats['duplicate'] += 1
                continue
            seen.add(normalized)
            result.append(record)
            stats['added_to_train'] += 1
            accepted_by_label[label] += 1
    if per_class is not None:
        candidates = result[len(train):]
        result = list(train)
        by_label = defaultdict(lambda: defaultdict(list))
        for record in candidates:
            by_label[int(record['label'])][group_key(record)].append(record)
        for label, parents_for_label in sorted(by_label.items()):
            rng = random.Random(seed + label)
            parent_keys = sorted(parents_for_label)
            rng.shuffle(parent_keys)
            for records in parents_for_label.values():
                rng.shuffle(records)
            count = 0
            while parent_keys and count < per_class:
                remaining = []
                for parent_key in parent_keys:
                    records = parents_for_label[parent_key]
                    result.append(records.pop())
                    count += 1
                    if records:
                        remaining.append(parent_key)
                    if count == per_class:
                        break
                parent_keys = remaining
        stats['eligible_before_limit'] = len(candidates)
        stats['excluded_by_limit'] = len(candidates) - (len(result) - len(train))
        stats['added_to_train'] = len(result) - len(train)
        accepted_by_label = Counter(str(r['label']) for r in result[len(train):])
    return result, {
        **dict(stats),
        'per_class_limit': per_class,
        'sampling_seed': seed,
        'available_by_category': dict(sorted(available_by_label.items())),
        'excluded_no_train_parent_by_category': dict(sorted(excluded_by_label.items())),
        'added_by_category': dict(sorted(accepted_by_label.items())),
    }
