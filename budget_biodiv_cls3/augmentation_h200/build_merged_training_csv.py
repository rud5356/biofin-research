"""Build a new v1 training dataset; never modify source CSVs or JSONLs."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        return reader.fieldnames, list(reader)


def build(original_path, label_path, augmentation_dir, output_dir):
    original_path, label_path, augmentation_dir, output_dir = map(Path, (original_path, label_path, augmentation_dir, output_dir))
    original_hash, label_hash = sha(original_path), sha(label_path)
    fields, originals = read_rows(original_path)
    label_fields, labels = read_rows(label_path)
    by_key = {r['business_key']: r for r in originals}
    gold = {r['business_key']: r for r in labels}
    if len(by_key) != len(originals) or len(gold) != len(labels) or '' in by_key or '' in gold:
        raise ValueError('business_key must be nonempty and unique in both source CSVs')
    for row in labels:
        if row['1차 카테고리'] not in {str(i) for i in range(10)}:
            raise ValueError('Invalid primary category')
    files = sorted(augmentation_dir.glob('aug-*.jsonl'))
    if not files:
        raise ValueError('No augmentation files found')
    # Refuse overwriting a previous export or accidentally writing into an input directory.
    output_dir.mkdir(parents=True, exist_ok=False)
    target = output_dir / 'open_fiscal_2023_augmented.csv'
    extras = ['BIOFIN 1차 카테고리', 'row_type', 'training_eligible', 'augmented_text',
              'augmentation_id', 'augmentation_source_file', 'augmentation_source_line',
              'source_business_key', 'source_csv_sha256', 'needs_review', 'generation_model']
    columns = list(dict.fromkeys(fields + label_fields + extras))
    stats = Counter()
    original_counts, augmented_counts = Counter(), Counter()
    seen = set()
    def base(row, kind):
        key = row['business_key']
        # Preserve every original fiscal column, enrich only missing columns from gold CSV.
        merged = {**gold.get(key, {}), **row}
        merged.update(row_type=kind, training_eligible='1' if key in gold else '0',
                      source_business_key=key, source_csv_sha256=label_hash)
        merged['BIOFIN 1차 카테고리'] = gold.get(key, {}).get('1차 카테고리', '')
        return merged
    temporary = target.with_suffix('.csv.partial')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle, (output_dir / 'excluded_augmentations.jsonl').open('w', encoding='utf-8') as rejected:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in originals:
            merged = base(row, 'original')
            writer.writerow(merged)
            stats['original_rows'] += 1
            if merged['training_eligible'] == '1':
                stats['labeled_original_rows'] += 1
                original_counts[merged['BIOFIN 1차 카테고리']] += 1
            else:
                stats['unlabeled_original_rows'] += 1
        for file in files:
            with file.open(encoding='utf-8-sig') as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    stats['augmentation_input_rows'] += 1
                    try:
                        row = json.loads(line)
                        key = row['source_business_key']
                        if key not in by_key or key not in gold:
                            raise ValueError('unmatched_source_business')
                        if row.get('source_csv_sha256') != label_hash:
                            raise ValueError('source_csv_hash_mismatch')
                        label = gold[key]['1차 카테고리']
                        if str(row['label']) != label or str(row.get('primary_category')) != label:
                            raise ValueError('label_mismatch')
                        if str(row.get('subcategory')) != gold[key]['하위 카테고리']:
                            raise ValueError('subcategory_mismatch')
                        if row.get('backend') == 'mock':
                            raise ValueError('mock_data')
                        text = row['text']
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError('empty_text')
                        digest = hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
                        if digest in seen:
                            raise ValueError('duplicate_text')
                        seen.add(digest)
                        merged = base(by_key[key], 'augmentation')
                        merged.update(augmented_text=text.strip(), augmentation_id=digest,
                                      augmentation_source_file=file.name, augmentation_source_line=line_number,
                                      needs_review='1', generation_model=row.get('model', ''))
                        writer.writerow(merged)
                        augmented_counts[label] += 1
                        stats['augmentation_kept_rows'] += 1
                    except (ValueError, KeyError, TypeError) as exc:
                        reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                        stats['excluded_' + reason] += 1
                        rejected.write(json.dumps({'file': file.name, 'line': line_number, 'reason': reason}, ensure_ascii=False) + '\n')
    if sha(original_path) != original_hash or sha(label_path) != label_hash:
        raise RuntimeError('Source changed during export; partial output not published')
    os.replace(temporary, target)
    report = {'csv': str(target.resolve()), 'original_csv': str(original_path.resolve()),
              'original_sha256': original_hash, 'gold_csv': str(label_path.resolve()),
              'gold_sha256': label_hash, 'augmentation_files': len(files),
              'counts': dict(stats), 'original_by_category': dict(sorted(original_counts.items())),
              'augmentation_by_category': dict(sorted(augmented_counts.items())),
              'note': 'All originals preserved. Unlabeled originals excluded from training. Augmentations only join train after original group split; semantic review still required.'}
    (output_dir / 'merge_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-csv', required=True)
    parser.add_argument('--label-csv', required=True)
    parser.add_argument('--augmentation-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.original_csv, args.label_csv, args.augmentation_dir, args.output_dir), ensure_ascii=False, indent=2))
