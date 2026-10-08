import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import classify_biofin_category_with_ollama as ollama
import classify_biofin_category_with_vllm as vllm


class RevisionTests(unittest.TestCase):
    def test_backends_share_prompt(self):
        self.assertEqual(ollama.SYSTEM_PROMPT, vllm.SYSTEM_PROMPT)
        self.assertEqual(ollama.PROMPT_TEMPLATE, vllm.PROMPT_TEMPLATE)
        for module in (ollama, vllm):
            module.build_prompt({}, '사업목적 본문')  # JSON braces survive format().

    def test_review_response_is_distinct_from_zero_and_failure(self):
        for module in (ollama, vllm):
            with self.subTest(backend=module.__name__):
                result = module.parse_jsonish_response(json.dumps({
                    'label': None, 'classification_status': '검토 필요',
                    'reason': '근거 부족: 검토 필요', 'confidence': 0.99,
                }))
                self.assertEqual(result['label'], '')
                record = {**result, 'prompt_version': module.PROMPT_VERSION}
                self.assertTrue(module.valid_cached_label(record))
                self.assertFalse(module.valid_cached_label({**record, 'classification_status': '실패'}))
                self.assertFalse(module.valid_cached_label({**record, 'prompt_version': 'old'}))
                zero = module.parse_jsonish_response('{"label":0}')
                self.assertEqual(zero['classification_status'], '무관')
                for response in [
                    {'label': 0, 'classification_status': '검토 필요'},
                    {'label': None}, {'classification_status': '검토 필요'},
                    {'label': 7, 'classification_status': '무관'},
                ]:
                    with self.assertRaises(ValueError):
                        module.parse_jsonish_response(json.dumps(response))
                with self.assertRaises(ValueError):
                    module.parse_jsonish_response('7 > 8 > 4 규칙을 검토해야 함')

    def test_review_csv_cache_and_evaluation(self):
        for module in (ollama, vllm):
            with self.subTest(backend=module.__name__), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                args = argparse.Namespace(
                    label_col='prediction', gold_label_col='gold',
                    output_file=out/'result.csv', audit_csv=out/'audit.csv',
                    review_csv=out/'review.csv', output_dir=out, review_threshold=0.7,
                    model='test', input_file=out/'input.csv', max_tokens=100,
                    enable_thinking=False,
                )
                rows = [{'business_key': 'review', 'gold': '7'},
                        {'business_key': 'zero', 'gold': '0'}]
                items = module.collect_items(rows)
                cache = {}
                for key, item in items.items():
                    review = item['row']['business_key'] == 'review'
                    cache[key] = {
                        'key_hash': key, 'label': '' if review else 0,
                        'classification_status': '검토 필요' if review else '무관',
                        'confidence': 0.99, 'prompt_version': module.PROMPT_VERSION,
                    }
                module.save_cache(out/'cache.csv', cache)
                loaded = module.load_cache(out/'cache.csv')
                self.assertTrue(all(module.valid_cached_label(r) for r in loaded.values()))
                module.write_outputs(list(rows[0]), rows, items, loaded, args)
                _, results, _ = module.read_csv(args.output_file)
                self.assertEqual(results[0]['prediction'], '')
                self.assertEqual(results[0]['classification_status'], '검토 필요')
                self.assertEqual(results[1]['prediction'], '0')
                _, review_rows, _ = module.read_csv(args.review_csv)
                self.assertEqual(len(review_rows), 1)
                self.assertEqual(review_rows[0]['classification_status'], '검토 필요')
                metrics = json.loads((out/'evaluation_metrics.json').read_text('utf-8'))
                self.assertEqual(metrics['evaluated_rows'], 1)
                self.assertEqual(metrics['skipped_review_needed'], 1)
                self.assertEqual(metrics['incorrect_rows'], 0)

    def test_legal_basis_boundaries_and_prompt_budget(self):
        for module in (ollama, vllm):
            extract = module.extract_legal_basis
            for heading in ['법적 근거', '나. 사업근거 및 추진경위', '□ 지원근거', '추진근거']:
                self.assertEqual(extract(heading+'\n보전법 제1조\n사업내용\n다른 내용'), '보전법 제1조')
            self.assertEqual(extract('법적 근거\n끝 경계 없음'), '')
            self.assertEqual(extract('법적 근거를 설명하는 문장\n본문\n사업내용'), '')
            for purpose in ['', '사업목적\n'+'목적'*200+'\n']:
                document = purpose+'법적 근거\n보전법 제1조\n사업내용\n제외할 내용'
                args = argparse.Namespace(no_document_text=False, max_document_chars=100)
                with patch.object(module, 'resolve_document_path', return_value=Path('sample.hwpx')), \
                     patch.object(module, 'extract_document', return_value=document):
                    text, status, _, chars = module.load_document_for_prompt({}, args)
                self.assertIn('보전법 제1조', text)
                self.assertNotIn('제외할 내용', text)
                self.assertLessEqual(chars, 100)
                self.assertEqual(status, 'PURPOSE_EXTRACTED' if purpose else 'LEGAL_BASIS_EXTRACTED')


if __name__ == '__main__':
    unittest.main()
