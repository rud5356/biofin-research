import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import classify_biofin_category_with_vllm as vllm


class FieldFilterTests(unittest.TestCase):
    def test_local_main_saves_zero_before_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source = out/'local.csv'
            rows = [
                {'business_key': 'x', '소관명': '강원본청', '분야명': '교육', '지방자치단체코드': '4200000'},
                {'business_key': 'y', '소관명': '강원본청', '분야명': '산업·중소기업및에너지', '지방자치단체코드': '4200000'},
            ]
            vllm.write_csv(source, list(rows[0]), rows)
            def check(items, cache, args):
                self.assertEqual(args.fiscal_type, 'local')
                _, saved, _ = vllm.read_csv(args.output_file)
                self.assertEqual([r[args.label_col] for r in saved], ['0', ''])
                self.assertEqual([i['row']['business_key'] for i in items.values()], ['y'])
                self.assertEqual(cache, {})
            with patch('sys.argv', ['classifier', '--input-file', str(source), '--output-dir', str(out)]), \
                    patch.object(vllm, 'classify_items', side_effect=check) as call:
                self.assertEqual(vllm.main(), 0)
            call.assert_called_once()

    def test_local_missing_identifiers_rejected(self):
        with self.assertRaises(ValueError):
            vllm.detect_fiscal_type(['지방자치단체코드'], [{'지방자치단체코드': ''}])

    def test_main_saves_zeros_before_call_and_keeps_them_out_of_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source = out/'input.csv'
            rows = [
                {'business_key': 'excluded', '소관명': '대상외', '분야명': '환경'},
                {'business_key': 'included', '소관명': vllm.TARGET_MINISTRIES[0], '분야명': '환경'},
            ]
            vllm.write_csv(source, list(rows[0]), rows)
            output = out/'input_llm_classified.csv'

            def check_before_call(items, cache, args):
                _, saved, _ = vllm.read_csv(output)
                self.assertEqual([r[args.label_col] for r in saved], ['0', ''])
                self.assertEqual(len(items), 1)
                self.assertEqual(next(iter(items.values()))['row']['business_key'], 'included')
                self.assertEqual(cache, {})
                self.assertEqual(vllm.load_cache(args.cache_csv), {})

            with patch('sys.argv', ['classifier', '--input-file', str(source), '--output-dir', str(out)]), \
                    patch.object(vllm, 'classify_items', side_effect=check_before_call) as call, \
                    patch.object(vllm, 'print_document_match_summary') as scan:
                self.assertEqual(vllm.main(), 0)
            call.assert_called_once()
            scan.assert_not_called()
            _, saved, _ = vllm.read_csv(output)
            self.assertEqual(len(saved), 2)
            self.assertEqual(saved[0][vllm.DEFAULT_LABEL_COLUMN], '0')

    def test_filter_overrides_cache_preserves_rows_and_skips_llm(self):
        rows = [
            {"business_key": "excluded", "소관명": "outside", "분야명": "환경"},
            {"business_key": "included", "소관명": "field0", "분야명": "환경"},
            {"business_key": "excluded", "소관명": "outside", "분야명": "환경"},
        ]
        items = vllm.collect_items(rows)
        excluded = vllm.hash_key(vllm.build_key(rows[0]))
        included = vllm.hash_key(vllm.build_key(rows[1]))
        cache = {
            excluded: {"label": 7, "prompt_version": vllm.PROMPT_VERSION},
            included: {"label": 0, "model": "ministry-filter", "prompt_version": vllm.PROMPT_VERSION},
        }
        response = dict(label=9, classification_status="관련", review_needed=False,
                        review_reason="", 검토필요="", confidence=0.9, reason="test",
                        evidence="[]", document_status="DISABLED", document_path="",
                        document_chars=0, raw_response="")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = argparse.Namespace(
                target_ministries=tuple(f"field{i}" for i in range(11)), limit_keys=1,
                overwrite=False, workers=1, save_every=0, cache_csv=out/'cache.csv',
                model="test", label_col="prediction", gold_label_col="gold",
                output_file=out/'output.csv', output_dir=out, audit_csv=out/'audit.csv',
                review_csv=out/'review.csv', review_threshold=0.7, max_tokens=100,
                enable_thinking=False, input_file=out/'input.csv',
            )
            with patch.object(vllm, 'classify', return_value=response) as call:
                vllm.classify_items(items, cache, args)
            call.assert_called_once_with(rows[1], args)
            self.assertEqual(cache[excluded]['label'], 0)
            self.assertEqual(cache[included]['label'], 9)
            self.assertTrue(vllm.valid_cached_label(cache[excluded]))
            vllm.write_outputs(list(rows[0]), rows, items, cache, args)
            _, output, _ = vllm.read_csv(args.output_file)
            self.assertEqual([r['business_key'] for r in output], ['excluded', 'included', 'excluded'])
            self.assertEqual([r['prediction'] for r in output], ['0', '9', '0'])
            _, review, _ = vllm.read_csv(args.review_csv)
            self.assertEqual(review, [])

    def test_all_excluded_never_calls_model(self):
        items = vllm.collect_items([{'business_key': 'x', '소관명': 'outside'}])
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(target_ministries=tuple(f'f{i}' for i in range(11)),
                                      limit_keys=1, overwrite=True, workers=1,
                                      cache_csv=Path(tmp)/'cache.csv')
            with patch.object(vllm, 'classify') as call:
                cache = {}
                vllm.classify_items(items, cache, args)
            call.assert_not_called()
            self.assertEqual(next(iter(cache.values()))['label'], 0)


class OllamaFieldFilterTests(FieldFilterTests):
    def setUp(self):
        import classify_biofin_category_with_ollama as ollama
        replacement = patch(__name__ + '.vllm', ollama)
        replacement.start()
        self.addCleanup(replacement.stop)


if __name__ == '__main__':
    unittest.main()
