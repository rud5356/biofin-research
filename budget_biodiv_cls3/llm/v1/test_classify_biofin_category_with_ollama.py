import argparse
import json
import unittest
from unittest.mock import MagicMock, patch

import classify_biofin_category_with_ollama as classifier


class ClassifierRegressionTests(unittest.TestCase):
    def test_actual_gold_header(self):
        self.assertEqual(
            classifier.resolve_gold_label_column(['No.', '1차 카테고리'], None),
            '1차 카테고리',
        )
        with self.assertRaises(ValueError):
            classifier.resolve_gold_label_column(['1차 카테고리'], '1차')
        with self.assertRaises(ValueError):
            classifier.resolve_gold_label_column(['1차', '1차 카테고리'], None)

    def test_zero_and_invalid_labels(self):
        self.assertEqual(classifier.parse_valid_label(0), 0)
        for label in range(10):
            self.assertEqual(
                classifier.parse_jsonish_response(json.dumps({'label': label}))['label'],
                label,
            )
        for text in ['', '[]', 'null', '{"label": true}', '{"label": 2.9}', '{"label": 10}']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                classifier.parse_jsonish_response(text)

    def args(self):
        return argparse.Namespace(
            model='test', num_ctx=16384, no_json_format=False,
            ollama_url='http://localhost:11434', timeout=1,
            retries=0, retry_delay=0, delay=0,
        )

    def test_http_response_validation_and_preservation(self):
        for body in [
            {'thinking': 'internal text', 'response': ''},
            {'message': {'content': '{"label": 2}'}},
            {'error': 'model unavailable'}, [],
        ]:
            raw = json.dumps(body)
            response = MagicMock()
            response.__enter__.return_value.read.return_value = raw.encode()
            with patch.object(classifier.request, 'urlopen', return_value=response):
                with self.assertRaises(classifier.OllamaResponseError) as caught:
                    classifier.call_ollama('test', self.args())
                self.assertEqual(caught.exception.raw_response, raw)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"response":"{\\"label\\":0}"}'
        with patch.object(classifier.request, 'urlopen', return_value=response):
            self.assertEqual(classifier.call_ollama('test', self.args()), '{"label":0}')

    def test_classification_failure_retains_response(self):
        with patch.object(classifier, 'load_document_for_prompt', return_value=('', 'NOT_FOUND', '', 0)), \
             patch.object(classifier, 'build_prompt', return_value='test'), \
             patch.object(classifier, 'call_ollama', return_value='cannot classify'):
            result = classifier.classify({}, self.args())
        self.assertEqual(result['label'], '')
        self.assertEqual(result['raw_response'], 'cannot classify')
        raw = '{"response":"","thinking":"text"}'
        with patch.object(classifier, 'load_document_for_prompt', return_value=('', 'NOT_FOUND', '', 0)), \
             patch.object(classifier, 'build_prompt', return_value='test'), \
             patch.object(classifier, 'call_ollama', side_effect=classifier.OllamaResponseError('empty', raw)):
            result = classifier.classify({}, self.args())
        self.assertEqual(result['raw_response'], raw)


if __name__ == '__main__':
    unittest.main()
