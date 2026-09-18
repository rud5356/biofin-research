import argparse
import json
import unittest
from unittest.mock import MagicMock, patch

import classify_biofin_category_with_ollama as classifier


class ClassifierRegressionTests(unittest.TestCase):
    def test_purpose_section_boundaries(self):
        extract = classifier.extract_business_purpose
        self.assertEqual(
            extract('사업개요\n가. 사업목적\n○ 생물다양성 보전\n- 서식지 보호\n나. 사업내용\n예산 정보'),
            '○ 생물다양성 보전\n- 서식지 보호',
        )
        self.assertEqual(extract('□ 사 업 목 적 : 자연 보전\n□ 추진근거\n법률'), '자연 보전')
        self.assertEqual(extract('사업목적을 설명하는 문장\n일반 본문\n사업내용'), '')
        self.assertEqual(extract('사업목적\n내용\n경계가 없는 나머지 문서'), '')
        self.assertEqual(extract('사업목적\n사업내용\n다른 내용'), '')

    def test_only_purpose_reaches_prompt(self):
        args = argparse.Namespace(no_document_text=False, max_document_chars=100)
        from pathlib import Path
        with patch.object(classifier, 'resolve_document_path', return_value=Path('sample.txt')), \
             patch.object(classifier, 'extract_document', return_value='사업목적\n자연 보전\n사업내용\n제외할 내용'):
            text, status, _, chars = classifier.load_document_for_prompt({}, args)
        self.assertEqual((text, status, chars), ('자연 보전', 'PURPOSE_EXTRACTED', 5))
        with patch.object(classifier, 'resolve_document_path', return_value=Path('sample.txt')), \
             patch.object(classifier, 'extract_document', return_value='일반 본문만 존재'):
            text, status, _, chars = classifier.load_document_for_prompt({}, args)
        self.assertEqual((status, chars), ('PURPOSE_NOT_FOUND', 0))
        self.assertNotIn('일반 본문만 존재', text)

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
        with patch.object(classifier.request, 'urlopen', return_value=response) as send:
            self.assertEqual(classifier.call_ollama('test', self.args()), '{"label":0}')
            payload = json.loads(send.call_args.args[0].data)
            self.assertIs(payload['think'], False)
            self.assertIs(payload['stream'], False)
            self.assertEqual(payload['format'], 'json')

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
