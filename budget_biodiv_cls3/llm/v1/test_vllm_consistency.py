import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import classify_biofin_category_with_vllm as m

def response(label, reason, review=False):
    return json.dumps(dict(label=label, reason=reason, confidence=.9, evidence=[], review_needed=review, review_reason='확인 필요' if review else ''), ensure_ascii=False)

class ConsistencyTest(unittest.TestCase):
    def check(self, label, reason, name, reply):
        with patch.object(m, 'call_vllm', return_value=reply) as call:
            result=m.enforce_result_consistency(m.parse_jsonish_response(response(label,reason)), {'세부사업명':name}, 'prompt', SimpleNamespace())
            return result,call.call_count
    def test_related_rnd(self):
        r,n=self.check(9,'가축 방역 관련 기술개발','백신 개발(R&D)',response(2,'관련 R&D이므로 2를 적용한다.'))
        self.assertEqual((r['label'],r['model_original_label'],n),(2,9,1))
    def test_unrelated_rnd_stays_zero(self):
        r,n=self.check(0,'일반 산업 기술','양자 개발(R&D)','')
        self.assertEqual((r['label'],n),(0,0))
    def test_no_numbers_from_comparison(self):
        r,n=self.check(4,'2가 아닌 4를 적용한다.','일반 전환', '')
        self.assertEqual((r['label'],n),(4,0))
    def test_label_reason_conflict(self):
        r,n=self.check(4,'4가 아닌 2를 적용한다.','선박 개발(R&D)',response(2,'관련 R&D이므로 2를 적용한다.'))
        self.assertEqual((r['label'],n),(2,1))
    def test_unresolved_keeps_numeric_review(self):
        r,n=self.check(6,'오염관리 기술','정화(R&D)',response(6,'6을 적용한다.'))
        self.assertEqual(r['label'],2);self.assertTrue(r['review_needed']);self.assertEqual(n,1)
    def test_existing_review_preserved(self):
        with patch.object(m,'call_vllm',return_value=response(2,'2를 적용한다.')):
            r=m.enforce_result_consistency(m.parse_jsonish_response(response(9,'관련 연구',True)),{'세부사업명':'연구(R&D)'},'p',SimpleNamespace())
        self.assertTrue(r['review_needed'])
    def test_transport_failure_preserves_first(self):
        with patch.object(m,'call_vllm',side_effect=m.VLLMResponseError('length','raw-http')):
            r=m.enforce_result_consistency(m.parse_jsonish_response(response(9,'관련 연구')),{'세부사업명':'연구(R&D)'},'p',SimpleNamespace())
        self.assertEqual(r['label'],2);self.assertEqual(r['consistency_response'],'raw-http');self.assertIn('관련 연구',r['model_original_response'])

if __name__=='__main__':unittest.main()
