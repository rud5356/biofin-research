import csv
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / 'transformer/v1/outputs/20260909_2023data'
L = ROOT / 'llm/v1/outputs/20260910_2023data'
def read(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
t = read(T / 'test_predictions.csv')
l = read(L / '260812_2023data_llm_classified.csv')
lookup = {r['business_key']: r for r in l}
assert len(lookup) == len(l)
rows = []
for r in t:
    if r['file_path'].startswith('metadata://'):
        s = l[int(r['file_path'].split('#row=')[1])-2]
        assert s['세부사업명'] == r['activity_name']
    else:
        candidates = [s for s in l if s['사업설명자료_파일명'] == r['file_path'].split('/')[-1]]
        assert len(candidates) == 1, (r['activity_name'],len(candidates))
        s = candidates[0]
    key = s['business_key']
    assert r['true_label'] == s['BIOFIN 1차 카테고리']
    assert r['ministry'] == s['소관명']
    rows.append(dict(key=key, ministry=r['ministry'], gold=r['true_label'], transformer=r['pred_label'], llm=s['LLM BIOFIN 1차 카테고리']))
all_l = [dict(gold=r['BIOFIN 1차 카테고리'], llm=r['LLM BIOFIN 1차 카테고리'], ministry=r['소관명']) for r in l]
names = ['비해당', '유전자원 접근 및 이익 공유(ABS)', '인식 제고 및 연구', '생물안전성', '녹색경제와 생물다양성', '생물다양성 기획 및 재정', '오염 관리', '보호지역 및 기타 보전조치', '복원', '지속 가능한 이용과 생물다양성']
descs = ['생물다양성 관련 포함 조건에 해당하지 않는 사업. 일반 산업·유통 지원, 상수 공급 인프라 등.', '유전자원 탐사·정보관리, 접근·이용 계약, 이익 공유 및 나고야의정서 이행.', '생물다양성 교육·홍보·전시, 과학연구·데이터·통계 및 지식·정보 공유.', '침입외래종 유입 차단·검역, GMO/LMO 안전관리 및 카르타헤나의정서 이행.', '녹색 공급망·순환경제, 지속가능 에너지·교통, 생태관광 및 녹색 기반시설.', '관련 법률·계획·조정, 재원·기금, 전략환경평가, 공간계획 및 국제협력.', '토양·수질 등 오염 저감, 하수·폐기물 처리, 해양오염 방제, 오염 측정·감시.', '보호지역 지정·관리, 보호구역 밖 보전, 야생종 보호 및 밀렵·불법거래 방지.', '훼손 생태계·서식처 회복, 종 재도입, 생태하천·갯벌·산림 복원 및 사후관리.', '농업·양식·어업·임업·담수·해양 등 생물자원의 지속가능한 이용·관리.']
def count(rs, model):
    return sum(r['gold'] == r[model] for r in rs)
def cell(rs, model):
    return f'{count(rs, model):,}/{len(rs):,} ({count(rs, model)/len(rs)*100:.2f}%)' if rs else '—'
def table(headers, data):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] + ['| ' + ' | '.join(map(str,r)) + ' |' for r in data])
out = ['# 2023년 예산사업 BIOFIN 1차 분류 결과 비교', '', '## 1. 분류 카테고리 설명', '', '아래 설명은 이번 실행에 사용된 저장소의 LLM v1 분류 프롬프트를 요약한 것이다. 국제 BIOFIN 기준 원문을 별도로 검증한 표는 아니며, 세부 포함·제외 및 예외 규칙은 해당 코드에 따른다.', '', table(['코드','카테고리','설명'], [(i,names[i],descs[i]) for i in range(10)]), '', '## 2. 평가 기준', '', '- 정답: 원본 데이터의 BIOFIN 1차 카테고리(0~9). 정확도는 정답과 예측 카테고리가 일치한 사업 수 ÷ 평가 사업 수로 계산한다.', '- 주 비교 대상: Transformer 테스트 397건. 사업 식별자(business_key)로 LLM 결과와 397건 모두 일대일 연결하고, 정답 및 부처 일치를 확인했다.', '- 카테고리별 정확도는 실제 정답이 해당 카테고리인 사업 중 정분류 비율(재현율)이다. 예측 카테고리 기준의 정밀도와 구분한다.', '- 부처별 정확도는 해당 부처의 테스트 사업 중 정분류 비율이다. 예산액 가중치 없이 사업 행 단위로 계산한다.', '- LLM 미분류(빈 예측)는 주 비교에서 오답으로 포함한다. 유효 예측만을 분모로 한 수치는 별도로 표시한다.', '', '## 3. 전체 정확도', '', table(['평가 범위','Transformer','LLM'], [('동일 테스트 397건',cell(rows,'transformer'),cell(rows,'llm')), ('LLM 전체 입력 3,972건','해당 전체 예측 파일 없음',cell(all_l,'llm')), ('LLM 전체 중 유효 예측만','—',cell([r for r in all_l if r['llm']!=''],'llm'))]), '', f"동일 테스트에서 LLM 미분류는 {sum(r['llm']=='' for r in rows)}건이다. 전체 입력에서 LLM 미분류는 22건이며, 유효 예측은 3,950건(99.45%)이다.", '', '### 기존 LLM 평가 수치와의 차이', '', '저장된 evaluation_metrics.json은 2,740건 중 341건 정답(12.45%)으로 기록되어 있다. 그러나 원본 CSV에는 0번 예측이 1,210건 있으며, 기존 지표는 이 1,210건과 실제 미분류 22건을 합친 1,232건을 평가에서 제외했다. 본 보고서는 0번을 유효한 카테고리로 포함하여 CSV에서 재계산한 값을 사용한다. 기존 12.45%를 최종 정확도로 인용하면 안 된다.', '', '## 4. 카테고리별 정확도 — 동일 테스트 397건', '', table(['코드','카테고리','사업 수','Transformer 정답/전체 (정확도)','LLM 정답/전체 (정확도)'], [(i,names[i],len(sub),cell(sub,'transformer'),cell(sub,'llm')) for i in range(10) for sub in [[r for r in rows if r['gold']==str(i)]]]), '', '## 5. 부처별 정확도 — 동일 테스트 397건', '', table(['부처','사업 수','Transformer 정답/전체 (정확도)','LLM 정답/전체 (정확도)'], [(m,len(sub),cell(sub,'transformer'),cell(sub,'llm')) for m in sorted({r['ministry'] for r in rows}) for sub in [[r for r in rows if r['ministry']==m]]]), '', '## 6. 해석 시 유의사항', '', f"테스트 397건 중 비해당(0번)은 344건(86.65%)이다. 모든 사업을 비해당으로 예측하는 단순 기준의 정확도는 86.65%이므로, 전체 정확도만으로 분류 성능을 판단하기 어렵다. 생물다양성 해당 사업(1~9번, 53건)의 정확도는 Transformer {cell([r for r in rows if r['gold']!='0'],'transformer')}, LLM {cell([r for r in rows if r['gold']!='0'],'llm')}이다.", '', '카테고리 1·3은 각각 1건, 8은 2건, 7은 3건으로 표본이 작다. 카테고리·부처별 비율은 반드시 사업 수와 함께 해석해야 한다. LLM 전체 결과와 Transformer 테스트 결과는 평가 범위가 다르므로, 모델 간 비교에는 동일 테스트 표를 사용한다.', '', '## 7. 원본 및 재현', '', f'- Transformer: `{T}`', f'- LLM: `{L}`', '- 분류 설명: `llm/v1/classify_biofin_category_with_ollama.py`의 카테고리별 기준.', '- 재계산: `python budget_biodiv_cls3/reports/summarize_202609_models.py`', '- 대조한 사업별 결과: `20260911_common_test_comparison.csv`', '']
out = [s.replace('사업 식별자(business_key)로 LLM 결과와 397건 모두 일대일 연결하고, 정답 및 부처 일치를 확인했다.', '문서 파일명 및 원본 CSV 행 번호로 LLM 결과와 397건 모두 연결하고, 사업 식별자(business_key)·정답·부처를 대조했다.') for s in out]
out.extend(['## 부록. LLM 전체 입력 기준 세부 정확도', '', 'LLM 전체 3,972건 기준이며 빈 예측 22건을 오답으로 포함한다. 동일 테스트 비교와 평가 범위가 다르다.', '', table(['코드', '카테고리', 'LLM 정답/전체 (정확도)'], [(i, names[i], cell([r for r in all_l if r['gold']==str(i)], 'llm')) for i in range(10)]), '', table(['부처', 'LLM 정답/전체 (정확도)'], [(m, cell([r for r in all_l if r['ministry']==m], 'llm')) for m in sorted({r['ministry'] for r in all_l})]), ''])
assert len({r['key'] for r in rows}) == 397
assert count([r for r in all_l if r['llm'] not in ('', '0')], 'llm') == 341
dest = ROOT / 'reports/20260911_BIOFIN_분류결과_비교보고.md'
dest.write_text('\n'.join(out),encoding='utf-8')
with (dest.parent/'20260911_common_test_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(dest)
print('common:',cell(rows,'transformer'),cell(rows,'llm'))
print('LLM all:',cell(all_l,'llm'))
print('LLM valid:',cell([r for r in all_l if r['llm']!=''],'llm'))
assert count(rows,'transformer') / len(rows) == 0.6070528967254408
