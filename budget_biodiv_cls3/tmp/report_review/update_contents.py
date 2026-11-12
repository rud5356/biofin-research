from pathlib import Path
import zipfile,copy,re,json
from lxml import etree as E
base=Path('C:/repos/biofin-research/budget_biodiv_cls3')
src=base/'output/report_review/BIOFIN_최종보고서_초안_2026.11.08_제3장수정3.hwpx'
dst=base/'output/report_review/BIOFIN_최종보고서_초안_2026.11.08_제3장수정4_차례정리.hwpx'
pages=json.loads((base/'tmp/report_review/pagination.json').read_text(encoding='utf-8'))
nums={120:43,121:45,122:46,123:46,124:48,125:52,126:56,127:56,128:57,129:58,130:59,
131:64,132:64,133:64,134:66,135:67,136:68,137:69,138:69,139:69,140:70,141:71,
142:74,143:76,144:76,145:77,146:78,147:79,148:81,149:83,150:85,151:85,152:86,153:92,154:97,155:100,156:101,157:103}
ns={'h':'http://www.hancom.co.kr/hwpml/2011/paragraph'}
log=[]
with zipfile.ZipFile(src) as z:
 raw=z.read('Contents/section0.xml').decode('utf-8');root=E.fromstring(raw.encode());ps=root.findall('.//h:p',ns)
 for idx in list(range(168,187))+list(range(192,212)):
  t=ps[idx].find('./h:run/h:t',ns);label=t.text
  hits=[i-11 for i,txt in enumerate(pages) if i>=54 and label in txt.splitlines()]
  assert len(hits)==1,(label,hits)
  nums[idx]=hits[0]
 for idx,new in sorted(nums.items(),reverse=True):
  t=ps[idx].find('./h:run/h:t',ns);tab=t.find('h:tab',ns)
  assert tab is not None and (tab.tail or '').isdigit(),idx
  old=tab.tail
  starts=[m.start() for m in re.finditer(r'<hp:p(?=\s)',raw)]
  start=starts[idx];end=raw.index('</hp:p>',start)+len('</hp:p>');block=raw[start:end]
  updated,n=re.subn(r'(<hp:tab\b[^>]*/>)'+old+r'(?=</hp:t>)',lambda m:m[1]+str(new),block)
  assert n==1,idx
  raw=raw[:start]+updated+raw[end:]
  log.append((t.text,old,str(new)))
 replacements={
 '제2절 AI 모델 BIOFIN 기반 예산 지출 분류의 절차':'제2절 AI 모델을 활용한 BIOFIN 기반 예산 지출 분류 절차',
 '개발 및 시범평가':'개발 및 시범 평가',
 'Qwen3.8 27B(W4A16 양자화) 모델':'Qwen 계열 모델',
 'LLM 기반 모델(Qwen3.8 27B)':'Qwen 계열 LLM 모델',
 }
 for old,new in replacements.items():
  assert old in raw,old
  raw=raw.replace(old,new)
 E.fromstring(raw.encode())
 with zipfile.ZipFile(dst,'w') as w:
  for info in z.infolist():w.writestr(copy.copy(info),raw.encode() if info.filename=='Contents/section0.xml' else z.read(info.filename))
 with zipfile.ZipFile(dst) as w:
  assert w.testzip() is None
  rr=E.fromstring(w.read('Contents/section0.xml'));pp=rr.findall('.//h:p',ns)
  assert len(pp)==len(ps)
  for idx,new in nums.items():assert pp[idx].find('./h:run/h:t/h:tab',ns).tail==str(new)
  assert 'Qwen3.8' not in raw and '개발 및 시범평가' not in raw
  for name in z.namelist():
   if name!='Contents/section0.xml':assert z.read(name)==w.read(name)
(dst.parent/'차례_쪽번호_수정내역.txt').write_text('\n'.join(f'{t}: {a} → {b}' for t,a,b in reversed(log)),encoding='utf-8')
print(dst);print('Contents entries checked:',len(nums))
