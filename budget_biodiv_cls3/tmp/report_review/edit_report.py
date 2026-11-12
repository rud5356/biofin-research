from pathlib import Path
import zipfile, re, hashlib, json, copy
from lxml import etree as E
from xml.sax.saxutils import escape

src = Path('C:/Users/lenovo/Desktop/BIOFIN_최종보고서_초안_2026.11.08_코아수정1_부록Ⅱ개정.hwpx')
outdir = Path('C:/repos/biofin-research/budget_biodiv_cls3/output/report_review')
outdir.mkdir(parents=True, exist_ok=True)
dst = outdir / 'BIOFIN_최종보고서_초안_2026.11.08_제3장수정.hwpx'
changes = [
('AIP에서는 중앙정부 및 지방정부의 예산사업 자료를 입력받아', 'AIP는 중앙·지방정부의 예산사업 기본정보(CSV)와 사업설명자료를 입력받는다. 사용자가 자료 구분과 회계연도를 선택하면 실행 도우미가 해당 자료의 저장 위치에서 입력 CSV와 사업설명자료 폴더를 찾는다.'),
('예산 데이터의 입력과 처리 과정은', '예산 데이터의 입력과 처리는 Python 기반 메뉴형 실행 도구인 biofin_menu.py로 수행한다. 서버 주소와 경로, 모델 ID 등 환경별 설정은 config.json에서, 추론과 귀속 예산 계산 등의 실행 옵션은 defaults.json에서 관리한다. 실행 도우미는 두 설정 파일의 값을 읽어 실행 명령어에 반영한다.'),
('AIP 실행 과정에서는 먼저', '사용자는 작업과 모델을 선택한 뒤 자료 구분과 회계연도를 선택하며, 지방재정 자료는 지역도 선택한다. 실행 도우미는 해당 폴더의 CSV가 하나이면 자동으로 선택하고, 여러 개이면 목록을 제시하여 사용자가 선택하게 한다. 사업설명자료 폴더는 같은 자료 경로에서 자동으로 지정하며, 예산사업 정보와 설명자료는 사업 식별정보를 기준으로 연계한다.'),
('모델 실행 시에는 입력 데이터', '모델 서버 주소와 모델 ID는 config.json에서 읽고, 사업목적 최대 문자 수, 응답 최대 토큰 수, 동시 요청 수, 타임아웃 및 재시도 횟수는 defaults.json의 값을 적용한다. 실행 중에는 추론 옵션을 묻지 않으며, 조건을 변경할 때는 설정 파일을 수정한 후 실행 도우미를 다시 시작한다.'),
('입력자료와 실행 옵션 설정이 완료되면', '자료 선택이 끝나면 실행 도우미는 자동 지정된 경로와 설정 파일의 값을 반영한 Docker 실행 명령어를 표시한다. 사용자가 서버 주소, 모델 ID, 입력·결과 경로와 실행 옵션을 확인하고 y를 입력하면 분류를 시작하며, 그 외의 입력은 실행을 취소한다.'),
('분류 작업이 완료되면', '결과 저장 폴더는 모델별 outputs 경로 아래에 날짜·회계연도·자료 구분·번호를 조합하여 자동 생성한다. 같은 날 같은 자료를 다시 실행하면 번호가 증가하므로 이전 결과를 덮어쓰지 않는다. 분류가 끝나면 귀속률 적용 계산을 이어서 수행하고, 최종 결과를 *_bar.csv로 저장한다.'),
('따라서 중앙정부 예산은 LLM', '따라서 중앙정부 예산은 LLM을 주 분류기로 사용하되, LLM이 비해당으로 판정한 사업 가운데 누락 가능성이 있는 사업을 보완 검토하는 방식이 적합하다. Transformer는 LLM 구동이 어려운 하드웨어 환경에서 제한적으로 대체 활용할 수 있다. 중앙정부와 지방정부에는 동일한 분류 프롬프트를 사용하였다. 지방정부는 사업 설명이 없거나 짧은 경우가 많으므로, 판단 근거가 부족한 사업에 대한 담당자 검토를 병행할 필요가 있다.'),
('【검토 메모】 울산 2024년 세부사업 수가', None),
]
before_hash = hashlib.sha256(src.read_bytes()).hexdigest()
with zipfile.ZipFile(src) as z:
    raw = z.read('Contents/section0.xml').decode('utf-8')
    root = E.fromstring(raw.encode())
    ns = {'hp':'http://www.hancom.co.kr/hwpml/2011/paragraph'}
    paras = root.findall('.//hp:p',ns)
    edits = []
    for prefix, new in changes:
        found = [p for p in paras if ''.join(p.xpath('./hp:run/hp:t/text()',namespaces=ns)).startswith(prefix)]
        assert len(found) == 1, prefix
        p = found[0]
        old = ''.join(p.xpath('./hp:run/hp:t/text()',namespaces=ns))
        assert len(p.findall('./hp:run/hp:t',ns)) == 1
        token = '<hp:t>' + escape(old) + '</hp:t>'
        assert raw.count(token) == 1
        pos = raw.index(token)
        start = raw.rfind('<hp:p ',0,pos)
        end = raw.index('</hp:p>',pos)+len('</hp:p>')
        block = raw[start:end]
        assert '<hp:p ' not in block[len('<hp:p '):]
        if new is None:
            replacement = ''
        else:
            replacement = block.replace(token,'<hp:t>'+escape(new)+'</hp:t>')
            # Drop stale line-layout cache only for changed paragraphs.
            replacement = re.sub(r'<hp:linesegarray>.*?</hp:linesegarray>','',replacement,flags=re.S)
        raw = raw[:start]+replacement+raw[end:]
        edits.append({'before':old,'after':new})
    E.fromstring(raw.encode('utf-8'))
    with zipfile.ZipFile(dst,'w') as w:
        for info in z.infolist():
            data = raw.encode('utf-8') if info.filename == 'Contents/section0.xml' else z.read(info.filename)
            w.writestr(copy.copy(info),data)
    with zipfile.ZipFile(dst) as w:
        assert w.testzip() is None
        assert z.namelist() == w.namelist()
        for name in z.namelist():
            if name != 'Contents/section0.xml': assert z.read(name) == w.read(name),name
        newroot = E.fromstring(w.read('Contents/section0.xml'))
        oldtexts = [''.join(p.xpath('./hp:run/hp:t/text()',namespaces=ns)) for p in paras]
        expected = oldtexts[:]
        for e in edits:
            idx = expected.index(e['before'])
            if e['after'] is None: expected.pop(idx)
            else: expected[idx] = e['after']
        actual = [''.join(p.xpath('./hp:run/hp:t/text()',namespaces=ns)) for p in newroot.findall('.//hp:p',ns)]
        assert actual == expected, 'Unexpected paragraph differences'
assert hashlib.sha256(src.read_bytes()).hexdigest() == before_hash
log = ['# 제3장 수정 내역','', '제3장 7개 문단 수정 및 울산 사업 수 검토 메모 1개 삭제. 그 외 문단과 문서 구성 파일은 원본과 동일함.','']
for i,e in enumerate(edits,1):
    log.extend([f'## {i}', '', '**수정 전**', e['before'], '', '**수정 후**', e['after'] or '(메모 삭제)', ''])
(outdir/'제3장_수정내역.md').write_text('\n'.join(log),encoding='utf-8')
print(json.dumps({'output':str(dst),'changed_paragraphs':7,'deleted_memos':1,'validation':'ZIP/XML valid; all other paragraphs and ZIP members unchanged; original unchanged'},ensure_ascii=False))
