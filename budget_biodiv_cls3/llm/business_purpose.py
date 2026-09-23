"""사업목적 또는 사업목적·내용 결합 항목을 추출한다."""
import re
import unicodedata

PREFIX = r'\s*(?:[□■○◦●ㅇᄋ▪▫◆◇▣▶▷※*\-–①-⑳]\s*|\(?\d+[.)]\s*|\([가나다라마바사아자차카타파하]\)\s*|[가나다라마바사아자차카타파하][.)]\s*)*'
START = re.compile(r'^' + PREFIX + r'(?:사\s*업\s*(?:의\s*)?)?목\s*적(?:\s*(?:[·ㆍᆞ‧・/]|및)\s*(?:사\s*업\s*)?내\s*용)?(?:\s*[:：]\s*(.*)|\s*)$')
END = re.compile(
    r'^' + PREFIX + r'(?:'
    r'(?:사\s*업\s*)?(?:추\s*진\s*)?근\s*거|'
    r'사\s*업\s*근\s*거\s*(?:및|[·ㆍ‧])\s*추\s*진\s*경\s*위|'
    r'(?:사\s*업\s*)?(?:내\s*용|개\s*요|기\s*간|규\s*모|대\s*상|예\s*산|효\s*과)|'
    r'추\s*진\s*(?:경\s*위|체\s*계|실\s*적|계\s*획|현\s*황)|'
    r'지\s*원\s*(?:근\s*거|대\s*상|조\s*건)|'
    r'법\s*적\s*근\s*거|성\s*과\s*목\s*표|기\s*대\s*효\s*과|주\s*요\s*내\s*용'
    r')(?:\s*[:：].*|\s*)$'
)


def extract_business_purpose(text: str) -> str:
    """제목 경계가 확인된 첫 목적 항목을 반환한다. 경계 불명확 시 공란."""
    lines = unicodedata.normalize('NFKC', text).splitlines()
    for start, line in enumerate(lines):
        match = START.fullmatch(line.strip())
        if not match:
            continue
        content = [match.group(1)] if match.group(1) else []
        for following in lines[start + 1:]:
            if END.fullmatch(following.strip()) or START.fullmatch(following.strip()):
                result = '\n'.join(content).strip()
                if result:
                    return result
                break
            content.append(following)
        # 종료 경계가 없으면 나머지 문서를 목적이라고 간주하지 않는다.
    return ''
