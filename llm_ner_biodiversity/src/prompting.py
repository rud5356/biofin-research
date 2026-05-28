"""
LLM NER(개체명 인식)을 위한 프롬프트를 생성하는 모듈.

Few-shot 프롬프팅이란?
  LLM에게 작업 예시(예: 입력 → 출력 쌍)를 몇 가지 보여주면
  별도 학습 없이도 같은 형식으로 답변하는 방식입니다.
  예시가 없으면 "zero_shot", 예시가 있으면 "few_shot"이라고 합니다.

이 모듈에서 정의된 규칙들:
- SPECIES: 종명은 이명법(속명+종명) 형식 (예: Rana coreana)
- LOCATION: 실제 지명만 (생태 용어 제외)
- DATE: 구체적인 날짜/기간만
"""

from pathlib import Path


# ─── 기본 지침 (LLM에게 역할과 규칙을 설명) ─────────────────────────────────
# 영어로 작성된 이유: LLM(llama3.1:8b)이 영어 지침에 더 잘 반응하기 때문
BASE_INSTRUCTION = """You are an expert in biodiversity literature analysis.
Extract only the following entity types from the text:
- SPECIES: species names, preferably scientific names
- LOCATION: place names, regions, habitat names
- DATE: observation dates or time periods

Rules:
1. Return valid JSON only.
2. Do not add information that is not present in the text.
3. If an entity is uncertain, omit it.
4. Use this schema: {"entities": [{"type": "...", "text": "..."}]}

SPECIES rules:
- SPECIES must be in genus+species binomial format (e.g., Rana coreana, Mustela sibirica).
- Do NOT extract: taxonomic ranks above species (Caudata, Anura, Amphibia, Mammalia, Cervidae, etc.)
- Do NOT extract: gene names or protein names (e.g., SOX10, PLP1, MAG, SIRT2, MBP, GPX4, COPG1)
- Do NOT extract: chemical compounds or drug names (e.g., Morusin, Benztropine, Rb1, Rg1, Rg3, cisplatin)
- Do NOT extract: cell line names (e.g., HepG2, PANC-1, HeLa, MCF-7, HaCaT, RAW)
- Do NOT extract: single-word common names or vague organism references
- Common names (e.g., "Korean water deer", "Siberian roe deer", "Eurasian otter") must be converted to
  their scientific name if you can confirm it from the text or your knowledge (e.g., Hydropotes inermis,
  Capreolus pygargus, Lutra lutra). If the scientific name cannot be confirmed, omit the entity entirely.
- Abbreviated species names (e.g., D. suweonensis) should only be extracted if the full genus can be confirmed
  from context. If confirmable, restore the full name (e.g., Dryophytes suweonensis).

LOCATION rules:
- Extract only actual named places: country names, city names, province/state names, named regions, protected areas.
- Do NOT extract: general ecological terms (wetland, forest, mountain stream, freshwater ecosystem, etc.)
- Do NOT extract: vague or abstract expressions (multiple experimental systems, various regions, etc.)

DATE rules:
- Extract specific observation dates or study periods (e.g., "April 2023", "2018–2024").
- Do NOT extract: developmental time points (e.g., "gestation day 8", "day 14") or treatment durations.
"""

# ─── Few-shot 예시 (LLM이 참고할 입출력 예시들) ──────────────────────────────
# 다양한 케이스(생물다양성 논문, 의학 논문, 약어 종명 등)를 포함합니다.
FEW_SHOT_EXAMPLES = """
Example 1 (biodiversity paper — extract correctly):
Text: "Rana coreana was observed in wetlands near Suwon in April 2023."
Output: {"entities": [{"type": "SPECIES", "text": "Rana coreana"}, {"type": "LOCATION", "text": "Suwon"}, {"type": "DATE", "text": "April 2023"}]}

Example 2 (species with abbreviated genus — restore full name):
Text: "The Korean salamander (Hynobius leechii) inhabits mountain streams in Gangwon province. H. leechii is endemic to Korea."
Output: {"entities": [{"type": "SPECIES", "text": "Hynobius leechii"}, {"type": "LOCATION", "text": "Gangwon province"}, {"type": "LOCATION", "text": "Korea"}]}

Example 3 (biomedical paper — do NOT extract gene/compound names as SPECIES):
Text: "SOX10-inducible OPC differentiation showed that Morusin enhanced MBP, PLP1, MAG expression in multiple experimental systems."
Output: {"entities": []}

Example 4 (biomedical paper — do NOT extract cell lines or drugs as SPECIES):
Text: "MCF-7 and MDA-MB-231 breast cancer cells were treated with cisplatin and Platycodon grandiflorus extract."
Output: {"entities": [{"type": "SPECIES", "text": "Platycodon grandiflorus"}]}

Example 5 (common name — convert to scientific name):
Text: "The Eurasian otter and Korean water deer were surveyed along rivers in South Korea between 2018 and 2024."
Output: {"entities": [{"type": "SPECIES", "text": "Lutra lutra"}, {"type": "SPECIES", "text": "Hydropotes inermis"}, {"type": "LOCATION", "text": "South Korea"}, {"type": "DATE", "text": "2018-2024"}]}

Example 6 (no relevant entities):
Text: "No significant findings were reported in this region."
Output: {"entities": []}
"""


def build_prompt(text: str, mode: str = "few_shot") -> str:
    """
    NER 프롬프트를 생성합니다.

    Args:
        text: 개체명을 추출할 논문 초록 텍스트
        mode: "few_shot" (예시 포함) 또는 "zero_shot" (예시 없음)

    Returns:
        LLM에 전달할 완성된 프롬프트 문자열
    """
    if mode == "few_shot":
        # 기본 지침 + 예시 + 처리할 텍스트
        return f"{BASE_INSTRUCTION}\n{FEW_SHOT_EXAMPLES}\nText: {text}\nOutput:"
    # zero_shot: 예시 없이 지침과 텍스트만 제공
    return f"{BASE_INSTRUCTION}\nText: {text}\nOutput:"


def save_prompt_template(prompts_dir: Path, mode: str) -> Path:
    """
    프롬프트 템플릿을 파일로 저장합니다.

    실제 텍스트 자리에는 '<TEXT>' 플레이스홀더를 사용합니다.
    저장된 파일은 프롬프트 검토나 디버깅에 사용됩니다.

    Args:
        prompts_dir: 저장할 폴더 경로
        mode: "few_shot" 또는 "zero_shot"

    Returns:
        저장된 파일 경로
    """
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompts_dir / f"ner_prompt_{mode}.txt"
    # <TEXT> 자리에 실제 텍스트가 들어갈 것임을 표시
    prompt_path.write_text(build_prompt("<TEXT>", mode=mode), encoding="utf-8")
    return prompt_path
