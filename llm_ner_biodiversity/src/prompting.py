from pathlib import Path


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
    if mode == "few_shot":
        return f"{BASE_INSTRUCTION}\n{FEW_SHOT_EXAMPLES}\nText: {text}\nOutput:"
    return f"{BASE_INSTRUCTION}\nText: {text}\nOutput:"


def save_prompt_template(prompts_dir: Path, mode: str) -> Path:
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompts_dir / f"ner_prompt_{mode}.txt"
    prompt_path.write_text(build_prompt("<TEXT>", mode=mode), encoding="utf-8")
    return prompt_path
