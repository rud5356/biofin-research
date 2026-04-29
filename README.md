# biofin-research

국가생물다양성 재정(BIOFIN) 연구를 위한 데이터 분석 및 AI 파이프라인 모음입니다.  
열린재정 예산 데이터에서 생물다양성 관련 사업을 식별하고, 문서를 분류·분석합니다.

---

## 서브 프로젝트

| 폴더 | 설명 |
|------|------|
| `budget_biodiv_cls` | 재정자료 생물다양성 관련 여부 이진 분류 (Ollama LLM + KoBERT) |
| `budget_field_cls` | 재정자료 분야별 다중 분류 — HWP/PDF 문서를 16개 예산 분야로 분류 (KoBERT, Docker 지원) |
| `biodiversity_rag` | 생물다양성 논문 초록 기반 RAG 질의응답 시스템 |
| `budget_matcher` | 예산 CSV와 열린재정 파일 폴더 매칭 |
| `llm_ner_biodiversity` | LLM으로 생물다양성 논문에서 개체명(종, 지역 등) 추출 |

---

## budget_biodiv_cls

예산 사업 정보(분야명, 부문명, 프로그램명, 세부사업명)를 Ollama LLM에 입력해  
생물다양성 관련 여부를 `1 / 0 / -1(실패)` 로 라벨링합니다.

```bash
cd budget_biodiv_cls
pip install -r requirements.txt

python src/make_biodiv_labels.py \
  --input-csv <입력CSV경로> \
  --output-csv <출력CSV경로> \
  --model llama3.2:latest \
  --limit 100          # 테스트 시 행 수 제한
```

- Ollama가 로컬에서 실행 중이어야 합니다 (`http://localhost:11434`)
- 중단 후 재실행하면 이어서 처리합니다

---

## budget_field_cls

HWP/PDF 예산 문서에서 텍스트를 추출하고 `klue/bert-base`를 fine-tuning해  
16개 예산 분야(환경, 농림수산, 교육 등)를 자동 분류합니다.

자세한 내용은 [budget_field_cls/README.md](budget_field_cls/README.md) 참조.

---

## biodiversity_rag

PubMed 생물다양성 논문 초록을 ChromaDB에 인덱싱하고  
자연어 질문에 대해 관련 논문을 검색·인용하며 답변합니다.

자세한 내용은 [biodiversity_rag/README.md](biodiversity_rag/README.md) 참조.

---

## budget_matcher

예산 CSV(열린재정)와 로컬 HWP/PDF 파일 폴더를 대조해  
사업명 기준으로 파일을 매칭하고 결과를 CSV로 출력합니다.

자세한 내용은 [budget_matcher/README.md](budget_matcher/README.md) 참조.

---

## llm_ner_biodiversity

PubMed에서 생물다양성 논문 초록을 수집하고  
LLM(GPT, Llama 등)으로 종명·지역명 등 개체명을 추출합니다.

자세한 내용은 [llm_ner_biodiversity/README.md](llm_ner_biodiversity/README.md) 참조.

---

## 공통 요구사항

- Python 3.10+
- 각 서브 프로젝트의 `requirements.txt` 또는 `environment.yml` 참조
- 일부 프로젝트는 Ollama 또는 OpenAI API 키 필요
