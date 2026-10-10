# biofin-research

국가생물다양성 재정(BIOFIN) 연구를 위한 데이터 분석 및 AI 파이프라인 모음입니다.  
열린재정 예산 데이터에서 생물다양성 관련 사업을 식별하고, 문서를 분류·분석합니다.

---

## 서브 프로젝트

| 폴더 | 설명 |
|------|------|
| `budget_biodiv_cls` | 재정자료 생물다양성 관련 여부 이진 분류 (Ollama LLM + KoBERT) |
| `budget_biodiv_cls2` | `budget_biodiv_cls` 2세대 — BIOFIN 9대 분류 기준 프롬프트(v1~v4) + 해시 캐시 기반 라벨링 |
| `budget_biodiv_cls3` | 예산 CSV + 사업설명자료 전체 본문 결합 — BIOFIN 1차 카테고리(0~9) Transformer(Attention Pooling) 분류 |
| `budget_field_cls` | 재정자료 분야별 다중 분류 — HWP/PDF 문서를 16개 예산 분야로 분류 (KoBERT, Docker 지원) |
| `biodiversity_rag` | 생물다양성 논문 초록 기반 RAG 질의응답 시스템 |
| `budget_matcher` | 예산 CSV와 열린재정 파일 폴더 매칭 |
| `llm_ner_biodiversity` | LLM으로 생물다양성 논문에서 개체명(종, 지역 등) 추출 |
| `biofin_qwen` | Qwen2.5 LoRA 파인튜닝 감정분류 실습/데모 |
| `crawlers` | 재정 데이터 크롤러 — `open_fiscal`(중앙재정), `local_fiscal`(지방재정365) |
| `presentation` | 생물다양성 분류 결과 리포트 산출물(HTML/PDF) |

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

## budget_biodiv_cls2

`budget_biodiv_cls`의 2세대 파이프라인입니다. BIOFIN 9대 분류 기준 프롬프트(v1→v4)로 고도화했고,
사업 조합(소관명/분야명/부문명/프로그램명/단위사업명/세부사업명)의 SHA256 해시 캐시로 동일 사업의 중복 LLM 호출을 방지합니다.

```bash
cd budget_biodiv_cls2
python label_biodiv_with_ollama.py --limit-keys 50 --dry-run   # 테스트
python label_biodiv_with_ollama.py                              # 전체 실행
python apply_cache_labels.py                                    # 캐시를 다른 CSV에 재적용
```

- Ollama 로컬 서버(`llama3.1:8b` 권장) 필요
- 자세한 내용은 [budget_biodiv_cls2/README.md](budget_biodiv_cls2/README.md) 참조

---

## budget_biodiv_cls3

`budget_biodiv_cls2`와 달리 예산 CSV 메타데이터뿐 아니라 매칭된 사업설명자료(HWP) 전체
본문을 함께 사용해 BIOFIN 1차 카테고리(0~9)를 Transformer로 분류합니다. 긴 문서는 512
token chunk로 나눠 Attention Pooling으로 문서 단위 분류를 수행합니다.

```powershell
cd budget_biodiv_cls3
python src/train_attention_classifier.py --document_only --class_weight   # 학습
python src/predict_attention_classifier_v2.py `
  --model_dir outputs/model_results `
  --doc_dir document/2023/사업설명자료 `
  --budget_file document/2023biofin_label.csv `
  --output_dir outputs/predictions `
  --no-heatmap                                                            # 새 CSV 분류
```

자세한 내용은 [budget_biodiv_cls3/README.md](budget_biodiv_cls3/README.md) 참조.

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

## biofin_qwen

`Qwen2.5-0.5B-Instruct`를 LoRA로 파인튜닝해 한국어 문장 감정분류(긍정/부정)를 실습하는 데모 프로젝트입니다.
소규모(20문장) 예시 데이터라 일반화된 성능 지표로 해석하지 않습니다.

```bash
cd biofin_qwen
python -m venv .venv && .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python convert_csv.py     # 데이터 변환
python train_lora.py      # LoRA 학습
python compare.py         # 학습 전/후 비교
```

자세한 내용은 [biofin_qwen/README.md](biofin_qwen/README.md) 참조.

---

## crawlers

재정 데이터의 출처별로 분리된 크롤러 모음입니다. Playwright 기반 브라우저 자동화를 사용합니다.

| 서브폴더 | 설명 |
|------|------|
| `open_fiscal` | 열린재정(중앙재정) 세부사업 목록·사업설명자료(HWP) 수집 |
| `local_fiscal` | 지방재정365 광역지역 7곳 세부사업 목록·상세화면 PDF 수집 |

```bash
# 중앙재정
cd crawlers/open_fiscal
python crawl_business_docs.py --year 2024 --limit 5

# 지방재정
cd crawlers/local_fiscal
python crawl_business_docs.py --regions 서울 부산 --limit 5
```

자세한 내용은 [crawlers/README.md](crawlers/README.md), [crawlers/open_fiscal/README.md](crawlers/open_fiscal/README.md), [crawlers/local_fiscal/README.md](crawlers/local_fiscal/README.md) 참조.

---

## presentation

`budget_biodiv_cls2` 분류 결과를 정리한 리포트 산출물(HTML/PDF) 폴더입니다. 별도 실행 스크립트는 없습니다.

---

## 내역사업 분류 워크플로우 (루트 스크립트)

`국가생물다양성_열린재정 데이터/`의 HWP 사업설명자료에서 내역사업(세부사업 하위 항목) 포함 여부를 판별·분류하는 3단계 파이프라인입니다.

```bash
python add_naeyeok_column.py    # 1단계: HWP에서 "내역사업명" 키워드 파싱 → 내역사업포함여부 컬럼 추가
python copy_naeyeok_files.py    # 2단계: 내역사업포함여부=1인 HWP를 분야명별 폴더로 복사
python classify_naeyeok.py [--copy]   # 3단계: 내역사업 1개/여러개 패턴 분류
```

- `tmp_count_probe.py`, `tmp_debug_b.py`는 위 파이프라인 디버깅용 임시 스크립트로, 정식 워크플로우에 포함되지 않습니다.

---

## 원본 데이터

`국가생물다양성_열린재정 데이터/`, `국가생물다양성_열린재정_데이터.tar`(~6.4GB)는 2024년도 예산·결산 HWP 원본 문서 모음입니다.
용량이 크므로 Git 추적 대상에서 제외하는 것을 권장합니다.

---

## 공통 요구사항

- Python 3.10+
- 각 서브 프로젝트의 `requirements.txt` 또는 `environment.yml` 참조
- 일부 프로젝트는 Ollama 또는 OpenAI API 키 필요
