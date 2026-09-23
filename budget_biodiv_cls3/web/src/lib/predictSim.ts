import type { BudgetProgram, ModelPrediction, PredictionRun } from '../types'
import { SUB_CATEGORIES, TOP_CATEGORIES, subCategoriesForTop } from '../data/categories'

function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)]
}

export type SimulatableModelKey = Exclude<PredictionRun['modelKey'], 'combined'>

const MODEL_LABEL: Record<SimulatableModelKey, string> = {
  transformer_v1: 'Transformer v1',
  transformer_v2: 'Transformer v2',
  llm_v1: 'LLM v1',
  llm_v2: 'LLM v2',
  pipeline: '단계별 파이프라인',
}

export function simulatePrediction(program: BudgetProgram, modelKey: SimulatableModelKey): ModelPrediction {
  const usesSub = modelKey === 'transformer_v2' || modelKey === 'llm_v2' || modelKey === 'pipeline'
  const top = pick(TOP_CATEGORIES.filter((c) => c.code !== '0' || Math.random() < 0.2))
  const subs = subCategoriesForTop(top.code)
  const sub = usesSub && top.code !== '0' && subs.length > 0 ? pick(SUB_CATEGORIES.filter((s) => s.topCode === top.code)) : null
  const isLlm = modelKey === 'llm_v1' || modelKey === 'llm_v2'

  return {
    modelLabel: MODEL_LABEL[modelKey],
    topCategory: top.code,
    subCategory: sub ? sub.code : null,
    confidence: 0.55 + Math.random() * 0.4,
    confidenceType: isLlm ? 'self_reported' : 'probability',
    reasoning: `[신규 실행 데모] "${program.programName}" 사업설명자료 및 예산 항목을 바탕으로 "${top.name}"로 분류했습니다.`,
    evidenceSentences: program.documentStatus === '연결됨' ? [`"...${program.programName} 관련 세부 추진계획을 포함한다..."`] : [],
    attentionSpans: program.documentStatus === '연결됨' ? [top.name] : [],
    inputBasis: program.documentStatus === '연결됨' ? 'document' : 'budget_only',
    purposeExtractionFailed: program.documentStatus === '추출실패',
  }
}
