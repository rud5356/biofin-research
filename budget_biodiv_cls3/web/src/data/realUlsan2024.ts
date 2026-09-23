import type { BudgetProgram } from '../types'
import { emptyReview } from './seed'
import rawRows from './realUlsan2024.json'

// 실제 데이터: 울산광역시 환경분야 2024년 세부사업 355건.
// crawlers/open_fiscal로 수집한 세부사업 목록 + Transformer v1 / LLM v1 실제 예측 결과입니다.
// (scripts/convert-ulsan-csv.mjs로 "울산_환경_2024 모델 카테고리 분류 결과.CSV"에서 변환)
// 전문가 검토는 아직 이루어지지 않아 전부 검토대기 상태로 시작합니다.

export const RUN_REAL_ULSAN_2024 = 'run-real-ulsan-2024'

type RawRow = Omit<BudgetProgram, 'predictionRunId' | 'transformer_v2' | 'llm_v2' | 'pipeline' | 'expertReview' | 'assignee' | 'goldTopCategory' | 'goldSubCategory'>

export function buildRealUlsanPrograms(): BudgetProgram[] {
  return (rawRows as RawRow[]).map((row) => ({
    ...row,
    predictionRunId: RUN_REAL_ULSAN_2024,
    transformer_v2: null,
    llm_v2: null,
    pipeline: null,
    expertReview: emptyReview(),
    assignee: null,
    goldTopCategory: null,
    goldSubCategory: null,
  }))
}
