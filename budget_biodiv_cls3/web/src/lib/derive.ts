import type { BudgetProgram } from '../types'

export const HIGH_VALUE_THRESHOLD = 3_000_000_000 // 30억원 — 데모 기준값

export function isMismatch(p: BudgetProgram): boolean {
  const topMismatch =
    !!p.transformer_v1 &&
    !!p.llm_v1 &&
    p.transformer_v1.topCategory !== null &&
    p.llm_v1.topCategory !== null &&
    p.transformer_v1.topCategory !== p.llm_v1.topCategory
  const subMismatch =
    !!p.transformer_v2 &&
    !!p.llm_v2 &&
    p.transformer_v2.subCategory !== null &&
    p.llm_v2.subCategory !== null &&
    p.transformer_v2.subCategory !== p.llm_v2.subCategory
  return topMismatch || subMismatch
}

export function isUnclassified(p: BudgetProgram): boolean {
  return !p.transformer_v1 && !p.llm_v1 && !p.pipeline
}

export function isMissingDocument(p: BudgetProgram): boolean {
  return p.documentStatus !== '연결됨'
}

export function isHighValueUnreviewed(p: BudgetProgram): boolean {
  return (p.budgetAmount ?? 0) >= HIGH_VALUE_THRESHOLD && p.expertReview.status === '검토대기'
}

export function amountByBasis(p: BudgetProgram, basis: 'budget' | 'expenditure' | 'balance'): number | null {
  switch (basis) {
    case 'budget':
      return p.budgetAmount
    case 'expenditure':
      return p.expenditureAmount
    case 'balance':
      return p.balanceAmount
  }
}
