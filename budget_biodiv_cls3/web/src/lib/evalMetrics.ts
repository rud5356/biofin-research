import type { BudgetProgram, TopCode } from '../types'
import { EVAL_GOLD_TOP } from '../data/evalGold'
import { TOP_CATEGORIES } from '../data/categories'

export interface CategoryMetric {
  code: TopCode
  precision: number | null
  recall: number | null
  support: number
}

export interface EvalResult {
  evalDenominator: number
  missingPredictionCount: number
  accuracy: number | null
  macroF1: number | null
  perCategory: CategoryMetric[]
  confusion: Record<string, Record<string, number>> // confusion[gold][pred] = count
  evaluable: boolean
}

export function evaluateModel(programs: BudgetProgram[], getPred: (p: BudgetProgram) => TopCode | null): EvalResult {
  const evalSet = programs.filter((p) => EVAL_GOLD_TOP[p.id] !== undefined)
  const withPred = evalSet.filter((p) => getPred(p) !== null)
  const missingPredictionCount = evalSet.length - withPred.length

  if (withPred.length === 0) {
    return { evalDenominator: evalSet.length, missingPredictionCount, accuracy: null, macroF1: null, perCategory: [], confusion: {}, evaluable: false }
  }

  const confusion: Record<string, Record<string, number>> = {}
  let correct = 0
  withPred.forEach((p) => {
    const gold = EVAL_GOLD_TOP[p.id]
    const pred = getPred(p)!
    confusion[gold] = confusion[gold] ?? {}
    confusion[gold][pred] = (confusion[gold][pred] ?? 0) + 1
    if (gold === pred) correct += 1
  })
  const accuracy = correct / withPred.length

  const perCategory: CategoryMetric[] = TOP_CATEGORIES.map((c) => {
    const support = withPred.filter((p) => EVAL_GOLD_TOP[p.id] === c.code).length
    if (support === 0) return { code: c.code, precision: null, recall: null, support: 0 }
    let tp = 0
    let fp = 0
    let fn = 0
    withPred.forEach((p) => {
      const gold = EVAL_GOLD_TOP[p.id]
      const pred = getPred(p)!
      if (pred === c.code && gold === c.code) tp += 1
      else if (pred === c.code && gold !== c.code) fp += 1
      else if (pred !== c.code && gold === c.code) fn += 1
    })
    const precision = tp + fp > 0 ? tp / (tp + fp) : null
    const recall = tp + fn > 0 ? tp / (tp + fn) : null
    return { code: c.code, precision, recall, support }
  })

  const f1Scores = perCategory
    .filter((c) => c.support > 0 && c.precision !== null && c.recall !== null)
    .map((c) => (c.precision! + c.recall! > 0 ? (2 * c.precision! * c.recall!) / (c.precision! + c.recall!) : 0))
  const macroF1 = f1Scores.length > 0 ? f1Scores.reduce((a, b) => a + b, 0) / f1Scores.length : null

  return { evalDenominator: evalSet.length, missingPredictionCount, accuracy, macroF1, perCategory, confusion, evaluable: true }
}
