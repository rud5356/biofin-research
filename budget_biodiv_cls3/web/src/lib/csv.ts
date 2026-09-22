import type { BudgetProgram, PredictionSource } from '../types'
import { subCategoryName, topCategoryName } from '../data/categories'
import { formatDateTime } from './format'

function csvEscape(value: string | number | null | undefined): string {
  if (value === null || value === undefined) return ''
  const str = String(value)
  if (/[",\n]/.test(str)) return `"${str.replace(/"/g, '""')}"`
  return str
}

const SOURCE_LABEL: Record<PredictionSource, string> = {
  transformer_v1: 'Transformer v1',
  transformer_v2: 'Transformer v2',
  llm_v1: 'LLM v1',
  llm_v2: 'LLM v2',
  pipeline: '단계별 파이프라인',
  manual: '전문가 수동',
}

export function programsToCsv(programs: BudgetProgram[]): string {
  const headers = [
    '원본행ID',
    '실행ID',
    '연도',
    '소관명',
    '세부사업명',
    '예산액(원)',
    '지출액(원)',
    'Transformer v1 예측',
    'Transformer v2 예측',
    'LLM v1 예측',
    'LLM v2 예측',
    '파이프라인 예측',
    '전문가 최종판정(상위)',
    '전문가 최종판정(하위)',
    '승인출처',
    '검토상태',
    '검토의견',
    '수정사유',
    '검토자',
    '검토일시',
  ]

  const rows = programs.map((p) => {
    const predCell = (pred: BudgetProgram['transformer_v1']) =>
      pred && pred.topCategory ? `${topCategoryName(pred.topCategory)}${pred.subCategory ? ' / ' + subCategoryName(pred.subCategory) : ''}` : '미분류'
    return [
      p.id,
      p.predictionRunId ?? '',
      p.year,
      p.ministry,
      p.programName,
      p.budgetAmount ?? '',
      p.expenditureAmount ?? '',
      predCell(p.transformer_v1),
      predCell(p.transformer_v2),
      predCell(p.llm_v1),
      predCell(p.llm_v2),
      predCell(p.pipeline),
      p.expertReview.finalTopCategory ? topCategoryName(p.expertReview.finalTopCategory) : '',
      p.expertReview.finalSubCategory ? subCategoryName(p.expertReview.finalSubCategory) : '',
      p.expertReview.approvedSource ? SOURCE_LABEL[p.expertReview.approvedSource] : '',
      p.expertReview.status,
      p.expertReview.opinion,
      p.expertReview.reason,
      p.expertReview.reviewer ?? '',
      p.expertReview.reviewedAt ? formatDateTime(p.expertReview.reviewedAt) : '',
    ]
      .map(csvEscape)
      .join(',')
  })

  return '﻿' + [headers.map(csvEscape).join(','), ...rows].join('\r\n')
}

export function downloadCsv(filename: string, csv: string) {
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.setAttribute('download', filename)
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}
