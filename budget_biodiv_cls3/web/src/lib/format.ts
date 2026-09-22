import type { AmountUnit } from '../types'

export function formatAmount(value: number | null, unit: AmountUnit): string {
  if (value === null || value === undefined) return '정보 없음'
  let scaled: number
  switch (unit) {
    case 'won':
      scaled = value
      break
    case 'million':
      scaled = value / 1_000_000
      break
    case '100million':
      scaled = value / 100_000_000
      break
  }
  const rounded = unit === 'won' ? Math.round(scaled) : Math.round(scaled * 10) / 10
  return rounded.toLocaleString('ko-KR')
}

export function amountUnitLabel(unit: AmountUnit): string {
  switch (unit) {
    case 'won':
      return '원'
    case 'million':
      return '백만원'
    case '100million':
      return '억원'
  }
}

export function formatPercent(value: number | null): string {
  if (value === null || value === undefined) return '미제공'
  return `${Math.round(value * 1000) / 10}%`
}

export function formatDateTime(iso: string | null): string {
  if (!iso) return '-'
  const d = new Date(iso)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

export function formatDate(iso: string | null): string {
  if (!iso) return '-'
  const d = new Date(iso)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
}
