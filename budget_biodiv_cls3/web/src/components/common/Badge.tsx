import type { DocumentStatus, ReviewStatus } from '../../types'

const REVIEW_STYLE: Record<ReviewStatus, string> = {
  검토대기: 'bg-amber-50 text-amber-800 border-amber-200',
  검토완료: 'bg-teal-100 text-teal-700 border-teal-500/40',
  보류: 'bg-orange-50 text-orange-700 border-orange-200',
  재검토요청: 'bg-red-50 text-red-700 border-red-200',
}

const DOC_STYLE: Record<DocumentStatus, string> = {
  연결됨: 'bg-slate-50 text-slate-700 border-slate-200',
  없음: 'bg-slate-100 text-slate-500 border-slate-200',
  추출실패: 'bg-red-50 text-red-700 border-red-200',
}

export function ReviewStatusBadge({ status }: { status: ReviewStatus }) {
  return (
    <span className={`inline-flex items-center rounded border px-1.5 py-0.5 text-xs font-medium whitespace-nowrap ${REVIEW_STYLE[status]}`}>
      {status}
    </span>
  )
}

export function DocumentStatusBadge({ status }: { status: DocumentStatus }) {
  return (
    <span className={`inline-flex items-center rounded border px-1.5 py-0.5 text-xs font-medium whitespace-nowrap ${DOC_STYLE[status]}`}>
      문서 {status}
    </span>
  )
}

export function MismatchBadge() {
  return (
    <span className="inline-flex items-center gap-1 rounded border border-red-200 bg-red-50 px-1.5 py-0.5 text-xs font-medium text-red-700 whitespace-nowrap">
      ⚠ 모델 불일치
    </span>
  )
}

export function UnclassifiedBadge() {
  return (
    <span className="inline-flex items-center rounded border border-slate-300 bg-slate-50 px-1.5 py-0.5 text-xs font-medium text-slate-600 whitespace-nowrap">
      미분류
    </span>
  )
}

export function Badge({ tone = 'neutral', children }: { tone?: 'neutral' | 'navy' | 'teal'; children: React.ReactNode }) {
  const style =
    tone === 'navy'
      ? 'bg-navy-900/5 text-navy-900 border-navy-900/15'
      : tone === 'teal'
        ? 'bg-teal-100 text-teal-700 border-teal-500/30'
        : 'bg-slate-100 text-slate-600 border-slate-200'
  return <span className={`inline-flex items-center rounded border px-1.5 py-0.5 text-xs font-medium whitespace-nowrap ${style}`}>{children}</span>
}
