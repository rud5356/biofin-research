import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAppStore } from '../store/useAppStore'
import { isHighValueUnreviewed, isMismatch, isMissingDocument, isUnclassified } from '../lib/derive'
import { formatAmount, amountUnitLabel } from '../lib/format'
import { StatCard } from '../components/dashboard/StatCard'
import { CategoryChart } from '../components/dashboard/CategoryChart'

type AggBasis = 'model' | 'expert'

export function DashboardPage() {
  const programs = useAppStore((s) => s.programs)
  const navigate = useNavigate()
  const [metric, setMetric] = useState<'count' | 'amount'>('count')
  const [aggBasis, setAggBasis] = useState<AggBasis>('model')

  const stats = useMemo(() => {
    const totalBudget = programs.reduce((sum, p) => sum + (p.budgetAmount ?? 0), 0)
    return {
      total: programs.length,
      totalBudget,
      predicted: programs.filter((p) => p.transformer_v1 !== null).length,
      unclassified: programs.filter(isUnclassified).length,
      pending: programs.filter((p) => p.expertReview.status === '검토대기').length,
      completed: programs.filter((p) => p.expertReview.status === '검토완료').length,
      mismatch: programs.filter(isMismatch).length,
      highValueUnreviewed: programs.filter(isHighValueUnreviewed).length,
      missingDoc: programs.filter(isMissingDocument).length,
    }
  }, [programs])

  const categoryAgg = useMemo(() => {
    const map = new Map<string, { count: number; amount: number }>()
    programs.forEach((p) => {
      const code = aggBasis === 'model' ? p.transformer_v1?.topCategory ?? null : p.expertReview.finalTopCategory
      if (code === null) return
      const cur = map.get(code) ?? { count: 0, amount: 0 }
      cur.count += 1
      cur.amount += p.budgetAmount ?? 0
      map.set(code, cur)
    })
    return Array.from(map.entries()).map(([code, v]) => ({ code, ...v }))
  }, [programs, aggBasis])

  return (
    <div className="space-y-4 p-5">
      <div>
        <h1 className="text-base font-semibold text-navy-900">현황 대시보드</h1>
        <p className="mt-0.5 text-xs text-slate-500">집계 대상: 전체 등록 사업 {stats.total.toLocaleString('ko-KR')}건 (현재 필터 없음, 전체 기준)</p>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-7">
        <StatCard label="전체 사업 수" value={`${stats.total.toLocaleString('ko-KR')}건`} />
        <StatCard label="예산 합계" value={formatAmount(stats.totalBudget, '100million')} sub={amountUnitLabel('100million')} />
        <StatCard label="예측 완료" value={`${stats.predicted.toLocaleString('ko-KR')}건`} />
        <StatCard label="미분류" value={`${stats.unclassified.toLocaleString('ko-KR')}건`} tone="warning" linkTo="/results?quick=unclassified" />
        <StatCard label="검토 대기" value={`${stats.pending.toLocaleString('ko-KR')}건`} tone="warning" linkTo="/results?reviewStatus=검토대기" />
        <StatCard label="검토 완료" value={`${stats.completed.toLocaleString('ko-KR')}건`} tone="good" />
        <StatCard label="모델 불일치" value={`${stats.mismatch.toLocaleString('ko-KR')}건`} tone="critical" linkTo="/results?quick=mismatch" />
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
        <button
          onClick={() => navigate('/results?quick=highValue')}
          className="rounded-lg border border-slate-200 bg-white p-4 text-left hover:shadow-md"
        >
          <div className="text-xs font-medium text-slate-500">고액 미검토</div>
          <div className="mt-1 text-lg font-semibold text-amber-700">{stats.highValueUnreviewed.toLocaleString('ko-KR')}건</div>
          <div className="mt-0.5 text-2xs text-slate-400">예산 30억원 이상 · 검토대기 사업 바로가기</div>
        </button>
        <button
          onClick={() => navigate('/results?quick=mismatch')}
          className="rounded-lg border border-slate-200 bg-white p-4 text-left hover:shadow-md"
        >
          <div className="text-xs font-medium text-slate-500">모델 불일치</div>
          <div className="mt-1 text-lg font-semibold text-red-600">{stats.mismatch.toLocaleString('ko-KR')}건</div>
          <div className="mt-0.5 text-2xs text-slate-400">Transformer·LLM 상위 카테고리 불일치 바로가기</div>
        </button>
        <button
          onClick={() => navigate('/results?quick=missingDoc')}
          className="rounded-lg border border-slate-200 bg-white p-4 text-left hover:shadow-md"
        >
          <div className="text-xs font-medium text-slate-500">문서 누락</div>
          <div className="mt-1 text-lg font-semibold text-slate-700">{stats.missingDoc.toLocaleString('ko-KR')}건</div>
          <div className="mt-0.5 text-2xs text-slate-400">사업설명자료 미연결·추출실패 바로가기</div>
        </button>
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-sm font-semibold text-navy-900">카테고리별 분포</h2>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 rounded-md border border-slate-300 bg-white p-0.5 text-2xs">
              <button
                onClick={() => setMetric('count')}
                className={`rounded px-2 py-1 font-medium ${metric === 'count' ? 'bg-navy-900 text-white' : 'text-slate-600'}`}
              >
                사업 수
              </button>
              <button
                onClick={() => setMetric('amount')}
                className={`rounded px-2 py-1 font-medium ${metric === 'amount' ? 'bg-navy-900 text-white' : 'text-slate-600'}`}
              >
                예산 합계
              </button>
            </div>
            <div className="flex items-center gap-1 rounded-md border border-slate-300 bg-white p-0.5 text-2xs">
              <button
                onClick={() => setAggBasis('model')}
                className={`rounded px-2 py-1 font-medium ${aggBasis === 'model' ? 'bg-teal-600 text-white' : 'text-slate-600'}`}
              >
                모델 예측 기준
              </button>
              <button
                onClick={() => setAggBasis('expert')}
                className={`rounded px-2 py-1 font-medium ${aggBasis === 'expert' ? 'bg-teal-600 text-white' : 'text-slate-600'}`}
              >
                전문가 확정 기준
              </button>
            </div>
          </div>
        </div>
        <p className="mt-1 text-2xs text-slate-400">
          {aggBasis === 'model' ? 'Transformer v1 예측 상위 카테고리 기준 분포입니다.' : '전문가가 최종 확정한 상위 카테고리 기준 분포입니다.'} 카테고리별 금액은 분류된 사업의 예산 합계입니다 (생물다양성 인정금액이 아닙니다).
        </p>
        <div className="mt-2">
          <CategoryChart metric={metric} data={categoryAgg} />
        </div>
      </div>
    </div>
  )
}
