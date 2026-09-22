import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAppStore } from '../store/useAppStore'
import { isMismatch } from '../lib/derive'
import { evaluateModel } from '../lib/evalMetrics'
import { formatAmount, formatPercent } from '../lib/format'
import { topCategoryName, TOP_CATEGORIES } from '../data/categories'
import { Button } from '../components/common/Button'
import { MismatchBadge } from '../components/common/Badge'

export function CompareEvalPage() {
  const programs = useAppStore((s) => s.programs)
  const navigate = useNavigate()
  const [confusionModel, setConfusionModel] = useState<'transformer_v1' | 'llm_v1'>('transformer_v1')

  const mismatched = useMemo(() => programs.filter(isMismatch), [programs])

  const transformerEval = useMemo(() => evaluateModel(programs, (p) => p.transformer_v1?.topCategory ?? null), [programs])
  const llmEval = useMemo(() => evaluateModel(programs, (p) => p.llm_v1?.topCategory ?? null), [programs])

  const activeEval = confusionModel === 'transformer_v1' ? transformerEval : llmEval

  const beforeAfter = useMemo(() => {
    const map = new Map<string, { beforeCount: number; beforeAmount: number; afterCount: number; afterAmount: number }>()
    programs.forEach((p) => {
      const before = p.transformer_v1?.topCategory ?? null
      const after = p.expertReview.finalTopCategory
      if (before !== null) {
        const cur = map.get(before) ?? { beforeCount: 0, beforeAmount: 0, afterCount: 0, afterAmount: 0 }
        cur.beforeCount += 1
        cur.beforeAmount += p.budgetAmount ?? 0
        map.set(before, cur)
      }
      if (after !== null) {
        const cur = map.get(after) ?? { beforeCount: 0, beforeAmount: 0, afterCount: 0, afterAmount: 0 }
        cur.afterCount += 1
        cur.afterAmount += p.budgetAmount ?? 0
        map.set(after, cur)
      }
    })
    return TOP_CATEGORIES.filter((c) => map.has(c.code)).map((c) => ({ code: c.code, name: c.name, ...map.get(c.code)! }))
  }, [programs])

  return (
    <div className="space-y-4 p-5">
      <div>
        <h1 className="text-base font-semibold text-navy-900">모델 비교·평가</h1>
        <p className="mt-0.5 text-xs text-slate-500">모든 평가 수치는 데모 지표입니다. 정답이 확보되지 않은 사업은 정확도 계산에서 제외되고 "평가 불가"로 표시됩니다.</p>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <EvalCard title="Transformer v1" result={transformerEval} />
        <EvalCard title="LLM v1" result={llmEval} />
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-navy-900">카테고리별 정밀도·재현율·표본 수 (데모 지표)</h2>
        </div>
        <table className="mt-2 w-full text-xs">
          <thead className="text-2xs text-slate-400">
            <tr>
              <th className="border-b border-slate-200 px-2 py-1.5 text-left">카테고리</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-right">Tf 정밀도</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-right">Tf 재현율</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-right">LLM 정밀도</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-right">LLM 재현율</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-right">표본 수</th>
            </tr>
          </thead>
          <tbody>
            {TOP_CATEGORIES.map((c) => {
              const t = transformerEval.perCategory.find((m) => m.code === c.code)
              const l = llmEval.perCategory.find((m) => m.code === c.code)
              if (!t || t.support === 0) return null
              return (
                <tr key={c.code} className="border-b border-slate-100">
                  <td className="px-2 py-1.5 text-slate-700">
                    {c.code}. {c.name}
                  </td>
                  <td className="px-2 py-1.5 text-right tabular-nums text-slate-500">{formatPercent(t.precision)}</td>
                  <td className="px-2 py-1.5 text-right tabular-nums text-slate-500">{formatPercent(t.recall)}</td>
                  <td className="px-2 py-1.5 text-right tabular-nums text-slate-500">{formatPercent(l?.precision ?? null)}</td>
                  <td className="px-2 py-1.5 text-right tabular-nums text-slate-500">{formatPercent(l?.recall ?? null)}</td>
                  <td className="px-2 py-1.5 text-right tabular-nums text-slate-500">{t.support}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-navy-900">혼동행렬 (데모 지표)</h2>
          <div className="flex items-center gap-1 rounded-md border border-slate-300 bg-white p-0.5 text-2xs">
            {(['transformer_v1', 'llm_v1'] as const).map((m) => (
              <button
                key={m}
                onClick={() => setConfusionModel(m)}
                className={`rounded px-2 py-1 font-medium ${confusionModel === m ? 'bg-navy-900 text-white' : 'text-slate-600'}`}
              >
                {m === 'transformer_v1' ? 'Transformer v1' : 'LLM v1'}
              </button>
            ))}
          </div>
        </div>
        {activeEval.evaluable ? (
          <div className="mt-2 overflow-x-auto">
            <table className="text-2xs">
              <thead>
                <tr>
                  <th className="border border-slate-200 bg-slate-50 px-2 py-1">정답\예측</th>
                  {TOP_CATEGORIES.map((c) => (
                    <th key={c.code} className="border border-slate-200 bg-slate-50 px-2 py-1">
                      {c.code}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {TOP_CATEGORIES.filter((g) => activeEval.confusion[g.code]).map((g) => (
                  <tr key={g.code}>
                    <td className="border border-slate-200 bg-slate-50 px-2 py-1 font-medium">{g.code}</td>
                    {TOP_CATEGORIES.map((p) => {
                      const v = activeEval.confusion[g.code]?.[p.code] ?? 0
                      return (
                        <td key={p.code} className={`border border-slate-100 px-2 py-1 text-center tabular-nums ${g.code === p.code ? 'bg-teal-50 font-semibold text-teal-700' : v > 0 ? 'bg-red-50 text-red-600' : 'text-slate-300'}`}>
                          {v || ''}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="mt-2 text-xs text-slate-400">평가 불가 — 정답이 확보된 사업이 없습니다.</div>
        )}
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="text-sm font-semibold text-navy-900">Transformer·LLM 불일치 사업</h2>
        <p className="mt-0.5 text-2xs text-slate-400">상위 카테고리 또는 하위 코드가 서로 다른 사업입니다. 클릭 시 검증 화면으로 이동합니다.</p>
        <div className="mt-2 max-h-72 overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-white text-2xs text-slate-400">
              <tr>
                <th className="border-b border-slate-200 px-2 py-1.5 text-left">사업명</th>
                <th className="border-b border-slate-200 px-2 py-1.5 text-left">Transformer</th>
                <th className="border-b border-slate-200 px-2 py-1.5 text-left">LLM</th>
                <th className="border-b border-slate-200 px-2 py-1.5 text-left">검토 상태</th>
                <th className="border-b border-slate-200 px-2 py-1.5"></th>
              </tr>
            </thead>
            <tbody>
              {mismatched.map((p) => (
                <tr key={p.id} className="border-b border-slate-100">
                  <td className="px-2 py-1.5 text-slate-700">
                    <div className="flex items-center gap-1.5">
                      {p.programName} <MismatchBadge />
                    </div>
                  </td>
                  <td className="px-2 py-1.5 text-slate-500">{p.transformer_v1 ? topCategoryName(p.transformer_v1.topCategory) : '-'}</td>
                  <td className="px-2 py-1.5 text-slate-500">{p.llm_v1 ? topCategoryName(p.llm_v1.topCategory) : '-'}</td>
                  <td className="px-2 py-1.5 text-slate-500">{p.expertReview.status}</td>
                  <td className="px-2 py-1.5 text-right">
                    <Button size="sm" variant="outline" onClick={() => navigate(`/results?id=${p.id}`)}>
                      검증 화면으로
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="text-sm font-semibold text-navy-900">전문가 수정 전후 카테고리별 사업 수·금액 비교</h2>
        <p className="mt-0.5 text-2xs text-slate-400">수정 전: Transformer v1 예측 기준 · 수정 후: 전문가 최종 확정 기준</p>
        <table className="mt-2 w-full text-xs">
          <thead className="text-2xs text-slate-400">
            <tr>
              <th className="border-b border-slate-200 px-2 py-1.5 text-left">카테고리</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-right">수정 전 건수</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-right">수정 전 금액(억원)</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-right">수정 후 건수</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-right">수정 후 금액(억원)</th>
            </tr>
          </thead>
          <tbody>
            {beforeAfter.map((row) => (
              <tr key={row.code} className="border-b border-slate-100">
                <td className="px-2 py-1.5 text-slate-700">
                  {row.code}. {row.name}
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums text-slate-500">{row.beforeCount}</td>
                <td className="px-2 py-1.5 text-right tabular-nums text-slate-500">{formatAmount(row.beforeAmount, '100million')}</td>
                <td className="px-2 py-1.5 text-right tabular-nums text-teal-700">{row.afterCount}</td>
                <td className="px-2 py-1.5 text-right tabular-nums text-teal-700">{formatAmount(row.afterAmount, '100million')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function EvalCard({ title, result }: { title: string; result: ReturnType<typeof evaluateModel> }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <h3 className="text-sm font-semibold text-navy-900">{title} · 데모 지표</h3>
      <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
        <Metric label="정확도" value={result.evaluable ? formatPercent(result.accuracy) : '평가 불가'} />
        <Metric label="Macro F1" value={result.evaluable ? formatPercent(result.macroF1) : '평가 불가'} />
        <Metric label="평가 분모" value={`${result.evalDenominator}건`} />
        <Metric label="예측 누락" value={`${result.missingPredictionCount}건`} />
      </div>
      <p className="mt-2 text-2xs text-slate-400">평가 분모는 정답이 확보된 사업 수이며, 예측이 없는 사업은 분모에서 제외 후 별도로 표시합니다.</p>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-slate-100 bg-slate-50 px-2.5 py-1.5">
      <div className="text-3xs text-slate-400">{label}</div>
      <div className="text-sm font-semibold tabular-nums text-navy-900">{value}</div>
    </div>
  )
}
