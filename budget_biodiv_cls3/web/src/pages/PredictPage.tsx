import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAppStore } from '../store/useAppStore'
import type { PredictionRun } from '../types'
import { isUnclassified } from '../lib/derive'
import { simulatePrediction, type SimulatableModelKey } from '../lib/predictSim'
import { Button } from '../components/common/Button'
import { formatDateTime } from '../lib/format'

const MODEL_OPTIONS: { key: SimulatableModelKey; title: string; desc: string }[] = [
  { key: 'transformer_v1', title: 'Transformer v1', desc: '상위 카테고리 0~9 분류' },
  { key: 'transformer_v2', title: 'Transformer v2', desc: '계층형 하위 코드 분류 (예: 6.05)' },
  { key: 'llm_v1', title: 'LLM v1', desc: 'Ollama 기반 상위 카테고리 0~9 분류' },
  { key: 'llm_v2', title: 'LLM v2', desc: 'Ollama 기반 비해당 0 및 39개 하위 코드 분류' },
  { key: 'pipeline', title: '단계별 파이프라인', desc: '규칙 → Transformer → LLM 라우팅 (상위 분류 데모)' },
]

const TABS = [
  { key: 'registered', label: '등록 데이터 선택' },
  { key: 'csv', label: 'CSV 등록 데모' },
  { key: 'manual', label: '단일 사업 직접 입력' },
] as const

export function PredictPage() {
  const programs = useAppStore((s) => s.programs)
  const runs = useAppStore((s) => s.runs)
  const datasets = useAppStore((s) => s.datasets)
  const createRun = useAppStore((s) => s.createRun)
  const setRunStatus = useAppStore((s) => s.setRunStatus)
  const setRunProgress = useAppStore((s) => s.setRunProgress)
  const finishRunFinishedAt = useAppStore((s) => s.finishRunFinishedAt)
  const applyRunResult = useAppStore((s) => s.applyRunResult)
  const retryRun = useAppStore((s) => s.retryRun)
  const navigate = useNavigate()

  const [tab, setTab] = useState<(typeof TABS)[number]['key']>('registered')
  const [modelKey, setModelKey] = useState<SimulatableModelKey>('transformer_v1')
  const [runName, setRunName] = useState('신규 예측 실행')
  const [scope, setScope] = useState<PredictionRun['scope']>('전체')
  const [datasetName, setDatasetName] = useState(datasets[0]?.name ?? '')
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [simulateError, setSimulateError] = useState(false)
  const [activeRunId, setActiveRunId] = useState<string | null>(null)
  const [manualText, setManualText] = useState('')
  const intervalRef = useRef<number | null>(null)

  const activeRun = runs.find((r) => r.id === activeRunId) ?? null

  const targetIds = useMemo(() => {
    if (scope === '전체') return programs.map((p) => p.id)
    if (scope === '미분류 사업') return programs.filter(isUnclassified).map((p) => p.id)
    return Array.from(selectedIds)
  }, [scope, programs, selectedIds])

  useEffect(() => {
    return () => {
      if (intervalRef.current) window.clearInterval(intervalRef.current)
    }
  }, [])

  function runInterval(runId: string, total: number) {
    let processed = 0
    let failed = 0
    intervalRef.current = window.setInterval(() => {
      if (simulateError && processed >= Math.floor(total * 0.6)) {
        window.clearInterval(intervalRef.current!)
        setRunStatus(runId, '오류')
        finishRunFinishedAt(runId)
        return
      }
      const step = Math.max(1, Math.round(total / 12))
      processed = Math.min(total, processed + step)
      if (Math.random() < 0.08) failed += 1
      setRunProgress(runId, processed, failed)

      if (processed >= total) {
        window.clearInterval(intervalRef.current!)
        const run = useAppStore.getState().runs.find((r) => r.id === runId)
        if (run && run.modelKey !== 'combined') {
          const runModelKey = run.modelKey
          run.targetIds.forEach((pid) => {
            const program = useAppStore.getState().programs.find((p) => p.id === pid)
            if (!program) return
            const pred = simulatePrediction(program, runModelKey)
            applyRunResult(runId, pid, runModelKey, pred)
          })
        }
        setRunStatus(runId, '완료')
        finishRunFinishedAt(runId)
      }
    }, 250)
  }

  function handleStart() {
    if (targetIds.length === 0) {
      alert('처리 대상 사업이 없습니다. 처리 범위를 확인해 주세요.')
      return
    }
    const id = createRun({ name: runName, modelKey, scope, targetIds, datasetName: datasetName || '수동 입력' })
    setActiveRunId(id)
    setRunStatus(id, '진행중')
    runInterval(id, targetIds.length)
  }

  function handleCancel() {
    if (!activeRunId) return
    if (intervalRef.current) window.clearInterval(intervalRef.current)
    setRunStatus(activeRunId, '취소')
    finishRunFinishedAt(activeRunId)
  }

  function handleRetry(runId: string) {
    const newId = retryRun(runId)
    setActiveRunId(newId)
    const run = useAppStore.getState().runs.find((r) => r.id === newId)
    if (run) {
      setRunStatus(newId, '진행중')
      runInterval(newId, run.totalCount)
    }
  }

  const recentRuns = runs.slice(0, 8)

  return (
    <div className="space-y-4 p-5">
      <div>
        <h1 className="text-base font-semibold text-navy-900">예측 실행</h1>
        <p className="mt-0.5 text-xs text-slate-500">실제 파이프라인 모델 연결은 준비 중이며, 아래 실행은 데모 데이터로 진행 상태를 시뮬레이션합니다.</p>
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <div className="flex gap-1 border-b border-slate-200 pb-2">
          {TABS.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`rounded-t-md px-3 py-1.5 text-xs font-medium ${tab === t.key ? 'bg-navy-900 text-white' : 'text-slate-500 hover:bg-slate-100'}`}
            >
              {t.label}
            </button>
          ))}
        </div>

        <div className="mt-3 space-y-4">
          {tab === 'registered' && (
            <div>
              <label className="mb-1 block text-xs font-medium text-slate-600">등록 데이터셋</label>
              <select className="h-8 w-full max-w-md rounded border border-slate-300 bg-white px-2 text-xs" value={datasetName} onChange={(e) => setDatasetName(e.target.value)}>
                {datasets.map((d) => (
                  <option key={d.id} value={d.name}>
                    {d.name} ({d.rowCount.toLocaleString('ko-KR')}행)
                  </option>
                ))}
              </select>
            </div>
          )}
          {tab === 'csv' && (
            <div className="rounded border border-dashed border-slate-300 p-6 text-center text-xs text-slate-500">
              CSV 업로드 데모 — <b>데이터·문서 관리</b> 메뉴의 업로드·컬럼 매핑 화면과 연결됩니다. (여기서는 실행 대상 데이터로 위 등록 데이터셋을 사용합니다.)
            </div>
          )}
          {tab === 'manual' && (
            <div>
              <label className="mb-1 block text-xs font-medium text-slate-600">사업 설명 직접 입력 (데모)</label>
              <textarea
                className="w-full rounded border border-slate-300 p-2 text-xs"
                rows={3}
                value={manualText}
                onChange={(e) => setManualText(e.target.value)}
                placeholder="사업명·사업 목적을 입력하면 단일 사업 분류 데모에 사용됩니다. (실제 저장되지 않는 데모 입력입니다)"
              />
            </div>
          )}

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div>
              <label className="mb-1.5 block text-xs font-medium text-slate-600">모델 선택</label>
              <div className="space-y-1.5">
                {MODEL_OPTIONS.map((m) => (
                  <label
                    key={m.key}
                    className={`flex cursor-pointer items-start gap-2 rounded-md border p-2 text-xs ${
                      modelKey === m.key ? 'border-navy-700 bg-navy-900/5' : 'border-slate-200 hover:bg-slate-50'
                    }`}
                  >
                    <input type="radio" name="model" className="mt-0.5" checked={modelKey === m.key} onChange={() => setModelKey(m.key)} />
                    <span>
                      <span className="block font-medium text-slate-800">{m.title}</span>
                      <span className="block text-2xs text-slate-400">{m.desc}</span>
                    </span>
                  </label>
                ))}
              </div>
            </div>

            <div className="space-y-3">
              <div>
                <label className="mb-1 block text-xs font-medium text-slate-600">실행 이름</label>
                <input className="h-8 w-full rounded border border-slate-300 px-2 text-xs" value={runName} onChange={(e) => setRunName(e.target.value)} />
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-slate-600">처리 범위</label>
                <div className="flex gap-1.5">
                  {(['전체', '선택 사업', '미분류 사업'] as PredictionRun['scope'][]).map((s) => (
                    <button
                      key={s}
                      onClick={() => setScope(s)}
                      className={`rounded-md border px-2.5 py-1 text-xs font-medium ${scope === s ? 'border-navy-900 bg-navy-900 text-white' : 'border-slate-300 bg-white text-slate-600'}`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
                <div className="mt-1 text-2xs text-slate-400">대상 {targetIds.length.toLocaleString('ko-KR')}건</div>
              </div>

              {scope === '선택 사업' && (
                <div className="max-h-40 space-y-1 overflow-y-auto rounded border border-slate-200 p-2">
                  {programs.map((p) => (
                    <label key={p.id} className="flex items-center gap-2 text-2xs text-slate-600">
                      <input
                        type="checkbox"
                        checked={selectedIds.has(p.id)}
                        onChange={(e) => {
                          const next = new Set(selectedIds)
                          if (e.target.checked) next.add(p.id)
                          else next.delete(p.id)
                          setSelectedIds(next)
                        }}
                      />
                      {p.programName}
                    </label>
                  ))}
                </div>
              )}

              <label className="flex items-center gap-1.5 text-2xs text-slate-500">
                <input type="checkbox" checked={simulateError} onChange={(e) => setSimulateError(e.target.checked)} />
                오류 상황 데모(진행 중 오류 발생시키기)
              </label>

              <Button variant="primary" onClick={handleStart} disabled={!!activeRun && (activeRun.status === '진행중')}>
                예측 시작
              </Button>
            </div>
          </div>
        </div>
      </div>

      {activeRun && (
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-navy-900">{activeRun.name}</h2>
            <RunStatusBadge status={activeRun.status} />
          </div>
          <div className="mt-2 h-3 w-full overflow-hidden rounded-full bg-slate-100">
            <div
              className={`h-full rounded-full ${activeRun.status === '오류' ? 'bg-red-400' : activeRun.status === '취소' ? 'bg-slate-400' : 'bg-teal-500 progress-stripe'}`}
              style={{ width: `${activeRun.totalCount ? Math.min(100, (activeRun.processedCount / activeRun.totalCount) * 100) : 0}%` }}
            />
          </div>
          <div className="mt-1.5 flex items-center justify-between text-2xs text-slate-500">
            <span>
              처리 {activeRun.processedCount.toLocaleString('ko-KR')} / {activeRun.totalCount.toLocaleString('ko-KR')}건 · 실패 {activeRun.failedCount}건
            </span>
            <div className="flex gap-1.5">
              {activeRun.status === '진행중' && (
                <Button size="sm" variant="danger" onClick={handleCancel}>
                  취소
                </Button>
              )}
              {(activeRun.status === '오류' || activeRun.status === '취소') && (
                <Button size="sm" variant="outline" onClick={() => handleRetry(activeRun.id)}>
                  재시도
                </Button>
              )}
              {activeRun.status === '완료' && (
                <Button size="sm" variant="secondary" onClick={() => navigate(`/results?run=${activeRun.id}`)}>
                  이 실행 결과 보기 →
                </Button>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="mb-2 text-sm font-semibold text-navy-900">최근 실행</h2>
        <table className="w-full text-xs">
          <thead className="text-2xs text-slate-400">
            <tr>
              <th className="border-b border-slate-200 px-2 py-1.5 text-left">실행명</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-left">모델</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-left">범위</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-left">상태</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-right">처리/전체</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-left">생성일시</th>
              <th className="border-b border-slate-200 px-2 py-1.5"></th>
            </tr>
          </thead>
          <tbody>
            {recentRuns.map((r) => (
              <tr key={r.id} className="border-b border-slate-100">
                <td className="px-2 py-1.5 text-slate-700">{r.name}</td>
                <td className="px-2 py-1.5 text-slate-500">{r.modelLabel}</td>
                <td className="px-2 py-1.5 text-slate-500">{r.scope}</td>
                <td className="px-2 py-1.5">
                  <RunStatusBadge status={r.status} />
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums text-slate-500">
                  {r.processedCount}/{r.totalCount}
                </td>
                <td className="px-2 py-1.5 text-slate-400">{formatDateTime(r.createdAt)}</td>
                <td className="px-2 py-1.5 text-right">
                  <button className="text-teal-700 hover:underline" onClick={() => navigate(`/results?run=${r.id}`)}>
                    결과 보기
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function RunStatusBadge({ status }: { status: PredictionRun['status'] }) {
  const style: Record<PredictionRun['status'], string> = {
    대기: 'bg-slate-100 text-slate-600 border-slate-200',
    진행중: 'bg-teal-100 text-teal-700 border-teal-500/30',
    완료: 'bg-teal-100 text-teal-700 border-teal-500/30',
    오류: 'bg-red-50 text-red-700 border-red-200',
    취소: 'bg-slate-100 text-slate-500 border-slate-200',
  }
  return <span className={`inline-flex items-center rounded border px-1.5 py-0.5 text-2xs font-medium ${style[status]}`}>{status}</span>
}
