import { useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { useAppStore } from '../store/useAppStore'
import { TOP_CATEGORIES, SUB_CATEGORIES } from '../data/categories'
import { formatDateTime } from '../lib/format'
import { Button } from '../components/common/Button'

const TABS = [
  { key: 'train', label: '학습' },
  { key: 'augment', label: '데이터 증강' },
  { key: 'pipeline', label: '파이프라인' },
  { key: 'criteria', label: '분류 기준' },
  { key: 'runs', label: '실행 이력' },
  { key: 'assignee', label: '담당자 관리' },
] as const
type TabKey = (typeof TABS)[number]['key']

export function ResearchAdminPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const tab = (searchParams.get('tab') as TabKey) ?? 'train'

  function setTab(t: TabKey) {
    const next = new URLSearchParams(searchParams)
    next.set('tab', t)
    setSearchParams(next)
  }

  return (
    <div className="space-y-4 p-5">
      <div>
        <h1 className="text-base font-semibold text-navy-900">연구·관리</h1>
        <p className="mt-0.5 text-xs text-slate-500">학습·증강·파이프라인·분류기준·실행이력·담당자 관리를 위한 연구용 화면입니다.</p>
      </div>

      <div className="flex gap-1 border-b border-slate-200">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`rounded-t-md px-3 py-2 text-xs font-medium ${tab === t.key ? 'border-b-2 border-navy-900 text-navy-900' : 'text-slate-500 hover:text-slate-700'}`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'train' && <TrainTab />}
      {tab === 'augment' && <AugmentTab />}
      {tab === 'pipeline' && <PipelineTab />}
      {tab === 'criteria' && <CriteriaTab />}
      {tab === 'runs' && <RunsTab />}
      {tab === 'assignee' && <AssigneeTab />}
    </div>
  )
}

function TrainTab() {
  const [modelVersion, setModelVersion] = useState('Transformer v2')
  const [docOnly, setDocOnly] = useState(false)
  const [classWeight, setClassWeight] = useState(true)
  const [oversampling, setOversampling] = useState('없음')
  const [split, setSplit] = useState({ train: 70, val: 15, test: 15 })
  const [logs, setLogs] = useState<string[]>([])
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<{ acc: number; f1: number } | null>(null)
  const intervalRef = useRef<number | null>(null)

  useEffect(() => () => { if (intervalRef.current) window.clearInterval(intervalRef.current) }, [])

  function start() {
    setLogs([])
    setResult(null)
    setRunning(true)
    let epoch = 0
    intervalRef.current = window.setInterval(() => {
      epoch += 1
      setLogs((l) => [...l, `[epoch ${epoch}/8] train_loss=${(1.2 - epoch * 0.1).toFixed(3)} val_loss=${(1.3 - epoch * 0.09).toFixed(3)}`])
      if (epoch >= 8) {
        window.clearInterval(intervalRef.current!)
        setRunning(false)
        setResult({ acc: 0.81 + Math.random() * 0.08, f1: 0.74 + Math.random() * 0.08 })
      }
    }, 350)
  }

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="mb-3 text-sm font-semibold text-navy-900">학습 설정</h2>
        <div className="space-y-3 text-xs">
          <div>
            <label className="mb-1 block font-medium text-slate-600">모델 버전</label>
            <select className="h-8 w-full rounded border border-slate-300 bg-white px-2" value={modelVersion} onChange={(e) => setModelVersion(e.target.value)}>
              <option>Transformer v1</option>
              <option>Transformer v2</option>
            </select>
          </div>
          <div>
            <label className="mb-1 block font-medium text-slate-600">데이터 분할 (train/val/test %)</label>
            <div className="flex gap-2">
              {(['train', 'val', 'test'] as const).map((k) => (
                <input
                  key={k}
                  type="number"
                  className="h-8 w-20 rounded border border-slate-300 px-2"
                  value={split[k]}
                  onChange={(e) => setSplit((s) => ({ ...s, [k]: Number(e.target.value) }))}
                />
              ))}
            </div>
          </div>
          <label className="flex items-center gap-1.5 text-slate-600">
            <input type="checkbox" checked={docOnly} onChange={(e) => setDocOnly(e.target.checked)} /> 문서만 사용(예산정보 대체 입력 제외)
          </label>
          <label className="flex items-center gap-1.5 text-slate-600">
            <input type="checkbox" checked={classWeight} onChange={(e) => setClassWeight(e.target.checked)} /> 클래스 가중치 적용
          </label>
          <div>
            <label className="mb-1 block font-medium text-slate-600">다수 클래스 샘플링</label>
            <select className="h-8 w-full rounded border border-slate-300 bg-white px-2" value={oversampling} onChange={(e) => setOversampling(e.target.value)}>
              <option>없음</option>
              <option>언더샘플링</option>
              <option>오버샘플링</option>
            </select>
          </div>
          <Button variant="primary" onClick={start} disabled={running}>
            학습 시작(데모)
          </Button>
        </div>
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="mb-2 text-sm font-semibold text-navy-900">진행 로그·결과</h2>
        <div className="h-40 overflow-y-auto rounded bg-navy-950 p-2 font-mono text-2xs text-teal-300">
          {logs.length === 0 && <div className="text-white/30">학습을 시작하면 로그가 표시됩니다.</div>}
          {logs.map((l, i) => (
            <div key={i}>{l}</div>
          ))}
        </div>
        {result && (
          <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
            <div className="rounded border border-slate-100 bg-slate-50 px-2.5 py-1.5">
              <div className="text-3xs text-slate-400">검증 정확도 (데모)</div>
              <div className="text-sm font-semibold text-navy-900">{(result.acc * 100).toFixed(1)}%</div>
            </div>
            <div className="rounded border border-slate-100 bg-slate-50 px-2.5 py-1.5">
              <div className="text-3xs text-slate-400">검증 Macro F1 (데모)</div>
              <div className="text-sm font-semibold text-navy-900">{(result.f1 * 100).toFixed(1)}%</div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function AugmentTab() {
  const rows = TOP_CATEGORIES.filter((c) => c.code !== '0').map((c, i) => ({
    code: c.code,
    name: c.name,
    original: 20 + i * 3,
    generated: 40 + i * 5,
    needsReview: i % 3 === 0,
  }))
  const [sample, setSample] = useState<string | null>(null)

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="mb-2 text-sm font-semibold text-navy-900">카테고리별 원본·생성 건수</h2>
        <table className="w-full text-xs">
          <thead className="text-2xs text-slate-400">
            <tr>
              <th className="border-b border-slate-200 px-2 py-1.5 text-left">카테고리</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-right">원본</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-right">생성</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-left">분포(원본→증강)</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-left">검수 상태</th>
              <th className="border-b border-slate-200 px-2 py-1.5"></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.code} className="border-b border-slate-100">
                <td className="px-2 py-1.5 text-slate-700">
                  {r.code}. {r.name}
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums text-slate-500">{r.original}</td>
                <td className="px-2 py-1.5 text-right tabular-nums text-teal-700">{r.generated}</td>
                <td className="px-2 py-1.5">
                  <div className="h-1.5 w-32 overflow-hidden rounded-full bg-slate-100">
                    <div className="h-full bg-teal-500" style={{ width: `${Math.min(100, (r.original / r.generated) * 100)}%` }} />
                  </div>
                </td>
                <td className="px-2 py-1.5">
                  {r.needsReview ? (
                    <span className="rounded border border-amber-200 bg-amber-50 px-1.5 py-0.5 text-2xs text-amber-700">검수 필요</span>
                  ) : (
                    <span className="rounded border border-teal-500/30 bg-teal-100 px-1.5 py-0.5 text-2xs text-teal-700">검수 완료</span>
                  )}
                </td>
                <td className="px-2 py-1.5 text-right">
                  <button className="text-teal-700 hover:underline" onClick={() => setSample(r.code)}>
                    원문·생성문 비교
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {sample && (
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <h3 className="mb-2 text-sm font-semibold text-navy-900">원문·생성문 비교 (데모) — {sample}</h3>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="rounded border border-slate-200 p-2.5">
              <div className="mb-1 text-3xs text-slate-400">원문</div>
              <p className="text-slate-600">"...본 사업은 관련 생태계 조사 및 보전 활동을 목적으로 하며, 연차별 세부 추진계획을 포함한다..."</p>
            </div>
            <div className="rounded border border-teal-200 bg-teal-50/50 p-2.5">
              <div className="mb-1 text-3xs text-teal-600">생성문(데모 증강 텍스트)</div>
              <p className="text-slate-600">"...본 사업의 목적은 대상 생태계의 서식 현황을 조사하고 훼손 구간을 복원하는 데 있으며, 단계적 추진 로드맵을 수립한다..."</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function PipelineTab() {
  const [threshold, setThreshold] = useState(0.85)
  const [probCondition, setProbCondition] = useState(true)

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="mb-3 text-sm font-semibold text-navy-900">라우팅 설정</h2>
        <div className="space-y-3 text-xs">
          <div>
            <div className="mb-1 font-medium text-slate-600">비해당 규칙</div>
            <p className="rounded border border-slate-200 bg-slate-50 p-2 text-2xs text-slate-500">
              사업 목적·세부내용에 생물다양성 관련 키워드(보전, 복원, 서식지, 생태계 등)가 전혀 확인되지 않는 경우 규칙 단계에서 "0. 비해당"으로 즉시 확정합니다.
            </p>
          </div>
          <div>
            <label className="mb-1 block font-medium text-slate-600">Transformer 신뢰 임계값: {threshold.toFixed(2)}</label>
            <input type="range" min={0.5} max={0.99} step={0.01} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className="w-full" />
            <p className="mt-1 text-2xs text-slate-400">임계값 이상이면 Transformer 결과를 채택하고, 미만이면 LLM으로 라우팅합니다(데모 설정값).</p>
          </div>
          <label className="flex items-center gap-1.5 text-slate-600">
            <input type="checkbox" checked={probCondition} onChange={(e) => setProbCondition(e.target.checked)} /> 선택적 확률 조건 사용(상위 2개 카테고리 확률차 반영)
          </label>
        </div>
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="mb-3 text-sm font-semibold text-navy-900">LLM 전달 흐름 미리보기</h2>
        <div className="flex flex-col items-center gap-2 text-xs">
          <FlowBox label="입력 사업" />
          <FlowArrow />
          <FlowBox label="비해당 규칙 판정" />
          <FlowArrow />
          <FlowBox label={`Transformer 신뢰도 ≥ ${threshold.toFixed(2)}?`} />
          <div className="flex w-full justify-between px-6">
            <div className="flex flex-col items-center gap-1">
              <span className="text-3xs text-teal-600">예</span>
              <FlowBox label="Transformer 결과 채택" small />
            </div>
            <div className="flex flex-col items-center gap-1">
              <span className="text-3xs text-amber-600">아니오</span>
              <FlowBox label="LLM 판단으로 라우팅" small />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function FlowBox({ label, small }: { label: string; small?: boolean }) {
  return <div className={`rounded-md border border-navy-700/30 bg-navy-900/5 px-3 py-1.5 text-center font-medium text-navy-900 ${small ? 'text-2xs' : 'text-xs'}`}>{label}</div>
}
function FlowArrow() {
  return <div className="text-slate-300">↓</div>
}

function CriteriaTab() {
  const [expandedTop, setExpandedTop] = useState<string | null>(null)
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-navy-900">분류 기준</h2>
        <span className="rounded border border-slate-200 bg-slate-50 px-2 py-0.5 text-2xs text-slate-500">기준 버전 v2026.09</span>
      </div>
      <div className="space-y-2">
        {TOP_CATEGORIES.map((c) => (
          <div key={c.code} className="rounded border border-slate-200">
            <button
              className="flex w-full items-center justify-between px-3 py-2 text-left text-xs font-medium text-slate-700"
              onClick={() => setExpandedTop(expandedTop === c.code ? null : c.code)}
            >
              <span>
                {c.code}. {c.name}
              </span>
              <span className="text-slate-400">{expandedTop === c.code ? '접기 ▲' : '펼치기 ▼'}</span>
            </button>
            {expandedTop === c.code && (
              <div className="border-t border-slate-100 px-3 py-2 text-2xs text-slate-500">
                <p className="mb-1">
                  <b>정의:</b> {c.code === '0' ? '생물다양성 목적과 무관한 사업' : `${c.name} 관련 활동을 주된 목적으로 하는 사업`}
                </p>
                <p className="mb-1">
                  <b>포함 기준:</b> 사업설명자료의 목적·세부내용에 해당 활동이 명시적으로 기술된 경우
                </p>
                <p className="mb-2">
                  <b>제외 기준:</b> 시설·행정 목적이 주이고 생물다양성 활동은 부수적으로만 언급된 경우
                </p>
                {c.code !== '0' && (
                  <>
                    <div className="mb-1 font-medium text-slate-600">하위 코드</div>
                    <ul className="grid grid-cols-2 gap-x-3 gap-y-0.5">
                      {SUB_CATEGORIES.filter((s) => s.topCode === c.code).map((s) => (
                        <li key={s.code}>
                          {s.code} {s.name}
                        </li>
                      ))}
                    </ul>
                  </>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function RunsTab() {
  const runs = useAppStore((s) => s.runs)
  const navigate = useNavigate()
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="mb-2 text-sm font-semibold text-navy-900">실행 이력</h2>
      <table className="w-full text-xs">
        <thead className="text-2xs text-slate-400">
          <tr>
            <th className="border-b border-slate-200 px-2 py-1.5 text-left">실행명</th>
            <th className="border-b border-slate-200 px-2 py-1.5 text-left">데이터 버전</th>
            <th className="border-b border-slate-200 px-2 py-1.5 text-left">모델 버전</th>
            <th className="border-b border-slate-200 px-2 py-1.5 text-left">상태</th>
            <th className="border-b border-slate-200 px-2 py-1.5 text-left">생성일시</th>
            <th className="border-b border-slate-200 px-2 py-1.5"></th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id} className="border-b border-slate-100">
              <td className="px-2 py-1.5 text-slate-700">{r.name}</td>
              <td className="px-2 py-1.5 text-slate-500">{r.datasetName}</td>
              <td className="px-2 py-1.5 text-slate-500">{r.modelLabel}</td>
              <td className="px-2 py-1.5 text-slate-500">{r.status}</td>
              <td className="px-2 py-1.5 text-slate-400">{formatDateTime(r.createdAt)}</td>
              <td className="px-2 py-1.5 text-right">
                <button className="text-teal-700 hover:underline" onClick={() => navigate(`/results?run=${r.id}`)}>
                  결과 열기
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function AssigneeTab() {
  const programs = useAppStore((s) => s.programs)
  const reviewers = useAppStore((s) => s.reviewers)

  const stats = useMemo(
    () =>
      reviewers.map((r) => {
        const assigned = programs.filter((p) => p.assignee === r.name)
        return {
          ...r,
          assignedCount: assigned.length,
          completedCount: assigned.filter((p) => p.expertReview.status === '검토완료').length,
          pendingCount: assigned.filter((p) => p.expertReview.status === '검토대기').length,
        }
      }),
    [programs, reviewers],
  )

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="mb-2 text-sm font-semibold text-navy-900">담당자 관리 (가상 배정 현황)</h2>
      <table className="w-full text-xs">
        <thead className="text-2xs text-slate-400">
          <tr>
            <th className="border-b border-slate-200 px-2 py-1.5 text-left">담당자</th>
            <th className="border-b border-slate-200 px-2 py-1.5 text-left">소속팀</th>
            <th className="border-b border-slate-200 px-2 py-1.5 text-right">배정 건수</th>
            <th className="border-b border-slate-200 px-2 py-1.5 text-right">검토 완료</th>
            <th className="border-b border-slate-200 px-2 py-1.5 text-right">검토 대기</th>
          </tr>
        </thead>
        <tbody>
          {stats.map((r) => (
            <tr key={r.id} className="border-b border-slate-100">
              <td className="px-2 py-1.5 font-medium text-slate-700">{r.name}</td>
              <td className="px-2 py-1.5 text-slate-500">{r.team}</td>
              <td className="px-2 py-1.5 text-right tabular-nums text-slate-500">{r.assignedCount}</td>
              <td className="px-2 py-1.5 text-right tabular-nums text-teal-700">{r.completedCount}</td>
              <td className="px-2 py-1.5 text-right tabular-nums text-amber-700">{r.pendingCount}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
