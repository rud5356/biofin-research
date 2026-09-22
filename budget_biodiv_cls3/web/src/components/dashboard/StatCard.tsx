import { useNavigate } from 'react-router-dom'

export function StatCard({
  label,
  value,
  sub,
  tone = 'default',
  linkTo,
}: {
  label: string
  value: string
  sub?: string
  tone?: 'default' | 'warning' | 'critical' | 'good'
  linkTo?: string
}) {
  const navigate = useNavigate()
  const toneCls =
    tone === 'warning'
      ? 'text-amber-700'
      : tone === 'critical'
        ? 'text-red-600'
        : tone === 'good'
          ? 'text-teal-700'
          : 'text-navy-900'

  return (
    <button
      onClick={() => linkTo && navigate(linkTo)}
      disabled={!linkTo}
      className={`flex flex-col items-start gap-1 rounded-lg border border-slate-200 bg-white p-3.5 text-left transition-shadow ${
        linkTo ? 'cursor-pointer hover:shadow-md' : 'cursor-default'
      }`}
    >
      <span className="text-2xs font-medium text-slate-500">{label}</span>
      <span className={`text-xl font-semibold tabular-nums ${toneCls}`}>{value}</span>
      {sub && <span className="text-2xs text-slate-400">{sub}</span>}
    </button>
  )
}
