import { NavLink } from 'react-router-dom'

const NAV_ITEMS = [
  { to: '/dashboard', icon: '📊', label: '현황 대시보드' },
  { to: '/results', icon: '✅', label: '분류 결과·전문가 검증' },
  { to: '/predict', icon: '⚙️', label: '예측 실행' },
  { to: '/data', icon: '🗂️', label: '데이터·문서 관리' },
  { to: '/compare', icon: '📈', label: '모델 비교·평가' },
  { to: '/admin', icon: '🧪', label: '연구·관리' },
]

export function Sidebar() {
  return (
    <aside className="flex w-56 shrink-0 flex-col bg-navy-900 text-white">
      <div className="flex h-14 items-center gap-2 border-b border-white/10 px-4">
        <div className="flex h-7 w-7 items-center justify-center rounded bg-teal-500 text-sm font-bold">B</div>
        <div className="leading-tight">
          <div className="text-sm font-semibold">BIOFIN 분류·검증</div>
          <div className="text-3xs text-white/50">예산사업 전문가 검증 시스템</div>
        </div>
      </div>
      <nav className="flex-1 space-y-0.5 p-2">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `flex items-center gap-2.5 rounded-md px-3 py-2 text-xs font-medium transition-colors ${
                isActive ? 'bg-teal-600 text-white' : 'text-white/70 hover:bg-white/10 hover:text-white'
              }`
            }
          >
            <span className="text-sm">{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
      </nav>
      <div className="border-t border-white/10 p-3 text-3xs leading-relaxed text-white/40">
        BIOFIN 예산사업 분류 프로토타입
        <br />
        budget_biodiv_cls3 · UI 데모
      </div>
    </aside>
  )
}
