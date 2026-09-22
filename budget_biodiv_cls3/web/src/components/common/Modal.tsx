import type { ReactNode } from 'react'

export function Modal({ title, onClose, children, width = 'max-w-lg' }: { title: string; onClose: () => void; children: ReactNode; width?: string }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4" onClick={onClose}>
      <div
        className={`w-full ${width} overflow-y-auto rounded-lg bg-white shadow-xl`}
        style={{ maxHeight: '85vh' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-slate-200 px-5 py-3.5">
          <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
          <button onClick={onClose} className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600" aria-label="닫기">
            ✕
          </button>
        </div>
        <div className="p-5">{children}</div>
      </div>
    </div>
  )
}

export function EmptyState({ title, description }: { title: string; description?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-1.5 py-16 text-center">
      <div className="text-2xl">🔍</div>
      <div className="text-sm font-medium text-slate-700">{title}</div>
      {description && <div className="text-xs text-slate-500">{description}</div>}
    </div>
  )
}

export function LoadingState({ label = '불러오는 중입니다…' }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-16 text-center">
      <div className="h-5 w-5 animate-spin rounded-full border-2 border-slate-300 border-t-navy-700" />
      <div className="text-xs text-slate-500">{label}</div>
    </div>
  )
}
