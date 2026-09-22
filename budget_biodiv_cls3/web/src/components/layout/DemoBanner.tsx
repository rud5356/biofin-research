import { useState } from 'react'
import { useAppStore } from '../../store/useAppStore'

export function DemoBanner() {
  const resetDemo = useAppStore((s) => s.resetDemo)
  const [confirming, setConfirming] = useState(false)

  return (
    <div className="flex h-8 shrink-0 items-center justify-between bg-navy-950 px-4 text-2xs text-white">
      <div className="flex items-center gap-2">
        <span className="rounded bg-teal-500/90 px-1.5 py-0.5 font-semibold tracking-wide">DEMO</span>
        <span className="text-white/85">데모 · 실제 모델 미연결 — 모든 예측·평가 수치는 가상 데이터입니다.</span>
      </div>
      {!confirming ? (
        <button className="rounded px-2 py-0.5 text-white/70 hover:bg-white/10 hover:text-white" onClick={() => setConfirming(true)}>
          데모 상태 초기화
        </button>
      ) : (
        <div className="flex items-center gap-2">
          <span className="text-white/80">검토 이력을 포함한 모든 데모 상태를 초기화할까요?</span>
          <button
            className="rounded bg-red-500/90 px-2 py-0.5 font-medium hover:bg-red-500"
            onClick={() => {
              resetDemo()
              setConfirming(false)
            }}
          >
            초기화
          </button>
          <button className="rounded px-2 py-0.5 text-white/70 hover:bg-white/10" onClick={() => setConfirming(false)}>
            취소
          </button>
        </div>
      )}
    </div>
  )
}
