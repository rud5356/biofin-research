import { Outlet } from 'react-router-dom'
import { DemoBanner } from './DemoBanner'
import { Sidebar } from './Sidebar'

export function AppLayout() {
  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <DemoBanner />
      <div className="flex min-h-0 flex-1">
        <Sidebar />
        <main className="min-w-0 flex-1 overflow-y-auto bg-page">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
