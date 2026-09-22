import { Navigate, Route, Routes } from 'react-router-dom'
import { AppLayout } from './components/layout/AppLayout'
import { DashboardPage } from './pages/DashboardPage'
import { ResultsPage } from './pages/results/ResultsPage'
import { PredictPage } from './pages/PredictPage'
import { DataManagementPage } from './pages/DataManagementPage'
import { CompareEvalPage } from './pages/CompareEvalPage'
import { ResearchAdminPage } from './pages/ResearchAdminPage'

function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Navigate to="/results" replace />} />
        <Route path="dashboard" element={<DashboardPage />} />
        <Route path="results" element={<ResultsPage />} />
        <Route path="predict" element={<PredictPage />} />
        <Route path="data" element={<DataManagementPage />} />
        <Route path="compare" element={<CompareEvalPage />} />
        <Route path="admin" element={<ResearchAdminPage />} />
        <Route path="*" element={<Navigate to="/results" replace />} />
      </Route>
    </Routes>
  )
}

export default App
