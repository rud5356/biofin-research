import { useMemo, useState } from 'react'
import { useAppStore } from '../store/useAppStore'
import { formatDate } from '../lib/format'
import { Button } from '../components/common/Button'
import { DocumentStatusBadge } from '../components/common/Badge'
import { LoadingState } from '../components/common/Modal'

const REQUIRED_FIELDS = ['회계연도', '소관명', '사업명', '예산액', '사업 ID', '정답 상위 카테고리', '정답 하위 카테고리'] as const

export function DataManagementPage() {
  const datasets = useAppStore((s) => s.datasets)
  const programs = useAppStore((s) => s.programs)
  const linkDocument = useAppStore((s) => s.linkDocument)
  const addDataset = useAppStore((s) => s.addDataset)
  const realDataLoaded = useAppStore((s) => s.realDataLoaded)
  const loadRealDataset = useAppStore((s) => s.loadRealDataset)

  const [encoding, setEncoding] = useState<'UTF-8' | 'CP949'>('UTF-8')
  const [csvHeaders, setCsvHeaders] = useState<string[]>([])
  const [csvRows, setCsvRows] = useState<string[][]>([])
  const [fileName, setFileName] = useState<string | null>(null)
  const [mapping, setMapping] = useState<Record<string, string>>({})
  const [uploaded, setUploaded] = useState(false)
  const [isParsing, setIsParsing] = useState(false)
  const [parseError, setParseError] = useState<string | null>(null)
  const [isLoadingReal, setIsLoadingReal] = useState(false)

  async function handleFile(file: File) {
    setFileName(file.name)
    setUploaded(false)
    setParseError(null)
    setCsvHeaders([])
    setCsvRows([])
    setIsParsing(true)
    try {
      let text: string
      try {
        const buf = await file.arrayBuffer()
        text = new TextDecoder(encoding === 'CP949' ? 'euc-kr' : 'utf-8').decode(buf)
      } catch {
        text = await file.text()
      }
      const lines = text.split(/\r\n|\n/).filter((l) => l.trim().length > 0).slice(0, 21)
      if (lines.length < 2) {
        setParseError('헤더와 데이터 행을 확인할 수 없습니다. 올바른 CSV 파일인지, 인코딩 선택이 맞는지 확인해 주세요.')
        return
      }
      const parsed = lines.map((l) => l.split(','))
      setCsvHeaders(parsed[0] ?? [])
      setCsvRows(parsed.slice(1))
      const initialMap: Record<string, string> = {}
      REQUIRED_FIELDS.forEach((f) => {
        initialMap[f] = ''
      })
      setMapping(initialMap)
    } catch {
      setParseError('파일을 읽는 중 오류가 발생했습니다. 다른 파일로 다시 시도해 주세요.')
    } finally {
      setIsParsing(false)
    }
  }

  function handleLoadReal() {
    setIsLoadingReal(true)
    window.setTimeout(() => {
      loadRealDataset()
      setIsLoadingReal(false)
    }, 600)
  }

  const primaryCol = mapping['정답 상위 카테고리']
  const subCol = mapping['정답 하위 카테고리']
  const primaryIdx = csvHeaders.indexOf(primaryCol)
  const subIdx = csvHeaders.indexOf(subCol)

  const issues = useMemo(() => {
    const amountCol = mapping['예산액']
    const amountIdx = csvHeaders.indexOf(amountCol)
    const idCol = mapping['사업 ID']
    const idIdx = csvHeaders.indexOf(idCol)
    const seen = new Set<string>()
    const dupRows: number[] = []
    const missingRows: number[] = []
    const amountFormatRows: number[] = []
    csvRows.forEach((row, i) => {
      if (Object.values(mapping).some((col) => col && row[csvHeaders.indexOf(col)] === undefined)) missingRows.push(i)
      if (amountIdx >= 0 && row[amountIdx] && !/^[0-9,]+$/.test(row[amountIdx].trim())) amountFormatRows.push(i)
      if (idIdx >= 0 && row[idIdx]) {
        if (seen.has(row[idIdx])) dupRows.push(i)
        seen.add(row[idIdx])
      }
    })
    return { missingRows, amountFormatRows, dupRows }
  }, [csvRows, csvHeaders, mapping])

  function handleRegister() {
    if (!fileName) return
    addDataset({
      id: `ds-${Date.now()}`,
      name: fileName,
      rowCount: csvRows.length,
      year: new Date().getFullYear(),
      registeredAt: new Date().toISOString(),
      amountUnit: '원',
      encoding,
    })
    setUploaded(true)
  }

  const docGroups = useMemo(() => {
    return {
      연결됨: programs.filter((p) => p.documentStatus === '연결됨'),
      없음: programs.filter((p) => p.documentStatus === '없음'),
      추출실패: programs.filter((p) => p.documentStatus === '추출실패'),
    }
  }, [programs])

  return (
    <div className="space-y-4 p-5">
      <div>
        <h1 className="text-base font-semibold text-navy-900">데이터·문서 관리</h1>
        <p className="mt-0.5 text-xs text-slate-500">등록 데이터와 사업설명자료 연결 상태를 관리합니다. 업로드는 데모이며 서버에 저장되지 않습니다.</p>
      </div>

      <div className="rounded-lg border border-teal-500/30 bg-teal-50/50 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-navy-900">실제 데이터 — 울산광역시 환경분야 2024</h2>
            <p className="mt-0.5 text-2xs text-slate-500">
              crawlers/open_fiscal로 수집한 실제 세부사업 355건 + Transformer v1·LLM v1 실제 예측 결과. 아직 전문가 검토 전 상태로 불러와
              분류결과·전문가 검증 화면에서 실제로 검토할 수 있습니다.
            </p>
          </div>
          {realDataLoaded ? (
            <span className="rounded border border-teal-500/40 bg-white px-3 py-1.5 text-xs font-medium text-teal-700">불러오기 완료 (355건 반영됨)</span>
          ) : isLoadingReal ? (
            <span className="flex items-center gap-2 rounded border border-teal-500/40 bg-white px-3 py-1.5 text-xs font-medium text-teal-700">
              <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-teal-200 border-t-teal-600" />
              불러오는 중…
            </span>
          ) : (
            <Button variant="secondary" onClick={handleLoadReal}>
              실제 데이터 불러오기 →
            </Button>
          )}
        </div>
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="mb-2 text-sm font-semibold text-navy-900">등록 데이터 목록</h2>
        <table className="w-full text-xs">
          <thead className="text-2xs text-slate-400">
            <tr>
              <th className="border-b border-slate-200 px-2 py-1.5 text-left">파일명</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-right">행 수</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-left">연도</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-left">등록일</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-left">금액 단위</th>
              <th className="border-b border-slate-200 px-2 py-1.5 text-left">인코딩</th>
            </tr>
          </thead>
          <tbody>
            {datasets.map((d) => (
              <tr key={d.id} className="border-b border-slate-100">
                <td className="px-2 py-1.5 text-slate-700">{d.name}</td>
                <td className="px-2 py-1.5 text-right tabular-nums text-slate-500">{d.rowCount.toLocaleString('ko-KR')}</td>
                <td className="px-2 py-1.5 text-slate-500">{d.year}</td>
                <td className="px-2 py-1.5 text-slate-400">{formatDate(d.registeredAt)}</td>
                <td className="px-2 py-1.5 text-slate-500">{d.amountUnit}</td>
                <td className="px-2 py-1.5 text-slate-500">{d.encoding}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="mb-2 text-sm font-semibold text-navy-900">가상 CSV 업로드 · 컬럼 매핑</h2>
        <div className="flex flex-wrap items-center gap-3">
          <input
            type="file"
            accept=".csv"
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            className="text-xs"
          />
          <div className="flex items-center gap-1 rounded-md border border-slate-300 bg-white p-0.5 text-2xs">
            {(['UTF-8', 'CP949'] as const).map((e) => (
              <button key={e} onClick={() => setEncoding(e)} className={`rounded px-2 py-1 font-medium ${encoding === e ? 'bg-navy-900 text-white' : 'text-slate-600'}`}>
                {e}
              </button>
            ))}
          </div>
        </div>

        {isParsing && <LoadingState label="CSV 파일을 읽는 중입니다…" />}

        {parseError && (
          <div className="mt-3 rounded border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">{parseError}</div>
        )}

        {!isParsing && csvHeaders.length > 0 && (
          <div className="mt-4 space-y-4">
            <div>
              <h3 className="mb-1.5 text-xs font-semibold text-slate-600">컬럼 매핑</h3>
              <div className="grid grid-cols-2 gap-2 md:grid-cols-3">
                {REQUIRED_FIELDS.map((field) => (
                  <div key={field}>
                    <label className="mb-0.5 block text-2xs text-slate-500">{field}</label>
                    <select
                      className="h-8 w-full rounded border border-slate-300 bg-white px-2 text-xs"
                      value={mapping[field] ?? ''}
                      onChange={(e) => setMapping((m) => ({ ...m, [field]: e.target.value }))}
                    >
                      <option value="">매핑 안 함</option>
                      {csvHeaders.map((h) => (
                        <option key={h} value={h}>
                          {h}
                        </option>
                      ))}
                    </select>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h3 className="mb-1.5 text-xs font-semibold text-slate-600">원본 행 미리보기 ({fileName})</h3>
              <div className="max-h-60 overflow-auto rounded border border-slate-200">
                <table className="w-full text-2xs">
                  <thead className="bg-slate-50 text-slate-500">
                    <tr>
                      {csvHeaders.map((h) => (
                        <th key={h} className="border-b border-slate-200 px-2 py-1 text-left whitespace-nowrap">
                          {h}
                        </th>
                      ))}
                      <th className="border-b border-slate-200 px-2 py-1 text-left">결합 코드 미리보기</th>
                    </tr>
                  </thead>
                  <tbody>
                    {csvRows.slice(0, 10).map((row, i) => (
                      <tr key={i} className="border-b border-slate-100">
                        {csvHeaders.map((_, j) => (
                          <td key={j} className="px-2 py-1 whitespace-nowrap text-slate-600">
                            {row[j] ?? ''}
                          </td>
                        ))}
                        <td className="px-2 py-1 font-medium text-teal-700">
                          {primaryIdx >= 0 && subIdx >= 0 && row[primaryIdx] && row[subIdx] ? `${row[primaryIdx]}.${String(row[subIdx]).padStart(2, '0')}` : '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="flex flex-wrap gap-3 text-2xs">
              <span className="rounded border border-amber-200 bg-amber-50 px-2 py-1 text-amber-700">필수값 누락 {issues.missingRows.length}행</span>
              <span className="rounded border border-red-200 bg-red-50 px-2 py-1 text-red-700">금액 형식 오류 {issues.amountFormatRows.length}행</span>
              <span className="rounded border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600">중복 후보 {issues.dupRows.length}행</span>
            </div>

            <Button variant="primary" onClick={handleRegister}>
              변환 결과로 등록(데모)
            </Button>
            {uploaded && <span className="ml-2 text-xs text-teal-700">등록되었습니다. 원본 데이터와 변환 결과는 별도로 관리됩니다.</span>}
          </div>
        )}
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="mb-2 text-sm font-semibold text-navy-900">사업설명자료 연결 현황</h2>
        <div className="mb-3 flex gap-3 text-2xs">
          <span>연결됨 {docGroups.연결됨.length}건</span>
          <span>없음 {docGroups.없음.length}건</span>
          <span>추출실패 {docGroups.추출실패.length}건</span>
        </div>
        <div className="max-h-72 overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-white text-2xs text-slate-400">
              <tr>
                <th className="border-b border-slate-200 px-2 py-1.5 text-left">사업명</th>
                <th className="border-b border-slate-200 px-2 py-1.5 text-left">문서 상태</th>
                <th className="border-b border-slate-200 px-2 py-1.5 text-left">사용 범위</th>
                <th className="border-b border-slate-200 px-2 py-1.5"></th>
              </tr>
            </thead>
            <tbody>
              {programs
                .filter((p) => p.documentStatus !== '연결됨')
                .map((p) => (
                  <tr key={p.id} className="border-b border-slate-100">
                    <td className="px-2 py-1.5 text-slate-700">{p.programName}</td>
                    <td className="px-2 py-1.5">
                      <DocumentStatusBadge status={p.documentStatus} />
                    </td>
                    <td className="px-2 py-1.5 text-slate-400">{p.documentStatus === '없음' ? '예산정보만 사용' : '예산정보만 사용(추출 실패)'}</td>
                    <td className="px-2 py-1.5 text-right">
                      <Button size="sm" variant="outline" onClick={() => linkDocument(p.id)}>
                        수동 문서 연결(데모)
                      </Button>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
