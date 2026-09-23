// 실제 데이터 연결용 1회성 변환 스크립트.
// 울산_환경_2024 모델 카테고리 분류 결과.CSV(CP949, Transformer+LLM 예측 포함)를
// 프런트엔드에서 그대로 import할 수 있는 JSON(BudgetProgram 형태)으로 변환합니다.
// 모델·학습 코드는 건드리지 않고, 이미 생성된 결과 CSV만 읽습니다.
import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const SRC_CSV = path.resolve(__dirname, '../../울산_환경_2024 모델 카테고리 분류 결과.CSV')
const OUT_JSON = path.resolve(__dirname, '../src/data/realUlsan2024.json')

function parseCsv(text) {
  const rows = []
  let row = []
  let field = ''
  let inQuotes = false
  let i = 0
  const n = text.length
  while (i < n) {
    const c = text[i]
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"'
          i += 2
          continue
        }
        inQuotes = false
        i++
        continue
      }
      field += c
      i++
      continue
    }
    if (c === '"') {
      inQuotes = true
      i++
      continue
    }
    if (c === ',') {
      row.push(field)
      field = ''
      i++
      continue
    }
    if (c === '\r') {
      i++
      continue
    }
    if (c === '\n') {
      row.push(field)
      rows.push(row)
      row = []
      field = ''
      i++
      continue
    }
    field += c
    i++
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field)
    rows.push(row)
  }
  return rows
}

const buf = readFileSync(SRC_CSV)
const text = new TextDecoder('euc-kr').decode(buf)
const rows = parseCsv(text).filter((r) => r.length > 1)
const header = rows[0]
const idx = (name) => header.indexOf(name)

const col = {
  year: idx('회계연도'),
  ministry: idx('소관명'),
  accountName: idx('회계명'),
  field: idx('분야명'),
  programName: idx('세부사업명'),
  budget: idx('예산액'),
  businessKey: idx('business_key'),
  docFile: idx('사업설명자료_파일명'),
  docStatus: idx('다운로드상태'),
  national: idx('국비'),
  metro: idx('시도비'),
  local: idx('시군구비'),
  other: idx('기타재원'),
  expenditure: idx('지출액'),
  balance: idx('잔액'),
  transformerTop: idx('Transformer BIOFIN 카테고리'),
  llmTop: idx('LLM BIOFIN 1차 카테고리'),
  llmConfidence: idx('LLM 신뢰도'),
  llmReason: idx('LLM 분류사유'),
  llmEvidence: idx('LLM 근거'),
}

function toInt(v) {
  if (v === undefined || v === null || v === '') return null
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}
function toTopCode(v) {
  if (v === undefined || v === '' || v === null) return null
  const n = Number(v)
  if (!Number.isFinite(n) || n < 0 || n > 9) return null
  return String(n)
}

const docStatusCounts = {}

const programs = rows.slice(1).map((r, i) => {
  const docStatusRaw = r[col.docStatus]
  docStatusCounts[docStatusRaw] = (docStatusCounts[docStatusRaw] ?? 0) + 1
  const hasDoc = docStatusRaw === 'success'
  const documentStatus = hasDoc ? '연결됨' : docStatusRaw === 'no_document' ? '없음' : '추출실패'

  const budgetAmount = toInt(r[col.budget])
  const national = toInt(r[col.national]) ?? 0
  const metro = toInt(r[col.metro]) ?? 0
  const local = toInt(r[col.local]) ?? 0
  const other = toInt(r[col.other]) ?? 0

  const transformerTop = toTopCode(r[col.transformerTop])
  const llmTop = toTopCode(r[col.llmTop])
  const llmConfidence = toInt(r[col.llmConfidence] === '' ? null : r[col.llmConfidence]) !== null ? Number(r[col.llmConfidence]) : null

  return {
    id: r[col.businessKey] || `ULSAN-2024-${i}`,
    year: toInt(r[col.year]) ?? 2024,
    ministry: r[col.ministry],
    field: r[col.field],
    programName: r[col.programName],
    budgetAmount,
    expenditureAmount: toInt(r[col.expenditure]),
    balanceAmount: toInt(r[col.balance]),
    fundSource:
      budgetAmount !== null
        ? [
            { label: '국비', amount: national },
            { label: '시도비', amount: metro },
            { label: '시군구비', amount: local },
            { label: '기타재원', amount: other },
          ]
        : null,
    documentStatus,
    documentPreview: hasDoc && r[col.docFile] ? `사업설명자료 파일: ${r[col.docFile]}` : null,
    accountName: r[col.accountName],
    transformer_v1:
      transformerTop === null
        ? null
        : {
            modelLabel: 'Transformer v1',
            topCategory: transformerTop,
            subCategory: null,
            confidence: null,
            confidenceType: null,
            reasoning: null,
            evidenceSentences: [],
            attentionSpans: [],
            inputBasis: hasDoc ? 'document' : 'budget_only',
            purposeExtractionFailed: false,
          },
    llm_v1:
      llmTop === null
        ? null
        : {
            modelLabel: 'LLM v1',
            topCategory: llmTop,
            subCategory: null,
            confidence: llmConfidence,
            confidenceType: llmConfidence === null ? null : 'self_reported',
            reasoning: r[col.llmReason] || null,
            evidenceSentences: r[col.llmEvidence] ? [r[col.llmEvidence]] : [],
            attentionSpans: [],
            inputBasis: hasDoc ? 'document' : 'budget_only',
            purposeExtractionFailed: false,
          },
  }
})

writeFileSync(OUT_JSON, JSON.stringify(programs, null, 0))
console.log('rows:', programs.length)
console.log('docStatusCounts:', docStatusCounts)
console.log('written to', OUT_JSON)
