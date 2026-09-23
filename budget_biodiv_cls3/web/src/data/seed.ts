import type {
  BudgetProgram,
  DocumentStatus,
  ExpertReview,
  ModelPrediction,
  PredictionRun,
  RegisteredDataset,
  Reviewer,
  ReviewHistoryEntry,
  TopCode,
} from '../types'
import { subCategoryName, topCategoryName } from './categories'

// 결정론적 의사난수 — 새로고침/초기화 시에도 동일한 데모 데이터를 재현합니다.
function mulberry32(seed: number) {
  let a = seed
  return function () {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
const rng = mulberry32(20260922)

export const REVIEWERS: Reviewer[] = [
  { id: 'rv-01', name: '김도윤', team: '분류검증1팀', assignedCount: 0, completedCount: 0 },
  { id: 'rv-02', name: '이서연', team: '분류검증1팀', assignedCount: 0, completedCount: 0 },
  { id: 'rv-03', name: '박지훈', team: '분류검증2팀', assignedCount: 0, completedCount: 0 },
  { id: 'rv-04', name: '최유나', team: '분류검증2팀', assignedCount: 0, completedCount: 0 },
  { id: 'rv-05', name: '정민재', team: '분류검증1팀', assignedCount: 0, completedCount: 0 },
]

export const RUN_TV1 = 'run-transformer_v1-2609'
export const RUN_TV2 = 'run-transformer_v2-2609'
export const RUN_LLM1 = 'run-llm_v1-2609'
export const RUN_LLM2 = 'run-llm_v2-2609'
export const RUN_PIPE = 'run-pipeline-2609'

type MatchType = 'agree' | 'mismatch'
type ReviewOutcome =
  | 'approved'
  | 'pending'
  | 'corrected'
  | 'hold'
  | 'redo'
  | 'manual_unclassified'
  | 'pending_unclassified'
type DocCase = 'normal' | 'no_doc' | 'extract_fail' | 'budget_only_with_doc'

interface RowConfig {
  id: string
  year: number
  ministry: string
  field: string
  programName: string
  budget: number
  expenditure: number | null
  matchType: MatchType | 'none'
  reviewOutcome: ReviewOutcome
  top: TopCode | null
  sub: string | null
  altTop?: TopCode
  altSub?: string
  docCase: DocCase
  reviewerIdx: number | null
}

const ROWS: RowConfig[] = [
  { id: 'BP-2024-0001', year: 2024, ministry: '환경부', field: '환경', programName: '멸종위기 야생생물 보전대책 추진', budget: 8_450_000_000, expenditure: 3_120_000_000, matchType: 'agree', reviewOutcome: 'pending', top: '1', sub: '1.01', docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2023-0002', year: 2023, ministry: '환경부', field: '환경', programName: '생태계교란 생물 방제사업', budget: 1_230_000_000, expenditure: 1_180_000_000, matchType: 'agree', reviewOutcome: 'approved', top: '1', sub: '1.02', docCase: 'normal', reviewerIdx: 0 },
  { id: 'BP-2024-0003', year: 2024, ministry: '환경부', field: '환경', programName: '도시생태축 복원사업', budget: 640_000_000, expenditure: 210_000_000, matchType: 'agree', reviewOutcome: 'pending', top: '2', sub: '2.03', docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2022-0004', year: 2022, ministry: '환경부', field: '환경', programName: '습지보호지역 관리 및 복원', budget: 2_050_000_000, expenditure: 2_010_000_000, matchType: 'mismatch', reviewOutcome: 'corrected', top: '3', sub: '3.02', altTop: '2', altSub: '2.02', docCase: 'normal', reviewerIdx: 1 },
  { id: 'BP-2023-0005', year: 2023, ministry: '환경부', field: '환경', programName: '국립공원 자연생태계 정밀조사', budget: 980_000_000, expenditure: 950_000_000, matchType: 'agree', reviewOutcome: 'approved', top: '5', sub: '5.02', docCase: 'normal', reviewerIdx: 2 },
  { id: 'BP-2024-0006', year: 2024, ministry: '환경부', field: '환경', programName: '기후변화 생물다양성 영향평가 연구', budget: 6_700_000_000, expenditure: 1_500_000_000, matchType: 'agree', reviewOutcome: 'pending', top: '5', sub: '5.05', docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2023-0007', year: 2023, ministry: '환경부', field: '환경', programName: '자연환경 국민체험교육 확대', budget: 410_000_000, expenditure: 380_000_000, matchType: 'mismatch', reviewOutcome: 'hold', top: '7', sub: '7.04', altTop: '7', altSub: '7.02', docCase: 'normal', reviewerIdx: 3 },
  { id: 'BP-2024-0008', year: 2024, ministry: '환경부', field: '환경', programName: '생물다양성 통계 인프라 구축', budget: 320_000_000, expenditure: 90_000_000, matchType: 'agree', reviewOutcome: 'pending', top: '9', sub: '9.02', docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2022-0009', year: 2022, ministry: '환경부', field: '환경', programName: '외래생물 유입 관리체계 구축', budget: 560_000_000, expenditure: 540_000_000, matchType: 'mismatch', reviewOutcome: 'pending', top: '1', sub: '1.02', altTop: '3', altSub: '3.03', docCase: 'budget_only_with_doc', reviewerIdx: null },
  { id: 'BP-2023-0010', year: 2023, ministry: '환경부', field: '환경', programName: '야생동물 질병관리 시스템 운영', budget: 275_000_000, expenditure: null, matchType: 'agree', reviewOutcome: 'redo', top: '1', sub: '1.04', docCase: 'normal', reviewerIdx: 4 },

  { id: 'BP-2024-0011', year: 2024, ministry: '해양수산부', field: '해양수산', programName: '해양보호구역 지정 및 관리', budget: 3_400_000_000, expenditure: 3_300_000_000, matchType: 'agree', reviewOutcome: 'approved', top: '3', sub: '3.04', docCase: 'normal', reviewerIdx: 0 },
  { id: 'BP-2023-0012', year: 2023, ministry: '해양수산부', field: '해양수산', programName: '연안습지 복원사업', budget: 7_800_000_000, expenditure: 2_200_000_000, matchType: 'agree', reviewOutcome: 'pending', top: '2', sub: '2.02', docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2022-0013', year: 2022, ministry: '해양수산부', field: '해양수산', programName: '해양생태계 기본조사', budget: 1_150_000_000, expenditure: 1_100_000_000, matchType: 'agree', reviewOutcome: 'pending', top: '5', sub: '5.01', docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2024-0014', year: 2024, ministry: '해양수산부', field: '해양수산', programName: '수산자원 지속가능이용 기술개발', budget: 1_890_000_000, expenditure: 700_000_000, matchType: 'mismatch', reviewOutcome: 'pending', top: '4', sub: '4.02', altTop: '9', altSub: '9.01', docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2023-0015', year: 2023, ministry: '해양수산부', field: '해양수산', programName: '해양생물 유전자원 정보구축', budget: 890_000_000, expenditure: 860_000_000, matchType: 'mismatch', reviewOutcome: 'corrected', top: '5', sub: '5.03', altTop: '6', altSub: '6.02', docCase: 'normal', reviewerIdx: 1 },

  { id: 'BP-2024-0016', year: 2024, ministry: '산림청', field: '산림', programName: '산림생태복원사업', budget: 2_600_000_000, expenditure: 900_000_000, matchType: 'agree', reviewOutcome: 'approved', top: '2', sub: '2.04', docCase: 'normal', reviewerIdx: 2 },
  { id: 'BP-2022-0017', year: 2022, ministry: '산림청', field: '산림', programName: '산림유전자원보호구역 관리', budget: 470_000_000, expenditure: 460_000_000, matchType: 'mismatch', reviewOutcome: 'pending', top: '3', sub: '3.01', altTop: '6', altSub: '6.02', docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2023-0018', year: 2023, ministry: '산림청', field: '산림', programName: '도시숲 생태축 조성', budget: 350_000_000, expenditure: 340_000_000, matchType: 'agree', reviewOutcome: 'pending', top: '2', sub: '2.03', docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2024-0019', year: 2024, ministry: '산림청', field: '산림', programName: '산림생물다양성 모니터링', budget: 5_200_000_000, expenditure: 1_100_000_000, matchType: 'agree', reviewOutcome: 'pending', top: '5', sub: '5.02', docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2023-0020', year: 2023, ministry: '산림청', field: '산림', programName: '산불피해지 생태복원', budget: 1_320_000_000, expenditure: 400_000_000, matchType: 'mismatch', reviewOutcome: 'pending', top: '2', sub: '2.01', altTop: '3', altSub: '3.01', docCase: 'extract_fail', reviewerIdx: null },

  { id: 'BP-2022-0021', year: 2022, ministry: '농림축산식품부', field: '농업', programName: '농업생물다양성 보전사업', budget: 940_000_000, expenditure: 900_000_000, matchType: 'agree', reviewOutcome: 'approved', top: '1', sub: '1.03', docCase: 'normal', reviewerIdx: 3 },
  { id: 'BP-2023-0022', year: 2023, ministry: '농림축산식품부', field: '농업', programName: '전통 농업유산 보전 및 활용', budget: 610_000_000, expenditure: 590_000_000, matchType: 'agree', reviewOutcome: 'pending', top: '4', sub: '4.04', docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2024-0023', year: 2024, ministry: '농림축산식품부', field: '농업', programName: '친환경 농업생태계 서비스 지불제', budget: 4_100_000_000, expenditure: 1_800_000_000, matchType: 'mismatch', reviewOutcome: 'pending', top: '4', sub: '4.01', altTop: '9', altSub: '9.01', docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2023-0024', year: 2023, ministry: '농림축산식품부', field: '농업', programName: '농촌지역 생태관광 활성화', budget: 380_000_000, expenditure: 360_000_000, matchType: 'agree', reviewOutcome: 'approved', top: '4', sub: '4.03', docCase: 'normal', reviewerIdx: 4 },

  { id: 'BP-2024-0025', year: 2024, ministry: '문화체육관광부', field: '문화관광', programName: '국립공원 생태관광 콘텐츠 개발', budget: 520_000_000, expenditure: 100_000_000, matchType: 'mismatch', reviewOutcome: 'hold', top: '4', sub: '4.03', altTop: '7', altSub: '7.05', docCase: 'normal', reviewerIdx: 0 },
  { id: 'BP-2023-0026', year: 2023, ministry: '문화체육관광부', field: '문화관광', programName: '생물다양성 홍보콘텐츠 제작', budget: 190_000_000, expenditure: 185_000_000, matchType: 'agree', reviewOutcome: 'pending', top: '7', sub: '7.05', docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2022-0027', year: 2022, ministry: '문화체육관광부', field: '문화관광', programName: '청소년 자연생태 체험교육', budget: 260_000_000, expenditure: 255_000_000, matchType: 'agree', reviewOutcome: 'approved', top: '7', sub: '7.04', docCase: 'normal', reviewerIdx: 1 },

  { id: 'BP-2024-0028', year: 2024, ministry: '국토교통부', field: '국토', programName: '도시개발지 생태면적률 관리', budget: 9_300_000_000, expenditure: 2_400_000_000, matchType: 'mismatch', reviewOutcome: 'pending', top: '2', sub: '2.03', altTop: '0', altSub: undefined, docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2023-0029', year: 2023, ministry: '국토교통부', field: '국토', programName: '하천생태계 복원사업', budget: 3_050_000_000, expenditure: 1_200_000_000, matchType: 'agree', reviewOutcome: 'pending', top: '2', sub: '2.01', docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2022-0030', year: 2022, ministry: '국토교통부', field: '국토', programName: '그린인프라 조성사업', budget: 1_680_000_000, expenditure: 1_650_000_000, matchType: 'agree', reviewOutcome: 'approved', top: '0', sub: null, docCase: 'normal', reviewerIdx: 2 },

  { id: 'BP-2024-0031', year: 2024, ministry: '행정안전부', field: '안전', programName: '지자체 재난안전 인프라 구축', budget: 12_400_000_000, expenditure: 5_100_000_000, matchType: 'agree', reviewOutcome: 'approved', top: '0', sub: null, docCase: 'normal', reviewerIdx: 3 },
  { id: 'BP-2023-0032', year: 2023, ministry: '행정안전부', field: '안전', programName: '지역사회 안전기반 조성', budget: 870_000_000, expenditure: 850_000_000, matchType: 'agree', reviewOutcome: 'approved', top: '0', sub: null, docCase: 'no_doc', reviewerIdx: 4 },
  { id: 'BP-2024-0033', year: 2024, ministry: '행정안전부', field: '안전', programName: '지자체 생물다양성 이행지원', budget: 430_000_000, expenditure: 120_000_000, matchType: 'agree', reviewOutcome: 'pending', top: '9', sub: '9.04', docCase: 'normal', reviewerIdx: null },

  { id: 'BP-2023-0034', year: 2023, ministry: '문화재청', field: '문화재', programName: '천연기념물 보호 및 복원', budget: 2_950_000_000, expenditure: 700_000_000, matchType: 'agree', reviewOutcome: 'pending', top: '1', sub: '1.01', docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2024-0035', year: 2024, ministry: '문화재청', field: '문화재', programName: '명승지 생태경관 보전', budget: 660_000_000, expenditure: 200_000_000, matchType: 'mismatch', reviewOutcome: 'pending', top: '2', sub: '2.01', altTop: '3', altSub: '3.01', docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2022-0036', year: 2022, ministry: '문화재청', field: '문화재', programName: '문화재구역 생태조사', budget: 310_000_000, expenditure: 300_000_000, matchType: 'agree', reviewOutcome: 'approved', top: '5', sub: '5.04', docCase: 'normal', reviewerIdx: 0 },

  { id: 'BP-2024-0037', year: 2024, ministry: '환경부', field: '환경', programName: '생물다양성협약(CBD) 이행지원', budget: 1_050_000_000, expenditure: 300_000_000, matchType: 'agree', reviewOutcome: 'pending', top: '8', sub: '8.01', docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2023-0038', year: 2023, ministry: '환경부', field: '환경', programName: '개발도상국 생물다양성 역량강화지원', budget: 5_600_000_000, expenditure: 1_900_000_000, matchType: 'agree', reviewOutcome: 'pending', top: '8', sub: '8.02', docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2022-0039', year: 2022, ministry: '환경부', field: '환경', programName: '국외반출 승인 및 유전자원 접근관리', budget: 220_000_000, expenditure: 215_000_000, matchType: 'agree', reviewOutcome: 'approved', top: '6', sub: '6.03', docCase: 'normal', reviewerIdx: 1 },
  { id: 'BP-2024-0040', year: 2024, ministry: '환경부', field: '환경', programName: '생물유전자원 이익공유 계약지원', budget: 175_000_000, expenditure: 90_000_000, matchType: 'mismatch', reviewOutcome: 'corrected', top: '6', sub: '6.04', altTop: '6', altSub: '6.01', docCase: 'normal', reviewerIdx: 2 },

  { id: 'BP-2023-0041', year: 2023, ministry: '해양수산부', field: '해양수산', programName: '해양생물자원 국제공동연구', budget: 990_000_000, expenditure: 400_000_000, matchType: 'mismatch', reviewOutcome: 'pending', top: '8', sub: '8.03', altTop: '5', altSub: '5.01', docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2024-0042', year: 2024, ministry: '산림청', field: '산림', programName: '국제산림협력사업 분담금', budget: 340_000_000, expenditure: 340_000_000, matchType: 'agree', reviewOutcome: 'pending', top: '8', sub: '8.04', docCase: 'normal', reviewerIdx: null },

  { id: 'BP-2024-0043', year: 2024, ministry: '환경부', field: '환경', programName: '사업설명자료 미제출 시범사업 A', budget: 150_000_000, expenditure: null, matchType: 'none', reviewOutcome: 'pending_unclassified', top: null, sub: null, docCase: 'no_doc', reviewerIdx: null },
  { id: 'BP-2023-0044', year: 2023, ministry: '환경부', field: '환경', programName: '사업설명자료 미제출 시범사업 B', budget: 95_000_000, expenditure: null, matchType: 'none', reviewOutcome: 'manual_unclassified', top: '9', sub: '9.05', docCase: 'no_doc', reviewerIdx: 4 },
  { id: 'BP-2024-0045', year: 2024, ministry: '국토교통부', field: '국토', programName: '도로건설 안전시설 확충', budget: 3_800_000_000, expenditure: 1_500_000_000, matchType: 'agree', reviewOutcome: 'pending', top: '0', sub: null, docCase: 'normal', reviewerIdx: null },
  { id: 'BP-2022-0046', year: 2022, ministry: '행정안전부', field: '안전', programName: '지역축제 지원사업', budget: 205_000_000, expenditure: null, matchType: 'mismatch', reviewOutcome: 'pending', top: '0', sub: null, altTop: '9', altSub: '9.03', docCase: 'no_doc', reviewerIdx: null },
]

function documentStatusOf(docCase: DocCase): DocumentStatus {
  if (docCase === 'no_doc') return '없음'
  if (docCase === 'extract_fail') return '추출실패'
  return '연결됨'
}

function reasoningFor(topCode: TopCode, subCode: string | undefined, budgetOnly: boolean): string {
  const name = topCategoryName(topCode)
  const subName = subCode ? subCategoryName(subCode) : null
  if (budgetOnly) {
    return `사업설명자료 확인이 제한되어 사업명·예산 항목명을 근거로 "${name}"${subName ? `(${subName})` : ''}로 판단했습니다. 문서 근거가 아닌 예산정보 기반 추정입니다.`
  }
  return `사업설명자료의 목적·세부내용 서술에서 "${name}"${subName ? `(${subName})` : ''} 관련 활동이 확인되어 해당 카테고리로 분류했습니다.`
}

function evidenceFor(programName: string, budgetOnly: boolean): string[] {
  if (budgetOnly) return [`(예산서 항목명) "${programName}"`]
  return [
    `"...본 사업은 ${programName.replace(/사업$|추진$|구축$|관리$/, '')} 관련 조사·복원 활동을 목적으로 한다..."`,
    `"...사업 대상지 현황 및 향후 관리계획을 포함한다..."`,
  ]
}

function attentionFor(topCode: TopCode): string[] {
  const map: Record<string, string[]> = {
    '0': ['시설 확충', '인프라 구축'],
    '1': ['멸종위기', '보전', '증식·복원'],
    '2': ['훼손지 복원', '생태축 조성', '습지 조성'],
    '3': ['보호지역', '지정 관리', '보호구역'],
    '4': ['지속가능', '이용', '전통지식'],
    '5': ['조사', '모니터링', '영향평가'],
    '6': ['유전자원', '접근·이익공유', '국외반출'],
    '7': ['교육', '인식증진', '체험'],
    '8': ['국제협력', '이행지원', '공동연구'],
    '9': ['주류화', '통계', '지자체 이행'],
  }
  return map[topCode] ?? []
}

function mkPrediction(
  modelLabel: string,
  topCode: TopCode | null,
  subCode: string | null,
  confidence: number | null,
  confidenceType: 'probability' | 'self_reported' | null,
  programName: string,
  budgetOnly: boolean,
  purposeExtractionFailed: boolean,
  hasDoc: boolean,
): ModelPrediction {
  if (topCode === null) {
    return {
      modelLabel,
      topCategory: null,
      subCategory: null,
      confidence: null,
      confidenceType: null,
      reasoning: '입력 자료에서 분류 근거를 찾지 못해 미분류로 처리되었습니다.',
      evidenceSentences: [],
      attentionSpans: [],
      inputBasis: hasDoc ? 'document' : 'none',
      purposeExtractionFailed,
    }
  }
  return {
    modelLabel,
    topCategory: topCode,
    subCategory: subCode,
    confidence,
    confidenceType,
    reasoning: reasoningFor(topCode, subCode ?? undefined, budgetOnly),
    evidenceSentences: evidenceFor(programName, budgetOnly),
    attentionSpans: budgetOnly ? [] : attentionFor(topCode),
    inputBasis: budgetOnly ? 'budget_only' : 'document',
    purposeExtractionFailed,
  }
}

export function emptyReview(): ExpertReview {
  return {
    status: '검토대기',
    finalTopCategory: null,
    finalSubCategory: null,
    opinion: '',
    reason: '',
    reviewer: null,
    reviewedAt: null,
    approvedSource: null,
    history: [],
  }
}

function historyEntry(
  action: ReviewHistoryEntry['action'],
  reviewer: string,
  before: { top: TopCode | null; sub: string | null },
  after: { top: TopCode | null; sub: string | null },
  reason: string,
  opinion: string,
  approvedSource: ReviewHistoryEntry['approvedSource'],
  daysAgo: number,
): ReviewHistoryEntry {
  const ts = new Date(Date.UTC(2026, 8, 22))
  ts.setUTCDate(ts.getUTCDate() - daysAgo)
  return {
    id: `hist-${Math.round(rng() * 1e9)}`,
    timestamp: ts.toISOString(),
    reviewer,
    action,
    beforeTop: before.top,
    beforeSub: before.sub,
    afterTop: after.top,
    afterSub: after.sub,
    reason,
    opinion,
    approvedSource,
  }
}

export function buildSeedPrograms(): BudgetProgram[] {
  return ROWS.map((row, idx) => {
    const documentStatus = documentStatusOf(row.docCase)
    const hasDoc = documentStatus === '연결됨'
    const budgetOnly = row.docCase === 'budget_only_with_doc' || documentStatus !== '연결됨'
    const purposeExtractionFailed = row.docCase === 'extract_fail'
    const unclassified = row.matchType === 'none'

    const tConf = 0.62 + rng() * 0.33
    const lConf = 0.55 + rng() * 0.4

    let transformer_v1: ModelPrediction | null = null
    let transformer_v2: ModelPrediction | null = null
    let llm_v1: ModelPrediction | null = null
    let llm_v2: ModelPrediction | null = null
    let pipeline: ModelPrediction | null = null

    if (!unclassified && row.top) {
      const primaryTop = row.top
      const primarySub = row.sub ?? null
      const altTop = row.matchType === 'mismatch' ? row.altTop ?? primaryTop : primaryTop
      const altSub = row.matchType === 'mismatch' ? row.altSub ?? null : primarySub

      transformer_v1 = mkPrediction('Transformer v1', primaryTop, null, tConf, 'probability', row.programName, budgetOnly, purposeExtractionFailed, hasDoc)
      transformer_v2 = mkPrediction('Transformer v2', primaryTop, primarySub, Math.max(0.4, tConf - 0.05), 'probability', row.programName, budgetOnly, purposeExtractionFailed, hasDoc)
      llm_v1 = mkPrediction('LLM v1', altTop, null, lConf, 'self_reported', row.programName, budgetOnly, purposeExtractionFailed, hasDoc)
      llm_v2 = mkPrediction('LLM v2', altTop, altSub, Math.max(0.35, lConf - 0.08), 'self_reported', row.programName, budgetOnly, purposeExtractionFailed, hasDoc)

      const useTransformerForPipeline = row.matchType === 'agree' || tConf > 0.85
      pipeline = {
        modelLabel: '단계별 파이프라인',
        topCategory: useTransformerForPipeline ? primaryTop : altTop,
        subCategory: useTransformerForPipeline ? primarySub : altSub,
        confidence: null,
        confidenceType: null,
        reasoning:
          primaryTop === '0'
            ? '비해당 규칙 조건(사업 목적에 생물다양성 관련 서술 없음)에 해당하여 규칙 단계에서 즉시 확정되었습니다.'
            : useTransformerForPipeline
              ? 'Transformer 신뢰도가 임계값 이상이어서 Transformer 예측을 그대로 채택했습니다.'
              : 'Transformer 신뢰도가 임계값 미만이어서 LLM 판단으로 라우팅된 결과입니다.',
        evidenceSentences: [],
        attentionSpans: [],
        inputBasis: budgetOnly ? 'budget_only' : 'document',
        purposeExtractionFailed,
      }
    }

    const review = emptyReview()
    const reviewer = row.reviewerIdx !== null ? REVIEWERS[row.reviewerIdx].name : null
    const primaryTop = row.top
    const primarySub = row.sub ?? null
    const altTop = row.altTop ?? primaryTop ?? null
    const altSub = row.altSub ?? null

    switch (row.reviewOutcome) {
      case 'approved': {
        review.status = '검토완료'
        review.finalTopCategory = primaryTop
        review.finalSubCategory = primarySub
        review.reviewer = reviewer
        review.opinion = '모델 예측 근거가 타당하여 예측을 그대로 승인합니다.'
        review.approvedSource = 'transformer_v1'
        const hApproved = historyEntry('예측 승인', reviewer!, { top: null, sub: null }, { top: primaryTop, sub: primarySub }, '', review.opinion, 'transformer_v1', 3 + (idx % 5))
        review.reviewedAt = hApproved.timestamp
        review.history = [hApproved]
        break
      }
      case 'corrected': {
        review.status = '검토완료'
        review.finalTopCategory = primaryTop
        review.finalSubCategory = primarySub
        review.reviewer = reviewer
        review.reason = `Transformer·LLM 판단이 엇갈려 사업설명자료 원문을 재확인한 결과 "${topCategoryName(primaryTop)}"가 사업 목적에 더 부합합니다.`
        review.opinion = '원문 근거를 재확인하여 카테고리를 수정 확정합니다.'
        review.approvedSource = 'manual'
        const h = historyEntry('수정 확정', reviewer!, { top: altTop, sub: altSub }, { top: primaryTop, sub: primarySub }, review.reason, review.opinion, 'manual', 2 + (idx % 4))
        review.reviewedAt = h.timestamp
        review.history = [h]
        break
      }
      case 'hold': {
        review.status = '보류'
        review.reviewer = reviewer
        review.reason = '모델 간 판단이 달라 사업설명자료 보완 요청 후 재검토가 필요합니다.'
        review.opinion = '추가 자료 확인 전까지 보류합니다.'
        const h = historyEntry('보류', reviewer!, { top: null, sub: null }, { top: null, sub: null }, review.reason, review.opinion, null, 1 + (idx % 3))
        review.reviewedAt = h.timestamp
        review.history = [h]
        break
      }
      case 'redo': {
        review.status = '재검토요청'
        review.reviewer = reviewer
        review.reason = '초기 검토 근거가 불충분하여 담당자 변경 후 재검토를 요청합니다.'
        review.opinion = '재검토 요청'
        const h = historyEntry('재검토 요청', reviewer!, { top: null, sub: null }, { top: null, sub: null }, review.reason, review.opinion, null, 1)
        review.reviewedAt = h.timestamp
        review.history = [h]
        break
      }
      case 'manual_unclassified': {
        review.status = '검토완료'
        review.finalTopCategory = row.top
        review.finalSubCategory = row.sub ?? null
        review.reviewer = reviewer
        review.reason = '모델 예측이 없어 사업 담당부서 문의를 통해 수동으로 분류했습니다.'
        review.opinion = '수동 분류 승인'
        review.approvedSource = 'manual'
        const h = historyEntry('수정 확정', reviewer!, { top: null, sub: null }, { top: row.top ?? null, sub: row.sub ?? null }, review.reason, review.opinion, 'manual', 1)
        review.reviewedAt = h.timestamp
        review.history = [h]
        break
      }
      case 'pending_unclassified':
      case 'pending':
      default:
        break
    }

    const fundSource: import('../types').FundSource | null =
      row.budget && row.docCase !== 'no_doc'
        ? [
            { label: '일반회계', amount: Math.round(row.budget * 0.6) },
            { label: '특별회계', amount: Math.round(row.budget * 0.25) },
            { label: '기금', amount: row.budget - Math.round(row.budget * 0.6) - Math.round(row.budget * 0.25) },
          ]
        : null

    return {
      id: row.id,
      year: row.year,
      ministry: row.ministry,
      field: row.field,
      programName: row.programName,
      budgetAmount: row.budget,
      expenditureAmount: row.expenditure,
      balanceAmount: row.expenditure === null ? null : row.budget - row.expenditure,
      fundSource,
      documentStatus,
      documentPreview:
        documentStatus === '연결됨'
          ? `[사업설명자료 발췌] ${row.programName} 사업의 목적, 추진배경, 연차별 추진계획, 기대효과가 기술되어 있습니다. (데모용 미리보기 텍스트)`
          : null,
      predictionRunId: unclassified ? null : RUN_TV1,
      transformer_v1,
      transformer_v2,
      llm_v1,
      llm_v2,
      pipeline,
      expertReview: review,
      assignee: reviewer,
      goldTopCategory: null,
      goldSubCategory: null,
    } satisfies BudgetProgram
  })
}

export function buildSeedRuns(programs: BudgetProgram[]): PredictionRun[] {
  const withPred = programs.filter((p) => p.transformer_v1)
  const withV2 = programs.filter((p) => p.transformer_v2)
  const base = (id: string, name: string, modelKey: PredictionRun['modelKey'], modelLabel: string, offsetDays: number, count: number): PredictionRun => {
    const created = new Date(Date.UTC(2026, 8, 22))
    created.setUTCDate(created.getUTCDate() - offsetDays)
    const finished = new Date(created)
    finished.setUTCMinutes(finished.getUTCMinutes() + 24)
    return {
      id,
      name,
      modelKey,
      modelLabel,
      scope: '전체',
      status: '완료',
      createdAt: created.toISOString(),
      finishedAt: finished.toISOString(),
      totalCount: count,
      processedCount: count,
      failedCount: 0,
      targetIds: programs.slice(0, count).map((p) => p.id),
      datasetName: 'BIOFIN_2023_취합_2026.09.14',
    }
  }
  return [
    base(RUN_TV1, '2609 정기 예측 · Transformer v1', 'transformer_v1', 'Transformer v1', 12, withPred.length),
    base(RUN_TV2, '2609 정기 예측 · Transformer v2', 'transformer_v2', 'Transformer v2', 10, withV2.length),
    base(RUN_LLM1, '2609 정기 예측 · LLM v1', 'llm_v1', 'LLM v1', 8, withPred.length),
    base(RUN_LLM2, '2609 정기 예측 · LLM v2', 'llm_v2', 'LLM v2', 6, withV2.length),
    base(RUN_PIPE, '2609 정기 예측 · 단계별 파이프라인', 'pipeline', '단계별 파이프라인', 4, withPred.length),
  ]
}

export const SEED_DATASETS: RegisteredDataset[] = [
  { id: 'ds-01', name: 'BIOFIN_2023_취합_2026.09.14.csv', rowCount: 1842, year: 2023, registeredAt: '2026-09-14T02:10:00.000Z', amountUnit: '원', encoding: 'UTF-8' },
  { id: 'ds-02', name: 'open_fiscal_2024.csv', rowCount: 4210, year: 2024, registeredAt: '2026-07-28T05:00:00.000Z', amountUnit: '원', encoding: 'CP949' },
  { id: 'ds-03', name: '2023biofin_label_matched.csv', rowCount: 1560, year: 2023, registeredAt: '2026-07-28T09:00:00.000Z', amountUnit: '원', encoding: 'UTF-8' },
]
