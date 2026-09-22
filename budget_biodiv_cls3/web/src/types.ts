// BIOFIN 예산사업 분류·전문가 검증 시스템 — 프론트엔드 프로토타입 타입 정의
// 실제 모델/서버/DB와 연결되지 않는 데모 데이터 구조입니다.

export type TopCode = '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'

export interface TopCategory {
  code: TopCode
  name: string
}

export interface SubCategory {
  code: string // "1.01" 형식 — 문자열로 유지 (계층 코드 보존)
  topCode: TopCode
  name: string
}

export type ReviewStatus = '검토대기' | '검토완료' | '보류' | '재검토요청'
export type DocumentStatus = '연결됨' | '없음' | '추출실패'
export type PredictionSource = 'transformer_v1' | 'transformer_v2' | 'llm_v1' | 'llm_v2' | 'pipeline' | 'manual'
export type InputBasis = 'document' | 'budget_only' | 'none'
export type ConfidenceType = 'probability' | 'self_reported'

export interface ModelPrediction {
  modelLabel: string // 표시용, 예: "Transformer v1"
  topCategory: TopCode | null // null = 미분류
  subCategory: string | null // null = 미분류 또는 상위 모델만 지원
  confidence: number | null // 0~1, null = 미제공
  confidenceType: ConfidenceType | null
  reasoning: string | null
  evidenceSentences: string[]
  attentionSpans: string[] // "모델이 주목한 구간" — 확정적 근거 아님
  inputBasis: InputBasis
  purposeExtractionFailed: boolean
}

export interface ReviewHistoryEntry {
  id: string
  timestamp: string
  reviewer: string
  action: '예측 승인' | '수정 확정' | '보류' | '재검토 요청'
  beforeTop: TopCode | null
  beforeSub: string | null
  afterTop: TopCode | null
  afterSub: string | null
  reason: string
  opinion: string
  approvedSource: PredictionSource | null
}

export interface ExpertReview {
  status: ReviewStatus
  finalTopCategory: TopCode | null
  finalSubCategory: string | null
  opinion: string
  reason: string
  reviewer: string | null
  reviewedAt: string | null
  approvedSource: PredictionSource | null
  history: ReviewHistoryEntry[]
}

export interface FundSource {
  general: number // 일반회계 비중(원)
  special: number // 특별회계
  fund: number // 기금
}

export interface BudgetProgram {
  id: string // 원본 행 고유 ID, 자동 병합 금지
  year: number
  ministry: string // 소관명
  field: string // 분야
  programName: string // 세부사업명
  budgetAmount: number | null // 예산액(원), 정보 없음이면 null
  expenditureAmount: number | null
  balanceAmount: number | null
  fundSource: FundSource | null
  documentStatus: DocumentStatus
  documentPreview: string | null
  predictionRunId: string | null
  transformer_v1: ModelPrediction | null
  transformer_v2: ModelPrediction | null
  llm_v1: ModelPrediction | null
  llm_v2: ModelPrediction | null
  pipeline: ModelPrediction | null
  expertReview: ExpertReview
  assignee: string | null
  goldTopCategory: TopCode | null // 평가용 정답(있는 경우만), 모델 입력에는 사용 안 함
  goldSubCategory: string | null
}

export interface PredictionRun {
  id: string
  name: string
  modelKey: 'transformer_v1' | 'transformer_v2' | 'llm_v1' | 'llm_v2' | 'pipeline'
  modelLabel: string
  scope: '전체' | '선택 사업' | '미분류 사업'
  status: '대기' | '진행중' | '완료' | '오류' | '취소'
  createdAt: string
  finishedAt: string | null
  totalCount: number
  processedCount: number
  failedCount: number
  targetIds: string[]
  datasetName: string
}

export interface RegisteredDataset {
  id: string
  name: string
  rowCount: number
  year: number
  registeredAt: string
  amountUnit: string
  encoding: 'UTF-8' | 'CP949'
}

export interface Reviewer {
  id: string
  name: string
  team: string
  assignedCount: number
  completedCount: number
}

export type AmountBasis = 'budget' | 'expenditure' | 'balance'
export type AmountUnit = 'won' | 'million' | '100million'
export type SortDir = 'asc' | 'desc'
