import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type {
  BudgetProgram,
  PredictionRun,
  PredictionSource,
  RegisteredDataset,
  ReviewHistoryEntry,
  ReviewStatus,
  Reviewer,
  TopCode,
} from '../types'
import { buildSeedPrograms, buildSeedRuns, REVIEWERS, SEED_DATASETS } from '../data/seed'

const STORAGE_KEY = 'biofin-cls3-demo-store'

function nowIso() {
  return new Date().toISOString()
}

function makeInitialState() {
  const programs = buildSeedPrograms()
  const runs = buildSeedRuns(programs)
  return { programs, runs, datasets: SEED_DATASETS, reviewers: REVIEWERS.map((r) => ({ ...r })) }
}

export interface SaveReviewInput {
  programId: string
  action: '예측 승인' | '수정 확정' | '보류' | '재검토 요청'
  finalTopCategory: TopCode | null
  finalSubCategory: string | null
  opinion: string
  reason: string
  reviewer: string
  approvedSource: PredictionSource | null
}

export interface BulkApproveResult {
  appliedIds: string[]
  skipped: { id: string; reason: string }[]
}

interface AppState {
  programs: BudgetProgram[]
  runs: PredictionRun[]
  datasets: RegisteredDataset[]
  reviewers: Reviewer[]
  columnPrefs: Record<string, boolean>

  saveReview: (input: SaveReviewInput) => void
  bulkApprove: (ids: string[], source: PredictionSource) => BulkApproveResult
  setColumnPref: (key: string, visible: boolean) => void

  createRun: (input: { name: string; modelKey: PredictionRun['modelKey']; scope: PredictionRun['scope']; targetIds: string[]; datasetName: string }) => string
  setRunStatus: (id: string, status: PredictionRun['status']) => void
  setRunProgress: (id: string, processedCount: number, failedCount: number) => void
  finishRunFinishedAt: (id: string) => void
  applyRunResult: (runId: string, programId: string, modelKey: 'transformer_v1' | 'transformer_v2' | 'llm_v1' | 'llm_v2' | 'pipeline', prediction: BudgetProgram['transformer_v1']) => void
  retryRun: (id: string) => string

  addDataset: (ds: RegisteredDataset) => void
  linkDocument: (id: string) => void

  resetDemo: () => void
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      ...makeInitialState(),
      columnPrefs: {
        transformerConfidence: false,
        llmConfidence: false,
      },

      saveReview: (input) => {
        set((state) => ({
          programs: state.programs.map((p) => {
            if (p.id !== input.programId) return p
            const before = { top: p.expertReview.finalTopCategory, sub: p.expertReview.finalSubCategory }
            const after = { top: input.finalTopCategory, sub: input.finalSubCategory }
            const status: ReviewStatus =
              input.action === '예측 승인' || input.action === '수정 확정'
                ? '검토완료'
                : input.action === '보류'
                  ? '보류'
                  : '재검토요청'
            const historyEntry: ReviewHistoryEntry = {
              id: `hist-${Date.now()}-${Math.round(Math.random() * 1e6)}`,
              timestamp: nowIso(),
              reviewer: input.reviewer,
              action: input.action,
              beforeTop: before.top,
              beforeSub: before.sub,
              afterTop: after.top,
              afterSub: after.sub,
              reason: input.reason,
              opinion: input.opinion,
              approvedSource: input.approvedSource,
            }
            return {
              ...p,
              assignee: p.assignee ?? input.reviewer,
              expertReview: {
                status,
                finalTopCategory: after.top,
                finalSubCategory: after.sub,
                opinion: input.opinion,
                reason: input.reason,
                reviewer: input.reviewer,
                reviewedAt: nowIso(),
                approvedSource: input.approvedSource,
                history: [...p.expertReview.history, historyEntry],
              },
            }
          }),
        }))
      },

      bulkApprove: (ids, source) => {
        const state = get()
        const appliedIds: string[] = []
        const skipped: { id: string; reason: string }[] = []
        const targetSet = new Set(ids)

        set({
          programs: state.programs.map((p) => {
            if (!targetSet.has(p.id)) return p
            const pred = p[source as 'transformer_v1' | 'transformer_v2' | 'llm_v1' | 'llm_v2' | 'pipeline']
            if (!pred || pred.topCategory === null) {
              skipped.push({ id: p.id, reason: '선택한 예측 출처에 결과가 없어 대상에서 제외' })
              return p
            }
            appliedIds.push(p.id)
            const historyEntry: ReviewHistoryEntry = {
              id: `hist-${Date.now()}-${Math.round(Math.random() * 1e6)}`,
              timestamp: nowIso(),
              reviewer: p.expertReview.reviewer ?? '일괄승인',
              action: '예측 승인',
              beforeTop: p.expertReview.finalTopCategory,
              beforeSub: p.expertReview.finalSubCategory,
              afterTop: pred.topCategory,
              afterSub: pred.subCategory,
              reason: '',
              opinion: '일괄 승인 처리',
              approvedSource: source,
            }
            return {
              ...p,
              expertReview: {
                ...p.expertReview,
                status: '검토완료',
                finalTopCategory: pred.topCategory,
                finalSubCategory: pred.subCategory,
                approvedSource: source,
                reviewedAt: nowIso(),
                history: [...p.expertReview.history, historyEntry],
              },
            }
          }),
        })
        return { appliedIds, skipped }
      },

      setColumnPref: (key, visible) => set((state) => ({ columnPrefs: { ...state.columnPrefs, [key]: visible } })),

      createRun: (input) => {
        const id = `run-${input.modelKey}-${Date.now()}`
        const run: PredictionRun = {
          id,
          name: input.name,
          modelKey: input.modelKey,
          modelLabel:
            input.modelKey === 'transformer_v1'
              ? 'Transformer v1'
              : input.modelKey === 'transformer_v2'
                ? 'Transformer v2'
                : input.modelKey === 'llm_v1'
                  ? 'LLM v1'
                  : input.modelKey === 'llm_v2'
                    ? 'LLM v2'
                    : '단계별 파이프라인',
          scope: input.scope,
          status: '대기',
          createdAt: nowIso(),
          finishedAt: null,
          totalCount: input.targetIds.length,
          processedCount: 0,
          failedCount: 0,
          targetIds: input.targetIds,
          datasetName: input.datasetName,
        }
        set((state) => ({ runs: [run, ...state.runs] }))
        return id
      },

      setRunStatus: (id, status) => set((state) => ({ runs: state.runs.map((r) => (r.id === id ? { ...r, status } : r)) })),

      setRunProgress: (id, processedCount, failedCount) =>
        set((state) => ({ runs: state.runs.map((r) => (r.id === id ? { ...r, processedCount, failedCount } : r)) })),

      finishRunFinishedAt: (id) => set((state) => ({ runs: state.runs.map((r) => (r.id === id ? { ...r, finishedAt: nowIso() } : r)) })),

      applyRunResult: (runId, programId, modelKey, prediction) =>
        set((state) => ({
          programs: state.programs.map((p) => (p.id === programId ? { ...p, [modelKey]: prediction, predictionRunId: runId } : p)),
        })),

      retryRun: (id) => {
        const old = get().runs.find((r) => r.id === id)
        if (!old) return id
        const newId = `run-${old.modelKey}-${Date.now()}`
        const run: PredictionRun = {
          ...old,
          id: newId,
          name: `${old.name} (재시도)`,
          status: '대기',
          createdAt: nowIso(),
          finishedAt: null,
          processedCount: 0,
          failedCount: 0,
        }
        set((state) => ({ runs: [run, ...state.runs] }))
        return newId
      },

      addDataset: (ds) => set((state) => ({ datasets: [ds, ...state.datasets] })),

      linkDocument: (id) =>
        set((state) => ({
          programs: state.programs.map((p) =>
            p.id === id
              ? { ...p, documentStatus: '연결됨', documentPreview: `[사업설명자료 발췌] ${p.programName} 사업의 목적, 추진배경, 연차별 추진계획이 기술되어 있습니다. (수동 연결 데모)` }
              : p,
          ),
        })),

      resetDemo: () => {
        localStorage.removeItem(STORAGE_KEY)
        set({ ...makeInitialState(), columnPrefs: { transformerConfidence: false, llmConfidence: false } })
      },
    }),
    { name: STORAGE_KEY },
  ),
)
