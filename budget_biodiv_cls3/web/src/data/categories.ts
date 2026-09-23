import type { SubCategory, TopCategory } from '../types'

export const TOP_CATEGORIES: TopCategory[] = [
  { code: '0', name: '비해당' },
  { code: '1', name: '종·생태계 보전' },
  { code: '2', name: '서식지 복원·조성' },
  { code: '3', name: '보호지역 지정·관리' },
  { code: '4', name: '생물자원 지속가능이용' },
  { code: '5', name: '생물다양성 조사·연구' },
  { code: '6', name: '유전자원 접근·이익공유' },
  { code: '7', name: '교육·인식증진' },
  { code: '8', name: '국제협력·이행지원' },
  { code: '9', name: '주류화·기타' },
]

export const SUB_CATEGORIES: SubCategory[] = [
  { code: '1.01', topCode: '1', name: '멸종위기종 보전' },
  { code: '1.02', topCode: '1', name: '생태계교란종 관리' },
  { code: '1.03', topCode: '1', name: '자생생물 증식·복원' },
  { code: '1.04', topCode: '1', name: '야생동물 질병관리' },

  { code: '2.01', topCode: '2', name: '훼손지 복원' },
  { code: '2.02', topCode: '2', name: '습지 조성·복원' },
  { code: '2.03', topCode: '2', name: '도시생태축 조성' },
  { code: '2.04', topCode: '2', name: '산림생태복원' },

  { code: '3.01', topCode: '3', name: '국립공원 관리' },
  { code: '3.02', topCode: '3', name: '습지보호지역 관리' },
  { code: '3.03', topCode: '3', name: '야생생물보호구역 관리' },
  { code: '3.04', topCode: '3', name: '해양보호구역 관리' },

  { code: '4.01', topCode: '4', name: '지속가능 산림경영' },
  { code: '4.02', topCode: '4', name: '지속가능 수산자원관리' },
  { code: '4.03', topCode: '4', name: '생태관광 활성화' },
  { code: '4.04', topCode: '4', name: '전통지식 활용' },

  { code: '5.01', topCode: '5', name: '생물자원 발굴·조사' },
  { code: '5.02', topCode: '5', name: '생태계 모니터링' },
  { code: '5.03', topCode: '5', name: '유전자원 정보구축' },
  { code: '5.04', topCode: '5', name: '서식실태 조사' },
  { code: '5.05', topCode: '5', name: '기후변화 영향평가' },

  { code: '6.01', topCode: '6', name: '접근·이익공유 제도운영' },
  { code: '6.02', topCode: '6', name: '국가책임기관 운영' },
  { code: '6.03', topCode: '6', name: '국외반출 승인' },
  { code: '6.04', topCode: '6', name: '이익공유 계약지원' },

  { code: '7.01', topCode: '7', name: '생물다양성 교육프로그램' },
  { code: '7.02', topCode: '7', name: '대중 인식증진 캠페인' },
  { code: '7.03', topCode: '7', name: '전문인력 양성' },
  { code: '7.04', topCode: '7', name: '청소년 체험교육' },
  { code: '7.05', topCode: '7', name: '홍보콘텐츠 제작' },

  { code: '8.01', topCode: '8', name: '국제협약 이행지원' },
  { code: '8.02', topCode: '8', name: '개도국 역량강화지원' },
  { code: '8.03', topCode: '8', name: '국제공동연구' },
  { code: '8.04', topCode: '8', name: '다자기금 분담금' },

  { code: '9.01', topCode: '9', name: '생물다양성 주류화 정책' },
  { code: '9.02', topCode: '9', name: '통계·지표 개발' },
  { code: '9.03', topCode: '9', name: '민간참여 활성화' },
  { code: '9.04', topCode: '9', name: '지자체 이행지원' },
  { code: '9.05', topCode: '9', name: '기타 생물다양성 관련사업' },
]

export function topCategoryName(code: string | null): string {
  if (code === null) return '미분류'
  const found = TOP_CATEGORIES.find((c) => c.code === code)
  return found ? `${found.code}. ${found.name}` : code
}

export function subCategoryName(code: string | null): string {
  if (code === null) return '-'
  const found = SUB_CATEGORIES.find((c) => c.code === code)
  return found ? `${found.code} ${found.name}` : code
}

export function subCategoriesForTop(topCode: string | null): SubCategory[] {
  if (!topCode) return []
  return SUB_CATEGORIES.filter((c) => c.topCode === topCode)
}

// Transformer v2가 지원하는 하위 코드(라벨 맵 기준) — 데모에서는 전체 지원으로 설정
export const TRANSFORMER_V2_SUPPORTED_CODES = new Set(SUB_CATEGORIES.map((c) => c.code))

export const MINISTRIES = [
  '환경부',
  '해양수산부',
  '산림청',
  '농림축산식품부',
  '문화체육관광부',
  '국토교통부',
  '행정안전부',
  '문화재청',
]

export const FIELDS = ['환경', '해양수산', '산림', '농업', '문화관광', '국토', '안전', '문화재']
