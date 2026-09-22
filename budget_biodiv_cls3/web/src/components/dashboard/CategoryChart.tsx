import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { TOP_CATEGORIES } from '../../data/categories'
import { formatAmount } from '../../lib/format'

export function CategoryChart({
  metric,
  data,
}: {
  metric: 'count' | 'amount'
  data: { code: string; count: number; amount: number }[]
}) {
  const chartData = TOP_CATEGORIES.map((c) => {
    const found = data.find((d) => d.code === c.code)
    return {
      name: `${c.code}. ${c.name}`,
      value: metric === 'count' ? (found?.count ?? 0) : Math.round((found?.amount ?? 0) / 100_000_000),
    }
  })

  return (
    <ResponsiveContainer width="100%" height={360}>
      <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 24, top: 8, bottom: 8 }} barCategoryGap={6}>
        <CartesianGrid horizontal={false} stroke="#e1e0d9" />
        <XAxis type="number" tick={{ fontSize: 11, fill: '#898781' }} axisLine={{ stroke: '#c3c2b7' }} tickLine={false} />
        <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 11, fill: '#52514e' }} axisLine={{ stroke: '#c3c2b7' }} tickLine={false} />
        <Tooltip
          formatter={(value) => {
            const num = Number(value)
            return metric === 'count' ? [`${num.toLocaleString('ko-KR')}건`, '사업 수'] : [`${formatAmount(num * 100_000_000, '100million')}억원`, '예산 합계']
          }}
          contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e1e0d9' }}
        />
        <Bar dataKey="value" fill="#2a78d6" radius={[0, 4, 4, 0]} maxBarSize={18} />
      </BarChart>
    </ResponsiveContainer>
  )
}
