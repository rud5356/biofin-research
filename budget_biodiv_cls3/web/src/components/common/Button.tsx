import type { ButtonHTMLAttributes } from 'react'

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'outline'
type Size = 'sm' | 'md'

const VARIANT_STYLE: Record<Variant, string> = {
  primary: 'bg-navy-900 text-white border-navy-900 hover:bg-navy-800',
  secondary: 'bg-teal-600 text-white border-teal-600 hover:bg-teal-700',
  outline: 'bg-white text-navy-900 border-slate-300 hover:bg-slate-50',
  ghost: 'bg-transparent text-slate-600 border-transparent hover:bg-slate-100',
  danger: 'bg-white text-red-600 border-red-200 hover:bg-red-50',
}

const SIZE_STYLE: Record<Size, string> = {
  sm: 'px-2.5 py-1 text-xs',
  md: 'px-3.5 py-1.5 text-sm',
}

export function Button({
  variant = 'outline',
  size = 'md',
  className = '',
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: Size }) {
  return (
    <button
      className={`inline-flex items-center justify-center gap-1.5 rounded-md border font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${VARIANT_STYLE[variant]} ${SIZE_STYLE[size]} ${className}`}
      {...props}
    />
  )
}
