import * as React from 'react'
import { Slot } from '@radix-ui/react-slot'
import * as SliderPrimitive from '@radix-ui/react-slider'
import { cva, type VariantProps } from 'class-variance-authority'
import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: any[]) { return twMerge(clsx(inputs)) }

const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-md text-xs font-medium transition-colors disabled:opacity-40 disabled:pointer-events-none px-3 py-1.5',
  {
    variants: {
      variant: {
        default: 'bg-accent text-panel2 hover:bg-accent/80',
        outline: 'border border-edge bg-transparent text-ink hover:bg-panel',
        ghost: 'text-muted hover:bg-panel hover:text-ink',
        danger: 'bg-red-900/60 text-red-200 hover:bg-red-900',
      },
      size: { default: 'h-8', sm: 'h-6 px-2 text-[11px]', icon: 'h-8 w-8' },
    },
    defaultVariants: { variant: 'default', size: 'default' },
  },
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button'
    return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />
  },
)
Button.displayName = 'Button'

export function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block mb-2">
      <span className="text-[11px] text-muted block mb-0.5">{label}</span>
      {children}
    </label>
  )
}

export function NumberInput({ value, onChange, step = 0.001 }:
  { value: number; onChange: (v: number) => void; step?: number }) {
  return (
    <input type="number" step={step} value={Number.isFinite(value) ? value : 0}
      onChange={(e) => { const v = parseFloat(e.target.value); if (Number.isFinite(v)) onChange(v) }} />
  )
}

export function Vec3Input({ value, onChange, step = 0.001 }:
  { value: number[]; onChange: (v: number[]) => void; step?: number }) {
  return (
    <div className="grid grid-cols-3 gap-1">
      {['X', 'Y', 'Z'].map((axis, i) => (
        <div key={axis}>
          <span className="text-[10px] text-muted">{axis}</span>
          <NumberInput step={step} value={value[i] ?? 0}
            onChange={(v) => { const n = [...value]; n[i] = v; onChange(n) }} />
        </div>
      ))}
    </div>
  )
}

export function Slider({ value, min, max, step, onChange }:
  { value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <SliderPrimitive.Root className="relative flex items-center select-none touch-none w-full h-5"
      value={[value]} min={min} max={max} step={step}
      onValueChange={([v]) => onChange(v)}>
      <SliderPrimitive.Track className="bg-edge relative grow rounded-full h-1">
        <SliderPrimitive.Range className="absolute bg-accent rounded-full h-full" />
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb className="block w-3.5 h-3.5 bg-accent rounded-full border-2 border-panel2 focus:outline-none" />
    </SliderPrimitive.Root>
  )
}

export function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-3">
      <div className="text-[11px] uppercase tracking-wide text-accent mb-1.5 border-b border-edge pb-1">{title}</div>
      {children}
    </div>
  )
}