/**
 * Locale provider and copy helper.
 *
 * The Chinese source string is the dictionary key: components keep writing
 * natural Chinese and the English dictionary maps it to English. A missing
 * translation falls back to the Chinese source, so an untranslated screen shows
 * real copy instead of a bare key or an empty label.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { EN } from './en'

export type Locale = 'zh' | 'en'
export type Vars = Record<string, string | number>

const STORAGE_KEY = 'roboforge.locale.v1'

type LocaleContextValue = {
  locale: Locale
  setLocale: (next: Locale) => void
  toggle: () => void
  /** Translate a Chinese source string, filling `{name}` placeholders from vars. */
  t: (source: string, vars?: Vars) => string
}

const LocaleContext = createContext<LocaleContextValue | null>(null)

function initialLocale(): Locale {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved === 'zh' || saved === 'en') return saved
  } catch {
    /* Storage can be unavailable; fall through to the browser hint. */
  }
  // Only an explicitly English browser flips the default; everything else keeps
  // the primary language of the project.
  const nav = typeof navigator === 'undefined' ? '' : navigator.language.toLowerCase()
  return nav.startsWith('en') ? 'en' : 'zh'
}

export function LocaleProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(initialLocale)

  useEffect(() => {
    document.documentElement.lang = locale === 'en' ? 'en' : 'zh-CN'
  }, [locale])

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next)
    try {
      localStorage.setItem(STORAGE_KEY, next)
    } catch {
      /* A read-only profile keeps the in-memory locale only. */
    }
  }, [])

  const toggle = useCallback(() => {
    setLocaleState(current => {
      const next: Locale = current === 'zh' ? 'en' : 'zh'
      try {
        localStorage.setItem(STORAGE_KEY, next)
      } catch {
        /* Ignored for the same reason as above. */
      }
      return next
    })
  }, [])

  const t = useCallback((source: string, vars?: Vars) => {
    const text = locale === 'en' ? EN[source] ?? source : source
    if (!vars) return text
    return text.replace(/\{(\w+)\}/g, (match, name: string) =>
      name in vars ? String(vars[name]) : match)
  }, [locale])

  const value = useMemo(() => ({ locale, setLocale, toggle, t }), [locale, setLocale, toggle, t])
  return <LocaleContext.Provider value={value}>{children}</LocaleContext.Provider>
}

export function useI18n(): LocaleContextValue {
  const value = useContext(LocaleContext)
  if (!value) throw new Error('useI18n must be used inside LocaleProvider')
  return value
}

/** Two-state language control used by both shells. */
export function LanguageSwitch({ className }: { className?: string }) {
  const { locale, setLocale } = useI18n()
  return (
    <div className={className ?? 'lang-switch'} role="group" aria-label={locale === 'zh' ? '界面语言' : 'Interface language'}>
      <button type="button" className={locale === 'zh' ? 'selected' : ''} aria-pressed={locale === 'zh'} onClick={() => setLocale('zh')}>中文</button>
      <button type="button" className={locale === 'en' ? 'selected' : ''} aria-pressed={locale === 'en'} onClick={() => setLocale('en')}>EN</button>
    </div>
  )
}
