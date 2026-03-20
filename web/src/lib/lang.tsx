import { createContext, useContext } from 'react'
import type { I18nConfig } from '@/lib/api'

interface LangContextValue {
  i18n: I18nConfig | null
  activeLang: string | undefined
  setActiveLang: (lang: string | undefined) => void
}

export const LangContext = createContext<LangContextValue>({
  i18n: null,
  activeLang: undefined,
  setActiveLang: () => {},
})

export function useLang() {
  return useContext(LangContext)
}
