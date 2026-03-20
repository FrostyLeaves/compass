import { createContext, useContext, useEffect, useState } from 'react'

type ThemeMode = 'light' | 'dark' | 'auto'

interface ThemeContextValue {
  mode: ThemeMode
  setMode: (mode: ThemeMode) => void
  resolved: 'light' | 'dark'
}

const _AUTO_CHECK_INTERVAL = 60_000
const _DAY_START = 7
const _DAY_END = 19
const _STORAGE_KEY = 'compass-theme'

function resolveAuto(): 'light' | 'dark' {
  const h = new Date().getHours()
  return h >= _DAY_START && h < _DAY_END ? 'light' : 'dark'
}

function applyTheme(theme: 'light' | 'dark') {
  document.documentElement.setAttribute('data-theme', theme)
}

const ThemeContext = createContext<ThemeContextValue>({
  mode: 'light',
  setMode: () => {},
  resolved: 'light',
})

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [mode, setModeState] = useState<ThemeMode>(() => {
    try {
      return (localStorage.getItem(_STORAGE_KEY) as ThemeMode) || 'light'
    } catch {
      return 'light'
    }
  })

  const [resolved, setResolved] = useState<'light' | 'dark'>(() =>
    mode === 'auto' ? resolveAuto() : mode
  )

  const setMode = (m: ThemeMode) => {
    setModeState(m)
    try { localStorage.setItem(_STORAGE_KEY, m) } catch { /* storage unavailable */ }
  }

  useEffect(() => {
    if (mode === 'auto') {
      const r = resolveAuto()
      setResolved(r)
      applyTheme(r)
      const id = setInterval(() => {
        const next = resolveAuto()
        setResolved(next)
        applyTheme(next)
      }, _AUTO_CHECK_INTERVAL)
      return () => clearInterval(id)
    }
    setResolved(mode)
    applyTheme(mode)
  }, [mode])

  return (
    <ThemeContext.Provider value={{ mode, setMode, resolved }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  return useContext(ThemeContext)
}
