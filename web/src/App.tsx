import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { lazy, Suspense, useEffect, useState } from 'react'
import { fetchStatus, fetchI18n, type StatusResponse, type I18nConfig } from '@/lib/api'
import { LangContext } from '@/lib/lang'
import { ThemeProvider, useTheme } from '@/lib/theme'
import { MessageSquare, BookOpen, Globe, Sun, Moon, SunMoon, Plug } from 'lucide-react'

const QAPage = lazy(() => import('@/pages/QAPage'))
const PapersPage = lazy(() => import('@/pages/PapersPage'))
const PaperDetail = lazy(() => import('@/pages/PaperDetail'))
const MCPPage = lazy(() => import('@/pages/MCPPage'))

function StatusDot({ ok, label }: { ok?: boolean; label: string }) {
  const color = ok === undefined ? 'bg-gray-400' : ok ? 'bg-green-500' : 'bg-red-500'
  const status = ok === undefined ? 'unknown' : ok ? 'healthy' : 'unhealthy'
  return <span className={`inline-block w-2 h-2 rounded-full ${color}`} role="status" aria-label={`${label}: ${status}`} />
}

function ThemeToggle() {
  const { mode, setMode } = useTheme()
  const [spin, setSpin] = useState(false)
  const next = mode === 'light' ? 'dark' : mode === 'dark' ? 'auto' : 'light'
  const Icon = mode === 'light' ? Sun : mode === 'dark' ? Moon : SunMoon
  const label = mode === 'light' ? 'Light' : mode === 'dark' ? 'Dark' : 'Auto'
  return (
    <button
      onClick={() => { setSpin(true); setMode(next); setTimeout(() => setSpin(false), 300) }}
      className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground cursor-pointer"
      title={`Theme: ${label}`}
    >
      <Icon className={`h-3.5 w-3.5 transition-transform duration-300 ${spin ? 'rotate-180' : ''}`} />
    </button>
  )
}

function RouteFallback() {
  return <div className="flex justify-center py-20 text-sm text-muted-foreground animate-fade-in">Loading...</div>
}

export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [i18n, setI18n] = useState<I18nConfig | null>(null)
  const [activeLang, setActiveLang] = useState<string | undefined>(() => {
    try { return localStorage.getItem('compass-lang') || undefined } catch { return undefined }
  })

  useEffect(() => {
    fetchStatus().then(setStatus).catch(e => console.error('Failed to fetch status:', e))
    fetchI18n().then(setI18n).catch(e => console.error('Failed to fetch i18n:', e))
    const id = setInterval(() => fetchStatus().then(setStatus).catch(() => {}), 30000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    try {
      if (activeLang) localStorage.setItem('compass-lang', activeLang)
      else localStorage.removeItem('compass-lang')
    } catch { /* storage unavailable */ }
  }, [activeLang])

  const navItems = [
    { to: '/', icon: MessageSquare, label: 'Q&A' },
    { to: '/papers', icon: BookOpen, label: 'Papers' },
    { to: '/mcp-guide', icon: Plug, label: 'MCP' },
  ]

  return (
    <ThemeProvider>
    <BrowserRouter>
      <div className="min-h-screen bg-background text-foreground">
        <nav className="sticky top-0 z-50 h-16 border-b border-border flex items-center px-6 justify-between bg-background">
          <div className="flex items-center gap-6">
            <span className="font-bold text-lg">Compass</span>
            {navItems.map(({ to, icon: Icon, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-1.5 text-sm transition-colors duration-200 ${isActive ? 'text-foreground font-medium' : 'text-muted-foreground hover:text-foreground'}`
                }
              >
                <Icon className="h-4 w-4" /> {label}
              </NavLink>
            ))}
          </div>
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            <ThemeToggle />
            {i18n?.enabled && i18n.languages.length > 0 && (
              <select
                value={activeLang ?? ''}
                onChange={e => setActiveLang(e.target.value || undefined)}
                className="bg-transparent border border-border rounded px-1.5 py-0.5 text-xs cursor-pointer focus:outline-none"
              >
                <option value="">Original</option>
                {i18n.languages.map(l => (
                  <option key={l.code} value={l.code}>{l.name}</option>
                ))}
              </select>
            )}
            {i18n?.enabled && <Globe className="h-3 w-3" />}
            <span className="flex items-center gap-1"><StatusDot ok={status?.embedding.ok} label="Embedding" /> Embed</span>
            <span className="flex items-center gap-1"><StatusDot ok={status?.llm.ok} label="LLM" /> LLM</span>
            <span className="flex items-center gap-1"><StatusDot ok={status?.qdrant.ok} label="Database" /> DB</span>
          </div>
        </nav>
        <LangContext.Provider value={{ i18n, activeLang, setActiveLang }}>
          <Suspense fallback={<RouteFallback />}>
            <Routes>
              <Route path="/" element={<QAPage />} />
              <Route path="/papers" element={<PapersPage />} />
              <Route path="/papers/:paperId" element={<PaperDetail />} />
              <Route path="/mcp-guide" element={<MCPPage />} />
            </Routes>
          </Suspense>
        </LangContext.Provider>
      </div>
    </BrowserRouter>
    </ThemeProvider>
  )
}
