// API types matching FastAPI Pydantic models

export interface PaperItem {
  paper_id: string
  title: string
  display_title: string
  path: string
  chunks_count: number
  markdown_path: string
  pdf_path: string
  source_url: string
  keywords: string[]
  ingested_at: string
}

export interface SearchResultItem {
  paper_id: string
  title: string
  display_title: string
  text: string
  score: number
  source_path: string
  chunk_index: number
  markdown_path: string
  pdf_path: string
  keywords: string[]
  source_url: string
  ingested_at: string
}

export interface AskResponse {
  answer: string
  sources: SearchResultItem[]
}

export interface ServiceStatus {
  ok: boolean
  detail: string
}

export interface StatusResponse {
  embedding: ServiceStatus
  llm: ServiceStatus
  qdrant: ServiceStatus
}

// API functions

export async function fetchStatus(): Promise<StatusResponse> {
  const res = await fetch('/api/status')
  if (!res.ok) throw new Error(`Failed to fetch status (HTTP ${res.status})`)
  return res.json()
}

export async function fetchPapers(sortBy = 'ingested_at', order = 'desc', lang?: string): Promise<PaperItem[]> {
  const params = new URLSearchParams({ sort_by: sortBy, order })
  if (lang) params.set('lang', lang)
  const res = await fetch(`/api/papers?${params}`)
  if (!res.ok) throw new Error(`Failed to fetch papers (HTTP ${res.status})`)
  return res.json()
}

export interface I18nConfig {
  enabled: boolean
  languages: { code: string; name: string }[]
}

export async function fetchI18n(): Promise<I18nConfig> {
  const res = await fetch('/api/i18n')
  if (!res.ok) throw new Error(`Failed to fetch i18n config (HTTP ${res.status})`)
  return res.json()
}

export async function fetchPaperContent(paperId: string, lang?: string): Promise<{ paper_id: string; title: string; content: string; folder: string; source_url: string }> {
  const params = lang ? `?lang=${encodeURIComponent(lang)}` : ''
  const res = await fetch(`/api/papers/${encodeURIComponent(paperId)}/content${params}`)
  if (!res.ok) throw new Error(`Paper not found (HTTP ${res.status})`)
  return res.json()
}

export async function localizeTitles(markdownPaths: string[], lang: string): Promise<Record<string, string>> {
  const res = await fetch('/api/localize-titles', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ markdown_paths: markdownPaths, lang }),
  })
  if (!res.ok) throw new Error(`Failed to localize titles (HTTP ${res.status})`)
  return res.json()
}

export async function searchPapers(query: string, topK?: number, lang?: string, filterTitle?: string): Promise<SearchResultItem[]> {
  const body: Record<string, unknown> = { query, top_k: topK, lang }
  if (filterTitle) body.filter_title = filterTitle
  const res = await fetch('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`Search failed (HTTP ${res.status})`)
  return res.json()
}

export async function askQuestion(question: string, topK?: number, lang?: string): Promise<AskResponse> {
  const res = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK, lang }),
  })
  if (!res.ok) throw new Error(`Ask failed (HTTP ${res.status})`)
  return res.json()
}
