import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardTitle } from '@/components/ui/card'
import ScoreBadge from '@/components/ScoreBadge'
import { askQuestion, localizeTitles, type AskResponse, type SearchResultItem } from '@/lib/api'
import { useLang } from '@/lib/lang'
import { Loader2, Send, FileText } from 'lucide-react'

const PAGE_SIZE = 5

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: SearchResultItem[]
}

export default function QAPage() {
  const { activeLang } = useLang()
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [visibleMap, setVisibleMap] = useState<Record<number, number>>({})
  const [titleMap, setTitleMap] = useState<Record<string, string>>({})

  // Re-localize all source titles when language changes
  useEffect(() => {
    const paths = new Set<string>()
    for (const m of messages) {
      for (const s of m.sources ?? []) {
        if (s.markdown_path) paths.add(s.markdown_path)
      }
    }
    if (!paths.size) { setTitleMap({}); return }
    if (!activeLang) { setTitleMap({}); return }
    localizeTitles([...paths], activeLang).then(setTitleMap).catch(() => setTitleMap({}))
  }, [activeLang, messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const q = input.trim()
    if (!q || loading) return
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: q }])
    setLoading(true)
    try {
      const res: AskResponse = await askQuestion(q, 50, activeLang)
      setMessages(prev => [...prev, { role: 'assistant', content: res.answer, sources: res.sources }])
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'Unknown error'
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${detail}` }])
    } finally {
      setLoading(false)
    }
  }

  const getVisible = (idx: number) => visibleMap[idx] ?? PAGE_SIZE
  const showMore = (idx: number) => setVisibleMap(prev => ({ ...prev, [idx]: (prev[idx] ?? PAGE_SIZE) + PAGE_SIZE }))

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)]">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-muted-foreground mt-20 animate-fade-in">Ask a question to find relevant papers</div>
        )}
        {messages.map((m, i) => {
          const filtered = m.sources ?? []
          const vis = getVisible(i)
          return (
            <div key={i} className={`flex animate-fade-slide-up ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] rounded-lg px-4 py-3 ${m.role === 'user' ? 'bg-primary text-primary-foreground' : 'bg-muted'}`}>
                <p className="whitespace-pre-wrap text-sm">{m.content}</p>
                {filtered.length > 0 && (
                  <div className="mt-3 space-y-2">
                    <p className="text-xs text-muted-foreground font-medium">Related Papers ({filtered.length})</p>
                    {filtered.slice(0, vis).map((s) => (
                      <Card key={s.paper_id || s.markdown_path} className="flex items-center justify-between p-3">
                        <div className="flex items-center gap-3">
                          <CardTitle className="text-sm">{titleMap[s.markdown_path] || s.display_title || s.title}</CardTitle>
                          <ScoreBadge score={s.score} />
                        </div>
                        <Link to={`/papers/${encodeURIComponent(s.paper_id)}`}>
                          <Button variant="default" size="sm"><FileText className="h-4 w-4 mr-1" /> Read Paper</Button>
                        </Link>
                      </Card>
                    ))}
                    {filtered.length > vis && (
                      <Button variant="outline" size="sm" className="w-full" onClick={() => showMore(i)}>
                        Load more ({filtered.length - vis} remaining)
                      </Button>
                    )}
                  </div>
                )}
              </div>
            </div>
          )
        })}
        {loading && (
          <div className="flex justify-start animate-fade-in">
            <div className="bg-muted rounded-lg px-4 py-3">
              <Loader2 className="h-4 w-4 animate-spin" />
            </div>
          </div>
        )}
      </div>
      <form onSubmit={handleSubmit} className="border-t border-border p-4 flex gap-2">
        <Input value={input} onChange={e => setInput(e.target.value)} placeholder="Ask a question to find papers..." className="flex-1" aria-label="Question input" />
        <Button type="submit" disabled={loading} size="icon" aria-label="Send question"><Send className="h-4 w-4" /></Button>
      </form>
    </div>
  )
}
