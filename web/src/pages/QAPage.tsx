import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardTitle } from '@/components/ui/card'
import ScoreBadge from '@/components/ScoreBadge'
import { localizeTitles } from '@/lib/api'
import { useLang } from '@/lib/lang'
import { Loader2, Send, FileText, Plus, Trash2, MessageSquare } from 'lucide-react'
import MarkdownContent from '@/components/MarkdownContent'
import { cn } from '@/lib/utils'
import { useQA } from '@/lib/qa'

const PAGE_SIZE = 5

export default function QAPage() {
  const { activeLang } = useLang()
  const { chats, activeChat, loadingChatId, isLoading, newChat, selectChat, deleteChat, sendQuestion } = useQA()
  const [input, setInput] = useState('')
  const [visibleMap, setVisibleMap] = useState<Record<string, number>>({})
  const [titleMap, setTitleMap] = useState<Record<string, string>>({})
  const messages = activeChat?.messages ?? []

  useEffect(() => {
    setVisibleMap({})
  }, [activeChat?.id])

  // Re-localize all source titles when language changes
  useEffect(() => {
    const paths = new Set<string>()
    for (const m of activeChat?.messages ?? []) {
      for (const s of m.sources ?? []) {
        if (s.markdown_path) paths.add(s.markdown_path)
      }
    }
    if (!paths.size) { setTitleMap({}); return }
    if (!activeLang) { setTitleMap({}); return }
    localizeTitles([...paths], activeLang).then(setTitleMap).catch(() => setTitleMap({}))
  }, [activeLang, activeChat])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const q = input.trim()
    if (!q || isLoading || !activeChat) return
    setInput('')
    await sendQuestion(q)
  }

  const getVisible = (messageId: string) => visibleMap[messageId] ?? PAGE_SIZE
  const showMore = (messageId: string) => setVisibleMap(prev => ({ ...prev, [messageId]: (prev[messageId] ?? PAGE_SIZE) + PAGE_SIZE }))

  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col lg:flex-row">
      <aside className="shrink-0 border-b border-border bg-[linear-gradient(180deg,var(--color-muted),transparent)] lg:w-72 lg:border-b-0 lg:border-r">
        <div className="flex items-center gap-2 p-4">
          <Button className="flex-1" onClick={() => { newChat(); setInput('') }} disabled={isLoading}>
            <Plus className="h-4 w-4" /> New Chat
          </Button>
        </div>
        <div className="px-4 pb-4">
          <div className="flex gap-2 overflow-x-auto lg:flex-col lg:overflow-y-auto lg:max-h-[calc(100vh-10rem)]">
            {chats.map(chat => (
              <div key={chat.id} className="group relative min-w-60 lg:min-w-0">
                <button
                  type="button"
                  onClick={() => selectChat(chat.id)}
                  className={cn(
                    'w-full rounded-xl border px-3 py-3 pr-11 text-left transition-colors duration-150',
                    chat.id === activeChat?.id
                      ? 'border-primary/35 bg-primary/8 text-foreground shadow-sm'
                      : 'border-border bg-background hover:bg-accent',
                  )}
                >
                  <div className="truncate text-sm font-medium">{chat.title}</div>
                  <div className={cn(
                    'mt-1 text-xs',
                    'text-muted-foreground',
                  )}>
                    {chat.messages.length > 0 ? `${chat.messages.length} messages` : 'Empty chat'}
                    {loadingChatId === chat.id ? ' · Thinking' : ''}
                  </div>
                </button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="absolute top-2 right-2 h-7 w-7 opacity-0 transition-opacity duration-150 group-hover:opacity-100 focus-visible:opacity-100"
                  onClick={() => deleteChat(chat.id)}
                  disabled={loadingChatId === chat.id}
                  aria-label={`Delete ${chat.title}`}
                  title="Delete chat"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        </div>
      </aside>

      <div className="flex min-h-0 flex-1 flex-col">
        <div className="border-b border-border px-4 py-3">
          <div className="flex items-center justify-between gap-4">
            <div className="min-w-0">
              <div className="truncate text-sm font-medium">{activeChat?.title || 'New Chat'}</div>
              <div className="text-xs text-muted-foreground">
                {messages.length > 0 ? `${messages.length} messages saved in this chat` : 'This chat is empty'}
              </div>
            </div>
            {loadingChatId === activeChat?.id && (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Thinking
              </div>
            )}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && (
            <div className="mt-20 animate-fade-in text-center text-muted-foreground">
              <MessageSquare className="mx-auto mb-3 h-8 w-8 opacity-60" />
              Ask a question to start this chat. Your history will stay after refresh.
            </div>
          )}
          {messages.map((m) => {
            const filtered = m.sources ?? []
            const vis = getVisible(m.id)
            return (
              <div key={m.id} className={`flex animate-fade-slide-up ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] rounded-lg px-4 py-3 ${m.role === 'user' ? 'bg-primary text-primary-foreground' : 'bg-muted'}`}>
                  {m.role === 'assistant' && m.status === 'pending' ? (
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Thinking...
                    </div>
                  ) : m.role === 'assistant' ? (
                    <MarkdownContent
                      content={m.content}
                      className="chat-markdown prose prose-neutral prose-sm max-w-none text-sm"
                    />
                  ) : (
                    <p className="whitespace-pre-wrap text-sm">{m.content}</p>
                  )}
                  {filtered.length > 0 && (
                    <div className="mt-3 space-y-2">
                      <p className="text-xs font-medium text-muted-foreground">Related Papers ({filtered.length})</p>
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
                        <Button variant="outline" size="sm" className="w-full" onClick={() => showMore(m.id)}>
                          Load more ({filtered.length - vis} remaining)
                        </Button>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>

        <form onSubmit={handleSubmit} className="border-t border-border p-4 flex gap-2">
          <Input
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask a question to find papers..."
            className="flex-1"
            aria-label="Question input"
          />
          <Button type="submit" disabled={isLoading || !input.trim()} size="icon" aria-label="Send question">
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </div>
    </div>
  )
}
