import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import ScoreBadge from '@/components/ScoreBadge'
import { fetchPapers, searchPapers, type PaperItem, type SearchResultItem } from '@/lib/api'
import { useLang } from '@/lib/lang'
import { Loader2, ExternalLink, FileText, Search, X } from 'lucide-react'

const PAGE_SIZE = 5
const FETCH_K = 50

interface CardItem {
  paperId: string
  title: string
  displayTitle: string
  keywords: string[]
  ingested_at: string
  source_url: string
  score?: number
}

function PaperCard({ item, showScore, index }: { item: CardItem; showScore: boolean; index: number }) {
  return (
    <Card className="animate-fade-slide-up" style={{ animationDelay: `${index * 50}ms` }}>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">{item.displayTitle || item.title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {item.keywords.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {item.keywords.map(kw => (
              <Badge key={kw} variant="outline" className="text-xs font-normal">{kw}</Badge>
            ))}
          </div>
        )}
        <div className="flex items-center gap-3 text-sm text-muted-foreground">
          {item.ingested_at && <span>{item.ingested_at}</span>}
          {item.source_url && (
            <a href={item.source_url} target="_blank" rel="noopener noreferrer" className="hover:text-foreground inline-flex items-center gap-1 transition-colors duration-200">
              <ExternalLink className="h-3 w-3" /> Source
            </a>
          )}
          <Link to={`/papers/${encodeURIComponent(item.paperId)}`} className="hover:text-foreground inline-flex items-center gap-1 transition-colors duration-200">
            <FileText className="h-3 w-3" /> Read
          </Link>
          {showScore && item.score != null && <span className="ml-auto"><ScoreBadge score={item.score} /></span>}
        </div>
      </CardContent>
    </Card>
  )
}

function paperToCard(p: PaperItem): CardItem {
  return {
    paperId: p.paper_id,
    title: p.title,
    displayTitle: p.display_title,
    keywords: p.keywords ?? [],
    ingested_at: p.ingested_at,
    source_url: p.source_url,
  }
}

function searchToCard(r: SearchResultItem): CardItem {
  return {
    paperId: r.paper_id,
    title: r.title,
    displayTitle: r.display_title,
    keywords: r.keywords ?? [],
    ingested_at: r.ingested_at,
    source_url: r.source_url,
    score: r.score,
  }
}

export default function PapersPage() {
  const { activeLang } = useLang()
  const [papers, setPapers] = useState<PaperItem[]>([])
  const [loading, setLoading] = useState(true)
  const [sortBy, setSortBy] = useState<'ingested_at' | 'title'>('ingested_at')
  const [order, setOrder] = useState<'asc' | 'desc'>('desc')

  // Search state
  const [query, setQuery] = useState('')
  const [searchMode, setSearchMode] = useState<'rag' | 'title'>('rag')
  const [searchResults, setSearchResults] = useState<SearchResultItem[] | null>(null)
  const [searchLoading, setSearchLoading] = useState(false)
  const [visible, setVisible] = useState(PAGE_SIZE)

  useEffect(() => {
    setLoading(true)
    fetchPapers(sortBy, order, activeLang).then(setPapers).catch(() => setPapers([])).finally(() => setLoading(false))
  }, [sortBy, order, activeLang])

  // Re-search when language changes
  useEffect(() => {
    if (searchResults === null || !query.trim()) return
    const filterTitle = searchMode === 'title' ? query : undefined
    searchPapers(query, FETCH_K, activeLang, filterTitle).then(setSearchResults).catch(e => console.error('Re-search failed:', e))
  }, [activeLang]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) { clearSearch(); return }
    if (searchLoading) return
    setSearchLoading(true)
    setVisible(PAGE_SIZE)
    try {
      const filterTitle = searchMode === 'title' ? query : undefined
      const res = await searchPapers(query, FETCH_K, activeLang, filterTitle)
      setSearchResults(res)
    } catch {
      setSearchResults([])
    } finally {
      setSearchLoading(false)
    }
  }

  const clearSearch = () => {
    setQuery('')
    setSearchResults(null)
    setVisible(PAGE_SIZE)
  }

  const isSearching = searchResults !== null
  const cards: CardItem[] = isSearching
    ? searchResults.slice(0, visible).map(searchToCard)
    : papers.map(paperToCard)

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-4">
      {/* Search bar */}
      <form onSubmit={handleSearch} className="flex gap-2">
        <div className="flex rounded-md border border-input text-sm overflow-hidden">
          {(['rag', 'title'] as const).map(mode => (
            <button
              key={mode}
              type="button"
              onClick={() => setSearchMode(mode)}
              className={`px-3 py-2 transition-all duration-200 ${searchMode === mode ? 'bg-primary text-primary-foreground' : 'bg-background text-muted-foreground hover:text-foreground'}`}
            >
              {mode === 'rag' ? 'RAG' : 'Title'}
            </button>
          ))}
        </div>
        <div className="relative flex-1">
          <Input
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder={searchMode === 'rag' ? 'Semantic search...' : 'Search by title...'}
            className="pr-8"
          />
          {query && (
            <button type="button" onClick={clearSearch} className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors duration-200" aria-label="Clear search">
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
        <Button type="submit" disabled={searchLoading} size="icon" aria-label="Search">
          {searchLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
        </Button>
      </form>

      {/* Sort controls — only when showing all papers */}
      {!isSearching && (
        <div className="flex items-center gap-2 text-sm">
          <span className="text-muted-foreground">Sort by:</span>
          {(['ingested_at', 'title'] as const).map(s => (
            <Button key={s} variant={sortBy === s ? 'default' : 'outline'} size="sm" onClick={() => setSortBy(s)}>
              {s === 'ingested_at' ? 'Date' : 'Title'}
            </Button>
          ))}
          <Button variant="outline" size="sm" onClick={() => setOrder(o => o === 'asc' ? 'desc' : 'asc')}>
            {order === 'asc' ? '↑' : '↓'}
          </Button>
        </div>
      )}

      {/* Search result count */}
      {isSearching && !searchLoading && (
        <div className="text-sm text-muted-foreground animate-fade-in">
          <span>{searchResults.length} results for "{query}"</span>
        </div>
      )}

      {(loading || searchLoading) && <div className="text-center py-10"><Loader2 className="h-6 w-6 animate-spin mx-auto" /></div>}

      {/* Unified cards */}
      <div className="space-y-3">
        {cards.map((item, i) => (
          <PaperCard key={item.paperId || (isSearching ? i : item.title)} item={item} showScore={isSearching} index={i} />
        ))}
      </div>

      {/* Load more for search results */}
      {isSearching && searchResults.length > visible && (
        <div className="text-center">
          <Button variant="outline" onClick={() => setVisible(v => v + PAGE_SIZE)}>
            Load more ({searchResults.length - visible} remaining)
          </Button>
        </div>
      )}

      {!loading && !isSearching && papers.length === 0 && (
        <p className="text-center text-muted-foreground py-10 animate-fade-in">No papers ingested yet.</p>
      )}
    </div>
  )
}
