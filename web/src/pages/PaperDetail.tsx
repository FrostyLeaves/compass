import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { fetchPaperContent } from '@/lib/api'
import { useLang } from '@/lib/lang'
import { Loader2, ArrowLeft, ExternalLink } from 'lucide-react'
import { Button } from '@/components/ui/button'
import MarkdownContent from '@/components/MarkdownContent'

export default function PaperDetail() {
  const { paperId } = useParams<{ paperId: string }>()
  const { activeLang } = useLang()

  if (!paperId) {
    return <div className="text-center py-20 text-destructive">Paper not found.</div>
  }

  return <PaperDetailContent key={`${paperId}:${activeLang ?? ''}`} paperId={decodeURIComponent(paperId)} activeLang={activeLang} />
}

function PaperDetailContent({ paperId, activeLang }: { paperId: string; activeLang: string | undefined }) {
  const [content, setContent] = useState('')
  const [folder, setFolder] = useState('')
  const [sourceUrl, setSourceUrl] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    fetchPaperContent(paperId, activeLang)
      .then(res => {
        if (cancelled) return
        setContent(res.content)
        setFolder(res.folder)
        setSourceUrl(res.source_url)
      })
      .catch(() => {
        if (cancelled) return
        setError('Failed to load paper content.')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [paperId, activeLang])

  if (loading) return <div className="flex justify-center py-20"><Loader2 className="h-6 w-6 animate-spin" /></div>
  if (error) return <div className="text-center py-20 text-destructive">{error}</div>

  return (
    <div className="p-6 max-w-4xl mx-auto animate-fade-in">
      <div className="flex items-center justify-between mb-4">
        <Link to="/papers">
          <Button variant="ghost" size="sm"><ArrowLeft className="h-4 w-4 mr-1" /> Back</Button>
        </Link>
        {sourceUrl && (
          <a href={sourceUrl} target="_blank" rel="noopener noreferrer">
            <Button variant="outline" size="sm"><ExternalLink className="h-4 w-4 mr-1" /> Source</Button>
          </a>
        )}
      </div>
      <MarkdownContent content={content} folder={folder} className="prose prose-neutral max-w-none" />
    </div>
  )
}
