import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import remarkGfm from 'remark-gfm'
import rehypeKatex from 'rehype-katex'
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize'
import 'katex/dist/katex.min.css'

const sanitizeSchema = {
  ...defaultSchema,
  attributes: {
    ...defaultSchema.attributes,
    span: [...(defaultSchema.attributes?.span ?? []), 'id', 'className'],
    div: [...(defaultSchema.attributes?.div ?? []), 'className'],
    code: [...(defaultSchema.attributes?.code ?? []), 'className'],
    math: ['xmlns'],
  },
  tagNames: [...(defaultSchema.tagNames ?? []), 'math', 'semantics', 'mrow', 'mi', 'mo', 'mn', 'msup', 'msub', 'mfrac', 'annotation'],
}
import { fetchPaperContent } from '@/lib/api'
import { useLang } from '@/lib/lang'
import { Loader2, ArrowLeft, ExternalLink } from 'lucide-react'
import { Button } from '@/components/ui/button'

/** Convert single-line $$...$$ to multi-line so remark-math treats them as display math. */
function fixDisplayMath(md: string): string {
  return md.replace(/^\$\$(.+)\$\$$/gm, '$$$$\n$1\n$$$$')
}

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
      <article className="prose prose-neutral max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkMath, remarkGfm]}
          rehypePlugins={[[rehypeSanitize, sanitizeSchema], rehypeKatex]}
          components={{
            img: ({ src, alt, ...props }) => {
              let resolvedSrc = src || ''
              const safeFolder = folder.replace(/\.\./g, '').replace(/[/\\]/g, '')
              if (resolvedSrc && !resolvedSrc.startsWith('http') && !resolvedSrc.startsWith('/')) {
                resolvedSrc = `/static/papers/${safeFolder}/${resolvedSrc}`
              }
              return <img src={resolvedSrc} alt={alt || ''} {...props} className="max-w-full rounded" />
            },
            table: ({ children, ...props }) => (
              <div className="table-wrapper">
                <table {...props}>{children}</table>
              </div>
            ),
            span: ({ children, id, ...props }) => {
              // Strip empty anchor spans like <span id="page-2-1"></span>
              if (id && !children) return null
              return <span {...props}>{children}</span>
            },
          }}
        >
          {fixDisplayMath(content)}
        </ReactMarkdown>
      </article>
    </div>
  )
}
