import { useEffect, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Check, Copy, ExternalLink, Link2, Sparkles } from 'lucide-react'

const GUIDE_PATH = '/mcp-guide'

function getGuideUrl() {
  if (typeof window === 'undefined') return GUIDE_PATH
  return new URL(GUIDE_PATH, window.location.origin).toString()
}

function StepCard({
  index,
  title,
  description,
}: {
  index: string
  title: string
  description: string
}) {
  return (
    <div className="rounded-xl border border-border bg-background/80 p-4">
      <div className="mb-3 inline-flex h-8 w-8 items-center justify-center rounded-full bg-primary text-xs font-semibold text-primary-foreground">
        {index}
      </div>
      <h3 className="mb-1 text-sm font-medium">{title}</h3>
      <p className="text-sm text-muted-foreground">{description}</p>
    </div>
  )
}

export default function MCPPage() {
  const [guideUrl, setGuideUrl] = useState(GUIDE_PATH)
  const [copied, setCopied] = useState(false)
  const [copyError, setCopyError] = useState('')
  const textRef = useRef<HTMLTextAreaElement | null>(null)

  useEffect(() => {
    setGuideUrl(getGuideUrl())
  }, [])

  const copyText = async () => {
    try {
      await navigator.clipboard.writeText(guideUrl)
      setCopied(true)
      setCopyError('')
      setTimeout(() => setCopied(false), 2000)
    } catch {
      setCopyError('Clipboard access failed. Copy the link manually below.')
    }
  }

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6 animate-fade-in">
      <section className="relative overflow-hidden rounded-2xl border border-border bg-[linear-gradient(135deg,var(--color-muted),transparent_65%)] p-8">
        <div className="absolute right-0 top-0 h-28 w-28 -translate-y-6 translate-x-6 rounded-full bg-primary/8 blur-2xl" />
        <div className="relative space-y-4">
          <div className="inline-flex items-center gap-2 rounded-full border border-border bg-background/80 px-3 py-1 text-xs text-muted-foreground">
            <Sparkles className="h-3.5 w-3.5" />
            Fast handoff to another LLM
          </div>
          <div className="space-y-2">
            <h1 className="text-2xl font-semibold tracking-tight">Send Compass to your LLM</h1>
            <p className="max-w-2xl text-sm leading-6 text-muted-foreground">
              Copy the text below and paste it into the LLM you want to use. The text is just a guide link. That guide tells
              the model how to connect to Compass MCP, store a usage skill, and verify the setup.
            </p>
          </div>
        </div>
      </section>

      <section className="grid gap-3 md:grid-cols-3">
        <StepCard
          index="1"
          title="Copy the link"
          description="Use the button below. The copied text is the guide URL itself, with no extra prompt required."
        />
        <StepCard
          index="2"
          title="Paste it into the LLM"
          description="Send the link in the target chat so the model can read the guide directly."
        />
        <StepCard
          index="3"
          title="Let it configure itself"
          description="The guide asks the model to add Compass MCP and keep a short usage skill for later."
        />
      </section>

      <Card className="overflow-hidden">
        <CardHeader className="pb-4">
          <div className="flex items-start justify-between gap-4">
            <div className="space-y-1">
              <div className="inline-flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">
                <Link2 className="h-3.5 w-3.5" />
                Text To Copy
              </div>
              <CardTitle className="text-lg">Paste this into your LLM</CardTitle>
            </div>
            <div className="text-right text-xs text-muted-foreground">
              The text below is exactly what should be copied.
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <textarea
            ref={textRef}
            readOnly
            value={guideUrl}
            onFocus={(e) => e.currentTarget.select()}
            onClick={(e) => e.currentTarget.select()}
            className="min-h-[96px] w-full resize-none rounded-xl border border-border bg-muted/60 px-4 py-3 font-mono text-sm leading-6 focus:outline-none focus:ring-2 focus:ring-ring"
            aria-label="MCP guide link to copy"
          />

          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={copyText} className="min-w-[148px]">
              {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
              {copied ? 'Copied' : 'Copy link'}
            </Button>
            <Button variant="outline" asChild>
              <a href={guideUrl} target="_blank" rel="noopener noreferrer">
                <ExternalLink className="h-4 w-4" />
                Open raw guide
              </a>
            </Button>
            <span className="text-sm text-muted-foreground">
              If clipboard access is blocked, click the text box and copy manually.
            </span>
          </div>

          {copyError && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-600 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300">
              {copyError}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">What the guide tells the LLM to do</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <p>Configure the Compass MCP endpoint in a compatible client.</p>
          <p>Store a short skill prompt so future paper-related requests use Compass correctly.</p>
          <p>Verify the connection by listing tools or calling <code className="font-mono text-foreground">list_papers</code>.</p>
        </CardContent>
      </Card>
    </div>
  )
}
