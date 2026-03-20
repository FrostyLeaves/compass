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

function fixDisplayMath(markdown: string): string {
  return markdown.replace(/^\$\$(.+)\$\$$/gm, '$$$$\n$1\n$$$$')
}

interface MarkdownContentProps {
  content: string
  folder?: string
  className?: string
}

export default function MarkdownContent({ content, folder = '', className = '' }: MarkdownContentProps) {
  return (
    <article className={className}>
      <ReactMarkdown
        remarkPlugins={[remarkMath, remarkGfm]}
        rehypePlugins={[[rehypeSanitize, sanitizeSchema], rehypeKatex]}
        components={{
          img: ({ src, alt, ...props }) => {
            let resolvedSrc = src || ''
            const safeFolder = folder.replace(/\.\./g, '').replace(/[/\\]/g, '')
            if (resolvedSrc && safeFolder && !resolvedSrc.startsWith('http') && !resolvedSrc.startsWith('/')) {
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
            if (id && !children) return null
            return <span {...props}>{children}</span>
          },
        }}
      >
        {fixDisplayMath(content)}
      </ReactMarkdown>
    </article>
  )
}
