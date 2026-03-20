export default function ScoreBadge({ score }: { score: number }) {
  const pct = Math.round(score * 100)
  const hue = Math.round(score * 120)
  return (
    <span
      className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold text-white"
      style={{ backgroundColor: `hsl(${hue}, 70%, var(--score-lightness))` }}
    >
      {pct}%
    </span>
  )
}
