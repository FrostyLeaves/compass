export default function ScoreBadge({ score }: { score: number }) {
  const normalized = Math.max(0, Math.min(score, 1))
  const hue = Math.round(normalized * 120)
  return (
    <span
      className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold text-white"
      style={{ backgroundColor: `hsl(${hue}, 70%, var(--score-lightness))` }}
    >
      {score.toFixed(3)}
    </span>
  )
}
