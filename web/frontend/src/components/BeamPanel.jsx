import React from 'react'

const ACTION_NAMES = ['UP ↑', 'DOWN ↓', 'STAY —', 'OPEN ⬚']
const ACTION_COLORS = ['#50d278', '#dc5a50', '#b4b450', '#50a0f0']

function scoreColor(score, lo = -8, hi = 40) {
  let t = (score - lo) / Math.max(1e-6, hi - lo)
  t = Math.max(0, Math.min(1, t))
  if (t < 0.5) {
    const s = t / 0.5
    return `rgb(${Math.round(240 - 70 * s)}, ${Math.round(70 + 135 * s)}, 70)`
  }
  const s = (t - 0.5) / 0.5
  return `rgb(${Math.round(170 - 110 * s)}, ${Math.round(205 + 5 * s)}, ${Math.round(70 + 50 * s)})`
}

export default function BeamPanel({ data, state }) {
  if (!data) return null

  const { levels, best_path, best_score, current_beam_width, action_scores, last_pending } = data

  const bestPrefixes = new Set()
  for (let i = 1; i <= best_path.length; i++) {
    bestPrefixes.add(best_path.slice(0, i).join(','))
  }

  const W = 780
  const TREE_Y = [55, 140, 225, 310]
  const NODE_R = [16, 10, 7, 7]

  // Compute positions
  const pos = {}
  pos[''] = { x: W / 2, y: TREE_Y[0] }

  levels.forEach((level, d) => {
    if (!level.length) return
    const sorted = [...level].sort((a, b) => {
      const sa = a.seq.join(',')
      const sb = b.seq.join(',')
      return sa < sb ? -1 : sa > sb ? 1 : 0
    })
    const span = W - 60
    sorted.forEach((entry, i) => {
      const x = 30 + ((i + 0.5) * span) / Math.max(1, sorted.length)
      const y = TREE_Y[d + 1]
      pos[entry.seq.join(',')] = { x, y }
    })
  })

  // K tag
  const kTag = current_beam_width <= 2 ? 'narrow' : current_beam_width <= 4 ? 'medium' : 'wide'
  const kColor = current_beam_width >= 6 ? 'var(--red)' : current_beam_width >= 4 ? 'var(--yellow)' : 'var(--green)'

  return (
    <div>
      <div className="vis-title">Beam Search — adaptive K, depth 3</div>
      <div className="vis-subtitle">
        pending={last_pending} · K={current_beam_width} · best_score={best_score > 0 ? '+' : ''}{best_score} · chosen={best_path.length > 0 ? ACTION_NAMES[best_path[0]] : '—'}
      </div>

      <div className="beam-k-badge" style={{ borderColor: kColor, color: kColor }}>
        <span style={{ fontWeight: 800 }}>K={current_beam_width}</span>
        <span style={{ fontSize: 9, color: 'var(--dim)' }}>{kTag}</span>
      </div>

      <svg width={W} height={360} style={{ display: 'block' }}>
        {/* Depth labels */}
        {TREE_Y.map((y, d) => (
          <text key={d} x={6} y={y + 4} fill="var(--dim)" fontSize={9}
                fontFamily="var(--font)" fontWeight={600}>d{d}</text>
        ))}

        {/* Edges */}
        {levels.map((level, d) =>
          level.map((entry, idx) => {
            const seqKey = entry.seq.join(',')
            const parentKey = entry.seq.slice(0, -1).join(',')
            const from = pos[parentKey]
            const to = pos[seqKey]
            if (!from || !to) return null

            const isBest = bestPrefixes.has(seqKey)
            let col = 'var(--grid)'
            let lw = 1
            if (isBest) { col = 'var(--gold)'; lw = 3 }
            else if (entry.survived) { col = '#5a647a'; lw = 1 }

            return (
              <line key={`e-${d}-${idx}`}
                    x1={from.x} y1={from.y + NODE_R[d]}
                    x2={to.x} y2={to.y - NODE_R[d + 1]}
                    stroke={col} strokeWidth={lw} />
            )
          })
        )}

        {/* Nodes */}
        {levels.map((level, d) =>
          level.map((entry, idx) => {
            const seqKey = entry.seq.join(',')
            const p = pos[seqKey]
            if (!p) return null
            const r = NODE_R[d + 1]
            const isBest = bestPrefixes.has(seqKey)

            if (entry.survived) {
              const ncol = scoreColor(entry.score)
              return (
                <g key={`n-${d}-${idx}`}>
                  <circle cx={p.x} cy={p.y} r={r} fill={ncol} />
                  {isBest && (
                    <circle cx={p.x} cy={p.y} r={r + 3} fill="none"
                            stroke="var(--gold)" strokeWidth={2} />
                  )}
                  {d === 0 && (
                    <text x={p.x} y={p.y + r + 12} fill="var(--text)" fontSize={9}
                          fontFamily="var(--font)" fontWeight={600} textAnchor="middle">
                      {ACTION_NAMES[entry.seq[0]]?.replace(/ .+/, '')}
                    </text>
                  )}
                </g>
              )
            } else {
              return (
                <g key={`n-${d}-${idx}`}>
                  <circle cx={p.x} cy={p.y} r={r} fill="#2d3044"
                          stroke="#23263a" strokeWidth={1} />
                </g>
              )
            }
          })
        )}

        {/* Root */}
        {pos[''] && (
          <g>
            <circle cx={pos[''].x} cy={pos[''].y} r={NODE_R[0]} fill="#3c82dc"
                    stroke="var(--accent)" strokeWidth={2} />
            <text x={pos[''].x} y={pos[''].y + 4} fill="#fff" fontSize={10}
                  fontFamily="var(--font)" fontWeight={800} textAnchor="middle">
              F{state.floor}
            </text>
          </g>
        )}
      </svg>

      {/* Action score strip */}
      <div style={{ fontSize: 11, color: 'var(--dim)', fontWeight: 600, letterSpacing: 1,
                    textTransform: 'uppercase', marginBottom: 8 }}>
        Action Scores (best leaf reachable)
      </div>
      <div className="beam-strip">
        {ACTION_NAMES.map((name, i) => {
          const chosen = best_path.length > 0 && best_path[0] === i
          const score = action_scores[i]
          const barT = Math.max(0, Math.min(1, (score + 10) / 50))
          return (
            <div key={i} className={`beam-action-card ${chosen ? 'chosen' : ''}`}>
              <div className="beam-action-name" style={{ color: ACTION_COLORS[i] }}>{name}</div>
              <div className="beam-action-score" style={{ color: scoreColor(score) }}>
                {score > 0 ? '+' : ''}{score}
              </div>
              <div className="beam-action-bar">
                <div className="beam-action-bar-fill"
                     style={{ width: `${barT * 100}%`, background: scoreColor(score) }} />
              </div>
              <div className="beam-action-tag"
                   style={{ color: chosen ? 'var(--gold)' : 'var(--dim)' }}>
                {chosen ? 'CHOSEN' : 'rejected'}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
