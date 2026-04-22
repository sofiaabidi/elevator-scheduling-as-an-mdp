import React from 'react'

const ACTION_NAMES = ['UP ↑', 'DOWN ↓', 'STAY —', 'OPEN ⬚']
const ACTION_COLORS = ['#50c878', '#dc5a50', '#b4b450', '#50a0f0']

function rewardColor(r, valid) {
  if (!valid) return '#373a4b'
  if (r >= 10) return '#32c870'
  if (r >= 0) {
    const t = r / 10
    return `rgb(${Math.round(50 + 150 * t)}, ${Math.round(150 + 50 * t)}, ${Math.round(110 - 30 * t)})`
  }
  const t = Math.min(1, Math.abs(r) / 10)
  return `rgb(${Math.round(180 + 60 * t)}, ${Math.round(80 - 30 * t)}, ${Math.round(70 - 20 * t)})`
}

export default function DecisionTreePanel({ tree }) {
  if (!tree) return null

  const W = 820
  const H = 420
  const bestPath = tree.best_path || [0, 0]

  // Layout positions
  const rootX = W / 2, rootY = 50, rootR = 22
  const d1Y = 160, d1R = 16
  const d2Y = 310, d2R = 12

  const spread = (n, margin) => {
    const step = (W - 2 * margin) / Math.max(1, n - 1)
    return Array.from({ length: n }, (_, i) => margin + i * step)
  }

  const d1Xs = spread(4, 100)
  const d2Xs = spread(16, 30)

  const children = tree.children || []

  return (
    <div>
      <div className="vis-title">Decision Tree (depth-2 lookahead)</div>
      <div className="vis-subtitle">
        Current: {tree.state} · Best path reward: {tree.best_cumulative}
      </div>
      <div className="vis-subtitle" style={{ marginTop: -10 }}>
        greyed = invalid · gold border = best path
      </div>

      <svg width={W} height={H} style={{ display: 'block' }}>
        {/* Edges: root → d1 */}
        {children.map((c1, i) => {
          const isBest = i === bestPath[0]
          const col = c1.valid ? ACTION_COLORS[c1.action] : '#373a4b'
          return (
            <g key={`e1-${i}`}>
              <line
                x1={rootX} y1={rootY + rootR}
                x2={d1Xs[i]} y2={d1Y - d1R}
                stroke={col} strokeWidth={isBest ? 3 : 1} opacity={isBest ? 1 : 0.6}
              />
              {/* Action label on edge */}
              <text
                x={(rootX + d1Xs[i]) / 2}
                y={(rootY + d1Y) / 2 - 4}
                fill={col} fontSize={9} fontFamily="var(--font)" fontWeight={600}
                textAnchor="middle"
              >
                {ACTION_NAMES[c1.action]}
              </text>
            </g>
          )
        })}

        {/* Edges: d1 → d2 */}
        {children.map((c1, i) =>
          c1.children.map((c2, j) => {
            const gi = i * 4 + j
            const isBest = i === bestPath[0] && j === bestPath[1]
            const col = c2.valid ? ACTION_COLORS[c2.action] : '#373a4b'
            return (
              <line
                key={`e2-${gi}`}
                x1={d1Xs[i]} y1={d1Y + d1R}
                x2={d2Xs[gi]} y2={d2Y - d2R}
                stroke={col} strokeWidth={isBest ? 3 : 1} opacity={isBest ? 1 : 0.4}
              />
            )
          })
        )}

        {/* D2 nodes */}
        {children.map((c1, i) =>
          c1.children.map((c2, j) => {
            const gi = i * 4 + j
            const isBest = i === bestPath[0] && j === bestPath[1]
            const ncol = rewardColor(c2.cumulative, c2.valid)
            return (
              <g key={`n2-${gi}`}>
                <circle cx={d2Xs[gi]} cy={d2Y} r={d2R} fill={ncol}
                        stroke={isBest ? 'var(--gold)' : 'rgba(255,255,255,0.1)'}
                        strokeWidth={isBest ? 3 : 1} />
                <text x={d2Xs[gi]} y={d2Y + 3}
                      fill={c2.valid ? '#dce1f0' : '#5a5f7a'}
                      fontSize={9} fontFamily="var(--font)" fontWeight={700}
                      textAnchor="middle">
                  {c2.cumulative > 0 ? '+' : ''}{c2.cumulative}
                </text>
                <text x={d2Xs[gi]} y={d2Y + d2R + 12}
                      fill="var(--dim)" fontSize={8} fontFamily="var(--font)"
                      textAnchor="middle">
                  {c2.label}
                </text>
              </g>
            )
          })
        )}

        {/* D1 nodes (drawn on top) */}
        {children.map((c1, i) => {
          const isBest = i === bestPath[0]
          const ncol = rewardColor(c1.reward, c1.valid)
          return (
            <g key={`n1-${i}`}>
              <circle cx={d1Xs[i]} cy={d1Y} r={d1R} fill={ncol}
                      stroke={isBest ? 'var(--gold)' : 'rgba(255,255,255,0.1)'}
                      strokeWidth={isBest ? 3 : 1} />
              <text x={d1Xs[i]} y={d1Y + 4}
                    fill={c1.valid ? '#dce1f0' : '#5a5f7a'}
                    fontSize={11} fontFamily="var(--font)" fontWeight={700}
                    textAnchor="middle">
                {c1.reward > 0 ? '+' : ''}{c1.reward}
              </text>
              <text x={d1Xs[i]} y={d1Y + d1R + 13}
                    fill="var(--dim)" fontSize={9} fontFamily="var(--font)"
                    textAnchor="middle">
                F{c1.floor}
              </text>
            </g>
          )
        })}

        {/* Root node */}
        <circle cx={rootX} cy={rootY} r={rootR} fill="#3282dc"
                stroke="var(--accent)" strokeWidth={3} />
        <text x={rootX} y={rootY + 5}
              fill="#fff" fontSize={12} fontFamily="var(--font)" fontWeight={800}
              textAnchor="middle">
          ROOT
        </text>

        {/* Legend */}
        {ACTION_NAMES.map((name, i) => (
          <g key={`leg-${i}`}>
            <circle cx={30 + i * 130} cy={H - 16} r={6} fill={ACTION_COLORS[i]} />
            <text x={40 + i * 130} y={H - 12}
                  fill={ACTION_COLORS[i]} fontSize={10} fontFamily="var(--font)" fontWeight={600}>
              {name}
            </text>
          </g>
        ))}
      </svg>
    </div>
  )
}
