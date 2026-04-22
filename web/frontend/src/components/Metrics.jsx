import React from 'react'

export default function Metrics({ state }) {
  const {
    step, max_steps, served, avg_wait, total_waiting,
    inside_count, max_capacity, floor, direction,
    action_name, reward, step_history,
  } = state

  const dirLabel = ['↓', '—', '↑'][direction]
  const progress = step / Math.max(1, max_steps)

  const rows = [
    { label: 'Step',       value: `${step}/${max_steps}`,       cls: 'white' },
    { label: 'Served',     value: `${served}`,                  cls: 'green' },
    { label: 'Avg Wait',   value: `${avg_wait?.toFixed(1)} steps`, cls: 'yellow' },
    { label: 'Pending',    value: `${total_waiting}`,           cls: total_waiting > 4 ? 'red' : 'white' },
    { label: 'In Car',     value: `${inside_count}/${max_capacity}`, cls: 'accent' },
    { label: 'Floor',      value: `${floor}`,                   cls: 'white' },
    { label: 'Direction',  value: dirLabel,                     cls: 'accent' },
  ]

  if (action_name) {
    rows.push({ label: 'Last Action', value: action_name, cls: 'accent' })
    rows.push({ label: 'Reward', value: `${reward >= 0 ? '+' : ''}${reward?.toFixed(1)}`, cls: reward >= 0 ? 'green' : 'red' })
  }

  // Sparkline
  const hist = step_history || []
  const sparkW = 200
  const sparkH = 40

  let sparkPath = ''
  if (hist.length > 1) {
    const minV = Math.min(...hist)
    const maxV = Math.max(...hist)
    const range = maxV - minV || 1
    const pts = hist.map((v, i) => {
      const x = (i / (hist.length - 1)) * sparkW
      const y = sparkH - ((v - minV) / range) * (sparkH - 4) - 2
      return `${x},${y}`
    })
    sparkPath = 'M' + pts.join(' L')
  }

  return (
    <div className="metrics-list">
      {rows.map((r, i) => (
        <div key={i} className="metric-item">
          <div className="metric-label">{r.label}</div>
          <div className={`metric-value ${r.cls}`}>{r.value}</div>
        </div>
      ))}

      {/* Progress bar */}
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${progress * 100}%` }} />
      </div>

      {/* Sparkline */}
      {hist.length > 2 && (
        <div className="sparkline">
          <div className="sparkline-label">Reward (last {hist.length} steps)</div>
          <svg width={sparkW} height={sparkH} style={{ display: 'block' }}>
            <path d={sparkPath} fill="none" stroke="var(--accent)" strokeWidth="1.5" opacity="0.7" />
          </svg>
        </div>
      )}
    </div>
  )
}
