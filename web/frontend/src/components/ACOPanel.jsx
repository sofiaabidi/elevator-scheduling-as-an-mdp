import React from 'react'

function heatColor(val, vmin, vmax) {
  let t = (val - vmin) / Math.max(1e-6, vmax - vmin)
  t = Math.max(0, Math.min(1, t))
  if (t < 0.33) {
    const s = t / 0.33
    return `rgb(${Math.round(20 + 40 * s)}, ${Math.round(40 + 140 * s)}, ${Math.round(80 + 160 * s)})`
  } else if (t < 0.66) {
    const s = (t - 0.33) / 0.33
    return `rgb(${Math.round(60 + 195 * s)}, ${Math.round(180 + 25 * s)}, ${Math.round(240 - 130 * s)})`
  } else {
    const s = (t - 0.66) / 0.34
    return `rgb(255, ${Math.round(205 + 20 * s)}, ${Math.round(110 + 110 * s)})`
  }
}

export default function ACOPanel({ data, state }) {
  if (!data) return null

  const { tau, tau_min, tau_max, best_sequence, best_fitness, n_ants, evap_rate } = data
  const floors = tau.length

  const seqStr = best_sequence.map(f => `F${f}`).join(' → ')

  return (
    <div>
      <div className="vis-title">ACO — Ant Colony Optimisation</div>
      <div className="vis-subtitle">
        Heatmap: pheromone strength τ[from→to] · Path: best ant this step
      </div>

      <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap' }}>
        {/* Heatmap */}
        <div>
          <div style={{ fontSize: 11, color: 'var(--accent)', fontWeight: 700,
                        letterSpacing: 1, marginBottom: 8 }}>
            PHEROMONE τ[from→to]
          </div>
          {/* Column headers */}
          <div style={{ display: 'flex', marginLeft: 28 }}>
            {Array.from({ length: floors }, (_, i) => (
              <div key={i} className="heatmap-axis" style={{ width: 56 }}>F{i}</div>
            ))}
          </div>
          {/* Grid rows */}
          {tau.map((row, r) => (
            <div key={r} style={{ display: 'flex', alignItems: 'center' }}>
              <div className="heatmap-axis" style={{ width: 28 }}>F{r}</div>
              {row.map((val, c) => {
                const isDiag = r === c
                const bg = isDiag ? '#14161e' : heatColor(val, tau_min, tau_max)
                const textColor = isDiag ? 'var(--grid)' : (val > (tau_min + tau_max) / 2 ? '#fff' : 'var(--dim)')
                return (
                  <div
                    key={c}
                    className="heatmap-cell"
                    style={{ background: bg, width: 56, height: 48, margin: 1.5 }}
                  >
                    {isDiag ? (
                      <span style={{ color: 'var(--grid)', fontSize: 14 }}>—</span>
                    ) : (
                      <span style={{ color: textColor }}>{val.toFixed(1)}</span>
                    )}
                  </div>
                )
              })}
            </div>
          ))}

          {/* Colour scale */}
          <div style={{ display: 'flex', alignItems: 'center', marginTop: 8, marginLeft: 28 }}>
            <span style={{ fontSize: 9, color: 'var(--dim)', marginRight: 6 }}>{tau_min.toFixed(1)}</span>
            <div style={{
              flex: 1, height: 8, borderRadius: 4, maxWidth: floors * 56,
              background: `linear-gradient(to right, ${heatColor(tau_min, tau_min, tau_max)}, ${heatColor((tau_min + tau_max) / 2, tau_min, tau_max)}, ${heatColor(tau_max, tau_min, tau_max)})`
            }} />
            <span style={{ fontSize: 9, color: '#fff', marginLeft: 6 }}>{tau_max.toFixed(1)}</span>
          </div>
        </div>

        {/* Best ant path (mini building) */}
        <div style={{ minWidth: 180 }}>
          <div style={{ fontSize: 11, color: 'var(--accent)', fontWeight: 700,
                        letterSpacing: 1, marginBottom: 8 }}>
            BEST ANT PATH
          </div>
          <svg width={180} height={floors * 60 + 20} style={{ display: 'block' }}>
            {/* Floor lines + labels */}
            {Array.from({ length: floors }, (_, f) => {
              const y = (floors - 1 - f) * 60 + 20
              return (
                <g key={f}>
                  <line x1={10} y1={y} x2={170} y2={y} stroke="var(--grid)" strokeWidth={1} />
                  <text x={4} y={y + 4} fill="var(--dim)" fontSize={9}
                        fontFamily="var(--font)" fontWeight={600}>F{f}</text>
                </g>
              )
            })}
            {/* Shaft */}
            <rect x={78} y={5} width={24} height={floors * 60} rx={3}
                  fill="var(--bg-card)" />
            {/* Path edges */}
            {best_sequence.length >= 2 && best_sequence.slice(1).map((f, i) => {
              const prevF = best_sequence[i]
              const y0 = (floors - 1 - prevF) * 60 + 20
              const y1 = (floors - 1 - f) * 60 + 20
              const frac = (i + 1) / best_sequence.length
              return (
                <line key={i}
                      x1={90} y1={y0} x2={90} y2={y1}
                      stroke={`rgba(80, ${160 + Math.round(50 * (1 - frac))}, ${Math.round(255 * (1 - frac))}, 0.8)`}
                      strokeWidth={3} />
              )
            })}
            {/* Path nodes */}
            {best_sequence.map((f, i) => {
              const y = (floors - 1 - f) * 60 + 20
              return (
                <circle key={i} cx={90} cy={y} r={6}
                        fill="var(--accent)" opacity={1 - i * 0.1} />
              )
            })}
            {/* Elevator */}
            <rect x={76} y={(floors - 1 - state.floor) * 60 + 6} width={28} height={28}
                  rx={4} fill="var(--elev)" stroke="var(--accent)" strokeWidth={2} />
            <text x={90} y={(floors - 1 - state.floor) * 60 + 24}
                  fill="#fff" fontSize={10} fontFamily="var(--font)" fontWeight={700}
                  textAnchor="middle">
              F{state.floor}
            </text>
          </svg>
        </div>
      </div>

      {/* Path sequence */}
      <div className="aco-path">
        <div className="aco-path-label">Sequence</div>
        <div className="aco-path-seq">{seqStr || '—'}</div>
      </div>

      {/* Metrics strip */}
      <div className="aco-metrics-strip">
        {[
          { label: 'Colony', value: `${n_ants} ants`, color: 'var(--accent)' },
          { label: 'Evap ρ', value: evap_rate.toFixed(2), color: 'var(--yellow)' },
          { label: 'Best Fit', value: `${best_fitness > 0 ? '+' : ''}${best_fitness}`, color: best_fitness >= 0 ? 'var(--green)' : 'var(--red)' },
          { label: 'τ max', value: tau_max.toFixed(2), color: '#fff' },
          { label: 'τ min', value: tau_min.toFixed(2), color: 'var(--dim)' },
          { label: 'Pending', value: `${state.total_waiting}`, color: state.total_waiting > 4 ? 'var(--red)' : '#fff' },
          { label: 'Served', value: `${state.served}`, color: 'var(--green)' },
        ].map((m, i) => (
          <div key={i} className="aco-metric">
            <div className="aco-metric-label">{m.label}</div>
            <div className="aco-metric-value" style={{ color: m.color }}>{m.value}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
