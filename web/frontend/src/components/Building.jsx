import React from 'react'

function waitColor(age) {
  if (age < 20) return 'green'
  if (age < 50) return 'yellow'
  return 'red'
}

export default function Building({ state }) {
  const { floors, floor, direction, waiting, waiting_counts, hall_up, hall_down,
          inside_count, max_capacity, inside_dst } = state

  const dirArrow = ['↓', '—', '↑'][direction]
  const floorRows = []

  for (let f = floors - 1; f >= 0; f--) {
    const isCurrent = f === floor
    const paxList = waiting[f] || []
    const nWait = waiting_counts[f] || 0
    const showPax = paxList.slice(0, 8)

    floorRows.push(
      <div key={f} className="floor-row">
        <div className={`floor-label ${isCurrent ? 'current' : ''}`}>F{f}</div>
        <div className="shaft">
          {isCurrent && (
            <div className="elevator-car">
              <div className="dir-arrow">{dirArrow}</div>
              <div className="cap">{inside_count}/{max_capacity}</div>
            </div>
          )}
        </div>
        <div className="pax-area">
          {showPax.map((p, i) => (
            <div key={i} className={`pax-dot ${waitColor(p.age)}`}
                 title={`F${p.src}→F${p.dst} age:${p.age}`} />
          ))}
          {nWait > 8 && <span className="pax-overflow">+{nWait - 8}</span>}
        </div>
        <div className="hall-arrows">
          {hall_up[f] === 1 && <span className="arrow-up">▲</span>}
          {hall_down[f] === 1 && <span className="arrow-down">▼</span>}
        </div>
      </div>
    )
  }

  return (
    <div className="building">
      {floorRows}
      {/* State space mini-grid */}
      <div style={{ padding: '12px 0 0' }}>
        <div style={{ fontSize: 10, color: 'var(--dim)', fontWeight: 600,
                      letterSpacing: 1, textTransform: 'uppercase', marginBottom: 6 }}>
          State Flags
        </div>
        <div className="state-grid">
          <div className="state-grid-row">
            <div className="state-label"></div>
            <div className="state-header">IN</div>
            <div className="state-header">H↑</div>
            <div className="state-header">H↓</div>
          </div>
          {Array.from({ length: floors }, (_, i) => floors - 1 - i).map(f => (
            <div key={f} className="state-grid-row">
              <div className="state-label" style={{ color: f === floor ? 'var(--accent)' : undefined }}>
                F{f}
              </div>
              <div className={`state-cell ${inside_dst[f] ? 'on' : 'off'}`}
                   style={{ background: inside_dst[f] ? 'rgba(180,100,255,0.3)' : undefined }}>
                <div className="state-cell-dot"
                     style={{ background: inside_dst[f] ? 'var(--purple)' : 'var(--grid)' }} />
              </div>
              <div className={`state-cell ${hall_up[f] ? 'on' : 'off'}`}
                   style={{ background: hall_up[f] ? 'rgba(60,210,120,0.2)' : undefined }}>
                <div className="state-cell-dot"
                     style={{ background: hall_up[f] ? 'var(--green)' : 'var(--grid)' }} />
              </div>
              <div className={`state-cell ${hall_down[f] ? 'on' : 'off'}`}
                   style={{ background: hall_down[f] ? 'rgba(255,80,80,0.2)' : undefined }}>
                <div className="state-cell-dot"
                     style={{ background: hall_down[f] ? 'var(--red)' : 'var(--grid)' }} />
              </div>
            </div>
          ))}
        </div>
        {/* Lambda badge */}
        <div className={`lambda-badge ${state.phase === 'PEAK' ? 'peak' : 'offpeak'}`}>
          λ={state.lam?.toFixed(2)} {state.phase}
        </div>
      </div>
    </div>
  )
}
