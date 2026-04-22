import React from 'react'

const agents = [
  { key: 'ql',     num: '1', name: 'Q-Learning',  desc: 'Greedy policy from trained Q-table (decision tree panel)' },
  { key: 'aco',    num: '2', name: 'ACO',          desc: 'Ant Colony Optimisation (pheromone heatmap panel)' },
  { key: 'beam',   num: '3', name: 'Beam Search',  desc: 'Adaptive beam search, depth 3 (beam tree panel)' },
  { key: 'random', num: '4', name: 'Random',       desc: 'Uniform random baseline' },
]

export default function AgentSelect({ onSelect }) {
  return (
    <div className="select-screen">
      <div className="select-box">
        <div className="select-title">Elevator Simulator</div>
        <div className="select-sub">Scheduling as an MDP · Select Agent</div>
        <div className="agent-options">
          {agents.map((a) => (
            <div key={a.key} className="agent-option" onClick={() => onSelect(a.key)}>
              <div className="agent-option-num">{a.num}</div>
              <div>
                <div className="agent-option-name">{a.name}</div>
                <div className="agent-option-desc">{a.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
