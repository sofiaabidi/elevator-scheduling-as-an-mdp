import React, { useState, useEffect, useRef, useCallback } from 'react'
import Building from './components/Building.jsx'
import Metrics from './components/Metrics.jsx'
import DecisionTreePanel from './components/DecisionTreePanel.jsx'
import ACOPanel from './components/ACOPanel.jsx'
import BeamPanel from './components/BeamPanel.jsx'
import AgentSelect from './components/AgentSelect.jsx'

const API = '/api'

export default function App() {
  const [agentMode, setAgentMode] = useState(null)  // null = show select screen
  const [state, setState] = useState(null)
  const [running, setRunning] = useState(false)
  const [speed, setSpeed] = useState(6)  // steps per second
  const intervalRef = useRef(null)
  const runningRef = useRef(false)

  const doInit = useCallback(async (mode) => {
    try {
      const res = await fetch(`${API}/init`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent: mode }),
      })
      if (!res.ok) { console.error('init failed', res.status); return }
      const data = await res.json()
      setState(data.state)
      setAgentMode(mode)
      setRunning(false)
    } catch (err) { console.error('init error', err) }
  }, [])

  const stopRunning = useCallback(() => {
    setRunning(false)
    runningRef.current = false
    clearInterval(intervalRef.current)
  }, [])

  const doStep = useCallback(async () => {
    try {
      const res = await fetch(`${API}/step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      if (!res.ok) {
        console.error('step failed', res.status)
        if (res.status === 400) stopRunning()
        return
      }
      const data = await res.json()
      if (data.state) setState(data.state)
    } catch (err) {
      console.error('step error', err)
      stopRunning()
    }
  }, [stopRunning])

  const doMultiStep = useCallback(async (n = 10) => {
    try {
      const res = await fetch(`${API}/multi_step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n }),
      })
      if (!res.ok) { console.error('multi_step failed', res.status); return }
      const data = await res.json()
      if (data.state) setState(data.state)
    } catch (err) { console.error('multi_step error', err) }
  }, [])

  // Auto-step loop
  useEffect(() => {
    runningRef.current = running
    if (running) {
      intervalRef.current = setInterval(() => {
        if (runningRef.current) doStep()
      }, 1000 / speed)
    } else {
      clearInterval(intervalRef.current)
    }
    return () => clearInterval(intervalRef.current)
  }, [running, speed, doStep])

  if (!agentMode) {
    return <AgentSelect onSelect={(mode) => doInit(mode)} />
  }

  const agentLabels = {
    ql: 'Q-LEARNING',
    aco: 'ACO',
    beam: 'BEAM SEARCH',
    random: 'RANDOM',
  }

  return (
    <div className="app">
      {/* Header */}
      <div className="header">
        <div>
          <div className="header-title">Elevator Simulator</div>
          <div className="header-sub">
            Agent: {agentLabels[agentMode] || agentMode.toUpperCase()}
            {state && ` · Step ${state.step}/${state.max_steps}`}
          </div>
        </div>
        <div className="control-bar">
          <button
            className={`btn ${running ? 'red' : 'green'}`}
            onClick={() => setRunning(!running)}
          >
            {running ? '■ STOP' : '▶ RUN'}
          </button>
          <button className="btn" onClick={doStep} disabled={running}>
            STEP
          </button>
          <button className="btn" onClick={() => doMultiStep(20)} disabled={running}>
            +20
          </button>
          <button className="btn" onClick={() => doMultiStep(50)} disabled={running}>
            +50
          </button>
          <button
            className="btn"
            onClick={() => setSpeed(Math.max(1, speed - 2))}
          >
            −
          </button>
          <span className="speed-display">{speed} sps</span>
          <button
            className="btn"
            onClick={() => setSpeed(Math.min(30, speed + 2))}
          >
            +
          </button>
          <button
            className="btn"
            onClick={() => { setRunning(false); doInit(agentMode) }}
          >
            RESET
          </button>
          <button
            className="btn"
            onClick={() => { setRunning(false); setAgentMode(null); setState(null) }}
          >
            BACK
          </button>
        </div>
      </div>

      {/* Main area */}
      <div className="main-area">
        <div className="panel-building">
          <div className="panel-label">Building</div>
          {state && <Building state={state} />}
        </div>
        <div className="panel-metrics">
          <div className="panel-label">Metrics</div>
          {state && <Metrics state={state} />}
        </div>
        <div className="panel-vis">
          {state && agentMode === 'ql' && state.decision_tree && (
            <DecisionTreePanel tree={state.decision_tree} />
          )}
          {state && agentMode === 'aco' && state.aco && (
            <ACOPanel data={state.aco} state={state} />
          )}
          {state && agentMode === 'beam' && state.beam && (
            <BeamPanel data={state.beam} state={state} />
          )}
          {state && agentMode === 'random' && (
            <div>
              <div className="vis-title">Random Policy</div>
              <div className="vis-subtitle">No visualization — actions are uniformly random.</div>
              {state.action_name && (
                <div style={{ marginTop: 20 }}>
                  <span className="action-badge" style={{ fontSize: 16 }}>
                    Last action: {state.action_name}
                  </span>
                  <div style={{ marginTop: 8, fontSize: 13, color: 'var(--dim)' }}>
                    Reward: {state.reward}
                  </div>
                </div>
              )}
            </div>
          )}
          {state && !state.decision_tree && !state.aco && !state.beam && agentMode !== 'random' && (
            <div className="vis-subtitle">
              Press STEP or RUN to start the simulation.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
