"""
Flask backend for the Elevator Simulator web UI.
Uses the EXACT same ElevatorEnv, ACOAgent, BeamAgent, and Q-table logic.
"""

import sys, os

# Add parent dir so we can import the original modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
from collections import defaultdict
import copy
import traceback
import json

from elevator_env import ElevatorEnv, Passenger
from aco_agent import ACOAgent
from beam_agent import BeamAgent

app = Flask(__name__, static_folder="frontend/dist", static_url_path="")
CORS(app)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Global simulation state ──────────────────────────────────────────────────

env = None
agent_mode = None
Q = None
aco_agent_inst = None
beam_agent_inst = None
sim_running = False
sim_speed = 8  # steps per second (for frontend polling)
step_history = []  # last N steps of reward for sparkline

QTABLE_PATH = os.path.join(os.path.dirname(__file__), "..", "qtable.pkl")


def load_qtable(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return defaultdict(lambda: np.zeros(4), data)


def _serialize_env_state(env_obj, extra=None):
    """Serialize the full env state to JSON-friendly dict."""
    waiting = []
    for f in range(env_obj.floors):
        floor_pax = []
        for p in env_obj.waiting[f]:
            floor_pax.append({
                "src": p.src,
                "dst": p.dst,
                "age": env_obj._step - p.arrived_step,
            })
        waiting.append(floor_pax)

    inside = []
    for p in env_obj.inside:
        inside.append({
            "dst": p.dst,
            "boarded_step": p.boarded_step,
        })

    # hall flags
    hall_up = [0] * env_obj.floors
    hall_down = [0] * env_obj.floors
    for f in range(env_obj.floors):
        for p in env_obj.waiting[f]:
            if p.dst > p.src:
                hall_up[f] = 1
            else:
                hall_down[f] = 1

    inside_dst = [0] * env_obj.floors
    for p in env_obj.inside:
        inside_dst[p.dst] = 1

    lam = env_obj._current_lam()
    phase = "PEAK" if lam > 0.3 else "OFF-PEAK"

    result = {
        "floor": int(env_obj.floor),
        "direction": int(env_obj.direction),
        "step": int(env_obj._step),
        "max_steps": int(env_obj.max_steps),
        "floors": int(env_obj.floors),
        "max_capacity": int(env_obj.max_capacity),
        "inside_count": len(env_obj.inside),
        "inside": inside,
        "inside_dst": inside_dst,
        "waiting": waiting,
        "waiting_counts": [len(env_obj.waiting[f]) for f in range(env_obj.floors)],
        "total_waiting": int(env_obj.total_waiting()),
        "hall_up": hall_up,
        "hall_down": hall_down,
        "served": int(env_obj._served),
        "avg_wait": round(float(env_obj._total_wait / max(1, env_obj._served)), 2),
        "lam": round(float(lam), 2),
        "phase": phase,
        "step_history": step_history[-100:],
    }
    if extra:
        result.update(extra)
    return result


def _get_decision_tree(env_obj):
    """Compute depth-2 decision tree (same logic as decision_tree_vis.py)."""
    from decision_tree_vis import _env_to_state, _sim_step

    root_state = _env_to_state(env_obj)
    floors = env_obj.floors
    max_cap = env_obj.max_capacity
    action_names = ["UP", "DOWN", "STAY", "OPEN"]

    tree = {"state": _state_short(root_state), "children": []}
    best_cum = -9999
    best_path = (0, 0)

    for a1 in range(4):
        s1, r1, v1, lbl1 = _sim_step(root_state, a1, floors, max_cap)
        node1 = {
            "action": a1,
            "action_name": action_names[a1],
            "reward": round(r1, 1),
            "valid": v1,
            "label": lbl1,
            "cumulative": round(r1, 1),
            "floor": s1["floor"],
            "children": [],
        }
        for a2 in range(4):
            s2, r2, v2, lbl2 = _sim_step(s1, a2, floors, max_cap)
            cum = r1 + r2
            node2 = {
                "action": a2,
                "action_name": action_names[a2],
                "reward": round(r2, 1),
                "valid": v2,
                "label": lbl2,
                "cumulative": round(cum, 1),
                "floor": s2["floor"],
            }
            node1["children"].append(node2)
            if cum > best_cum:
                best_cum = cum
                best_path = (a1, a2)
        tree["children"].append(node1)

    tree["best_path"] = list(best_path)
    tree["best_cumulative"] = round(best_cum, 1)
    return tree


def _state_short(s):
    dirs = ["↓", "—", "↑"]
    hu = "".join(str(v) for v in s["hall_up"])
    hd = "".join(str(v) for v in s["hall_down"])
    inside = len(s["inside_dst"])
    return f"F{s['floor']}{dirs[s['direction']]} [{inside}] H↑{hu} H↓{hd}"


def _get_aco_data(agent, env_obj):
    """Extract ACO visualization data."""
    tau = agent.last_tau.tolist()
    return {
        "tau": tau,
        "tau_min": float(agent.last_tau.min()),
        "tau_max": float(agent.last_tau.max()),
        "best_sequence": agent.best_sequence,
        "best_fitness": round(float(agent.best_fitness), 1),
        "n_ants": agent.n_ants,
        "evap_rate": agent.evap_rate,
    }


def _get_beam_data(agent, env_obj):
    """Extract Beam Search visualization data."""
    levels = []
    for level in agent.levels:
        lvl = []
        for entry in level:
            lvl.append({
                "seq": list(entry["seq"]),
                "score": round(entry["score"], 1),
                "survived": entry["survived"],
            })
        levels.append(lvl)

    return {
        "levels": levels,
        "best_path": agent.best_path,
        "best_score": round(float(agent.best_score), 1),
        "current_beam_width": agent.current_beam_width,
        "action_scores": [round(float(s), 1) for s in agent.action_scores],
        "last_pending": agent.last_pending,
    }


# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/init", methods=["POST"])
def api_init():
    """Initialize (or reinitialize) the simulation."""
    global env, agent_mode, Q, aco_agent_inst, beam_agent_inst, step_history

    try:
        data = request.get_json(force=True, silent=True) or {}
        mode = data.get("agent", "ql")
        floors = data.get("floors", 5)
        steps = data.get("max_steps", 600)
        lam = data.get("lam", 0.25)
        peak = data.get("peak_offpeak", True)

        agent_mode = mode
        step_history = []

        # Create environment (no render mode — headless)
        env = ElevatorEnv(floors=floors, max_steps=steps, lam=lam, peak_offpeak=peak)

        # Setup agent
        Q = None
        aco_agent_inst = None
        beam_agent_inst = None

        if mode == "ql":
            try:
                Q = load_qtable(QTABLE_PATH)
            except FileNotFoundError:
                agent_mode = "random"
        elif mode == "aco":
            aco_agent_inst = ACOAgent(floors=floors)
        elif mode == "beam":
            beam_agent_inst = BeamAgent(floors=floors)

        obs, _ = env.reset()

        return jsonify(_safe_json({
            "status": "ok",
            "agent": agent_mode,
            "state": _serialize_env_state(env),
        }))

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _safe_json(obj):
    """Convert obj to JSON string handling numpy types."""
    return json.loads(json.dumps(obj, cls=NumpyEncoder))


@app.route("/api/step", methods=["POST"])
def api_step():
    """Execute one simulation step. Returns full state."""
    global env, step_history

    if env is None:
        return jsonify({"error": "Not initialized. Call /api/init first."}), 400

    try:
        obs = env._get_obs()

        # Choose action
        if agent_mode == "ql" and Q is not None:
            state = env.obs_to_state(obs)
            action = int(np.argmax(Q[state]))
        elif agent_mode == "aco" and aco_agent_inst is not None:
            action = aco_agent_inst.choose_action(env)
        elif agent_mode == "beam" and beam_agent_inst is not None:
            action = beam_agent_inst.choose_action(env)
        else:
            action = int(env.action_space.sample())

        action_names = ["UP", "DOWN", "STAY", "OPEN"]

        obs, reward, done, trunc, info = env.step(action)
        step_history.append(round(float(reward), 2))

        extra = {
            "action": int(action),
            "action_name": action_names[action],
            "reward": round(float(reward), 2),
            "done": bool(done),
            "truncated": bool(trunc),
            "info": {k: (round(float(v), 2) if isinstance(v, (float, np.floating)) else int(v))
                     for k, v in info.items()},
        }

        # Add agent-specific vis data
        if agent_mode == "ql" and Q is not None:
            extra["decision_tree"] = _get_decision_tree(env)
        elif agent_mode == "aco" and aco_agent_inst is not None:
            extra["aco"] = _get_aco_data(aco_agent_inst, env)
        elif agent_mode == "beam" and beam_agent_inst is not None:
            extra["beam"] = _get_beam_data(beam_agent_inst, env)

        # Auto-reset if done
        if done or trunc:
            obs, _ = env.reset()
            step_history = []

        result = _safe_json({
            "status": "ok",
            "state": _serialize_env_state(env, extra),
        })
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/multi_step", methods=["POST"])
def api_multi_step():
    """Execute N steps at once. Returns final state only (for fast-forward)."""
    global env, step_history

    if env is None:
        return jsonify({"error": "Not initialized"}), 400

    try:
        data = request.get_json(force=True, silent=True) or {}
        n = min(data.get("n", 10), 50)  # cap at 50 per call

        action_names = ["UP", "DOWN", "STAY", "OPEN"]
        last_action = 2
        last_reward = 0.0
        episode_resets = 0

        for _ in range(n):
            obs = env._get_obs()

            if agent_mode == "ql" and Q is not None:
                state = env.obs_to_state(obs)
                action = int(np.argmax(Q[state]))
            elif agent_mode == "aco" and aco_agent_inst is not None:
                action = aco_agent_inst.choose_action(env)
            elif agent_mode == "beam" and beam_agent_inst is not None:
                action = beam_agent_inst.choose_action(env)
            else:
                action = int(env.action_space.sample())

            obs, reward, done, trunc, info = env.step(action)
            step_history.append(round(float(reward), 2))
            last_action = int(action)
            last_reward = float(reward)

            if done or trunc:
                episode_resets += 1
                obs, _ = env.reset()
                step_history = []

        extra = {
            "action": int(last_action),
            "action_name": action_names[last_action],
            "reward": round(last_reward, 2),
            "done": False,
            "truncated": False,
            "episode_resets": episode_resets,
        }

        if agent_mode == "ql" and Q is not None:
            extra["decision_tree"] = _get_decision_tree(env)
        elif agent_mode == "aco" and aco_agent_inst is not None:
            extra["aco"] = _get_aco_data(aco_agent_inst, env)
        elif agent_mode == "beam" and beam_agent_inst is not None:
            extra["beam"] = _get_beam_data(beam_agent_inst, env)

        return jsonify(_safe_json({
            "status": "ok",
            "state": _serialize_env_state(env, extra),
        }))

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
