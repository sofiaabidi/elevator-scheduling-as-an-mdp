"""
run.py — elevator simulation runner.

Usage:
    python run.py                        # interactive agent selector
    python run.py --agent qtable.pkl     # Q-table greedy (decision tree panel)
    python run.py --agent aco            # Ant Colony Optimisation (pheromone panel)
    python run.py --agent beam           # Adaptive Beam Search (beam tree panel)
    python run.py --agent random         # random policy
    python run.py --steps 600 --fps 12
    python run.py --no-render            # headless benchmark
"""

import argparse
import pickle
import traceback
import numpy as np
from collections import defaultdict


def load_qtable(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return defaultdict(lambda: np.zeros(4), data)


def _select_agent_interactive() -> str:
    """
    Ask the user which agent to run. Returns 'ql', 'aco', or 'random'.
    """
    print()
    print("┌─────────────────────────────────────────┐")
    print("│   ELEVATOR SIMULATOR  —  Agent Select   │")
    print("├─────────────────────────────────────────┤")
    print("│  1. Q-Learning   (load qtable.pkl)       │")
    print("│  2. ACO          (Ant Colony Optimiser)  │")
    print("│  3. Beam Search  (adaptive, depth 3)     │")
    print("│  4. Random       (baseline)              │")
    print("└─────────────────────────────────────────┘")
    while True:
        choice = input("Select [1/2/3/4]: ").strip()
        if choice == "1":
            return "ql"
        if choice == "2":
            return "aco"
        if choice == "3":
            return "beam"
        if choice == "4":
            return "random"
        print("  Please enter 1, 2, 3, or 4.")


def run(
    agent_mode    = None,    # 'ql' | 'aco' | 'random' | None=ask
    qtable_path   = "qtable.pkl",
    steps         = 400,
    floors        = 5,
    lam           = 0.25,
    peak_offpeak  = True,
    render        = True,
    fps           = 8,
    print_every   = 50,
):
    # ── agent selection ────────────────────────────────────────────────
    if agent_mode is None:
        agent_mode = _select_agent_interactive()

    print(f"\n[agent] Mode: {agent_mode.upper()}")

    # ── environment ────────────────────────────────────────────────────
    if render:
        try:
            from elevator_env import ElevatorEnvRender as Env
            env = Env(floors=floors, lam=lam,
                      peak_offpeak=peak_offpeak, render_mode="human", fps=fps)
            print("[sim]  Main window: pygame")
        except Exception:
            traceback.print_exc()
            from elevator_env import ElevatorEnv as Env
            env = Env(floors=floors, lam=lam, peak_offpeak=peak_offpeak)
            render = False
            print("[sim]  pygame unavailable — headless")
    else:
        from elevator_env import ElevatorEnv as Env
        env = Env(floors=floors, lam=lam, peak_offpeak=peak_offpeak)

    # ── side-panel visualisation (depends on agent) ────────────────────
    dtree     = None
    aco_vis   = None
    aco_agent = None
    beam_vis   = None
    beam_agent = None
    Q         = None

    if agent_mode == "ql":
        # load Q-table
        try:
            Q = load_qtable(qtable_path)
            print(f"[agent] Q-table loaded from {qtable_path} ({len(Q)} states)")
        except FileNotFoundError:
            print(f"[agent] WARNING: {qtable_path} not found — falling back to random")
            agent_mode = "random"

        # decision-tree panel
        if render:
            try:
                from decision_tree_vis import DecisionTreeVis
                dtree = DecisionTreeVis(floors=floors, max_capacity=env.max_capacity)
                env._side_surf = dtree._screen
                print("[vis]  Decision tree panel: embedded")
            except Exception as e:
                print(f"[vis]  Decision tree skipped: {e}")

    elif agent_mode == "aco":
        from aco_agent import ACOAgent
        aco_agent = ACOAgent(floors=floors)
        print(f"[agent] ACO ready ({aco_agent.n_ants} ants, "
              f"evap={aco_agent.evap_rate}, alpha={aco_agent.alpha}, beta={aco_agent.beta})")

        # pheromone / path panel
        if render:
            try:
                from aco_vis import ACOVis
                aco_vis = ACOVis(floors=floors,
                                 n_ants=aco_agent.n_ants,
                                 evap_rate=aco_agent.evap_rate)
                env._side_surf = aco_vis._screen
                print("[vis]  ACO pheromone panel: embedded")
            except Exception as e:
                print(f"[vis]  ACO vis skipped: {e}")

    elif agent_mode == "beam":
        from beam_agent import BeamAgent
        beam_agent = BeamAgent(floors=floors)
        print(f"[agent] Beam Search ready (depth={beam_agent.DEPTH}, adaptive K ∈ {{2,4,8}})")

        # beam tree panel
        if render:
            try:
                from beam_vis import BeamVis
                beam_vis = BeamVis(floors=floors)
                env._side_surf = beam_vis._screen
                print("[vis]  Beam tree panel: embedded")
            except Exception as e:
                print(f"[vis]  Beam vis skipped: {e}")

    else:
        print("[agent] Random policy")

    # ── main loop ──────────────────────────────────────────────────────
    obs, _       = env.reset()
    total_reward = 0.0
    episode      = 1

    print(f"\n{'Step':>6}  {'Ep':>4}  {'Reward':>8}  {'Served':>6}  {'Pending':>7}  {'AvgWait':>8}")
    print("─" * 52)

    for step in range(steps):
        # ── choose action ──
        if agent_mode == "ql" and Q is not None:
            state  = env.obs_to_state(obs)
            action = int(np.argmax(Q[state]))
        elif agent_mode == "aco" and aco_agent is not None:
            action = aco_agent.choose_action(env)
        elif agent_mode == "beam" and beam_agent is not None:
            action = beam_agent.choose_action(env)
        else:
            action = env.action_space.sample()

        # ── update side panel BEFORE step ──
        if dtree:
            try:
                dtree.update(env)
            except Exception:
                pass
        if aco_vis and aco_agent:
            try:
                aco_vis.update(aco_agent, env)
            except Exception:
                pass
        if beam_vis and beam_agent:
            try:
                beam_vis.update(beam_agent, env)
            except Exception:
                pass

        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward

        if print_every and (step + 1) % print_every == 0:
            print(f"{step+1:>6}  {episode:>4}  {total_reward:>8.1f}  "
                  f"{info['total_served']:>6}  {info['pending']:>7}  "
                  f"{info['avg_wait']:>8.1f}")

        if done or trunc:
            episode += 1
            obs, _ = env.reset()

    # ── cleanup ────────────────────────────────────────────────────────
    if dtree:
        try: dtree.close()
        except: pass
    if aco_vis:
        try: aco_vis.close()
        except: pass
    if beam_vis:
        try: beam_vis.close()
        except: pass
    env.close()

    print(f"\n{'─'*50}")
    print(f"Agent:     {agent_mode.upper()}")
    print(f"Steps:     {steps}")
    print(f"Episodes:  {episode}")
    print(f"Reward:    {total_reward:.1f}")
    print(f"Served:    {info['total_served']}")
    print(f"Avg wait:  {info['avg_wait']:.1f} steps")
    print(f"{'─'*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elevator Simulator")
    parser.add_argument("--agent",        type=str,   default=None,
                        help="'ql', 'aco', 'random', or path to qtable.pkl (omit to be asked)")
    parser.add_argument("--steps",        type=int,   default=400)
    parser.add_argument("--floors",       type=int,   default=5)
    parser.add_argument("--lam",          type=float, default=0.25)
    parser.add_argument("--no-peak",      action="store_true", help="disable peak/off-peak lambda")
    parser.add_argument("--no-render",    action="store_true", help="headless mode")
    parser.add_argument("--fps",          type=int,   default=8,   help="render FPS (1-60, default 8)")
    parser.add_argument("--print-every",  type=int,   default=50,  help="console stats interval (0=off)")
    args = parser.parse_args()

    # resolve --agent flag → agent_mode + qtable_path
    agent_mode  = None
    qtable_path = "qtable.pkl"
    if args.agent is not None:
        a = args.agent.lower()
        if a in ("ql", "qtable", "q"):
            agent_mode = "ql"
        elif a == "aco":
            agent_mode = "aco"
        elif a == "beam":
            agent_mode = "beam"
        elif a == "random":
            agent_mode = "random"
        else:
            # treat as a file path to a qtable
            agent_mode  = "ql"
            qtable_path = args.agent

    run(
        agent_mode    = agent_mode,
        qtable_path   = qtable_path,
        steps         = args.steps,
        floors        = args.floors,
        lam           = args.lam,
        peak_offpeak  = not args.no_peak,
        render        = not args.no_render,
        fps           = args.fps,
        print_every   = args.print_every,
    )