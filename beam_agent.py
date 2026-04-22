"""
beam_agent.py — Adaptive Beam Search agent for the elevator MDP.

Every step:
  1. Snapshot env (deterministic rollout ignoring future arrivals).
  2. Expand 4 actions from the root; score each using the env's own
     reward shaping (dropoff + board bonuses, wall / idle penalties,
     per-step time + waiting penalties).
  3. Keep top-K survivors (adaptive):
        pending ≤ 1  → K = 2   (narrow / decisive)
        pending 2-4  → K = 4   (balanced)
        pending ≥ 5  → K = 8   (wide / thorough)
  4. Expand each survivor by 4 actions, score, prune to top-K. Repeat
     until depth 3.
  5. Best surviving leaf's first action is returned.

Exposed attributes for the visualisation:
  current_beam_width : int                 — adaptive K
  current_beam       : list[tuple[int,...]] — surviving leaf action-sequences
  levels             : list[list[dict]]     — per-depth candidates:
                                              {"seq", "score", "survived"}
  best_path          : list[int]            — chosen action sequence (len=DEPTH)
  best_score         : float                — cumulative score of chosen leaf
  action_scores      : list[float]          — best reachable score per 1st action
  last_pending       : int                  — pending pax (for vis banner)
"""

import copy


ACTION_UP   = 0
ACTION_DOWN = 1
ACTION_STAY = 2
ACTION_OPEN = 3

ACTION_NAMES = ["UP", "DOWN", "STAY", "OPEN"]


# ── lookahead primitives ───────────────────────────────────────────────────

def _snapshot(env) -> dict:
    """Plain-dict snapshot of env state for deterministic rollout."""
    return {
        "floor":     env.floor,
        "direction": env.direction,
        "inside":    [p.dst for p in env.inside],
        "waiting":   [[(p.src, p.dst) for p in env.waiting[f]]
                      for f in range(env.floors)],
        "floors":    env.floors,
        "max_cap":   env.max_capacity,
    }


def _sim_step(state: dict, action: int):
    """
    Apply `action` to a copy of state. Returns (new_state, reward).
    Mirrors ElevatorEnv.step() scoring EXCEPT Poisson arrivals are frozen.
    """
    s      = copy.deepcopy(state)
    reward = -0.5                          # per-step time penalty
    floor  = s["floor"]

    # auto dropoff
    n_drop     = sum(1 for d in s["inside"] if d == floor)
    s["inside"] = [d for d in s["inside"] if d != floor]
    reward += 15.0 * n_drop

    if action == ACTION_UP:
        if floor < s["floors"] - 1:
            s["floor"]    += 1
            s["direction"] = 2
        else:
            reward -= 2.0                  # wall

    elif action == ACTION_DOWN:
        if floor > 0:
            s["floor"]    -= 1
            s["direction"] = 0
        else:
            reward -= 2.0

    elif action == ACTION_STAY:
        s["direction"] = 1
        if sum(len(w) for w in s["waiting"]) > 0 or s["inside"]:
            reward -= 1.0                  # idling while work pending

    elif action == ACTION_OPEN:
        boarded   = 0
        remaining = []
        for (src, dst) in s["waiting"][floor]:
            if len(s["inside"]) < s["max_cap"]:
                s["inside"].append(dst)
                boarded += 1
            else:
                remaining.append((src, dst))
        s["waiting"][floor] = remaining
        reward += 5.0 * boarded if boarded > 0 else -2.0   # empty open = penalty

    # accumulated wait penalty on still-pending pax
    reward -= 0.05 * sum(len(w) for w in s["waiting"])

    return s, reward


# ── agent ──────────────────────────────────────────────────────────────────

class BeamAgent:
    DEPTH = 3
    N_ACT = 4

    def __init__(self, floors=5):
        self.floors             = floors

        # live state for vis
        self.current_beam_width = 4
        self.current_beam       = []      # list of surviving leaf seqs
        self.levels             = []      # per-depth candidate records
        self.best_path          = []      # chosen action sequence
        self.best_score         = 0.0
        self.action_scores      = [0.0] * self.N_ACT
        self.last_pending       = 0

    # ── adaptive beam width ────────────────────────────────────────────
    def _adaptive_k(self, env) -> int:
        p = env.total_waiting()
        self.last_pending = p
        if p <= 1:   return 2
        if p <= 4:   return 4
        return 8

    # ── main API ───────────────────────────────────────────────────────
    def choose_action(self, env) -> int:
        K = self._adaptive_k(env)
        self.current_beam_width = K

        root = _snapshot(env)
        beam = [(0.0, (), root)]           # (cumulative_score, seq, state)
        self.levels = []

        for _ in range(self.DEPTH):
            candidates = []
            for (cum, seq, st) in beam:
                for a in range(self.N_ACT):
                    new_st, r = _sim_step(st, a)
                    candidates.append((cum + r, seq + (a,), new_st))

            candidates.sort(key=lambda x: -x[0])
            survivors = candidates[:K]
            sv_seqs   = {c[1] for c in survivors}

            self.levels.append([
                {"seq": c[1], "score": c[0], "survived": c[1] in sv_seqs}
                for c in candidates
            ])
            beam = survivors
            if not beam:
                break

        # pick best surviving leaf
        if not beam:
            self.best_path    = [ACTION_STAY]
            self.best_score   = 0.0
            self.current_beam = []
            self.action_scores = [0.0] * self.N_ACT
            return ACTION_STAY

        best = max(beam, key=lambda x: x[0])
        self.best_path    = list(best[1])
        self.best_score   = best[0]
        self.current_beam = [seq for (_, seq, _) in beam]

        # per-action scores for the bottom strip of the vis
        NEG_INF = float("-inf")
        self.action_scores = [NEG_INF] * self.N_ACT
        for (cum, seq, _) in beam:
            if seq and cum > self.action_scores[seq[0]]:
                self.action_scores[seq[0]] = cum
        # actions that were pruned before reaching the final beam:
        # fall back to their depth-1 score so the strip still shows a number
        for entry in self.levels[0]:
            a = entry["seq"][0]
            if self.action_scores[a] == NEG_INF:
                self.action_scores[a] = entry["score"]
        # last safety fallback (shouldn't trigger)
        self.action_scores = [0.0 if x == NEG_INF else x for x in self.action_scores]

        return self.best_path[0]
