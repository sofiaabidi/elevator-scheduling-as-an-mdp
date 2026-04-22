"""
aco_agent.py — Ant Colony Optimization agent for the elevator MDP.

Pheromone matrix: tau[src_floor][dst_floor] — how desirable it is to
travel from src_floor to dst_floor as the next move.

Each step:
  1. Snapshot current env state (floor, waiting, inside).
  2. N_ANTS ants each build a visit sequence greedily-stochastic.
  3. Evaluate each sequence (simulated rollout, no side effects).
  4. Evaporate all pheromone, then deposit on edges of best sequence.
  5. Return action derived from best sequence's first recommended floor.
"""

import copy
import numpy as np
from collections import defaultdict


ACTION_UP   = 0
ACTION_DOWN = 1
ACTION_STAY = 2
ACTION_OPEN = 3


class ACOAgent:
    def __init__(
        self,
        floors       = 5,
        n_ants       = 12,
        evap_rate    = 0.15,    # ρ  — pheromone evaporation per round
        alpha        = 1.2,     # pheromone influence exponent
        beta         = 2.0,     # heuristic (closeness) influence exponent
        q_deposit    = 8.0,     # base pheromone deposited by best ant
        tau_init     = 1.0,     # initial pheromone on all edges
        tau_min      = 0.05,    # floor to prevent complete evaporation
        max_seq_len  = 12,      # max floors in one ant's sequence
    ):
        self.floors      = floors
        self.n_ants      = n_ants
        self.evap_rate   = evap_rate
        self.alpha       = alpha
        self.beta        = beta
        self.q_deposit   = q_deposit
        self.tau_min     = tau_min
        self.max_seq_len = max_seq_len

        # pheromone matrix: tau[i][j] = desirability of going from floor i to j
        self.tau = np.full((floors, floors), tau_init, dtype=float)
        np.fill_diagonal(self.tau, 0.0)   # no self-loops

        # diagnostics exposed to vis
        self.best_fitness    = 0.0
        self.best_sequence   = []
        self.last_tau        = self.tau.copy()

    # ── public API ──────────────────────────────────────────────────────

    def choose_action(self, env) -> int:
        """Main entry point — call every step."""
        snap = self._snapshot(env)

        # ── always run ACO so pheromones evolve & the vis stays live ──
        if snap["targets"]:
            sequences = [self._build_sequence(snap) for _ in range(self.n_ants)]
            scored    = [(self._evaluate(seq, snap), seq) for seq in sequences]
            scored.sort(key=lambda x: x[0], reverse=True)

            best_fit, best_seq = scored[0]
            self.best_fitness  = best_fit
            self.best_sequence = best_seq

            self._evaporate()
            self._deposit(best_seq, best_fit)
            self.last_tau = self.tau.copy()
        else:
            self.best_fitness  = 0.0
            self.best_sequence = [snap["floor"]]

        # ── decision hierarchy ──────────────────────────────────────────
        floor          = snap["floor"]
        waiting_here   = len(env.waiting[floor])
        capacity_avail = len(env.inside) < env.max_capacity

        # 1) pax are waiting on this floor AND we can take them → OPEN
        if waiting_here > 0 and capacity_avail:
            return ACTION_OPEN

        # 2) no work anywhere → STAY
        if not snap["targets"]:
            return ACTION_STAY

        # 3) car is full but dropoff is at the current floor → next step
        #    auto-drops them, so STAY one tick to let it happen cleanly.
        #    (only relevant when capacity full AND dst-of-inside == floor)
        if not capacity_avail and any(p.dst == floor for p in env.inside):
            return ACTION_STAY

        # 4) otherwise follow best ant sequence toward next floor
        return self._seq_to_action(self.best_sequence, floor)

    # ── snapshot ────────────────────────────────────────────────────────

    def _snapshot(self, env) -> dict:
        """Capture the minimal env state needed for ACO rollout."""
        waiting = []
        for f in range(self.floors):
            waiting.append([(p.src, p.dst) for p in env.waiting[f]])

        inside   = [p.dst for p in env.inside]
        # all floors that are worth visiting: have waiting pax OR are a destination of onboard pax
        targets  = set()
        for f in range(self.floors):
            if waiting[f]:
                targets.add(f)
        for dst in inside:
            targets.add(dst)

        return {
            "floor":    env.floor,
            "waiting":  waiting,
            "inside":   inside,
            "targets":  targets,
            "capacity": len(env.inside),
            "max_cap":  env.max_capacity,
        }

    # ── ant construction ────────────────────────────────────────────────

    def _build_sequence(self, snap: dict) -> list:
        """One ant builds a floor visit sequence."""
        current  = snap["floor"]
        visited  = set()
        seq      = [current]
        waiting  = copy.deepcopy(snap["waiting"])
        inside   = list(snap["inside"])
        capacity = snap["capacity"]
        max_cap  = snap["max_cap"]

        for _ in range(self.max_seq_len):
            # refresh targets from simulated state
            targets = set()
            for f in range(self.floors):
                if waiting[f]:
                    targets.add(f)
            for dst in inside:
                targets.add(dst)
            targets.discard(current)   # already here

            if not targets:
                break

            # build probability distribution over candidate next floors
            candidates = list(targets - visited) or list(targets)
            if not candidates:
                break

            probs = []
            for nxt in candidates:
                dist       = max(1, abs(nxt - current))
                heuristic  = 1.0 / dist ** self.beta
                pher       = max(self.tau_min, self.tau[current][nxt]) ** self.alpha
                probs.append(pher * heuristic)

            probs  = np.array(probs)
            probs /= probs.sum()
            nxt    = candidates[int(np.random.choice(len(candidates), p=probs))]

            # simulate transit: drop off inside pax at floors we pass through
            step_dir = 1 if nxt > current else -1
            for mid in range(current + step_dir, nxt + step_dir, step_dir):
                # drop off
                inside = [d for d in inside if d != mid]
                # open door / board at mid if on the way
                remaining = []
                for (src, dst) in waiting[mid]:
                    if capacity < max_cap:
                        inside.append(dst)
                        capacity += 1
                    else:
                        remaining.append((src, dst))
                waiting[mid] = remaining

            current = nxt
            visited.add(nxt)
            seq.append(nxt)

        return seq

    # ── evaluation ──────────────────────────────────────────────────────

    def _evaluate(self, seq: list, snap: dict) -> float:
        """
        Simulate following seq from snap state.
        Score = served * 20  - steps_taken * 0.5  - remaining_waiting
        """
        if len(seq) < 2:
            return 0.0

        waiting  = copy.deepcopy(snap["waiting"])
        inside   = list(snap["inside"])
        capacity = snap["capacity"]
        max_cap  = snap["max_cap"]
        current  = seq[0]
        served   = 0
        steps    = 0

        for nxt in seq[1:]:
            step_dir = 1 if nxt > current else -1
            for mid in range(current + step_dir, nxt + step_dir, step_dir):
                steps += 1
                # drop off
                dropped = [d for d in inside if d == mid]
                served += len(dropped)
                inside  = [d for d in inside if d != mid]
                capacity = len(inside)
                # board
                remaining = []
                for (src, dst) in waiting[mid]:
                    if capacity < max_cap:
                        inside.append(dst)
                        capacity += 1
                    else:
                        remaining.append((src, dst))
                waiting[mid] = remaining

            current = nxt

        remaining_total = sum(len(w) for w in waiting)
        return served * 20.0 - steps * 0.5 - remaining_total * 2.0

    # ── pheromone ───────────────────────────────────────────────────────

    def _evaporate(self):
        self.tau *= (1.0 - self.evap_rate)
        self.tau  = np.maximum(self.tau, self.tau_min)
        np.fill_diagonal(self.tau, 0.0)

    def _deposit(self, seq: list, fitness: float):
        if len(seq) < 2 or fitness <= 0:
            return
        deposit = self.q_deposit * (1.0 + fitness / 20.0)
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if a != b:
                self.tau[a][b] += deposit
                self.tau[b][a] += deposit * 0.5   # weaker reverse trail

    # ── action extraction ───────────────────────────────────────────────

    def _seq_to_action(self, seq: list, current_floor: int) -> int:
        if len(seq) < 2:
            return ACTION_STAY

        next_target = seq[1]

        if next_target == current_floor:
            return ACTION_OPEN
        if next_target > current_floor:
            return ACTION_UP
        return ACTION_DOWN
