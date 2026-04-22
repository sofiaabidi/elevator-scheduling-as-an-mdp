"""
decision_tree_vis.py — live decision tree in a separate pygame window.

Shows depth-2 lookahead from current elevator state.
  - Root = current state
  - Level 1 = 4 action branches (UP / DOWN / STAY / OPEN)
  - Level 2 = 4 sub-branches each  (16 leaves total)
  - Invalid actions shown greyed out with penalty label
  - Node color = reward signal (green=good, red=bad, grey=invalid)
  - Arrivals are FROZEN during lookahead (deterministic sim)

Usage: instantiate DecisionTreeVis, call .update(env) each step.
"""

import pygame
import copy
import numpy as np

# ── palette ────────────────────────────────────────────────────────────────
BG        = (10,  12,  20)
GRID_COL  = (28,  32,  48)
WHITE     = (220, 225, 240)
DIM       = (70,  75,  100)
ACCENT    = (80,  180, 255)
GREEN     = (50,  200, 110)
YELLOW    = (255, 205, 55)
RED       = (240, 70,  70)
GREY      = (55,  58,  75)
NODE_CURR = (60,  130, 220)   # current state node
EDGE_COL  = (45,  50,  70)

ACTION_NAMES  = ["UP ↑", "DOWN ↓", "STAY —", "OPEN ⬚"]
ACTION_COLORS = [
    (80,  200, 120),   # UP   → green
    (200, 100, 80),    # DOWN → red-ish
    (160, 160, 80),    # STAY → yellow-ish
    (80,  160, 240),   # OPEN → blue
]

W, H = 840, 620


# ── helpers ────────────────────────────────────────────────────────────────

def _sim_step(env_state: dict, action: int, floors: int, max_cap: int):
    """
    Pure-function lookahead step — no side effects on real env.
    env_state keys: floor, direction, inside_dst (list), hall_up (list),
                    hall_down (list), capacity (int), waiting (list[list[int]])
                    where waiting[f] = list of (src,dst) tuples
    Returns (new_state, reward, label)
    """
    import copy
    s       = copy.deepcopy(env_state)
    reward  = -0.5
    valid   = True
    label   = ""

    # auto-dropoff
    dropped = [dst for dst in s["inside_dst"] if dst == s["floor"]]
    s["inside_dst"] = [d for d in s["inside_dst"] if d != s["floor"]]
    s["capacity"]   = len(s["inside_dst"])
    reward += 15.0 * len(dropped)

    if action == 0:   # UP
        if s["floor"] < floors - 1:
            s["floor"]     += 1
            s["direction"]  = 2
            label = f"+{s['floor']}"
        else:
            reward -= 2.0
            valid   = False
            label   = "wall!"
    elif action == 1:  # DOWN
        if s["floor"] > 0:
            s["floor"]     -= 1
            s["direction"]  = 0
            label = f"-{s['floor']}"
        else:
            reward -= 2.0
            valid   = False
            label   = "wall!"
    elif action == 2:  # STAY
        s["direction"] = 1
        pending = sum(len(w) for w in s["waiting"])
        if pending > 0 or s["capacity"] > 0:
            reward -= 1.0
        label = "idle"
    elif action == 3:  # OPEN_DOOR
        boarded = 0
        remaining = []
        for (src, dst) in s["waiting"][s["floor"]]:
            if s["capacity"] < max_cap:
                s["inside_dst"].append(dst)
                s["capacity"] += 1
                boarded += 1
            else:
                remaining.append((src, dst))
        s["waiting"][s["floor"]] = remaining
        if boarded > 0:
            reward += 5.0 * boarded
            label   = f"+{boarded} pax"
        else:
            reward -= 2.0
            valid   = False
            label   = "empty!"

    # recompute hall flags
    s["hall_up"]   = [0] * floors
    s["hall_down"] = [0] * floors
    for f in range(floors):
        for (src, dst) in s["waiting"][f]:
            if dst > src: s["hall_up"][f]   = 1
            else:         s["hall_down"][f] = 1

    return s, reward, valid, label


def _env_to_state(env) -> dict:
    """Snapshot live env into a plain dict for lookahead."""
    floors = env.floors
    waiting_snap = []
    for f in range(floors):
        waiting_snap.append([(p.src, p.dst) for p in env.waiting[f]])
    hall_up   = [0] * floors
    hall_down = [0] * floors
    for f in range(floors):
        for (src, dst) in waiting_snap[f]:
            if dst > src: hall_up[f]   = 1
            else:         hall_down[f] = 1
    return {
        "floor":      env.floor,
        "direction":  env.direction,
        "inside_dst": [p.dst for p in env.inside],
        "hall_up":    hall_up,
        "hall_down":  hall_down,
        "capacity":   len(env.inside),
        "waiting":    waiting_snap,
    }


def _reward_color(r: float, valid: bool):
    if not valid:
        return GREY
    if r >= 10:
        return GREEN
    if r >= 0:
        t = r / 10.0
        return (int(50 + 150*t), int(150 + 50*t), int(110 - 30*t))
    # negative
    t = min(1.0, abs(r) / 10.0)
    return (int(180 + 60*t), int(80 - 30*t), int(70 - 20*t))


def _state_short(s: dict) -> str:
    dirs = ["↓", "—", "↑"]
    hu = "".join(str(v) for v in s["hall_up"])
    hd = "".join(str(v) for v in s["hall_down"])
    inside = len(s["inside_dst"])
    return f"F{s['floor']}{dirs[s['direction']]} [{inside}] H↑{hu} H↓{hd}"


# ── main class ─────────────────────────────────────────────────────────────

class DecisionTreeVis:
    """
    Opens a separate pygame window showing depth-2 decision tree.
    Call .update(env) after every env.step().
    Call .close() at end.
    """

    DEPTH   = 2
    N_ACT   = 4
    NODE_R  = 18      # node circle radius
    ROOT_R  = 24

    def __init__(self, floors=5, max_capacity=6):
        self.floors      = floors
        self.max_cap     = max_capacity

        self._screen = pygame.Surface((W, H))
        self._font_t = pygame.font.SysFont("monospace", 15, bold=True)
        self._font_m = pygame.font.SysFont("monospace", 12)
        self._font_s = pygame.font.SysFont("monospace", 10)

        self._tree   = None   # last computed tree

    # ── layout ─────────────────────────────────────────────────────────

    def _node_positions(self):
        """
        Returns dict: (depth, branch_idx) → (cx, cy)
        depth 0 = root (1 node)
        depth 1 = 4 nodes
        depth 2 = 16 nodes
        """
        pos = {}
        margin_top  = 80
        level_gap   = (H - margin_top - 60) // self.DEPTH

        # root
        pos[(0, 0)] = (W // 2, margin_top)

        # depth 1 — 4 children
        d1_y = margin_top + level_gap
        d1_xs = _spread(W, 4, 90)
        for i in range(4):
            pos[(1, i)] = (d1_xs[i], d1_y)

        # depth 2 — 16 children (4 per d1 node)
        d2_y = margin_top + 2 * level_gap
        d2_xs = _spread(W, 16, 40)
        for i in range(16):
            pos[(2, i)] = (d2_xs[i], d2_y)

        return pos

    # ── compute tree ───────────────────────────────────────────────────

    def _compute_tree(self, env):
        root_state = _env_to_state(env)
        tree = {"state": root_state, "children": []}

        for a1 in range(self.N_ACT):
            s1, r1, v1, lbl1 = _sim_step(root_state, a1, self.floors, self.max_cap)
            node1 = {
                "action": a1, "state": s1, "reward": r1,
                "valid": v1, "label": lbl1, "children": [],
                "cumulative": r1,
            }
            for a2 in range(self.N_ACT):
                s2, r2, v2, lbl2 = _sim_step(s1, a2, self.floors, self.max_cap)
                node2 = {
                    "action": a2, "state": s2, "reward": r2,
                    "valid": v2, "label": lbl2, "children": [],
                    "cumulative": r1 + r2,
                }
                node1["children"].append(node2)
            tree["children"].append(node1)

        # find best leaf path
        best_cum  = -9999
        best_path = (0, 0)
        for i, c1 in enumerate(tree["children"]):
            for j, c2 in enumerate(c1["children"]):
                if c2["cumulative"] > best_cum:
                    best_cum  = c2["cumulative"]
                    best_path = (i, j)
        tree["best_path"] = best_path
        return tree

    # ── draw ───────────────────────────────────────────────────────────

    def update(self, env):
        self._tree = self._compute_tree(env)
        self._screen.fill(BG)
        self._draw_tree(self._tree)
        self._draw_legend()

    def _draw_tree(self, tree):
        s    = self._screen
        pos  = self._node_positions()
        best = tree["best_path"]

        # ── title ──
        title = f"DECISION TREE  (depth-2 lookahead)   current: {_state_short(tree['state'])}"
        s.blit(self._font_t.render(title, True, ACCENT), (10, 12))
        s.blit(self._font_s.render("greyed = invalid action  |  gold border = best path", True, DIM), (10, 35))

        rx, ry = pos[(0, 0)]

        # ── depth-1 nodes + edges from root ──
        for i, c1 in enumerate(tree["children"]):
            cx1, cy1 = pos[(1, i)]
            is_best_branch = (i == best[0])

            # edge root → d1
            ecol = ACTION_COLORS[c1["action"]] if c1["valid"] else GREY
            pygame.draw.line(s, ecol, (rx, ry + self.ROOT_R),
                             (cx1, cy1 - self.NODE_R), 2 if is_best_branch else 1)

            # action label on edge
            mx, my = (rx + cx1) // 2, (ry + cy1) // 2
            albl = ACTION_NAMES[c1["action"]]
            s.blit(self._font_s.render(albl, True, ecol), (mx - 20, my - 8))

            # ── depth-2 nodes + edges from d1 ──
            for j, c2 in enumerate(c1["children"]):
                gi    = i * 4 + j
                cx2, cy2 = pos[(2, gi)]
                is_best  = is_best_branch and (j == best[1])

                # edge d1 → d2
                ecol2 = ACTION_COLORS[c2["action"]] if c2["valid"] else GREY
                lw    = 2 if is_best else 1
                pygame.draw.line(s, ecol2,
                                 (cx1, cy1 + self.NODE_R),
                                 (cx2, cy2 - self.NODE_R), lw)

                # d2 node
                ncol  = _reward_color(c2["cumulative"], c2["valid"])
                bcol  = YELLOW if is_best else (ncol[0]//2, ncol[1]//2, ncol[2]//2)
                pygame.draw.circle(s, ncol, (cx2, cy2), self.NODE_R)
                pygame.draw.circle(s, bcol, (cx2, cy2), self.NODE_R, 3 if is_best else 1)

                # reward text inside node
                rtxt = f"{c2['cumulative']:+.0f}"
                ts   = self._font_s.render(rtxt, True, WHITE if c2["valid"] else DIM)
                s.blit(ts, (cx2 - ts.get_width()//2, cy2 - ts.get_height()//2))

                # label below node
                ltxt = self._font_s.render(c2["label"], True, DIM)
                s.blit(ltxt, (cx2 - ltxt.get_width()//2, cy2 + self.NODE_R + 2))

            # ── d1 node (drawn after children so it's on top) ──
            ncol1 = _reward_color(c1["reward"], c1["valid"])
            bcol1 = YELLOW if is_best_branch else (ncol1[0]//2, ncol1[1]//2, ncol1[2]//2)
            pygame.draw.circle(s, ncol1, (cx1, cy1), self.NODE_R)
            pygame.draw.circle(s, bcol1, (cx1, cy1), self.NODE_R, 3 if is_best_branch else 1)

            rtxt1 = f"{c1['reward']:+.0f}"
            ts1   = self._font_m.render(rtxt1, True, WHITE if c1["valid"] else DIM)
            s.blit(ts1, (cx1 - ts1.get_width()//2, cy1 - ts1.get_height()//2))

            # floor label under d1
            ftxt = self._font_s.render(f"F{c1['state']['floor']}", True, DIM)
            s.blit(ftxt, (cx1 - ftxt.get_width()//2, cy1 + self.NODE_R + 2))

        # ── root node ──
        pygame.draw.circle(s, NODE_CURR, (rx, ry), self.ROOT_R)
        pygame.draw.circle(s, ACCENT,    (rx, ry), self.ROOT_R, 3)
        rtxt0 = f"F{tree['state']['floor']}"
        ts0   = self._font_m.render(rtxt0, True, WHITE)
        s.blit(ts0, (rx - ts0.get_width()//2, ry - ts0.get_height()//2))

    def _draw_legend(self):
        s  = self._screen
        lx = 10
        ly = H - 38
        s.blit(self._font_s.render("ACTIONS:", True, DIM), (lx, ly))
        for i, (name, col) in enumerate(zip(ACTION_NAMES, ACTION_COLORS)):
            pygame.draw.circle(s, col, (lx + 80 + i * 120, ly + 6), 7)
            s.blit(self._font_s.render(name, True, col), (lx + 92 + i * 120, ly))
        # reward scale
        rx2 = W - 200
        s.blit(self._font_s.render("reward: ", True, DIM), (rx2, ly))
        for val, label in [(-8, "bad"), (0, "neutral"), (12, "good")]:
            col = _reward_color(val, True)
            pygame.draw.rect(s, col, (rx2 + 65 + [0,55,110][[-8,0,12].index(val)], ly, 40, 14), border_radius=3)
            s.blit(self._font_s.render(label, True, WHITE),
                   (rx2 + 68 + [0,55,110][[-8,0,12].index(val)], ly + 1))

    def close(self):
        pass


# ── utility ────────────────────────────────────────────────────────────────

def _spread(total_w: int, n: int, min_gap: int) -> list:
    """Return n evenly-spaced x centres across total_w."""
    margin = max(min_gap, (total_w - n * min_gap) // (n + 1))
    step   = (total_w - 2 * margin) // max(1, n - 1)
    return [margin + i * step for i in range(n)]