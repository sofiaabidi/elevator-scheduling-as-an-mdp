"""
beam_vis.py — Adaptive Beam Search visualisation (pygame.Surface 840×620).

Layout (embedded into the main env window's PANEL_DTREE slot):

  ┌──────────────────────────────────────────────┬───┐
  │ BEAM SEARCH — adaptive K, depth 3            │ K │
  │                                              │ ▓ │
  │   ●    depth 0 (current state)               │ ▓ │
  │  ╱│╲╲                                         │ ▓ │
  │ ● ● ● ●  depth 1   (all 4 actions)           │ ▒ │
  │ │ │ │ │                                       │ ▒ │
  │ …tree of survivors / pruned …                │ ░ │
  │                                              │   │
  ├──────────────────────────────────────────────┴───┤
  │ Action scores   UP  DOWN  STAY  OPEN              │
  └──────────────────────────────────────────────────┘

Color key
  • green / yellow / red gradient  = heuristic score of candidate
  • bright = survived pruning at that depth
  • grey   = pruned (kept in the picture so you see what was rejected)
  • gold   = best path / chosen action
"""

import pygame

# ── palette ────────────────────────────────────────────────────────────────
BG     = (10, 12, 20)
GRID   = (28, 32, 48)
WHITE  = (220, 225, 240)
DIM    = (70, 75, 100)
ACCENT = (80, 180, 255)
GREEN  = (60, 210, 120)
YELLOW = (255, 205, 55)
RED    = (240, 70, 70)
GOLD   = (255, 200, 70)
PRUNE  = (45, 48, 66)

ACTION_LETTERS = ["↑", "↓", "—", "⬚"]
ACTION_NAMES   = ["UP ↑", "DOWN ↓", "STAY —", "OPEN ⬚"]
ACTION_COLORS  = [(80, 210, 120), (220, 90, 80), (180, 180, 80), (80, 160, 240)]

W, H = 840, 620

# tree area
TREE_X0, TREE_X1 = 30, 780
TREE_Y  = [70, 180, 290, 400]     # y for depth 0, 1, 2, 3
NODE_R  = [18, 11, 8, 8]          # radii per depth

# width-indicator bar (right side)
BAR_X, BAR_Y, BAR_W, BAR_H = 795, 60, 26, 360

# bottom strip
STRIP_Y = 460
STRIP_H = H - STRIP_Y - 10


# ── helpers ────────────────────────────────────────────────────────────────

def _score_color(score: float, lo: float = -8.0, hi: float = 40.0):
    """Red → yellow → green gradient."""
    t = (score - lo) / max(1e-6, hi - lo)
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        s = t / 0.5
        return (int(240 - 70 * s), int(70 + 135 * s), int(70))
    s = (t - 0.5) / 0.5
    return (int(170 - 110 * s), int(205 + 5 * s), int(70 + 50 * s))


# ── vis class ──────────────────────────────────────────────────────────────

class BeamVis:
    """Render beam-search state to an off-screen Surface blitted by env."""

    def __init__(self, floors=5):
        self.floors  = floors
        self._screen = pygame.Surface((W, H))
        self._font_t = pygame.font.SysFont("monospace", 15, bold=True)
        self._font_m = pygame.font.SysFont("monospace", 12)
        self._font_s = pygame.font.SysFont("monospace", 10)

    # ── public ─────────────────────────────────────────────────────────
    def update(self, agent, env):
        s = self._screen
        s.fill(BG)

        # ── title + sub-line ──
        s.blit(self._font_t.render("BEAM SEARCH  —  adaptive K, depth 3",
                                   True, ACCENT), (10, 10))
        sub = (f"pending={agent.last_pending}   "
               f"K={agent.current_beam_width}   "
               f"best_score={agent.best_score:+.1f}   "
               f"chosen={ACTION_NAMES[agent.best_path[0]] if agent.best_path else '—'}")
        s.blit(self._font_s.render(sub, True, DIM), (10, 32))

        self._draw_tree(s, agent, env)
        self._draw_width_bar(s, agent)
        self._draw_action_strip(s, agent)

    # ── tree ───────────────────────────────────────────────────────────
    def _compute_positions(self, agent):
        """seq tuple → (x, y). Sort each level by seq so siblings cluster."""
        pos = {(): (W // 2, TREE_Y[0])}
        for d, level in enumerate(agent.levels):
            if not level:
                continue
            ordered = sorted(level, key=lambda e: e["seq"])
            n = len(ordered)
            span = TREE_X1 - TREE_X0
            for i, entry in enumerate(ordered):
                x = TREE_X0 + int((i + 0.5) * span / max(1, n))
                y = TREE_Y[d + 1]
                pos[entry["seq"]] = (x, y)
        return pos

    def _draw_tree(self, s, agent, env):
        pos = self._compute_positions(agent)

        # best path as prefix set for quick lookup
        best_prefixes = set()
        for i in range(1, len(agent.best_path) + 1):
            best_prefixes.add(tuple(agent.best_path[:i]))

        # ── edges (draw first so nodes overlap them) ──
        for d, level in enumerate(agent.levels):
            for entry in level:
                seq    = entry["seq"]
                parent = seq[:-1]
                if parent not in pos:
                    continue
                x0, y0 = pos[parent]
                x1, y1 = pos[seq]
                if seq in best_prefixes:
                    col, lw = GOLD, 3
                elif entry["survived"]:
                    col, lw = (90, 100, 130), 1
                else:
                    col, lw = PRUNE, 1
                pygame.draw.line(s, col,
                                 (x0, y0 + NODE_R[d] - 1),
                                 (x1, y1 - NODE_R[d + 1] + 1), lw)

        # ── nodes ──
        for d, level in enumerate(agent.levels):
            for entry in level:
                x, y   = pos[entry["seq"]]
                r      = NODE_R[d + 1]
                is_best = entry["seq"] in best_prefixes

                if entry["survived"]:
                    ncol = _score_color(entry["score"])
                    pygame.draw.circle(s, ncol, (x, y), r)
                    if is_best:
                        pygame.draw.circle(s, GOLD, (x, y), r + 3, 2)
                    else:
                        pygame.draw.circle(s, (ncol[0]//2, ncol[1]//2, ncol[2]//2),
                                           (x, y), r, 1)
                else:
                    pygame.draw.circle(s, PRUNE, (x, y), r)
                    pygame.draw.circle(s, (35, 38, 52), (x, y), r, 1)

                # action letter for depth-1 nodes only (enough room)
                if d == 0:
                    letter = ACTION_LETTERS[entry["seq"][0]]
                    txt    = self._font_s.render(
                        letter, True,
                        WHITE if entry["survived"] else DIM)
                    s.blit(txt, (x - txt.get_width() // 2, y + r + 2))

        # ── root ──
        rx, ry = pos[()]
        pygame.draw.circle(s, (60, 130, 220), (rx, ry), NODE_R[0])
        pygame.draw.circle(s, ACCENT, (rx, ry), NODE_R[0], 2)
        rlbl = self._font_s.render(f"F{env.floor}", True, WHITE)
        s.blit(rlbl, (rx - rlbl.get_width() // 2, ry - rlbl.get_height() // 2))

        # ── depth labels on the left ──
        for d, y in enumerate(TREE_Y):
            s.blit(self._font_s.render(f"d{d}", True, DIM), (5, y - 5))

    # ── adaptive beam-width bar ────────────────────────────────────────
    def _draw_width_bar(self, s, agent):
        pygame.draw.rect(s, GRID, (BAR_X, BAR_Y, BAR_W, BAR_H), border_radius=4)

        # label
        s.blit(self._font_s.render("BEAM K", True, DIM), (BAR_X - 4, BAR_Y - 18))

        K     = agent.current_beam_width
        ratio = K / 8.0
        fill  = int(BAR_H * ratio)
        col   = RED if K >= 6 else (YELLOW if K >= 4 else GREEN)
        pygame.draw.rect(s, col,
                         (BAR_X + 2, BAR_Y + BAR_H - fill,
                          BAR_W - 4, fill), border_radius=3)

        # current value underneath
        vtxt = self._font_t.render(str(K), True, col)
        s.blit(vtxt, (BAR_X + BAR_W // 2 - vtxt.get_width() // 2,
                      BAR_Y + BAR_H + 4))

        # tick marks at 2 / 4 / 8
        for k in (2, 4, 8):
            ty = BAR_Y + BAR_H - int(BAR_H * k / 8.0)
            pygame.draw.line(s, DIM, (BAR_X - 5, ty), (BAR_X, ty), 1)
            s.blit(self._font_s.render(str(k), True, DIM),
                   (BAR_X - 16, ty - 5))

        # congestion tag
        if K <= 2:
            tag = "narrow"
        elif K <= 4:
            tag = "medium"
        else:
            tag = "wide"
        ttxt = self._font_s.render(tag, True, DIM)
        s.blit(ttxt, (BAR_X + BAR_W // 2 - ttxt.get_width() // 2,
                      BAR_Y + BAR_H + 22))

    # ── action score strip ─────────────────────────────────────────────
    def _draw_action_strip(self, s, agent):
        pygame.draw.line(s, GRID, (10, STRIP_Y - 4), (W - 10, STRIP_Y - 4), 1)
        s.blit(self._font_m.render("Action scores (best leaf reachable)",
                                   True, DIM), (10, STRIP_Y))

        best_a = agent.best_path[0] if agent.best_path else 0
        n      = len(ACTION_NAMES)
        margin = 15
        col_w  = (W - 2 * margin - (n - 1) * 8) // n

        for i, (name, acol) in enumerate(zip(ACTION_NAMES, ACTION_COLORS)):
            x      = margin + i * (col_w + 8)
            y      = STRIP_Y + 22
            score  = agent.action_scores[i]
            chosen = (i == best_a)

            # tile background
            if chosen:
                bg = (acol[0] // 3, acol[1] // 3, acol[2] // 3)
            else:
                bg = (28, 32, 46)
            pygame.draw.rect(s, bg, (x, y, col_w, 108), border_radius=6)

            # gold border for chosen action
            if chosen:
                pygame.draw.rect(s, GOLD, (x, y, col_w, 108), 2, border_radius=6)

            # name
            s.blit(self._font_m.render(name, True, acol), (x + 10, y + 8))

            # big score
            stxt = self._font_t.render(f"{score:+.1f}", True, _score_color(score))
            s.blit(stxt, (x + 10, y + 30))

            # score bar (relative within [-10, 40])
            bar_x = x + 10
            bar_y = y + 60
            bar_w = col_w - 20
            pygame.draw.rect(s, GRID, (bar_x, bar_y, bar_w, 8), border_radius=3)
            t = max(0.0, min(1.0, (score + 10) / 50.0))
            pygame.draw.rect(s, _score_color(score),
                             (bar_x, bar_y, int(bar_w * t), 8), border_radius=3)

            # tag
            tag  = "CHOSEN" if chosen else "rejected"
            tcol = GOLD if chosen else DIM
            s.blit(self._font_s.render(tag, True, tcol), (x + 10, y + 86))

    def close(self):
        pass
