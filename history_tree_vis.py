"""
history_tree_vis.py — episode history visualised in a separate pygame window.

Renders the elevator's state trajectory this episode as a 2D scatter:
  X axis  = floor  (0 … floors-1)
  Y axis  = timestep (bottom=recent, top=episode start)
  Node    = circle, size ∝ passengers_inside
  Color   = action taken (UP/DOWN/STAY/OPEN)
  Edges   = consecutive steps

Also shows:
  - Reward sparkline (right panel)
  - Action frequency bar chart (bottom)
  - "Oscillation detector" — highlights back-and-forth runs in orange

Call .update(env, action, reward) after every step.
Call .reset_episode() at episode start.
Call .close() at end.
"""

import pygame
import numpy as np
from collections import deque

# ── palette ────────────────────────────────────────────────────────────────
BG        = (10,  12,  20)
GRID_COL  = (22,  26,  40)
WHITE     = (220, 225, 240)
DIM       = (70,  75,  100)
ACCENT    = (80,  180, 255)
OSCILLATE = (255, 165,  40)   # orange highlight for oscillation

ACTION_COLORS = [
    (80,  210, 120),   # UP   → green
    (220, 90,  80),    # DOWN → red
    (180, 180, 80),    # STAY → yellow
    (80,  160, 240),   # OPEN → blue
]
ACTION_NAMES = ["UP", "DOWN", "STAY", "OPEN"]

W, H        = 780, 640
SCATTER_X0  = 70    # left edge of scatter area
SCATTER_X1  = 530   # right edge
SCATTER_Y0  = 50    # top
SCATTER_Y1  = 500   # bottom
SPARKLINE_X = 560   # right panel x start

MAX_HISTORY = 300   # cap to keep render fast


class HistoryTreeVis:
    """
    Opens a separate pygame window tracking the episode history.
    """

    def __init__(self, floors=5, max_steps=600):
        self.floors    = floors
        self.max_steps = max_steps

        pygame.init()
        self._window = pygame.Window("History — Episode Trajectory", (W, H))
        self._screen = self._window.get_surface()
        self._clock  = pygame.time.Clock()
        self._font_t = pygame.font.SysFont("monospace", 15, bold=True)
        self._font_m = pygame.font.SysFont("monospace", 12)
        self._font_s = pygame.font.SysFont("monospace", 10)

        self._history: deque = deque(maxlen=MAX_HISTORY)
        # each entry: {step, floor, action, reward, inside, pending}
        self._ep_rewards: deque = deque(maxlen=MAX_HISTORY)
        self._action_counts = [0, 0, 0, 0]
        self._total_steps   = 0

    # ── public API ─────────────────────────────────────────────────────

    def reset_episode(self):
        self._history.clear()
        self._ep_rewards.clear()
        self._action_counts = [0, 0, 0, 0]
        self._total_steps   = 0

    def update(self, env, action: int, reward: float):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.event.post(event)   # let env handle app-level quit
                return
            elif event.type == pygame.WINDOWCLOSE:
                if hasattr(event, 'window') and event.window is self._window:
                    self.close()
                    return

        self._total_steps += 1
        self._action_counts[action] += 1
        self._history.append({
            "step":    env._step,
            "floor":   env.floor,
            "action":  action,
            "reward":  reward,
            "inside":  len(env.inside),
            "pending": env.total_waiting(),
        })
        self._ep_rewards.append(reward)

        self._screen.fill(BG)
        self._draw_grid()
        self._draw_trajectory()
        self._draw_sparkline()
        self._draw_action_bars()
        self._draw_labels()
        self._window.flip()
        self._clock.tick(8)

    # ── drawing ────────────────────────────────────────────────────────

    def _sx(self, floor: int) -> int:
        """Map floor → scatter x pixel."""
        span = SCATTER_X1 - SCATTER_X0
        step = span / max(1, self.floors - 1)
        return int(SCATTER_X0 + floor * step)

    def _sy(self, step: int) -> int:
        """Map timestep → scatter y pixel (recent = bottom)."""
        history_len = len(self._history)
        if history_len <= 1:
            return SCATTER_Y1
        span = SCATTER_Y1 - SCATTER_Y0
        # most recent entry at bottom
        entries = list(self._history)
        idx = next((i for i, e in enumerate(entries) if e["step"] == step), 0)
        frac = idx / max(1, history_len - 1)
        return int(SCATTER_Y0 + frac * span)

    def _draw_grid(self):
        s = self._screen

        # Background area
        pygame.draw.rect(s, GRID_COL,
                         (SCATTER_X0 - 10, SCATTER_Y0 - 10,
                          SCATTER_X1 - SCATTER_X0 + 20,
                          SCATTER_Y1 - SCATTER_Y0 + 20), border_radius=6)

        # Vertical floor lines
        for f in range(self.floors):
            x = self._sx(f)
            pygame.draw.line(s, (35, 40, 58), (x, SCATTER_Y0), (x, SCATTER_Y1), 1)
            s.blit(self._font_s.render(f"F{f}", True, DIM), (x - 8, SCATTER_Y1 + 6))

        # Axes labels
        s.blit(self._font_m.render("Floor", True, DIM),
               ((SCATTER_X0 + SCATTER_X1) // 2 - 20, SCATTER_Y1 + 22))
        # Y axis label (rotated text via surface)
        ytxt = self._font_m.render("Time →", True, DIM)
        ytxt = pygame.transform.rotate(ytxt, 90)
        s.blit(ytxt, (SCATTER_X0 - 40, (SCATTER_Y0 + SCATTER_Y1) // 2 - 20))

    def _draw_trajectory(self):
        s       = self._screen
        entries = list(self._history)
        if not entries:
            return

        # detect oscillation runs (UP immediately followed by DOWN or vice versa)
        oscillating = set()
        for i in range(1, len(entries) - 1):
            a_prev = entries[i-1]["action"]
            a_curr = entries[i]["action"]
            if (a_prev == 0 and a_curr == 1) or (a_prev == 1 and a_curr == 0):
                oscillating.add(i-1)
                oscillating.add(i)

        # draw edges first
        for i in range(1, len(entries)):
            e0, e1 = entries[i-1], entries[i]
            x0, y0 = self._sx(e0["floor"]), self._sy(e0["step"])
            x1, y1 = self._sx(e1["floor"]), self._sy(e1["step"])
            col = OSCILLATE if i in oscillating else DIM
            lw  = 2 if i in oscillating else 1
            pygame.draw.line(s, col, (x0, y0), (x1, y1), lw)

        # draw nodes
        for i, e in enumerate(entries):
            x, y   = self._sx(e["floor"]), self._sy(e["step"])
            r      = 4 + min(e["inside"], 6)   # size ∝ passengers inside
            col    = OSCILLATE if i in oscillating else ACTION_COLORS[e["action"]]

            pygame.draw.circle(s, col, (x, y), r)

            # highlight most recent
            if i == len(entries) - 1:
                pygame.draw.circle(s, WHITE, (x, y), r + 3, 2)
                s.blit(self._font_s.render(f"now F{e['floor']}", True, WHITE),
                       (x + r + 4, y - 6))

        # oscillation warning
        if len(oscillating) > 4:
            warn = f"⚠ oscillation detected ({len(oscillating)} steps)"
            s.blit(self._font_m.render(warn, True, OSCILLATE), (SCATTER_X0, SCATTER_Y0 - 26))

    def _draw_sparkline(self):
        s      = self._screen
        rews   = list(self._ep_rewards)
        if len(rews) < 2:
            return

        sx0, sx1 = SPARKLINE_X + 10, W - 15
        sy0, sy1 = SCATTER_Y0, SCATTER_Y1
        sw, sh   = sx1 - sx0, sy1 - sy0

        # background
        pygame.draw.rect(s, GRID_COL, (sx0 - 5, sy0 - 10, sw + 10, sh + 20), border_radius=6)

        # title
        s.blit(self._font_m.render("Reward", True, DIM), (sx0, sy0 - 6))

        rmin, rmax = min(rews), max(rews)
        rrange     = max(1, rmax - rmin)

        def ry(r):
            return int(sy1 - (r - rmin) / rrange * sh)

        # zero line
        z = ry(0)
        if sy0 < z < sy1:
            pygame.draw.line(s, (50, 55, 70), (sx0, z), (sx1, z), 1)
            s.blit(self._font_s.render("0", True, DIM), (sx0 - 14, z - 6))

        # sparkline
        step = max(1, len(rews) // sw)
        pts  = []
        for i, r in enumerate(rews[::step]):
            x = int(sx0 + (i / max(1, len(rews[::step]) - 1)) * sw)
            y = ry(r)
            pts.append((x, y))

        if len(pts) >= 2:
            for i in range(1, len(pts)):
                col = ACTION_COLORS[0] if rews[i * step] >= 0 else ACTION_COLORS[1]
                pygame.draw.line(s, col, pts[i-1], pts[i], 1)

        # cumulative reward
        cum = sum(rews)
        s.blit(self._font_s.render(f"cum: {cum:.0f}", True, ACCENT),
               (sx0, sy1 + 6))

    def _draw_action_bars(self):
        s    = self._screen
        total = max(1, sum(self._action_counts))
        bx0  = SCATTER_X0
        by   = SCATTER_Y1 + 48
        bw   = (SCATTER_X1 - SCATTER_X0) // 4 - 8

        s.blit(self._font_m.render("Action distribution:", True, DIM), (bx0, by - 16))

        max_h = 40
        for i, (cnt, col, name) in enumerate(zip(self._action_counts, ACTION_COLORS, ACTION_NAMES)):
            frac = cnt / total
            bh   = max(2, int(frac * max_h))
            rx   = bx0 + i * (bw + 8)
            ry   = by + max_h - bh
            pygame.draw.rect(s, col, (rx, ry, bw, bh), border_radius=3)
            s.blit(self._font_s.render(name, True, col), (rx, by + max_h + 2))
            s.blit(self._font_s.render(f"{frac*100:.0f}%", True, WHITE), (rx, ry - 14))

    def _draw_labels(self):
        s = self._screen
        s.blit(self._font_t.render("EPISODE HISTORY", True, ACCENT), (10, 12))
        if self._history:
            last = list(self._history)[-1]
            info = (f"step {last['step']}  |  "
                    f"floor {last['floor']}  |  "
                    f"inside {last['inside']}  |  "
                    f"pending {last['pending']}")
            s.blit(self._font_s.render(info, True, DIM), (10, 32))

        # legend
        lx, ly = SCATTER_X1 + 30, SCATTER_Y1 + 48
        s.blit(self._font_s.render("node size = pax inside", True, DIM), (lx - 10, ly))
        s.blit(self._font_s.render("orange = oscillation",   True, OSCILLATE), (lx - 10, ly + 14))

    def close(self):
        try:
            self._window.destroy()
        except Exception:
            pass