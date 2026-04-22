"""
aco_vis.py — ACO visualisation rendered to a pygame.Surface (840×620).

Panels:
  LEFT  — 5×5 pheromone heatmap + colour scale
  RIGHT — mini building diagram with best ant's path animated
  BOTTOM — live metrics bar (evap rate, best fitness, colony size)

The env's render() blits this surface into PANEL_DTREE (x=640).
Call .update(agent, env) each step.
"""

import pygame
import numpy as np

# ── palette ────────────────────────────────────────────────────────────────
BG      = (10,  12,  20)
GRID    = (28,  32,  48)
WHITE   = (220, 225, 240)
DIM     = (70,  75,  100)
ACCENT  = (80,  180, 255)
GREEN   = (50,  200, 110)
YELLOW  = (255, 205, 55)
RED     = (240, 70,  70)
ORANGE  = (255, 140, 40)

W, H = 840, 620

# heatmap cell size
HMAP_COLS = 5
HMAP_ROWS = 5
HMAP_X0   = 30
HMAP_Y0   = 70
CELL_W    = 72
CELL_H    = 72

# mini building (right side)
BUILD_X0  = 470
BUILD_Y0  = 60
BUILD_W   = 120
BUILD_H   = 480
SHAFT_X   = BUILD_X0 + BUILD_W // 2

ACTION_NAMES = ["UP ↑", "DOWN ↓", "STAY —", "OPEN ⬚"]
ACTION_COLS  = [(80, 210, 120), (220, 90, 80), (180, 180, 80), (80, 160, 240)]


def _heat_color(val: float, vmin: float, vmax: float):
    """Map scalar → RGB heat colour (dark blue → cyan → yellow → white)."""
    t = (val - vmin) / max(1e-6, vmax - vmin)
    t = max(0.0, min(1.0, t))
    if t < 0.33:
        s = t / 0.33
        return (int(20 + 40 * s), int(40 + 140 * s), int(80 + 160 * s))
    elif t < 0.66:
        s = (t - 0.33) / 0.33
        return (int(60 + 195 * s), int(180 + 25 * s), int(240 - 130 * s))
    else:
        s = (t - 0.66) / 0.34
        return (int(255), int(205 + 20 * s), int(110 + 110 * s))


class ACOVis:
    """
    Renders ACO state to a pygame.Surface.
    No window management — env blits this surface each frame.
    """

    def __init__(self, floors=5, n_ants=12, evap_rate=0.15):
        self.floors    = floors
        self.n_ants    = n_ants
        self.evap_rate = evap_rate

        self._screen   = pygame.Surface((W, H))
        self._font_t   = pygame.font.SysFont("monospace", 15, bold=True)
        self._font_m   = pygame.font.SysFont("monospace", 13)
        self._font_s   = pygame.font.SysFont("monospace", 10)

        self._anim_t   = 0   # animation tick for path drawing

    # ── public ─────────────────────────────────────────────────────────

    def update(self, agent, env):
        self._anim_t += 1
        s = self._screen
        s.fill(BG)

        self._draw_heatmap(s, agent)
        self._draw_building(s, agent, env)
        self._draw_metrics(s, agent, env)
        self._draw_title(s)

    # ── heatmap ────────────────────────────────────────────────────────

    def _draw_heatmap(self, s, agent):
        tau  = agent.last_tau
        vmin = tau.min()
        vmax = tau.max()

        # section label
        s.blit(self._font_m.render("PHEROMONE  τ[from→to]", True, ACCENT),
               (HMAP_X0, HMAP_Y0 - 30))

        for row in range(self.floors):
            for col in range(self.floors):
                x = HMAP_X0 + col * CELL_W
                y = HMAP_Y0 + row * CELL_H
                val = tau[row][col]
                col_c = _heat_color(val, vmin, vmax) if row != col else (20, 22, 34)
                pygame.draw.rect(s, col_c, (x + 2, y + 2, CELL_W - 4, CELL_H - 4),
                                 border_radius=5)

                # value label
                if row != col:
                    lbl = self._font_s.render(f"{val:.1f}", True,
                                              WHITE if val > (vmin + vmax) / 2 else DIM)
                    s.blit(lbl, (x + CELL_W // 2 - lbl.get_width() // 2,
                                 y + CELL_H // 2 - lbl.get_height() // 2))
                else:
                    pygame.draw.line(s, (40, 44, 60),
                                     (x + 6, y + 6),
                                     (x + CELL_W - 8, y + CELL_H - 8), 1)

        # axis labels
        for i in range(self.floors):
            x = HMAP_X0 + i * CELL_W + CELL_W // 2 - 8
            s.blit(self._font_s.render(f"F{i}", True, DIM), (x, HMAP_Y0 - 14))
            y = HMAP_Y0 + i * CELL_H + CELL_H // 2 - 6
            s.blit(self._font_s.render(f"F{i}", True, DIM), (HMAP_X0 - 24, y))

        # colour scale bar
        scale_x = HMAP_X0
        scale_y = HMAP_Y0 + self.floors * CELL_H + 12
        scale_w = self.floors * CELL_W
        for px in range(scale_w):
            t = px / scale_w
            v = vmin + t * (vmax - vmin)
            pygame.draw.line(s, _heat_color(v, vmin, vmax),
                             (scale_x + px, scale_y),
                             (scale_x + px, scale_y + 10))
        pygame.draw.rect(s, DIM, (scale_x, scale_y, scale_w, 10), 1)
        s.blit(self._font_s.render(f"{vmin:.1f}", True, DIM), (scale_x, scale_y + 12))
        s.blit(self._font_s.render(f"{vmax:.1f}", True, WHITE),
               (scale_x + scale_w - 28, scale_y + 12))

    # ── mini building ──────────────────────────────────────────────────

    def _floor_y(self, floor: int) -> int:
        """Y centre of a floor in the mini building panel."""
        span      = BUILD_H - 40
        floor_gap = span // max(1, self.floors - 1)
        return BUILD_Y0 + BUILD_H - 20 - floor * floor_gap

    def _draw_building(self, s, agent, env):
        s.blit(self._font_m.render("BEST ANT PATH", True, ACCENT),
               (BUILD_X0, BUILD_Y0 - 30))

        # shaft
        pygame.draw.rect(s, GRID, (SHAFT_X - 18, BUILD_Y0, 36, BUILD_H), border_radius=4)

        # floor lines + labels
        for f in range(self.floors):
            fy = self._floor_y(f)
            pygame.draw.line(s, (40, 46, 64),
                             (BUILD_X0 - 4, fy), (BUILD_X0 + BUILD_W + 4, fy), 1)
            s.blit(self._font_s.render(f"F{f}", True, DIM), (BUILD_X0 - 30, fy - 6))

            # waiting pax dots
            n_wait = len(env.waiting[f])
            for i in range(min(n_wait, 4)):
                dx = BUILD_X0 + BUILD_W + 12 + i * 10
                pygame.draw.circle(s, YELLOW, (dx, fy), 4)
            if n_wait > 4:
                s.blit(self._font_s.render(f"+{n_wait-4}", True, YELLOW),
                       (BUILD_X0 + BUILD_W + 52, fy - 6))

        # best path edges (animated — draw up to anim_t % len nodes)
        seq = agent.best_sequence
        if len(seq) >= 2:
            visible = max(2, (self._anim_t // 4 % len(seq)) + 1)
            for i in range(1, min(visible, len(seq))):
                y0 = self._floor_y(seq[i - 1])
                y1 = self._floor_y(seq[i])
                frac  = i / len(seq)
                alpha = int(80 + 175 * (1 - frac))
                col   = (80, int(160 + 50 * (1 - frac)), int(255 * (1 - frac)))
                pygame.draw.line(s, col, (SHAFT_X, y0), (SHAFT_X, y1), 3)
                pygame.draw.circle(s, col, (SHAFT_X, y1), 7)
                s.blit(self._font_s.render(f"F{seq[i]}", True, col),
                       (SHAFT_X + 12, y1 - 6))

        # elevator position
        ey = self._floor_y(env.floor)
        pygame.draw.rect(s, (50, 130, 220),
                         (SHAFT_X - 16, ey - 14, 32, 28), border_radius=4)
        pygame.draw.rect(s, ACCENT,
                         (SHAFT_X - 16, ey - 14, 32, 28), 2, border_radius=4)
        elbl = self._font_s.render(f"F{env.floor}", True, WHITE)
        s.blit(elbl, (SHAFT_X - elbl.get_width() // 2, ey - elbl.get_height() // 2))

        # inside pax count
        n_in = len(env.inside)
        s.blit(self._font_s.render(f"{n_in}/{env.max_capacity} in car", True, ACCENT),
               (BUILD_X0, BUILD_Y0 + BUILD_H + 8))

        # sequence text
        seq_str = " → ".join(f"F{f}" for f in seq[:8])
        if len(seq) > 8:
            seq_str += " …"
        s.blit(self._font_s.render(seq_str or "—", True, DIM),
               (BUILD_X0 - 30, BUILD_Y0 + BUILD_H + 24))

    # ── metrics strip ──────────────────────────────────────────────────

    def _draw_metrics(self, s, agent, env):
        my = H - 72
        pygame.draw.line(s, GRID, (10, my - 8), (W - 10, my - 8), 1)

        items = [
            ("Colony",    f"{self.n_ants} ants",                    ACCENT),
            ("Evap ρ",    f"{self.evap_rate:.2f}",                  YELLOW),
            ("Best fit",  f"{agent.best_fitness:+.1f}",             GREEN if agent.best_fitness >= 0 else RED),
            ("τ max",     f"{agent.last_tau.max():.2f}",            WHITE),
            ("τ min",     f"{agent.last_tau.min():.2f}",            DIM),
            ("Pending",   str(env.total_waiting()),                  RED if env.total_waiting() > 4 else WHITE),
            ("Served",    str(env._served),                         GREEN),
        ]

        col_w = (W - 20) // len(items)
        for i, (lbl, val, col) in enumerate(items):
            x = 10 + i * col_w
            s.blit(self._font_s.render(lbl, True, DIM),  (x, my))
            s.blit(self._font_m.render(val, True, col),   (x, my + 16))

    # ── title ──────────────────────────────────────────────────────────

    def _draw_title(self, s):
        s.blit(self._font_t.render("ACO  —  Ant Colony Optimisation", True, ACCENT),
               (10, 12))
        s.blit(self._font_s.render(
            "heatmap: pheromone strength τ[from→to]   |   path: best ant this step",
            True, DIM), (10, 34))

    def close(self):
        pass
