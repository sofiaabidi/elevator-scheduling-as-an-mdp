"""
ElevatorEnv — Phase 0 upgrade.

Key changes from starter:
  - Passengers have src→dst pairs (hall call + car call model)
  - Poisson arrivals with optional peak/off-peak schedule
  - Capacity constraint (max_capacity passengers inside)
  - Wait time tracked per passenger
  - Auto-dropoff at destination, agent decides pickups
  - OPEN_DOOR = pick up waiting passengers at current floor
  - Reward shaped on wait time + travel time
  - State encodes: floor, direction, inside_destinations, hall_calls
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


# ---------------------------------------------------------------------------
# Passenger dataclass (plain object)
# ---------------------------------------------------------------------------

class Passenger:
    __slots__ = ("src", "dst", "arrived_step", "boarded_step")

    def __init__(self, src: int, dst: int, arrived_step: int):
        self.src          = src
        self.dst          = dst
        self.arrived_step = arrived_step   # step when they appeared on floor
        self.boarded_step = None           # step when they entered elevator


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ElevatorEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 8}

    UP        = 0
    DOWN      = 1
    STAY      = 2
    OPEN_DOOR = 3

    # Peak hours (step-based, assuming ~1 step per second feel)
    # Off-peak λ=0.15, peak λ=0.55
    PEAK_STEPS   = None   # set in __init__ if peak_offpeak=True

    def __init__(
        self,
        floors        = 5,
        max_steps     = 600,
        lam           = 0.25,          # Poisson λ (arrivals per step)
        peak_offpeak  = True,          # vary λ over episode
        max_capacity  = 6,
        render_mode   = None,
    ):
        super().__init__()
        self.floors       = floors
        self.max_steps    = max_steps
        self.lam          = lam
        self.peak_offpeak = peak_offpeak
        self.max_capacity = max_capacity
        self.render_mode  = render_mode

        # Observation space (for Gym compliance; Q-table uses obs_to_state)
        self.observation_space = spaces.Dict({
            "floor":       spaces.Discrete(floors),
            "direction":   spaces.Discrete(3),
            "inside_dst":  spaces.MultiBinary(floors),   # destinations of onboard passengers
            "hall_up":     spaces.MultiBinary(floors),
            "hall_down":   spaces.MultiBinary(floors),
            "capacity":    spaces.Discrete(max_capacity + 1),
        })
        self.action_space = spaces.Discrete(4)

        # State (reset in reset())
        self.floor      = 0
        self.direction  = 1          # 0=down 1=idle 2=up
        self._step      = 0
        self._served    = 0
        self._total_wait   = 0.0
        self._total_travel = 0.0

        self.inside: list[Passenger]         = []   # passengers in elevator
        self.waiting: list[list[Passenger]]  = [[] for _ in range(floors)]  # per floor

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_lam(self) -> float:
        """Return Poisson λ for this step (peak/off-peak schedule)."""
        if not self.peak_offpeak:
            return self.lam
        # Split episode into thirds: peak | off-peak | peak
        third = self.max_steps // 3
        if self._step < third or self._step >= 2 * third:
            return 0.55   # peak
        return 0.12       # off-peak

    def _inside_dst_flags(self) -> np.ndarray:
        flags = np.zeros(self.floors, dtype=np.int8)
        for p in self.inside:
            flags[p.dst] = 1
        return flags

    def _hall_flags(self):
        up   = np.zeros(self.floors, dtype=np.int8)
        down = np.zeros(self.floors, dtype=np.int8)
        for f in range(self.floors):
            for p in self.waiting[f]:
                if p.dst > p.src:
                    up[f] = 1
                else:
                    down[f] = 1
        return up, down

    def _get_obs(self) -> dict:
        up, down = self._hall_flags()
        return {
            "floor":      int(self.floor),
            "direction":  int(self.direction),
            "inside_dst": self._inside_dst_flags(),
            "hall_up":    up,
            "hall_down":  down,
            "capacity":   len(self.inside),
        }

    def obs_to_state(self, obs=None) -> tuple:
        """Flat hashable tuple for Q-table — all plain Python ints."""
        if obs is None:
            obs = self._get_obs()
        inside  = tuple(int(v) for v in obs["inside_dst"])
        hall    = tuple(
            int(obs["hall_up"][i]) * 2 + int(obs["hall_down"][i])
            for i in range(self.floors)
        )
        cap_bin = min(int(obs["capacity"]), 3)   # bucket: 0,1,2,3+
        return (int(obs["floor"]), int(obs["direction"]), cap_bin) + inside + hall

    @property
    def state_space_size(self) -> int:
        return self.floors * 3 * 4 * (2 ** self.floors) * (3 ** self.floors)

    def total_waiting(self) -> int:
        return sum(len(q) for q in self.waiting)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.floor     = int(self.np_random.integers(0, self.floors))
        self.direction = 1
        self._step     = 0
        self._served   = 0
        self._total_wait   = 0.0
        self._total_travel = 0.0
        self.inside  = []
        self.waiting = [[] for _ in range(self.floors)]

        # seed a few initial passengers
        for _ in range(3):
            self._spawn_passenger(force=True)

        return self._get_obs(), {}

    def step(self, action: int):
        self._step += 1
        reward = -0.5   # time penalty per step

        # ---- auto-dropoff passengers at current floor ----
        dropped = [p for p in self.inside if p.dst == self.floor]
        for p in dropped:
            travel = self._step - p.boarded_step
            wait   = (p.boarded_step - p.arrived_step)
            self._total_wait   += wait
            self._total_travel += travel
            reward += 15.0 - 0.3 * wait   # bonus decays with long wait
            self._served += 1
        self.inside = [p for p in self.inside if p.dst != self.floor]

        # ---- execute action ----
        if action == self.UP:
            if self.floor < self.floors - 1:
                self.floor += 1
                self.direction = 2
            else:
                reward -= 2.0

        elif action == self.DOWN:
            if self.floor > 0:
                self.floor -= 1
                self.direction = 0
            else:
                reward -= 2.0

        elif action == self.STAY:
            self.direction = 1
            if self.total_waiting() > 0 or len(self.inside) > 0:
                reward -= 1.0   # penalise idling when work exists

        elif action == self.OPEN_DOOR:
            # pick up waiting passengers at current floor (capacity limited)
            boarded = 0
            remaining = []
            for p in self.waiting[self.floor]:
                if len(self.inside) < self.max_capacity:
                    p.boarded_step = self._step
                    self.inside.append(p)
                    boarded += 1
                else:
                    remaining.append(p)
            self.waiting[self.floor] = remaining

            if boarded > 0:
                reward += 5.0 * boarded
            else:
                reward -= 2.0   # opened for nobody

        # ---- Poisson arrivals ----
        lam = self._current_lam()
        n_arrivals = self.np_random.poisson(lam)
        for _ in range(n_arrivals):
            self._spawn_passenger()

        # ---- wait time penalty (per waiting passenger per step) ----
        reward -= 0.05 * self.total_waiting()

        terminated = False
        truncated  = self._step >= self.max_steps

        obs  = self._get_obs()
        info = {
            "step":          self._step,
            "total_served":  self._served,
            "pending":       self.total_waiting(),
            "inside":        len(self.inside),
            "avg_wait":      self._total_wait / max(1, self._served),
            "lam":           lam,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spawn_passenger(self, force=False):
        """Spawn a passenger with a random src→dst pair."""
        max_pending = self.floors * self.max_capacity
        if self.total_waiting() >= max_pending:
            return
        src = int(self.np_random.integers(0, self.floors))
        # dst ≠ src, ground floor (0) is popular destination
        choices = [f for f in range(self.floors) if f != src]
        weights = np.array([3.0 if f == 0 else 1.0 for f in choices])
        weights /= weights.sum()
        dst = int(self.np_random.choice(choices, p=weights))
        self.waiting[src].append(Passenger(src, dst, self._step))

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Rendering subclass
# ---------------------------------------------------------------------------

class ElevatorEnvRender(ElevatorEnv):
    """Elevator env with rich pygame visualisation (Phase 0)."""

    # Layout constants
    W, H          = 1480, 620
    PANEL_BUILD   = (0,   0,   360, 620)   # x,y,w,h
    PANEL_METRICS = (370, 0,   260, 620)
    PANEL_DTREE   = (640, 0,   840, 620)

    FLOOR_H    = 100     # pixels per floor
    ELEV_W     = 54
    ELEV_H     = 70
    ELEV_X     = 150     # centre x of elevator shaft

    # Palette
    BG          = (15,  17,  26)
    GRID        = (30,  34,  50)
    WHITE       = (220, 225, 240)
    DIM         = (90,  95,  120)
    ACCENT      = (80,  180, 255)
    GREEN       = (60,  210, 120)
    YELLOW      = (255, 210, 60)
    RED         = (255, 80,  80)
    PURPLE      = (180, 100, 255)
    ELEV_COL    = (50,  130, 220)

    def __init__(self, floors=5, max_steps=600, lam=0.25,
                 peak_offpeak=True, max_capacity=6, render_mode="human", fps=8):
        super().__init__(floors=floors, max_steps=max_steps, lam=lam,
                         peak_offpeak=peak_offpeak, max_capacity=max_capacity,
                         render_mode=render_mode)
        self._fps = max(1, fps)
        if not HAS_PYGAME:
            raise ImportError("pygame required: pip install pygame")

        pygame.init()
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Elevator Simulator — Phase 0")
        self.clock  = pygame.time.Clock()

        self._font_lg = pygame.font.SysFont("monospace", 18, bold=True)
        self._font_md = pygame.font.SysFont("monospace", 14)
        self._font_sm = pygame.font.SysFont("monospace", 11)

        # Smooth elevator position
        self._render_y  = float(self._floor_y(0))
        self._side_surf  = None   # set externally: decision tree OR aco vis surface
        self._paused     = False

    # ------------------------------------------------------------------
    # coordinate helpers
    # ------------------------------------------------------------------

    def _floor_y(self, floor: int) -> int:
        """Top-left Y of elevator car when on `floor`."""
        # floor 0 = bottom of screen
        base_y = self.H - 40   # bottom margin
        return base_y - (floor + 1) * self.FLOOR_H + (self.FLOOR_H - self.ELEV_H) // 2

    def _floor_line_y(self, floor: int) -> int:
        base_y = self.H - 40
        return base_y - floor * self.FLOOR_H

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode != "human":
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self._fps = min(self._fps + 2, 60)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self._fps = max(1, self._fps - 2)
                elif event.key == pygame.K_SPACE:
                    self._paused = not self._paused

        if self._paused:
            self.clock.tick(15)
            return

        self.screen.fill(self.BG)
        self._draw_building()
        self._draw_metrics_panel()
        if self._side_surf is not None:
            dx = self.PANEL_DTREE[0]
            self.screen.blit(self._side_surf, (dx, 0))
            pygame.draw.line(self.screen, self.GRID, (dx, 0), (dx, self.H), 1)
        # FPS / pause hint
        hint = f"FPS {self._fps}  [+/-: speed  Space: pause]"
        if self._paused:
            hint = "PAUSED  [Space: resume]"
        self._blit(self.screen, hint, (self.W - 340, self.H - 20), self._font_sm, self.DIM)
        pygame.display.flip()
        self.clock.tick(self._fps)

    def _draw_building(self):
        s = self.screen
        bx, by, bw, bh = self.PANEL_BUILD

        # Panel label
        self._blit(s, "BUILDING", (bx + 10, by + 8), self._font_lg, self.ACCENT)

        # Smooth elevator movement
        target_y = float(self._floor_y(self.floor))
        self._render_y += (target_y - self._render_y) * 0.25

        # Shaft background
        shaft_x = self.ELEV_X - self.ELEV_W // 2 - 4
        shaft_w = self.ELEV_W + 8
        pygame.draw.rect(s, self.GRID, (shaft_x, 30, shaft_w, bh - 60))

        # Floor lines + labels
        for f in range(self.floors + 1):
            ly = self._floor_line_y(f)
            pygame.draw.line(s, self.GRID, (bx + 10, ly), (bx + bw - 10, ly), 1)

        for f in range(self.floors):
            mid_y = self._floor_line_y(f) - self.FLOOR_H // 2

            # Floor label
            self._blit(s, f"F{f}", (bx + 14, mid_y - 8), self._font_md, self.DIM)

            # Waiting passengers (left side)
            n_wait = len(self.waiting[f])
            for i in range(min(n_wait, 6)):
                age   = self._step - self.waiting[f][i].arrived_step
                col   = self._wait_color(age)
                px    = bx + 50 + i * 14
                self._draw_person(s, px, mid_y, col, size=9)

            if n_wait > 6:
                self._blit(s, f"+{n_wait-6}", (bx + 50 + 6*14, mid_y - 6),
                           self._font_sm, self.YELLOW)

            # Hall call arrows
            up_f, dn_f = False, False
            for p in self.waiting[f]:
                if p.dst > p.src: up_f = True
                else:             dn_f = True
            ax = self.ELEV_X + self.ELEV_W // 2 + 12
            if up_f:
                self._draw_arrow(s, ax, mid_y - 8, up=True,  col=self.GREEN)
            if dn_f:
                self._draw_arrow(s, ax, mid_y + 4, up=False, col=self.RED)

        # Elevator car
        ey = int(self._render_y)
        ex = self.ELEV_X - self.ELEV_W // 2
        pygame.draw.rect(s, self.ELEV_COL,  (ex, ey, self.ELEV_W, self.ELEV_H), border_radius=6)
        pygame.draw.rect(s, self.ACCENT,    (ex, ey, self.ELEV_W, self.ELEV_H), 2, border_radius=6)

        # Passengers inside elevator
        n_in = len(self.inside)
        for i in range(min(n_in, self.max_capacity)):
            px = ex + 6 + (i % 3) * 16
            py = ey + 10 + (i // 3) * 22
            self._draw_person(s, px, py, self.WHITE, size=8)

        # Capacity indicator
        cap_text = f"{n_in}/{self.max_capacity}"
        self._blit(s, cap_text, (ex + 4, ey + self.ELEV_H - 16), self._font_sm, self.ACCENT)

        # Direction arrow on elevator
        arr_x = ex + self.ELEV_W // 2
        if self.direction == 2:
            self._draw_arrow(s, arr_x, ey - 14, up=True,  col=self.GREEN, big=True)
        elif self.direction == 0:
            self._draw_arrow(s, arr_x, ey + self.ELEV_H + 4, up=False, col=self.RED, big=True)

        # Divider
        pygame.draw.line(s, self.GRID, (bx + bw, 0), (bx + bw, self.H), 1)

    def _draw_state_panel(self):
        s = self.screen
        px, py, pw, ph = self.PANEL_STATE
        self._blit(s, "STATE SPACE", (px + 10, py + 8), self._font_lg, self.ACCENT)

        cell_w = 54
        cell_h = (ph - 140) // self.floors

        headers = ["IN", "H↑", "H↓"]
        col_xs  = [px + 28, px + 28 + cell_w + 6, px + 28 + (cell_w + 6) * 2]
        for ci, h in enumerate(headers):
            self._blit(s, h, (col_xs[ci] + cell_w//2 - 8, py + 34),
                       self._font_sm, self.DIM)

        inside_dst         = self._inside_dst_flags()
        hall_up, hall_down = self._hall_flags()

        # draw rows: floor (floors-1) at top row, floor 0 at bottom row
        for row in range(self.floors):
            f  = (self.floors - 1) - row     # floor index: top row = highest floor
            ry = py + 52 + row * (cell_h + 4)

            # floor label — highlight if elevator is here
            lbl_col = self.ACCENT if f == self.floor else self.DIM
            self._blit(s, f"F{f}", (px + 6, ry + cell_h // 2 - 7),
                       self._font_sm, lbl_col)

            vals = [int(inside_dst[f]), int(hall_up[f]), int(hall_down[f])]
            base_cols = [self.PURPLE, self.GREEN, self.RED]

            for ci, (v, c) in enumerate(zip(vals, base_cols)):
                rx  = col_xs[ci]
                col = c if v else self.GRID
                pygame.draw.rect(s, col, (rx, ry, cell_w, cell_h), border_radius=4)
                # dot or empty ring
                cx_c = rx + cell_w // 2
                cy_c = ry + cell_h // 2
                if v:
                    pygame.draw.circle(s, self.WHITE, (cx_c, cy_c), 5)
                else:
                    pygame.draw.circle(s, self.DIM,   (cx_c, cy_c), 4, 1)

        # ── state tuple — clean, no np.int8 ──
        state     = self.obs_to_state()
        state_str = "(" + ", ".join(str(x) for x in state) + ")"
        # wrap at 28 chars
        half      = len(state_str) // 2
        line1     = state_str[:half]
        line2     = state_str[half:]
        self._blit(s, "state:", (px + 6, ph - 95), self._font_sm, self.DIM)
        self._blit(s, line1,    (px + 6, ph - 78), self._font_sm, self.ACCENT)
        self._blit(s, line2,    (px + 6, ph - 62), self._font_sm, self.ACCENT)

        # ── λ indicator ──
        lam   = self._current_lam()
        phase = "PEAK" if lam > 0.3 else "OFF-PEAK"
        col   = self.RED if lam > 0.3 else self.GREEN
        self._blit(s, f"λ={lam:.2f}  {phase}", (px + 6, ph - 38),
                   self._font_md, col)

        pygame.draw.line(s, self.GRID, (px + pw, 0), (px + pw, self.H), 1)

    def _draw_metrics_panel(self):
        s = self.screen
        px, py, pw, ph = self.PANEL_METRICS
        self._blit(s, "METRICS", (px + 10, py + 8), self._font_lg, self.ACCENT)

        served   = self._served
        avg_wait = self._total_wait / max(1, served)
        pending  = self.total_waiting()
        inside   = len(self.inside)

        rows = [
            ("Step",       f"{self._step}/{self.max_steps}",    self.WHITE),
            ("Served",     str(served),                          self.GREEN),
            ("Avg Wait",   f"{avg_wait:.1f} steps",              self.YELLOW),
            ("Pending",    str(pending),                          self.RED if pending > 4 else self.WHITE),
            ("In Car",     f"{inside}/{self.max_capacity}",      self.ACCENT),
            ("Floor",      str(self.floor),                       self.WHITE),
            ("Direction",  ["↓", "—", "↑"][self.direction],     self.ACCENT),
        ]

        for i, (label, val, col) in enumerate(rows):
            ry = py + 45 + i * 52
            self._blit(s, label, (px + 12, ry),       self._font_sm, self.DIM)
            self._blit(s, val,   (px + 12, ry + 18),  self._font_lg, col)
            pygame.draw.line(s, self.GRID,
                             (px + 8, ry + 40), (px + pw - 8, ry + 40), 1)

        # Progress bar
        bar_y = ph - 30
        bar_w = pw - 20
        prog  = self._step / max(1, self.max_steps)
        pygame.draw.rect(s, self.GRID,   (px + 10, bar_y, bar_w, 10), border_radius=5)
        pygame.draw.rect(s, self.ACCENT, (px + 10, bar_y, int(bar_w * prog), 10), border_radius=5)

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def _blit(self, surf, text, pos, font, color):
        surf.blit(font.render(str(text), True, color), pos)

    def _draw_person(self, surf, x, y, color, size=10):
        """Stick figure (circle head + line body)."""
        pygame.draw.circle(surf, color, (x, y), size // 2)
        pygame.draw.line(surf, color, (x, y + size // 2), (x, y + size + 4), 1)

    def _draw_arrow(self, surf, x, y, up: bool, col, big=False):
        sz = 8 if big else 5
        if up:
            pts = [(x, y - sz), (x - sz, y + sz//2), (x + sz, y + sz//2)]
        else:
            pts = [(x, y + sz), (x - sz, y - sz//2), (x + sz, y - sz//2)]
        pygame.draw.polygon(surf, col, pts)

    def _wait_color(self, age: int):
        if age < 20:  return self.GREEN
        if age < 50:  return self.YELLOW
        return self.RED

    # ------------------------------------------------------------------
    # Override step/reset to auto-render
    # ------------------------------------------------------------------

    def step(self, action):
        obs, rew, term, trunc, info = super().step(action)
        self.render()
        return obs, rew, term, trunc, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._render_y = float(self._floor_y(self.floor))
        self.render()
        return obs, info

    def close(self):
        if pygame.get_init():
            pygame.quit()
        super().close()