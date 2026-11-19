"""Minecraft-Style Pixel-Art Live Demo for Bayesian Deception MARL.

Unified Interactive Presentation Engine
---------------------------------------
Integrates the complete theoretical and algorithmic stack from the project:
1. Upgraded Recurrent Defender:
   - Shared GRU trunk (5-dim input: att_x, att_y, def_x, def_y, t/T_max).
   - Recurrent sequential belief screening + policy + value heads.
   - Hot-switchable in real time against Old Memoryless Guard via [D].
2. Exact Bayesian Receiver Teacher & KL Diagnostic:
   - Live recursive Bayes filter b_t(theta) proportional to b_{t-1}(theta) * pi_att(a_t | s_t, theta).
   - Real-time HUD telemetry of D_KL(GRU belief || Exact Bayes) in bits.
3. Scripted Trick Attacker Suite:
   - Interactive adversary cycling via [T]:
     RL Policy -> StraightTo7 -> FeintAndCut -> Zigzag -> MidPoolDash -> IdleRush -> NoiseJitter.
4. Adaptive Arms-Race & Caught-Route Archive:
   - Live Caught-Route Overlay toggleable via [C] (red translucent hazard tiles + cross markers).
   - Real-time Arms-Race Phase badge (Phase 1 / Phase 2 / Phase 3) and penalty mu_e.
   - Live learning: intercepting attacker immediately adds visited cells to the archive!
5. Presentation Playback & Navigation:
   - [P]: Pause / Resume simulation auto-advance.
   - [LEFT / RIGHT]: Frame-by-frame step backward and forward through episode history.
   - [1] / [2] / [R]: Force Goal A (Blue), Goal B (Red), or Uniform Random.
   - [SPACE]: Enter Arena / Rules Lobby.
   - [D]: Toggle Defender Engine (Upgraded Recurrent GRU vs Old Memoryless Guard).
   - [T]: Cycle Attacker Adversary Mode (RL Policy vs Scripted Tricks).
   - [C]: Toggle Caught-Route Hazard Overlay.
   - [ESC] / [Q]: Quit.
"""

import os
import sys
import pygame
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import EnvConfig
from envs.grid_world import DeceptionGridWorld as AsymmetricGridEnv
from models.networks import get_attacker_network, ActorCritic
from models.recurrent_defender import RecurrentDefender
from models.belief_head import BeliefHead
from algos.exact_bayes import ExactBayesFilter, compute_kl_diagnostic
from algos.dynamic_rl_engine import DynamicRLEngine
from envs.trick_attackers import (
    BaseTrickAttacker,
    StraightToKAttacker,
    FeintAttacker,
    ZigzagAttacker,
    MidPoolDashAttacker,
    IdleRushAttacker,
    NoiseJitterAttacker,
)

# ============================================================================
# CONSTANTS & DISPLAY LAYOUT
# ============================================================================
GRID_SIZE = 11
TILE_SIZE = 60
ARENA_WIDTH = GRID_SIZE * TILE_SIZE     # 660 px
ARENA_HEIGHT = GRID_SIZE * TILE_SIZE    # 660 px
HUD_WIDTH = 380                         # 380 px
SCREEN_WIDTH = ARENA_WIDTH + HUD_WIDTH  # 1040 px
SCREEN_HEIGHT = ARENA_HEIGHT            # 660 px

FPS = 60
STEP_DELAY_MS = 400                     # 0.4s per step auto-progression

# Colors
COLOR_BG_DARK = (20, 20, 26)           # #14141A
COLOR_HUD_BG = (30, 30, 36)            # #1E1E24
COLOR_HUD_PANEL = (22, 22, 28)         # #16161C
COLOR_HUD_BORDER = (55, 55, 68)        # #373744
COLOR_ACCENT_BLUE = (24, 144, 255)     # #1890FF Goal A
COLOR_ACCENT_RED = (245, 34, 45)       # #F5222D Goal B
COLOR_ACCENT_GOLD = (250, 173, 20)     # #FAAD14 Gold
COLOR_ACCENT_GREEN = (82, 196, 26)     # #52C41A High Entropy / Victory
COLOR_ACCENT_PURPLE = (180, 70, 240)   # Exact Bayes Teacher Accent
COLOR_TEXT_WHITE = (240, 240, 245)
COLOR_TEXT_MUTED = (160, 160, 175)

STATE_LOBBY = "LOBBY"
STATE_PLAYING = "PLAYING"
STATE_DONE = "DONE"


# ============================================================================
# PROCEDURAL PIXEL-ART SPRITES (16x16)
# ============================================================================

def make_surface_from_matrix(matrix: list[str], palette: dict, target_size: int = 48) -> pygame.Surface:
    """Convert an ASCII pixel matrix into a scaled, crisp Pygame surface."""
    h = len(matrix)
    w = len(matrix[0])
    raw_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(h):
        for x in range(w):
            color = palette.get(matrix[y][x], (0, 0, 0, 0))
            raw_surf.set_at((x, y), color)
    return pygame.transform.scale(raw_surf, (target_size, target_size))


NINJA_PALETTE = {
    ".": (0, 0, 0, 0),
    "K": (22, 22, 28),
    "D": (38, 40, 52),
    "M": (56, 60, 78),
    "L": (82, 88, 114),
    "R": (180, 30, 30),
    "r": (230, 60, 60),
    "S": (245, 200, 160),
    "W": (255, 255, 255),
    "E": (18, 18, 24),
}

NINJA_MATRIX = [
    ".....KKKKKK.....",
    "....KMMMMMMK....",
    "...KMLLLLLLMK...",
    "..KMLDDDDDDLMK..",
    "..KLDDrrrrrDDLK.",
    "..KLDDRRRRRDDLK.",
    "..KMDDSSSSDDMK..",
    "..KMDESSSSEMK...",
    "..KMDEWSSWEMK...",
    "..KMDEESSEEMK...",
    "..KMDDSSSSDDMK..",
    "..KLDDRRRRDDLK..",
    "..KLDDrrrrrDDLK.",
    "..KMLDDDDDDLMK..",
    "...KMLLLLLLMK...",
    "....KMMMMMMK....",
]

KNIGHT_PALETTE = {
    ".": (0, 0, 0, 0),
    "K": (25, 25, 30),
    "S": (75, 78, 88),
    "I": (140, 145, 160),
    "H": (205, 210, 225),
    "W": (250, 250, 255),
    "G": (210, 160, 30),
    "g": (245, 205, 70),
    "R": (200, 40, 40),
}

KNIGHT_MATRIX = [
    ".....KKKKKK.....",
    "....KgGGGGgK....",
    "...KgGGGGGGgK...",
    "..KKgGGGGGGgKK..",
    "..KIIHHHHHHIIK..",
    "..KIHWWWWWWWHIK.",
    "..KIHSSKKSSHIK..",
    "..KIHSRKKRSHIK..",
    "..KIHSSKKSSHIK..",
    "..KIHWWWWWWWHIK.",
    "..KIIHHHHHHIIK..",
    "..KSSIHHHHISIK..",
    "..KSSIIHHIIKSSK.",
    "..KKSSIIIIKKSSK.",
    "...KKSSSSKKSSK..",
    ".....KKKKKK.....",
]

GOAL_A_PALETTE = {
    ".": (0, 0, 0, 0),
    "B": (10, 50, 110),
    "D": (20, 90, 190),
    "L": (40, 150, 255),
    "C": (100, 200, 255),
    "W": (220, 245, 255),
}

GOAL_A_MATRIX = [
    "......BBBB......",
    "....BBDDDDBB....",
    "...BDDLLLLDDB...",
    "..BDLLCCCCLLDB..",
    "..BDLCCWWCCDLB..",
    ".BDLCCWWWWCCDLB.",
    ".BDLCWWWWWWCLDB.",
    ".BDLCWWWWWWCLDB.",
    ".BDLCCWWWWCCDLB.",
    "..BDLCCWWCCDLB..",
    "..BDLLCCCCLLDB..",
    "...BDDLLLLDDB...",
    "....BBDDDDBB....",
    "......BBBB......",
    "................",
    "................",
]

GOAL_B_PALETTE = {
    ".": (0, 0, 0, 0),
    "R": (110, 15, 20),
    "D": (185, 30, 40),
    "L": (240, 60, 70),
    "O": (255, 120, 80),
    "W": (255, 230, 220),
}

GOAL_B_MATRIX = [
    "......RRRR......",
    "....RRDDDDRR....",
    "...RDLLLLLLDR...",
    "..RDLLOOOLLDR...",
    "..RDLOOWWOOLDR..",
    ".RDLOOWWWWOOLDR.",
    ".RDLOOWWWWOOLDR.",
    ".RDLOOWWWWOOLDR.",
    ".RDLOOWWWWOOLDR.",
    "..RDLOOWWOOLDR..",
    "..RDLLOOOLLDR...",
    "...RDLLLLLLDR...",
    "....RRDDDDRR....",
    "......RRRR......",
    "................",
    "................",
]

TRAIL_PALETTE = {
    ".": (0, 0, 0, 0),
    "D": (180, 140, 20, 100),
    "Y": (250, 200, 40, 180),
    "W": (255, 255, 180, 230),
}

TRAIL_SPRITE_MATRIX = [
    "................",
    "................",
    "......DYYD......",
    "....DYYYYYYD....",
    "...DYYWWWWYYD...",
    "...YYWWWWWWYY...",
    "..DYYWWWWWWYYD..",
    "..YYYYWWWWYYYY..",
    "..YYYYWWWWYYYY..",
    "..DYYWWWWWWYYD..",
    "...YYWWWWWWYY...",
    "...DYYWWWWYYD...",
    "....DYYYYYYD....",
    "......DYYD......",
    "................",
    "................",
]


# ============================================================================
# PROCEDURAL MINECRAFT TILES (GRASS & COBBLESTONE)
# ============================================================================

def create_grass_tile(variant: int = 0) -> pygame.Surface:
    """Create textured Minecraft grass block."""
    surf = pygame.Surface((TILE_SIZE, TILE_SIZE))
    base_color = (78, 128, 49) if variant == 0 else (69, 117, 42)
    dark_speck = (62, 106, 37) if variant == 0 else (55, 94, 32)
    light_speck = (94, 149, 60) if variant == 0 else (84, 136, 52)
    border_color = (55, 95, 32) if variant == 0 else (48, 84, 27)

    surf.fill(base_color)
    rng = np.random.RandomState(variant * 100 + 42)
    for _ in range(18):
        x = int(rng.randint(2, TILE_SIZE - 4))
        y = int(rng.randint(2, TILE_SIZE - 4))
        pygame.draw.rect(surf, dark_speck, (x, y, 2, 2))
    for _ in range(14):
        x = int(rng.randint(2, TILE_SIZE - 4))
        y = int(rng.randint(2, TILE_SIZE - 4))
        pygame.draw.rect(surf, light_speck, (x, y, 2, 2))
    pygame.draw.rect(surf, border_color, (0, 0, TILE_SIZE, TILE_SIZE), 1)
    return surf


def create_cobblestone_tile() -> pygame.Surface:
    """Create textured cobblestone path tile for median column 5."""
    surf = pygame.Surface((TILE_SIZE, TILE_SIZE))
    base_color = (115, 117, 120)
    mortar_color = (76, 78, 81)
    highlight_color = (146, 148, 151)
    deep_shadow = (56, 57, 59)

    surf.fill(base_color)
    row_h = TILE_SIZE // 3
    for r in range(3):
        y0 = r * row_h
        pygame.draw.line(surf, mortar_color, (0, y0), (TILE_SIZE, y0), 2)
        shift = 15 if r % 2 == 1 else 0
        for bx in range(-shift, TILE_SIZE, 30):
            if bx >= 0:
                pygame.draw.line(surf, mortar_color, (bx, y0), (bx, y0 + row_h), 2)
            pygame.draw.line(surf, highlight_color, (bx + 2, y0 + 2), (bx + 28, y0 + 2), 1)
            pygame.draw.line(surf, deep_shadow, (bx + 2, y0 + row_h - 2), (bx + 28, y0 + row_h - 2), 1)
    pygame.draw.rect(surf, mortar_color, (0, 0, TILE_SIZE, TILE_SIZE), 2)
    return surf


# ============================================================================
# MAIN APPLICATION CONTROLLER
# ============================================================================

class MinecraftLiveDemo:
    """Interactive Game-Theoretic Presentation Suite."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Bayesian Deception MARL - Minecraft Arena")

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # Typography
        self.font_title = pygame.font.SysFont("impact", 30)
        self.font_subtitle = pygame.font.SysFont("trebuchetms", 17, bold=True)
        self.font_header = pygame.font.SysFont("trebuchetms", 15, bold=True)
        self.font_body = pygame.font.SysFont("segoeui", 13)
        self.font_small = pygame.font.SysFont("segoeui", 11, bold=True)
        self.font_mono = pygame.font.SysFont("consolas", 12, bold=True)

        print("[LiveDemo] Initializing procedural assets and neural architectures...")
        self.grass_tile_a = create_grass_tile(variant=0)
        self.grass_tile_b = create_grass_tile(variant=1)
        self.cobble_tile = create_cobblestone_tile()

        sprite_sz = int(TILE_SIZE * 0.8)  # 48px
        self.ninja_sprite = make_surface_from_matrix(NINJA_MATRIX, NINJA_PALETTE, target_size=sprite_sz)
        self.knight_sprite = make_surface_from_matrix(KNIGHT_MATRIX, KNIGHT_PALETTE, target_size=sprite_sz)
        self.goal_a_sprite = make_surface_from_matrix(GOAL_A_MATRIX, GOAL_A_PALETTE, target_size=sprite_sz)
        self.goal_b_sprite = make_surface_from_matrix(GOAL_B_MATRIX, GOAL_B_PALETTE, target_size=sprite_sz)
        self.trail_sprite = make_surface_from_matrix(TRAIL_SPRITE_MATRIX, TRAIL_PALETTE, target_size=int(TILE_SIZE * 0.45))

        # Hardware & Environment (Swept-sphere swap collision detection enabled)
        self.device = torch.device("cpu")
        self.env_cfg = EnvConfig(include_time_in_obs=True, detect_swaps=True)
        self.env = AsymmetricGridEnv(self.env_cfg)

        # ------------------------------------------------------------------ #
        # Load Trained Neural Models
        # ------------------------------------------------------------------ #
        models_dir = os.path.join(CURRENT_DIR, "models")
        att_path = os.path.join(models_dir, "attacker_net.pt")
        rec_def_path = os.path.join(models_dir, "recurrent_defender.pt")
        old_def_path = os.path.join(models_dir, "defender_net.pt")
        old_bh_path = os.path.join(models_dir, "belief_head.pt")

        # 1. Attacker (ActorCritic, obs_dim=3, act_dim=9)
        self.attacker_net = get_attacker_network(hidden_dim=64, action_dim=9).to(self.device)
        if os.path.exists(att_path):
            self.attacker_net.load_state_dict(torch.load(att_path, map_location=self.device))
        self.attacker_net.eval()

        # 2. Upgraded Recurrent Defender (RecurrentDefender, obs_dim=5, act_dim=9)
        self.recurrent_defender = RecurrentDefender(obs_dim=5, action_dim=9, hidden_dim=64).to(self.device)
        if os.path.exists(rec_def_path):
            self.recurrent_defender.load_state_dict(torch.load(rec_def_path, map_location=self.device))
        self.recurrent_defender.eval()

        # 3. Old Memoryless Defender (obs_dim=6, act_dim=9) & Old Belief Head (input_dim=4)
        self.old_defender_net = ActorCritic(obs_dim=6, action_dim=9, hidden_dim=64).to(self.device)
        if os.path.exists(old_def_path):
            self.old_defender_net.load_state_dict(torch.load(old_def_path, map_location=self.device))
        self.old_defender_net.eval()

        self.old_belief_head = BeliefHead(input_dim=4, hidden_dim=64).to(self.device)
        if os.path.exists(old_bh_path):
            self.old_belief_head.load_state_dict(torch.load(old_bh_path, map_location=self.device))
        self.old_belief_head.eval()

        # 4. Exact Bayes Receiver Teacher
        self.bayes_filter = ExactBayesFilter(self.attacker_net, device=str(self.device), eps=1e-6)

        # 5. Dynamic Online RL Co-Evolution Engine
        self.rl_engine = DynamicRLEngine(device=self.device)

        # 6. Attacker Adversary Suite (Dynamic RL + Curriculum Tricks)
        self.trick_adversaries: List[Tuple[str, any]] = [
            ("★ Dynamic RL Co-Evolution (Live Learning)", "DYNAMIC_RL"),
            ("Trained Static RL Policy (Ninja)", None),
            ("Curriculum: Straight-to-Row-7 Commit", StraightToKAttacker(k=7)),
            ("Curriculum: Straight-to-Row-8 Late Pivot", StraightToKAttacker(k=8)),
            ("Curriculum: Feint-and-Cut Sharp Turn", FeintAttacker(feint_steps=4)),
            ("Curriculum: Zigzag Oscillation", ZigzagAttacker()),
            ("Curriculum: Mid-Pool Dash (Row 5)", MidPoolDashAttacker()),
            ("Curriculum: Idle-Rush Stall Rush", IdleRushAttacker(idle_steps=3)),
            ("Curriculum: Stochastic Noise Jitter", NoiseJitterAttacker(noise_prob=0.25)),
        ]
        self.current_trick_idx = 0  # 0 is Dynamic Online RL

        # Interactive Mode Toggles
        self.use_recurrent_defender = True
        self.show_caught_overlay = True
        self.toast_msg = "LIVE RL ACTIVE: [E] NEXT EPOCH | [0] RESET | [L] AUTO-LEARN"
        self.toast_time = pygame.time.get_ticks()

        # Telemetry & Caught Archive Data
        self.arms_phase = 3
        self.arms_mu_e = 0.85
        self.caught_map: Optional[np.ndarray] = None
        self._load_arms_race_telemetry()

        # App State
        self.state = STATE_LOBBY
        self.episode_count = 0
        self.last_step_time = 0
        self.paused = False
        self.history: list[dict] = []
        self.view_idx: int = 0

        # Episode Real-time tracking
        self.theta = 0
        self.att_obs = None
        self.def_obs = None
        self.def_h = None
        self.old_def_h = None
        self.visited_tiles: list[tuple[int, int]] = []
        self.step_counter = 0

        self.current_gru_probs = np.array([0.5, 0.5], dtype=np.float32)
        self.current_bayes_probs = np.array([0.5, 0.5], dtype=np.float32)
        self.current_entropy = 1.0000
        self.current_kl_div = 0.0000

        self.final_outcome = ""
        self.final_status_color = COLOR_ACCENT_GREEN
        self.final_is_goal = False

    def _load_arms_race_telemetry(self) -> None:
        """Load caught-route archive from models/caught_routes.npz or generate high-heat fallback."""
        telemetry_path = os.path.join(CURRENT_DIR, "models", "caught_routes.npz")
        loaded = False
        if os.path.exists(telemetry_path):
            try:
                data = np.load(telemetry_path)
                self.caught_map = data["C_theta"]
                self.arms_phase = int(data.get("phase", 3))
                self.arms_mu_e = float(data.get("mu_e", 0.85))
                if self.caught_map is not None and self.caught_map.max() > 0:
                    loaded = True
                    print(f"[LiveDemo] Loaded Caught-Route Archive: Phase {self.arms_phase}, mu_e={self.arms_mu_e:.2f}")
            except Exception as e:
                print(f"[LiveDemo] Warning: Could not load caught_routes.npz: {e}")

        if not loaded:
            self.caught_map = np.zeros((2, GRID_SIZE, GRID_SIZE), dtype=np.float32)
            # Goal A dangerous cells: straight approach into row 8-9 and premature turns
            for r in range(6, 10):
                self.caught_map[0, r, 5] = 0.95
            self.caught_map[0, 8, 4] = 0.85
            self.caught_map[0, 9, 3] = 0.70
            self.caught_map[0, 8, 5] = 1.00
            self.caught_map[0, 9, 5] = 1.00
            self.caught_map[0, 3, 6] = 0.50
            self.caught_map[0, 4, 7] = 0.60

            # Goal B dangerous cells: straight approach into row 8-9 and premature turns
            for r in range(6, 10):
                self.caught_map[1, r, 5] = 0.95
            self.caught_map[1, 8, 6] = 0.85
            self.caught_map[1, 9, 7] = 0.70
            self.caught_map[1, 8, 5] = 1.00
            self.caught_map[1, 9, 5] = 1.00
            self.caught_map[1, 3, 4] = 0.50
            self.caught_map[1, 4, 3] = 0.60
            self.arms_phase = 3
            self.arms_mu_e = 0.85

    def grid_to_screen(self, row: float, col: float) -> tuple[int, int]:
        """Convert environment (row, col) coordinates to screen (x, y) pixels."""
        pixel_x = int(round(col * TILE_SIZE))
        pixel_y = int(round((10 - row) * TILE_SIZE))
        return pixel_x, pixel_y

    def _create_snapshot(self, step: int, done: bool = False, outcome_text: str = "") -> dict:
        """Capture complete snapshot of all visual and Bayesian telemetry for playback."""
        att_pos = self.env.attacker_pos.copy()
        def_pos = self.env.defender_pos.copy()
        att_r = int(round(att_pos[0]))
        att_c = int(round(att_pos[1]))
        def_r = int(round(def_pos[0]))
        def_c = int(round(def_pos[1]))
        dist = float(np.linalg.norm(att_pos - def_pos))

        phase_str = "Median Corridor (Col 5)" if (att_r <= 7 and att_c == 5) else "Diagonal Bifurcation Phase"
        phase_col = COLOR_ACCENT_GREEN if (att_r <= 7 and att_c == 5) else COLOR_ACCENT_GOLD

        return {
            "attacker_pos": (float(att_pos[0]), float(att_pos[1])),
            "defender_pos": (float(def_pos[0]), float(def_pos[1])),
            "visited_tiles": list(self.visited_tiles),
            "step": step,
            "p_a": float(np.clip(self.current_gru_probs[0], 0.0, 1.0)),
            "p_b": float(np.clip(self.current_gru_probs[1], 0.0, 1.0)),
            "bayes_a": float(np.clip(self.current_bayes_probs[0], 0.0, 1.0)),
            "bayes_b": float(np.clip(self.current_bayes_probs[1], 0.0, 1.0)),
            "entropy": float(self.current_entropy),
            "kl_div": float(self.current_kl_div),
            "done": done,
            "outcome_text": outcome_text,
            "att_r": att_r,
            "att_c": att_c,
            "def_r": def_r,
            "def_c": def_c,
            "dist": dist,
            "phase_str": phase_str,
            "phase_col": phase_col,
            "final_outcome": self.final_outcome,
            "final_status_color": self.final_status_color,
            "final_is_goal": self.final_is_goal,
            "defender_engine": "RECURRENT GRU" if self.use_recurrent_defender else "MEMORYLESS MLP",
            "attacker_mode": self.trick_adversaries[self.current_trick_idx][0],
            "rl_epoch": self.rl_engine.epoch,
            "rl_stage": self.rl_engine.current_stage,
            "rl_delta_w": self.rl_engine.weight_delta_mag,
            "rl_loss": self.rl_engine.last_loss_val,
            "rl_msg": self.rl_engine.last_update_msg,
            "rl_auto": self.rl_engine.learning_enabled,
        }

    def reset_episode(self, force_theta: Optional[int] = None) -> None:
        """Reset episode and initialize active trick attacker / bayes filters."""
        if force_theta is not None:
            desired_theta = int(force_theta)
        else:
            desired_theta = int(np.random.choice([0, 1]))

        obs_dict, info = self.env.reset(options={"theta": desired_theta})
        self.theta = info["theta"]
        self.att_obs = obs_dict["attacker"]
        self.def_obs = obs_dict["defender"]
        self.def_h = None
        self.old_def_h = None

        # Reset Exact Bayes filter to uniform prior
        self.bayes_filter.reset()

        # Reset active trick attacker if selected
        active_trick = self.trick_adversaries[self.current_trick_idx][1]
        if active_trick is not None and active_trick != "DYNAMIC_RL":
            active_trick.reset(theta=self.theta)

        self.visited_tiles = [(int(self.env.attacker_pos[0]), int(self.env.attacker_pos[1]))]
        self.step_counter = 0
        self.current_gru_probs = np.array([0.5, 0.5], dtype=np.float32)
        self.current_bayes_probs = np.array([0.5, 0.5], dtype=np.float32)
        self.current_entropy = 1.0000
        self.current_kl_div = 0.0000
        self.final_outcome = ""
        self.final_is_goal = False

        self.paused = False
        self.history = [self._create_snapshot(step=0, done=False, outcome_text=f"Target: Goal {'A (Blue)' if self.theta == 0 else 'B (Red)'}")]
        self.view_idx = 0
        self.state = STATE_PLAYING
        self.last_step_time = pygame.time.get_ticks()

    def advance_environment_step(self) -> None:
        """Execute one complete forward step with full Bayesian, recurrent, and trick suite logic."""
        att_pos = self.env.attacker_pos.copy()
        def_pos = self.env.defender_pos.copy()

        # ------------------------------------------------------------------ #
        # 1. Attacker Action Selection (Dynamic RL or Scripted Trick / Static)
        # ------------------------------------------------------------------ #
        active_trick = self.trick_adversaries[self.current_trick_idx][1]
        if active_trick == "DYNAMIC_RL":
            greedy_att_action = self.rl_engine.get_action(
                att_pos, def_pos, self.theta, caught_map=self.caught_map
            )
        elif active_trick is not None:
            greedy_att_action = active_trick.get_action(att_pos, def_pos)
        else:
            att_obs_t = torch.as_tensor(self.att_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                att_logits, _ = self.attacker_net(att_obs_t)
            greedy_att_action = int(torch.argmax(att_logits, dim=-1).item())

        # ------------------------------------------------------------------ #
        # 2. Exact Bayesian Receiver Teacher Update
        # ------------------------------------------------------------------ #
        self.current_bayes_probs = self.bayes_filter.update(att_pos, greedy_att_action)

        # ------------------------------------------------------------------ #
        # 3. Defender Action Selection (Upgraded Recurrent GRU vs Old Guard)
        # ------------------------------------------------------------------ #
        if self.use_recurrent_defender:
            # 5-dimensional recurrent observation [att_x, att_y, def_x, def_y, t/T_max]
            norm_t = float(self.step_counter) / float(self.env_cfg.horizon)
            def_obs_5d = np.array([
                att_pos[0] / 10.0,
                att_pos[1] / 10.0,
                def_pos[0] / 10.0,
                def_pos[1] / 10.0,
                norm_t,
            ], dtype=np.float32)

            def_obs_t = torch.as_tensor(def_obs_5d, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                act_t, _, _, _, _, probs_t, self.def_h = self.recurrent_defender.forward_step(
                    def_obs_t, self.def_h, deterministic=True
                )
            greedy_def_action = int(act_t.item())
            self.current_gru_probs = probs_t.squeeze(0).cpu().numpy()

            # Co-evolution hook in Dynamic RL mode
            active_trick = self.trick_adversaries[self.current_trick_idx][1]
            if active_trick == "DYNAMIC_RL":
                active_belief = self.current_bayes_probs if (hasattr(self, "current_bayes_probs") and self.current_bayes_probs is not None) else self.current_gru_probs
                greedy_def_action = self.rl_engine.get_defender_action(
                    att_pos, def_pos, active_belief, greedy_def_action, norm_t=norm_t
                )

        else:
            # Old Memoryless Defender logic: separate BeliefHead + 6D MLP
            def_obs_4d = np.array([
                att_pos[0] / 10.0,
                att_pos[1] / 10.0,
                def_pos[0] / 10.0,
                def_pos[1] / 10.0,
            ], dtype=np.float32)
            def_obs_t = torch.as_tensor(def_obs_4d, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                bh_logits, self.old_def_h = self.old_belief_head.step(def_obs_t, self.old_def_h)
                old_probs = F.softmax(bh_logits / 1.5, dim=-1).squeeze(0).cpu().numpy()

                def_obs_full = np.concatenate([def_obs_4d, old_probs], axis=-1).astype(np.float32)
                def_obs_full_t = torch.as_tensor(def_obs_full, dtype=torch.float32, device=self.device).unsqueeze(0)
                def_logits, _ = self.old_defender_net(def_obs_full_t)

            greedy_def_action = int(torch.argmax(def_logits, dim=-1).item())
            self.current_gru_probs = old_probs

        # ------------------------------------------------------------------ #
        # 4. Bayesian Metrics (Entropy & KL Divergence)
        # ------------------------------------------------------------------ #
        p = self.current_gru_probs
        self.current_entropy = -float(np.sum(p * np.log2(p + 1e-8)))
        self.current_kl_div = compute_kl_diagnostic(self.current_gru_probs, self.current_bayes_probs)

        # ------------------------------------------------------------------ #
        # 5. Advance Environment (Swept-Sphere Swap Physics)
        # ------------------------------------------------------------------ #
        next_obs_dict, _, terminated, truncated, step_info = self.env.step(
            greedy_att_action, greedy_def_action
        )
        self.step_counter += 1

        cur_pos = (int(round(self.env.attacker_pos[0])), int(round(self.env.attacker_pos[1])))
        if cur_pos not in self.visited_tiles:
            self.visited_tiles.append(cur_pos)

        is_done = bool(terminated or truncated)
        outcome = ""

        if is_done:
            self.state = STATE_DONE
            self.episode_count += 1

            if step_info.get("goal_reached", False):
                self.final_is_goal = True
                self.final_outcome = f"GOAL {'A (BLUE)' if self.theta == 0 else 'B (RED)'} CAPTURED!"
                self.final_status_color = COLOR_ACCENT_GREEN
            elif step_info.get("intercepted", False):
                self.final_is_goal = False
                self.final_outcome = "INTERCEPTED BY GUARD!"
                self.final_status_color = COLOR_ACCENT_RED
            else:
                self.final_is_goal = False
                self.final_outcome = "TIMEOUT REACHED"
                self.final_status_color = COLOR_ACCENT_GOLD

            # Dynamic Online RL Update (Epoch-by-Epoch Co-Evolution)
            active_trick = self.trick_adversaries[self.current_trick_idx][1]
            if active_trick == "DYNAMIC_RL" and self.rl_engine.learning_enabled:
                learn_res = self.rl_engine.execute_online_learning_step(
                    trajectory=self.visited_tiles,
                    theta=self.theta,
                    intercepted=bool(step_info.get("intercepted", False)),
                    goal_reached=bool(step_info.get("goal_reached", False)),
                    mean_entropy=self.current_entropy,
                    caught_map=self.caught_map,
                )
                self.toast_msg = learn_res["msg"]
                self.toast_time = pygame.time.get_ticks()
                self.arms_phase = min(3, (self.rl_engine.epoch // 5) + 1)
                self.arms_mu_e = learn_res["arms_mu_e"]
            elif self.caught_map is not None and step_info.get("intercepted", False):
                for r, c in self.visited_tiles:
                    self.caught_map[self.theta, r, c] = 1.0
                self.toast_msg = "NEW INTERCEPTION RECORDED TO CAUGHT ARCHIVE!"
                self.toast_time = pygame.time.get_ticks()

            outcome = self.final_outcome

        snapshot = self._create_snapshot(step=self.step_counter, done=is_done, outcome_text=outcome)
        self.history.append(snapshot)
        self.view_idx = len(self.history) - 1

        if not is_done:
            self.att_obs = next_obs_dict["attacker"]
            self.def_obs = next_obs_dict["defender"]

    # ========================================================================
    # RENDERING METHODS
    # ========================================================================

    def render_arena(self, snapshot: dict) -> None:
        """Render 660x660 Minecraft arena with terrain, trail, caught hazard tiles, and sprites."""
        # 1. Base Grid Tiles
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                px, py = self.grid_to_screen(r, c)
                if c == 5:
                    self.screen.blit(self.cobble_tile, (px, py))
                else:
                    tile = self.grass_tile_a if (r + c) % 2 == 0 else self.grass_tile_b
                    self.screen.blit(tile, (px, py))

        # 2. Footprint Trail
        trail_offset = (TILE_SIZE - self.trail_sprite.get_width()) // 2
        for r, c in snapshot["visited_tiles"]:
            tx, ty = self.grid_to_screen(r, c)
            self.screen.blit(self.trail_sprite, (tx + trail_offset, ty + trail_offset))

        # 3. Caught-Route Hazard Overlay
        if self.show_caught_overlay and self.caught_map is not None:
            c_map = self.caught_map[self.theta]
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    val = float(c_map[r, c])
                    if val > 0.05:
                        px, py = self.grid_to_screen(r, c)
                        overlay = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                        alpha = int(np.clip(val * 130 + 50, 70, 200))
                        overlay.fill((235, 25, 25, alpha))
                        self.screen.blit(overlay, (px, py))
                        pygame.draw.rect(self.screen, (255, 60, 60), (px, py, TILE_SIZE, TILE_SIZE), 2)

                        # Small hazard cross
                        cx, cy = px + TILE_SIZE // 2, py + TILE_SIZE // 2
                        cs = 5
                        pygame.draw.line(self.screen, (255, 220, 220), (cx - cs, cy - cs), (cx + cs, cy + cs), 2)
                        pygame.draw.line(self.screen, (255, 220, 220), (cx - cs, cy + cs), (cx + cs, cy - cs), 2)

        # 4. On-Screen Toast Notification
        curr_ticks = pygame.time.get_ticks()
        if hasattr(self, "toast_msg") and self.toast_msg and (curr_ticks - self.toast_time < 2800):
            toast_w, toast_h = 440, 36
            tx, ty = (ARENA_WIDTH - toast_w) // 2, 16
            t_rect = pygame.Rect(tx, ty, toast_w, toast_h)
            toast_surf = pygame.Surface((toast_w, toast_h), pygame.SRCALPHA)
            toast_surf.fill((16, 18, 24, 235))
            self.screen.blit(toast_surf, (tx, ty))
            border_col = (255, 60, 60) if self.show_caught_overlay else (180, 180, 190)
            pygame.draw.rect(self.screen, border_col, t_rect, width=2, border_radius=6)
            txt_col = (255, 80, 80) if self.show_caught_overlay else COLOR_TEXT_WHITE
            t_txt = self.font_small.render(self.toast_msg, True, txt_col)
            self.screen.blit(t_txt, t_txt.get_rect(center=t_rect.center))

        # 5. Goal Sprites
        sprite_offset = (TILE_SIZE - self.ninja_sprite.get_width()) // 2
        ga_x, ga_y = self.grid_to_screen(10, 0)
        gb_x, gb_y = self.grid_to_screen(10, 10)
        self.screen.blit(self.goal_a_sprite, (ga_x + sprite_offset, ga_y + sprite_offset))
        self.screen.blit(self.goal_b_sprite, (gb_x + sprite_offset, gb_y + sprite_offset))

        # 6. Defender (Knight)
        def_r, def_c = snapshot["defender_pos"]
        dx, dy = self.grid_to_screen(def_r, def_c)
        self.screen.blit(self.knight_sprite, (dx + sprite_offset, dy + sprite_offset))

        # 7. Attacker (Ninja)
        att_r, att_c = snapshot["attacker_pos"]
        ax, ay = self.grid_to_screen(att_r, att_c)
        self.screen.blit(self.ninja_sprite, (ax + sprite_offset, ay + sprite_offset))

        # Paused banner
        if self.paused:
            total_steps = self.history[-1]["step"]
            paused_rect = pygame.Rect(ARENA_WIDTH // 2 - 90, 12, 180, 30)
            pygame.draw.rect(self.screen, (22, 22, 28), paused_rect, border_radius=4)
            pygame.draw.rect(self.screen, COLOR_ACCENT_GOLD, paused_rect, width=1, border_radius=4)
            p_text = self.font_small.render(f"⏸ PAUSED [{snapshot['step']}/{total_steps}]", True, COLOR_ACCENT_GOLD)
            self.screen.blit(p_text, p_text.get_rect(center=paused_rect.center))

    def render_hud(self, snapshot: dict) -> None:
        """Render 380x660 Live HUD with Dual Belief bars, Exact Bayes, Entropy, and AI Engine controls."""
        hud_rect = pygame.Rect(ARENA_WIDTH, 0, HUD_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_HUD_BG, hud_rect)
        pygame.draw.line(self.screen, (40, 42, 52), (ARENA_WIDTH, 0), (ARENA_WIDTH, SCREEN_HEIGHT), 2)

        hud_x = ARENA_WIDTH + 16
        hud_w = HUD_WIDTH - 32

        # --- Section 1: Episode Header ---
        y = 14
        card_rect = pygame.Rect(hud_x, y, hud_w, 42)
        pygame.draw.rect(self.screen, COLOR_HUD_PANEL, card_rect, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_HUD_BORDER, card_rect, width=1, border_radius=6)

        ep_text = self.font_header.render(f"MATCH #{self.episode_count + 1:02d}", True, COLOR_TEXT_WHITE)
        step_text = self.font_header.render(f"STEP: {snapshot['step']:02d} / 25", True, COLOR_ACCENT_GOLD)
        self.screen.blit(ep_text, (hud_x + 12, y + 12))
        self.screen.blit(step_text, (hud_x + hud_w - step_text.get_width() - 12, y + 12))

        # --- Section 2: Secret Objective Card ---
        y = 62
        target_box = pygame.Rect(hud_x, y, hud_w, 36)
        if self.theta == 0:
            target_color = COLOR_ACCENT_BLUE
            target_title = "SECRET: GOAL A (BLUE) - TOP-LEFT"
            bg_card = (16, 32, 54)
        else:
            target_color = COLOR_ACCENT_RED
            target_title = "SECRET: GOAL B (RED) - TOP-RIGHT"
            bg_card = (54, 18, 22)

        pygame.draw.rect(self.screen, bg_card, target_box, border_radius=6)
        pygame.draw.rect(self.screen, target_color, target_box, width=2, border_radius=6)
        txt_target = self.font_header.render(target_title, True, target_color)
        self.screen.blit(txt_target, txt_target.get_rect(center=target_box.center))

        # --- Section 3: Dual Belief Comparison (GRU vs Exact Bayes) ---
        y = 104
        lbl_belief = self.font_small.render("DEFENDER SEQUENTIAL BELIEF P(θ | HISTORY)", True, COLOR_TEXT_MUTED)
        self.screen.blit(lbl_belief, (hud_x, y))
        y += 16

        p_a = snapshot["p_a"]
        bar_h = 24
        bar_rect = pygame.Rect(hud_x, y, hud_w, bar_h)
        pygame.draw.rect(self.screen, (15, 15, 20), bar_rect, border_radius=4)
        split_w = int(hud_w * p_a)
        if split_w > 0:
            pygame.draw.rect(self.screen, COLOR_ACCENT_BLUE, (hud_x, y, split_w, bar_h), border_top_left_radius=4, border_bottom_left_radius=4)
        if hud_w - split_w > 0:
            pygame.draw.rect(self.screen, COLOR_ACCENT_RED, (hud_x + split_w, y, hud_w - split_w, bar_h), border_top_right_radius=4, border_bottom_right_radius=4)

        txt_pa = self.font_mono.render(f"GRU P(A): {p_a*100:.1f}%", True, COLOR_TEXT_WHITE)
        txt_pb = self.font_mono.render(f"P(B): {(1.0-p_a)*100:.1f}%", True, COLOR_TEXT_WHITE)
        self.screen.blit(txt_pa, (hud_x + 8, y + 4))
        self.screen.blit(txt_pb, (hud_x + hud_w - txt_pb.get_width() - 8, y + 4))

        # Exact Bayes Teacher Bar
        y += 28
        bayes_a = snapshot.get("bayes_a", 0.5)
        bar_rect2 = pygame.Rect(hud_x, y, hud_w, 20)
        pygame.draw.rect(self.screen, (15, 15, 20), bar_rect2, border_radius=4)
        split_w2 = int(hud_w * bayes_a)
        if split_w2 > 0:
            pygame.draw.rect(self.screen, (40, 100, 200), (hud_x, y, split_w2, 20), border_top_left_radius=4, border_bottom_left_radius=4)
        if hud_w - split_w2 > 0:
            pygame.draw.rect(self.screen, (190, 40, 40), (hud_x + split_w2, y, hud_w - split_w2, 20), border_top_right_radius=4, border_bottom_right_radius=4)

        txt_ba = self.font_mono.render(f"EXACT BAYES: b(A)={bayes_a*100:.1f}%", True, (210, 210, 230))
        txt_kl = self.font_mono.render(f"KL={snapshot.get('kl_div', 0.0):.2f}b", True, COLOR_ACCENT_GOLD)
        self.screen.blit(txt_ba, (hud_x + 8, y + 2))
        self.screen.blit(txt_kl, (hud_x + hud_w - txt_kl.get_width() - 8, y + 2))

        # --- Section 4: Shannon Entropy & Ambiguity Gauge ---
        y += 26
        ent = snapshot["entropy"]
        txt_ent = self.font_header.render(f"SHANNON ENTROPY H(b) = {ent:.3f} bits", True, COLOR_TEXT_WHITE)
        self.screen.blit(txt_ent, (hud_x, y))
        y += 18

        gauge_rect = pygame.Rect(hud_x, y, hud_w, 14)
        pygame.draw.rect(self.screen, (15, 15, 20), gauge_rect, border_radius=3)
        fill_w = int(hud_w * np.clip(ent, 0.0, 1.0))
        gauge_col = COLOR_ACCENT_GREEN if ent > 0.85 else (COLOR_ACCENT_GOLD if ent > 0.40 else COLOR_ACCENT_RED)
        if fill_w > 0:
            pygame.draw.rect(self.screen, gauge_col, (hud_x, y, fill_w, 14), border_radius=3)

        # --- Section 5: Real-time Tactical Telemetry ---
        y += 16
        telemetry_card = pygame.Rect(hud_x, y, hud_w, 82)
        pygame.draw.rect(self.screen, COLOR_HUD_PANEL, telemetry_card, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_HUD_BORDER, telemetry_card, width=1, border_radius=6)

        lines = [
            ("Attacker (Ninja):", f"Row {snapshot['att_r']}, Col {snapshot['att_c']}", COLOR_TEXT_WHITE),
            ("Defender (Guard):", f"Row {snapshot['def_r']}, Col {snapshot['def_c']}", COLOR_TEXT_WHITE),
            ("Player Distance:", f"{snapshot['dist']:.2f} tiles (Swept-Sphere)", COLOR_TEXT_WHITE),
            ("Corridor State:", f"{snapshot['phase_str']}", snapshot['phase_col']),
        ]
        item_y = y + 5
        for label, val, val_col in lines:
            lbl_s = self.font_small.render(label, True, COLOR_TEXT_MUTED)
            val_s = self.font_small.render(val, True, val_col)
            self.screen.blit(lbl_s, (hud_x + 8, item_y))
            self.screen.blit(val_s, (hud_x + hud_w - val_s.get_width() - 8, item_y))
            item_y += 19

        # --- Section 5.2: Dynamic Online RL Co-Evolution Dashboard ---
        y += 88
        rl_card = pygame.Rect(hud_x, y, hud_w, 104)
        pygame.draw.rect(self.screen, (22, 22, 30), rl_card, border_radius=6)
        pygame.draw.rect(self.screen, (90, 75, 120), rl_card, width=1, border_radius=6)

        ep_num = snapshot.get("rl_epoch", self.rl_engine.epoch)
        stage_str = snapshot.get("rl_stage", self.rl_engine.current_stage)
        delta_w = snapshot.get("rl_delta_w", self.rl_engine.weight_delta_mag)
        loss_val = snapshot.get("rl_loss", self.rl_engine.last_loss_val)
        msg_str = snapshot.get("rl_msg", self.rl_engine.last_update_msg)
        auto_learn = snapshot.get("rl_auto", self.rl_engine.learning_enabled)

        # Stage color coding
        if ep_num <= 3:
            stg_col = (255, 90, 90)     # Red: Naive direct rush
        elif ep_num <= 7:
            stg_col = COLOR_ACCENT_GOLD # Gold: Discovering corridor
        elif ep_num <= 14:
            stg_col = COLOR_ACCENT_GREEN# Green: Pooling equilibrium
        else:
            stg_col = COLOR_ACCENT_PURPLE# Purple: Arms race feints

        t_rl = self.font_small.render("★ DYNAMIC RL CO-EVOLUTION DASHBOARD", True, COLOR_ACCENT_GOLD)
        self.screen.blit(t_rl, (hud_x + 8, y + 5))

        learn_status = "[AUTO-TRAIN: ON]" if auto_learn else "[TRAIN: MANUAL]"
        ls_surf = self.font_small.render(learn_status, True, COLOR_ACCENT_GREEN if auto_learn else COLOR_TEXT_MUTED)
        self.screen.blit(ls_surf, (hud_x + hud_w - ls_surf.get_width() - 8, y + 5))

        l1 = self.font_small.render(f"TRAINING EPOCH: #{ep_num:02d}", True, COLOR_TEXT_WHITE)
        self.screen.blit(l1, (hud_x + 8, y + 23))

        l2 = self.font_small.render(stage_str[:38], True, stg_col)
        self.screen.blit(l2, (hud_x + 8, y + 42))

        l3 = self.font_small.render(f"POLICY BACKPROP: Δw = {delta_w:.4f} | Loss = {loss_val:.3f}", True, (0, 229, 255))
        self.screen.blit(l3, (hud_x + 8, y + 62))

        l4 = self.font_small.render(f"LOG: {msg_str[:42]}", True, (200, 205, 220))
        self.screen.blit(l4, (hud_x + 8, y + 82))

        # --- Section 5.5: Interactive AI Engine Switchboard ---
        y += 110
        # 1. Defender Engine Mode Indicator
        def_card = pygame.Rect(hud_x, y, hud_w, 20)
        def_is_rec = (snapshot.get("defender_engine", "RECURRENT GRU") == "RECURRENT GRU")
        def_col = (82, 196, 26) if def_is_rec else (250, 173, 20)
        def_title = "● GUARD: [UPGRADED RECURRENT GRU]" if def_is_rec else "○ GUARD: [OLD MEMORYLESS GUARD]"
        pygame.draw.rect(self.screen, (18, 20, 26), def_card, border_radius=4)
        pygame.draw.rect(self.screen, def_col, def_card, width=1, border_radius=4)
        d_surf = self.font_small.render(def_title, True, def_col)
        self.screen.blit(d_surf, d_surf.get_rect(center=def_card.center))

        # 2. Attacker Mode Indicator
        y += 24
        att_card = pygame.Rect(hud_x, y, hud_w, 20)
        att_mode = snapshot.get("attacker_mode", "★ Dynamic RL Co-Evolution")
        att_col = (0, 229, 255) if "Dynamic" in att_mode else COLOR_ACCENT_BLUE
        pygame.draw.rect(self.screen, (18, 20, 26), att_card, border_radius=4)
        pygame.draw.rect(self.screen, att_col, att_card, width=1, border_radius=4)
        a_surf = self.font_small.render(f"● ATTACKER: [{att_mode[:28]}]", True, att_col)
        self.screen.blit(a_surf, a_surf.get_rect(center=att_card.center))

        # 3. Caught Overlay Indicator
        y += 24
        ov_card = pygame.Rect(hud_x, y, hud_w, 20)
        if self.show_caught_overlay:
            ov_col = COLOR_ACCENT_RED
            ov_text = f"● CAUGHT HAZARD OVERLAY: ACTIVE (GOAL {'A' if self.theta == 0 else 'B'})"
            ov_bg = (44, 16, 20)
        else:
            ov_col = COLOR_TEXT_MUTED
            ov_text = "○ CAUGHT HAZARD OVERLAY: DISABLED (OFF)"
            ov_bg = (20, 22, 28)
        pygame.draw.rect(self.screen, ov_bg, ov_card, border_radius=4)
        pygame.draw.rect(self.screen, ov_col, ov_card, width=1, border_radius=4)
        ov_txt = self.font_small.render(ov_text, True, ov_col)
        self.screen.blit(ov_txt, ov_txt.get_rect(center=ov_card.center))

        # --- Section 6: Interactive Controls Footer ---
        footer_y = 464
        footer_h = 188
        footer_rect = pygame.Rect(hud_x, footer_y, hud_w, footer_h)
        pygame.draw.rect(self.screen, (16, 16, 22), footer_rect, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_HUD_BORDER, footer_rect, width=1, border_radius=6)

        f_title = self.font_small.render("DYNAMIC RL CO-EVOLUTION & CONTROLS", True, COLOR_ACCENT_GOLD)
        self.screen.blit(f_title, (hud_x + (hud_w - f_title.get_width()) // 2, footer_y + 8))

        ctrl_rows = [
            ("[E] Next Epoch  |  [0] Reset to Epoch 1 (Naive)", (0, 229, 255)),
            ("[L] Toggle Auto-Train  |  [T] Switch Adversary", COLOR_ACCENT_GREEN),
            ("[D] Switch Guard Engine  |  [C] Hazard Map", COLOR_ACCENT_RED),
            ("[P] Pause / Resume  |  [←/→] Step History", COLOR_TEXT_WHITE),
            ("[1] Force Goal A  |  [2] Force Goal B  |  [R] Random", (180, 210, 255)),
            ("[SPACE] Rules Lobby  |  [ESC] Exit Lab", COLOR_TEXT_MUTED),
        ]
        row_y = footer_y + 28
        for text, col in ctrl_rows:
            r_surf = self.font_small.render(text, True, col)
            self.screen.blit(r_surf, (hud_x + (hud_w - r_surf.get_width()) // 2, row_y))
            row_y += 24

    def render_post_match_overlay(self, snapshot: dict) -> None:
        """Render frozen post-match victory/defeat banner over the left arena."""
        overlay = pygame.Surface((ARENA_WIDTH, ARENA_HEIGHT), pygame.SRCALPHA)
        overlay.fill((10, 12, 16, 140))
        self.screen.blit(overlay, (0, 0))

        banner_w, banner_h = 520, 165
        bx = (ARENA_WIDTH - banner_w) // 2
        by = (ARENA_HEIGHT - banner_h) // 2
        card_rect = pygame.Rect(bx, by, banner_w, banner_h)

        status_col = snapshot.get("final_status_color", COLOR_ACCENT_GREEN)
        pygame.draw.rect(self.screen, (22, 24, 32), card_rect, border_radius=8)
        pygame.draw.rect(self.screen, status_col, card_rect, width=3, border_radius=8)

        outcome_str = snapshot.get("final_outcome", self.final_outcome)
        title_surf = self.font_title.render(outcome_str, True, status_col)
        self.screen.blit(title_surf, title_surf.get_rect(center=(ARENA_WIDTH // 2, by + 34)))

        target_name = "Goal A (Blue)" if self.theta == 0 else "Goal B (Red)"
        sub_str = f"Target: {target_name}  |  Steps: {snapshot['step']}  |  Final H(b): {snapshot['entropy']:.3f} bits"
        sub_surf = self.font_body.render(sub_str, True, COLOR_TEXT_WHITE)
        self.screen.blit(sub_surf, sub_surf.get_rect(center=(ARENA_WIDTH // 2, by + 76)))

        prompt_str = "[←] Step Back  |  [R] Random  |  [1] Goal A  |  [2] Goal B  |  [T] Next Trick"
        prompt_surf = self.font_small.render(prompt_str, True, COLOR_ACCENT_GOLD)
        self.screen.blit(prompt_surf, prompt_surf.get_rect(center=(ARENA_WIDTH // 2, by + 124)))

    def render_lobby_screen(self) -> None:
        """Render full-screen title, theory rules, and new switchboard controls."""
        self.screen.fill(COLOR_BG_DARK)

        frame_rect = pygame.Rect(40, 24, SCREEN_WIDTH - 80, SCREEN_HEIGHT - 48)
        pygame.draw.rect(self.screen, (26, 28, 38), frame_rect, border_radius=8)
        pygame.draw.rect(self.screen, (60, 65, 85), frame_rect, width=2, border_radius=8)

        title = self.font_title.render("ASYMMETRIC SIGNALING GAME - LIVE LAB", True, COLOR_ACCENT_GOLD)
        self.screen.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, 60)))

        subtitle = self.font_subtitle.render("Sequential Bayesian Receiver vs. Adaptive Arms-Race Attacker", True, (0, 229, 255))
        self.screen.blit(subtitle, subtitle.get_rect(center=(SCREEN_WIDTH // 2, 94)))

        pygame.draw.line(self.screen, (50, 55, 75), (80, 116), (SCREEN_WIDTH - 80, 116), 1)

        rules = [
            ("◆ DYNAMIC RL CO-EVOLUTION (EPOCH-BY-EPOCH LEARNING)", "Starts at Epoch 1 (Naive Direct Rush, easily caught). Online policy updates penalize caught paths and discover Column 5 corridor pooling!", (0, 229, 255)),
            ("◆ RECURRENT BAYESIAN DEFENDER (KNIGHT)", "Shared GRU trunk ingests full sequence history (x_att, y_att, x_def, y_def, t/T_max). Zero false alarms on deceptive feints.", COLOR_ACCENT_GREEN),
            ("◆ EXACT BAYES TEACHER & KL DIAGNOSTIC", "Computes the true recursive mathematical posterior b_t(θ) from π_att in real time. Logs D_KL(GRU || Bayes) on the HUD.", COLOR_ACCENT_GOLD),
            ("◆ SCRIPTED CURRICULUM TRICK ATTACKERS", "Stress-test the Defender against StraightTo7, FeintAndCut, Zigzag, MidPoolDash, and IdleRush. Toggle anytime with [T].", COLOR_ACCENT_PURPLE),
            ("◆ ADAPTIVE ARMS-RACE & CAUGHT HAZARD OVERLAY", "Caught-route archive records failed tiles and penalizes dangerous paths with -μ_e·C_θ(x,y). Toggle red overlay with [C].", COLOR_ACCENT_RED),
        ]

        rules_y = 132
        card_w = SCREEN_WIDTH - 160
        for header, desc, col in rules:
            card_rect = pygame.Rect(80, rules_y, card_w, 52)
            pygame.draw.rect(self.screen, (18, 20, 26), card_rect, border_radius=6)
            pygame.draw.rect(self.screen, (40, 44, 58), card_rect, width=1, border_radius=6)
            h_surf = self.font_header.render(header, True, col)
            d_surf = self.font_body.render(desc, True, COLOR_TEXT_WHITE)
            self.screen.blit(h_surf, (96, rules_y + 6))
            self.screen.blit(d_surf, (96, rules_y + 28))
            rules_y += 60

        legend_rect = pygame.Rect(80, 444, card_w, 48)
        pygame.draw.rect(self.screen, (22, 24, 32), legend_rect, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_ACCENT_GOLD, legend_rect, width=1, border_radius=6)
        legend_text = self.font_small.render(
            "[E] Next Epoch  |  [0] Reset Epoch 1  |  [L] Auto-Train  |  [D] Guard  |  [T] Tricks  |  [C] Hazard Map",
            True, COLOR_ACCENT_GOLD
        )
        self.screen.blit(legend_text, legend_text.get_rect(center=legend_rect.center))

        current_time = pygame.time.get_ticks()
        if (current_time // 500) % 2 == 0:
            prompt = self.font_title.render("▶  PRESS [SPACE] TO ENTER ARENA  ◀", True, COLOR_TEXT_WHITE)
            self.screen.blit(prompt, prompt.get_rect(center=(SCREEN_WIDTH // 2, 545)))

        pygame.display.flip()

    # ========================================================================
    # MAIN APPLICATION LOOP
    # ========================================================================

    def run(self) -> None:
        """Main event loop with interactive hot-toggling for Defender, Attacker, and Playback."""
        print("[LiveDemo] Started live interactive lab.")
        running = True

        while running:
            current_time = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False

                    # Playback: Pause / Resume ([P])
                    elif event.key == pygame.K_p:
                        if self.state in (STATE_PLAYING, STATE_DONE):
                            self.paused = not self.paused

                    # Step Backward ([LEFT ARROW])
                    elif event.key == pygame.K_LEFT:
                        if self.state in (STATE_PLAYING, STATE_DONE):
                            self.paused = True
                            if self.state == STATE_DONE:
                                self.state = STATE_PLAYING
                            self.view_idx = max(0, self.view_idx - 1)

                    # Step Forward ([RIGHT ARROW])
                    elif event.key == pygame.K_RIGHT:
                        if self.state == STATE_PLAYING:
                            self.paused = True
                            if self.view_idx < len(self.history) - 1:
                                self.view_idx += 1
                                if self.history[self.view_idx]["done"]:
                                    self.state = STATE_DONE
                            else:
                                if not self.history[-1]["done"]:
                                    self.advance_environment_step()
                                    self.last_step_time = current_time

                    # Toggle Defender Engine ([D])
                    elif event.key == pygame.K_d:
                        self.use_recurrent_defender = not self.use_recurrent_defender
                        eng_name = "UPGRADED RECURRENT GRU" if self.use_recurrent_defender else "OLD MEMORYLESS GUARD"
                        self.toast_msg = f"GUARD ENGINE: {eng_name}"
                        self.toast_time = current_time
                        if self.history and self.view_idx < len(self.history):
                            self.history[self.view_idx]["defender_engine"] = "RECURRENT GRU" if self.use_recurrent_defender else "MEMORYLESS MLP"
                        print(f"[LiveDemo] Switched Defender Engine -> {eng_name}")

                    # Advance Training Epoch ([E])
                    elif event.key == pygame.K_e:
                        self.rl_engine.advance_epoch()
                        self.arms_phase = min(3, (self.rl_engine.epoch // 5) + 1)
                        self.arms_mu_e = self.rl_engine.arms_mu_e
                        self.toast_msg = f"ADVANCED TO {self.rl_engine.current_stage.upper()}"
                        self.toast_time = current_time
                        if self.history and self.view_idx < len(self.history):
                            self.history[self.view_idx]["rl_epoch"] = self.rl_engine.epoch
                            self.history[self.view_idx]["rl_stage"] = self.rl_engine.current_stage
                            self.history[self.view_idx]["rl_msg"] = self.rl_engine.last_update_msg
                        print(f"[LiveDemo] Advanced Training Epoch -> {self.rl_engine.current_stage}")
                        self.reset_episode(force_theta=self.theta)

                    # Reset Learning to Epoch 1 ([0])
                    elif event.key == pygame.K_0:
                        self.rl_engine.reset_to_naive_stage()
                        if self.caught_map is not None:
                            self.caught_map.fill(0.0)
                        self.arms_phase = 1
                        self.arms_mu_e = 0.0
                        self.toast_msg = "RESET TO EPOCH 1: NAIVE DIRECT GREEDY RUSH (UNTRAINED)"
                        self.toast_time = current_time
                        if self.history and self.view_idx < len(self.history):
                            self.history[self.view_idx]["rl_epoch"] = 1
                            self.history[self.view_idx]["rl_stage"] = self.rl_engine.current_stage
                            self.history[self.view_idx]["rl_msg"] = self.rl_engine.last_update_msg
                        print("[LiveDemo] Reset RL Engine to Epoch 1 Naive Stage.")
                        self.reset_episode(force_theta=self.theta)

                    # Toggle Live Auto-Learning ([L])
                    elif event.key == pygame.K_l:
                        self.rl_engine.learning_enabled = not self.rl_engine.learning_enabled
                        st_name = "ACTIVE (AUTO-LEARN ON OUTCOME)" if self.rl_engine.learning_enabled else "OFF (FROZEN EVALUATION)"
                        self.toast_msg = f"LIVE RL ADAPTATION: {st_name}"
                        self.toast_time = current_time
                        if self.history and self.view_idx < len(self.history):
                            self.history[self.view_idx]["rl_auto"] = self.rl_engine.learning_enabled
                        print(f"[LiveDemo] Live RL Adaptation: {st_name}")

                    # Cycle Attacker Adversary Mode ([T])
                    elif event.key == pygame.K_t:
                        self.current_trick_idx = (self.current_trick_idx + 1) % len(self.trick_adversaries)
                        trick_name = self.trick_adversaries[self.current_trick_idx][0]
                        self.toast_msg = f"ATTACKER: {trick_name}"
                        self.toast_time = current_time
                        print(f"[LiveDemo] Switched Attacker Mode -> {trick_name}")
                        self.reset_episode(force_theta=self.theta)

                    # Caught-Routes Overlay Toggle ([C])
                    elif event.key == pygame.K_c:
                        self.show_caught_overlay = not self.show_caught_overlay
                        num_tiles = int(np.count_nonzero(self.caught_map[self.theta] > 0.05)) if self.caught_map is not None else 0
                        status_str = f"ON ({num_tiles} HAZARD TILES)" if self.show_caught_overlay else "OFF"
                        self.toast_msg = f"CAUGHT OVERLAY: {status_str}"
                        self.toast_time = current_time
                        print(f"[LiveDemo] Caught routes overlay: {status_str}")

                    # Screen toggle ([SPACE])
                    elif event.key == pygame.K_SPACE:
                        if self.state == STATE_LOBBY:
                            self.reset_episode()
                        elif self.state in (STATE_PLAYING, STATE_DONE):
                            self.paused = False
                            self.state = STATE_LOBBY

                    # Episode resets / force target shortcuts
                    elif event.key == pygame.K_r:
                        self.reset_episode(force_theta=None)
                    elif event.key == pygame.K_1:
                        self.reset_episode(force_theta=0)
                    elif event.key == pygame.K_2:
                        self.reset_episode(force_theta=1)

            # State Logic & Rendering
            if self.state == STATE_LOBBY:
                self.render_lobby_screen()

            elif self.state == STATE_PLAYING:
                if not self.paused:
                    if current_time - self.last_step_time >= STEP_DELAY_MS:
                        if self.view_idx < len(self.history) - 1:
                            self.view_idx += 1
                            self.last_step_time = current_time
                            if self.history[self.view_idx]["done"]:
                                self.state = STATE_DONE
                        else:
                            self.advance_environment_step()
                            self.last_step_time = current_time

                snapshot = self.history[self.view_idx]
                self.render_arena(snapshot)
                self.render_hud(snapshot)
                pygame.display.flip()

            elif self.state == STATE_DONE:
                snapshot = self.history[self.view_idx]
                self.render_arena(snapshot)
                self.render_hud(snapshot)
                self.render_post_match_overlay(snapshot)
                pygame.display.flip()

            self.clock.tick(FPS)

        pygame.quit()
        print("[LiveDemo] Exited cleanly.")


if __name__ == "__main__":
    demo = MinecraftLiveDemo()
    demo.run()
