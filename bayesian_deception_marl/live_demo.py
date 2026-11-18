"""Minecraft-Style Pixel-Art Live Demo for Bayesian Deception MARL.

Presentation UI, Live HUD, and Step Playback Controls
------------------------------------------------------
- Window: 1040 x 660 px
    * Left Panel (0..660 px): 11x11 Minecraft arena, cobblestone corridor,
      procedural pixel-art sprites, and glowing Y-shaped trail.
    * Right Panel (660..1040 px): Dark-themed Live HUD (#1E1E24) with real-time
      Bayesian belief split bar, Shannon entropy gauge, telemetry, and status badges.
- 3-Screen State Machine:
    * STATE_LOBBY: Full-window Minecraft-style intro & rules screen.
    * STATE_PLAYING: Live 0.4s/step simulation with real-time HUD updates.
    * STATE_DONE: Frozen final frame with post-match victory/interception banner.
- Playback & Presenter Controls:
    * [P]: Pause / Resume simulation auto-advance.
    * [LEFT ARROW]: Step backward frame-by-frame through episode history.
    * [RIGHT ARROW]: Step forward frame-by-frame (or advance environment if at head).
    * [R]: Start episode with uniform random target (theta ~ Uniform({0, 1})).
    * [1]: Force episode with Goal A (Blue).
    * [2]: Force episode with Goal B (Red).
    * [SPACE]: Enter Arena / Toggle Rules Screen.
    * [ESC] / [Q]: Quit.
"""

import os
import sys
import pygame
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

# Ensure repository root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import EnvConfig
from envs.grid_world import DeceptionGridWorld as AsymmetricGridEnv
from models.networks import get_attacker_network, get_defender_network
from models.belief_head import BeliefHead

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================
GRID_SIZE = 11
TILE_SIZE = 60
ARENA_WIDTH = GRID_SIZE * TILE_SIZE     # 660 px
ARENA_HEIGHT = GRID_SIZE * TILE_SIZE    # 660 px
HUD_WIDTH = 380                         # 380 px
SCREEN_WIDTH = ARENA_WIDTH + HUD_WIDTH  # 1040 px
SCREEN_HEIGHT = ARENA_HEIGHT            # 660 px

FPS = 60
STEP_DELAY_MS = 400                     # 0.4 seconds per environment step

# Colors (Dark Minecraft Theme)
COLOR_BG_DARK = (20, 20, 26)           # #14141A
COLOR_HUD_BG = (30, 30, 36)            # #1E1E24
COLOR_HUD_PANEL = (22, 22, 28)         # #16161C
COLOR_HUD_BORDER = (55, 55, 68)        # #373744
COLOR_ACCENT_BLUE = (24, 144, 255)     # #1890FF Goal A
COLOR_ACCENT_RED = (245, 34, 45)       # #F5222D Goal B
COLOR_ACCENT_GOLD = (250, 173, 20)     # #FAAD14 Gold
COLOR_ACCENT_GREEN = (82, 196, 26)     # #52C41A High Entropy / Victory
COLOR_TEXT_WHITE = (240, 240, 245)
COLOR_TEXT_MUTED = (160, 160, 175)

# States
STATE_LOBBY = "LOBBY"
STATE_PLAYING = "PLAYING"
STATE_DONE = "DONE"


# ============================================================================
# PROCEDURAL PIXEL-ART MATRICES & PALETTES (16x16)
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
    # Nearest-neighbor scale preserves sharp, blocky Minecraft-style pixels
    return pygame.transform.scale(raw_surf, (target_size, target_size))


# --- 1. Attacker Sprite: Pixel Ninja / Rogue ---
NINJA_PALETTE = {
    ".": (0, 0, 0, 0),
    "K": (22, 22, 28),         # Dark hood edge / shadow
    "D": (38, 40, 52),         # Dark ninja tunic
    "M": (56, 60, 78),         # Hood mid-tone
    "L": (82, 88, 114),        # Hood highlight
    "R": (180, 30, 30),        # Red Headband
    "r": (230, 60, 60),        # Red Headband highlight
    "S": (225, 185, 150),      # Face skin tone
    "E": (0, 255, 230),        # Glowing Cyan Eyes
    "W": (255, 255, 255),      # Eye glint
    "B": (26, 26, 34),         # Belt / pants
    "G": (180, 190, 205),      # Steel buckle / shuriken
}

NINJA_SPRITE_MATRIX = [
    "....KKMMMMKK....",
    "...KMMLLLLMMK...",
    "..KMMLLLLLLMMK..",
    "..KMRrrrrrrRMK..",
    "..KMRRRRRRRRMK..",
    "..KMMSSSSSSMMK..",
    "..KMSEWSSWESMK..",
    "..KMMSSSSSSMMK..",
    "..KMMMMDDMMMMK..",
    "...KMDDDDMDDK...",
    "...KDDDGGDDDK...",
    "...KDDDBBDDDK...",
    "...KMDDBBDMMK...",
    "...KMDD..DDMK...",
    "...KMMK..KMMK...",
    "....KK....KK....",
]

# --- 2. Defender Sprite: Pixel Knight / Guard ---
KNIGHT_PALETTE = {
    ".": (0, 0, 0, 0),
    "H": (35, 40, 48),         # Iron outline
    "A": (130, 140, 155),      # Steel armor mid
    "B": (185, 195, 210),      # Steel armor highlight
    "D": (80, 88, 100),        # Steel armor shadow
    "P": (220, 40, 40),        # Plume Red
    "p": (255, 95, 95),        # Plume highlight
    "V": (15, 18, 24),         # Visor slit black
    "E": (255, 50, 40),        # Glowing red visor gaze
    "G": (245, 190, 35),       # Gold crest
    "g": (255, 230, 110),      # Gold highlight
    "S": (100, 110, 125),      # Shoulder plate
}

KNIGHT_SPRITE_MATRIX = [
    ".....HHppHH.....",
    "....HPPPppPH....",
    "...HPPPHHPPH....",
    "...HAAABBAAH....",
    "..HAAABBBBBAH...",
    "..HAABGggGBAH...",
    "..HAAAABBAAAH...",
    "..HDDVVEEVVDH...",
    "..HDDVEEEEVDH...",
    "..HDAAAAAAAD....",
    ".HSAABBGBAASH...",
    ".HSAADGgGDAASH..",
    ".HSAADGGGDDAAH..",
    "..HDAADDDDAADH..",
    "..HAAH....HAAH..",
    "...HH......HH...",
]

# --- 3. Goal A Sprite: Blue Diamond Treasure / Gem (Top-Left) ---
GOAL_A_PALETTE = {
    ".": (0, 0, 0, 0),
    "O": (12, 45, 90),         # Deep navy outline
    "C": (0, 160, 235),        # Diamond cyan
    "L": (70, 215, 255),       # Light cyan
    "W": (230, 250, 255),      # Sparkling white
    "D": (0, 100, 175),        # Diamond shadow
    "G": (245, 200, 40),       # Chest gold trim
    "B": (120, 75, 35),        # Chest wood
}

GOAL_A_SPRITE_MATRIX = [
    "......OOOO......",
    "....OOLLLLOO....",
    "...OLLWWLLLLO...",
    "..OLWWLLLLLLDO..",
    ".OLWLLLLLLLLDDO.",
    ".OLLLLCCDDDDDDO.",
    ".OLLCCDDDDDDDDO.",
    "..OCCDDDDDDDOO..",
    "..OGGGGGGGGGGO..",
    ".OGBBBBGGGBBBGO.",
    ".OGBWWGGGGWWGBGO",
    ".OGBBBBGGGBBBGO.",
    ".OGGGGGGGGGGGGO.",
    ".OGBBBBBBBBBBGO.",
    "..OGGGGGGGGGGO..",
    "...OOOOOOOOOO...",
]

# --- 4. Goal B Sprite: Red Ruby Treasure / Gem (Top-Right) ---
GOAL_B_PALETTE = {
    ".": (0, 0, 0, 0),
    "O": (70, 12, 22),         # Deep ruby outline
    "R": (225, 30, 60),        # Ruby red
    "L": (255, 95, 125),       # Light ruby
    "W": (255, 240, 245),      # Sparkling white
    "D": (150, 15, 40),        # Ruby shadow
    "G": (245, 200, 40),       # Chest gold trim
    "B": (120, 75, 35),        # Chest wood
}

GOAL_B_SPRITE_MATRIX = [
    "......OOOO......",
    "....OOLLLLOO....",
    "...OLLWWLLLLO...",
    "..OLWWLLLLLLDO..",
    ".OLWLLLLLLLLDDO.",
    ".OLLLLRRDDDDDDO.",
    ".OLLRRDDDDDDDDO.",
    "..ORRDDDDDDDOO..",
    "..OGGGGGGGGGGO..",
    ".OGBBBBGGGBBBGO.",
    ".OGBWWGGGGWWGBGO",
    ".OGBBBBGGGBBBGO.",
    ".OGGGGGGGGGGGGO.",
    ".OGBBBBBBBBBBGO.",
    "..OGGGGGGGGGGO..",
    "...OOOOOOOOOO...",
]

# --- 5. Trail Footprint / Glowing Rune ---
TRAIL_PALETTE = {
    ".": (0, 0, 0, 0),
    "Y": (255, 215, 60, 180),  # Glowing amber/gold rune
    "W": (255, 255, 240, 230), # Core bright glint
    "D": (200, 140, 30, 140),  # Soft edge glow
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
    """Create a textured 60x60 Minecraft grass block tile."""
    surf = pygame.Surface((TILE_SIZE, TILE_SIZE))
    if variant == 0:
        base_color = (78, 128, 49)      # #4e8031
        dark_speck = (62, 106, 37)      # #3e6a25
        light_speck = (94, 149, 60)     # #5e953c
        border_color = (55, 95, 32)
    else:
        base_color = (69, 117, 42)      # #45752a
        dark_speck = (55, 94, 32)       # #375e20
        light_speck = (84, 136, 52)     # #548834
        border_color = (48, 84, 27)

    surf.fill(base_color)

    rng = np.random.RandomState(variant * 100 + 42)
    for _ in range(18):
        x = int(rng.randint(2, TILE_SIZE - 4))
        y = int(rng.randint(2, TILE_SIZE - 4))
        w = int(rng.choice([2, 3]))
        pygame.draw.rect(surf, dark_speck, (x, y, w, w))

    for _ in range(14):
        x = int(rng.randint(2, TILE_SIZE - 4))
        y = int(rng.randint(2, TILE_SIZE - 4))
        w = int(rng.choice([2, 3]))
        pygame.draw.rect(surf, light_speck, (x, y, w, w))

    pygame.draw.rect(surf, border_color, (0, 0, TILE_SIZE, TILE_SIZE), 1)
    return surf


def create_cobblestone_tile() -> pygame.Surface:
    """Create a textured 60x60 Minecraft cobblestone path tile for Column 5."""
    surf = pygame.Surface((TILE_SIZE, TILE_SIZE))
    base_color = (115, 117, 120)       # #737578
    mortar_color = (76, 78, 81)        # #4c4e51 dark mortar
    highlight_color = (146, 148, 151)  # #929497 stone highlight
    deep_shadow = (56, 57, 59)         # #38393b

    surf.fill(base_color)

    row_height = TILE_SIZE // 3  # 20px per brick layer
    for r in range(3):
        y = r * row_height
        pygame.draw.line(surf, mortar_color, (0, y), (TILE_SIZE, y), 2)
        pygame.draw.line(surf, highlight_color, (0, y + 2), (TILE_SIZE, y + 2), 1)

        offset = 0 if r % 2 == 0 else TILE_SIZE // 2
        for col_idx in [offset, offset + TILE_SIZE // 2]:
            x = col_idx % TILE_SIZE
            pygame.draw.line(surf, mortar_color, (x, y), (x, y + row_height), 2)
            pygame.draw.line(surf, highlight_color, (x + 2, y), (x + 2, y + row_height), 1)

    rng = np.random.RandomState(555)
    for _ in range(25):
        px = int(rng.randint(2, TILE_SIZE - 2))
        py = int(rng.randint(2, TILE_SIZE - 2))
        c = highlight_color if rng.rand() > 0.5 else deep_shadow
        surf.set_at((px, py), c)

    pygame.draw.rect(surf, deep_shadow, (0, 0, TILE_SIZE, TILE_SIZE), 1)
    return surf


# ============================================================================
# LIVE DEMO APPLICATION CLASS (WITH STEP PLAYBACK & STATE HISTORY)
# ============================================================================

class MinecraftLiveDemo:
    """PyGame interactive presentation visualizer for Bayesian Deception MARL."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Bayesian Deception MARL - Minecraft Arena")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # Build procedural sprites and tiles
        print("[LiveDemo] Generating procedural pixel-art assets...")
        sprite_size = 46  # 46x46 fits centered inside 60x60 tile (7px margin)
        self.ninja_sprite = make_surface_from_matrix(NINJA_SPRITE_MATRIX, NINJA_PALETTE, target_size=sprite_size)
        self.knight_sprite = make_surface_from_matrix(KNIGHT_SPRITE_MATRIX, KNIGHT_PALETTE, target_size=sprite_size)
        self.goal_a_sprite = make_surface_from_matrix(GOAL_A_SPRITE_MATRIX, GOAL_A_PALETTE, target_size=sprite_size)
        self.goal_b_sprite = make_surface_from_matrix(GOAL_B_SPRITE_MATRIX, GOAL_B_PALETTE, target_size=sprite_size)
        self.trail_sprite = make_surface_from_matrix(TRAIL_SPRITE_MATRIX, TRAIL_PALETTE, target_size=28)

        # Pre-render tile surfaces
        self.grass_tile_a = create_grass_tile(variant=0)
        self.grass_tile_b = create_grass_tile(variant=1)
        self.cobble_tile = create_cobblestone_tile()

        # Fonts
        self.font_title = pygame.font.SysFont("consolas", 28, bold=True)
        self.font_subtitle = pygame.font.SysFont("consolas", 17, bold=True)
        self.font_header = pygame.font.SysFont("consolas", 15, bold=True)
        self.font_body = pygame.font.SysFont("consolas", 13, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 11, bold=True)
        self.font_large_stat = pygame.font.SysFont("consolas", 22, bold=True)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LiveDemo] Inference device: {self.device}")

        # Initialize Environment
        self.env = AsymmetricGridEnv(EnvConfig())

        # Load models
        print("[LiveDemo] Loading trained neural networks...")
        self.attacker_net = get_attacker_network(hidden_dim=64, action_dim=9).to(self.device)
        self.defender_net = get_defender_network(hidden_dim=64, action_dim=9).to(self.device)
        self.belief_head = BeliefHead(input_dim=4, hidden_dim=64).to(self.device)

        models_dir = os.path.join(CURRENT_DIR, "models")
        att_path = os.path.join(models_dir, "attacker_net.pt")
        def_path = os.path.join(models_dir, "defender_net.pt")
        bh_path = os.path.join(models_dir, "belief_head.pt")

        self.attacker_net.load_state_dict(torch.load(att_path, map_location=self.device))
        self.defender_net.load_state_dict(torch.load(def_path, map_location=self.device))
        self.belief_head.load_state_dict(torch.load(bh_path, map_location=self.device))

        self.attacker_net.eval()
        self.defender_net.eval()
        self.belief_head.eval()
        print("[LiveDemo] Models loaded successfully in eval() mode.")

        # Application state machine: STATE_LOBBY, STATE_PLAYING, STATE_DONE
        self.state = STATE_LOBBY
        self.episode_count = 0
        self.last_step_time = 0

        # Step Playback Controls & History Buffer
        self.paused = False
        self.history: list[dict] = []
        self.view_idx: int = 0

        # Current episode metrics & observation buffers
        self.theta = 0
        self.att_obs = None
        self.def_obs = None
        self.def_hidden = None
        self.visited_tiles: list[tuple[int, int]] = []
        self.step_counter = 0

        # Live Bayesian beliefs & entropy
        self.current_probs = np.array([0.5, 0.5], dtype=np.float32)
        self.current_entropy = 1.0000

        # Final match summary record for STATE_DONE
        self.final_outcome = ""
        self.final_status_color = COLOR_ACCENT_GREEN
        self.final_is_goal = False

    def grid_to_screen(self, row: float, col: float) -> tuple[int, int]:
        """Convert environment (row, col) coordinates to screen (x, y) pixels."""
        pixel_x = int(round(col * TILE_SIZE))
        pixel_y = int(round((10 - row) * TILE_SIZE))
        return pixel_x, pixel_y

    def _create_snapshot(self, step: int, done: bool = False, outcome_text: str = "") -> dict:
        """Capture a complete lightweight snapshot of the current state for playback."""
        att_pos = self.env.attacker_pos.copy()
        def_pos = self.env.defender_pos.copy()
        att_r = int(round(att_pos[0]))
        att_c = int(round(att_pos[1]))
        def_r = int(round(def_pos[0]))
        def_c = int(round(def_pos[1]))
        dist = float(np.linalg.norm(att_pos - def_pos))

        phase_str = "Pooling Corridor (Col 5)" if (att_r <= 7 and att_c == 5) else "Diagonal Bifurcation Phase"
        phase_col = COLOR_ACCENT_GREEN if (att_r <= 7 and att_c == 5) else COLOR_ACCENT_GOLD

        return {
            "attacker_pos": (float(att_pos[0]), float(att_pos[1])),
            "defender_pos": (float(def_pos[0]), float(def_pos[1])),
            "visited_tiles": list(self.visited_tiles),
            "step": step,
            "p_a": float(np.clip(self.current_probs[0], 0.0, 1.0)),
            "p_b": float(np.clip(self.current_probs[1], 0.0, 1.0)),
            "entropy": float(self.current_entropy),
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
        }

    def reset_episode(self, force_theta: Optional[int] = None) -> None:
        """Reset the match. If force_theta is None, sample uniformly from {0, 1}."""
        if force_theta is not None:
            desired_theta = int(force_theta)
        else:
            desired_theta = int(np.random.choice([0, 1]))

        obs_dict, info = self.env.reset(options={"theta": desired_theta})

        self.theta = info["theta"]
        self.att_obs = obs_dict["attacker"]
        self.def_obs = obs_dict["defender"]
        self.def_hidden = None

        self.visited_tiles = [(int(self.env.attacker_pos[0]), int(self.env.attacker_pos[1]))]
        self.step_counter = 0
        self.current_probs = np.array([0.5, 0.5], dtype=np.float32)
        self.current_entropy = 1.0000
        self.final_outcome = ""
        self.final_is_goal = False

        # Reset playback state
        self.paused = False
        self.history = [self._create_snapshot(step=0, done=False, outcome_text=f"Target: Goal {'A (Blue)' if self.theta == 0 else 'B (Red)'}")]
        self.view_idx = 0

        self.state = STATE_PLAYING
        self.last_step_time = pygame.time.get_ticks()

        target_name = "Goal A (Blue)" if self.theta == 0 else "Goal B (Red)"
        print(f"\n[Episode {self.episode_count + 1}] Started | Secret Target: {target_name}")

    def advance_environment_step(self) -> None:
        """Advance one greedy environment step and record to history."""
        # 1. Update Defender Belief Head
        def_obs_t = torch.as_tensor(self.def_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            belief_logits, self.def_hidden = self.belief_head.step(def_obs_t, self.def_hidden)
            probs = F.softmax(belief_logits, dim=-1)
            self.current_probs = probs.squeeze(0).cpu().numpy()

        # 2. Shannon Entropy: H(b) = - sum(p * log2(p + 1e-8))
        self.current_entropy = -float(np.sum(self.current_probs * np.log2(self.current_probs + 1e-8)))

        # 3. Construct Defender Actor-Critic input (dim=6)
        def_obs_full = np.concatenate([self.def_obs, self.current_probs], axis=-1).astype(np.float32)

        # 4. Greedy Action Selection (argmax over logits)
        att_obs_t = torch.as_tensor(self.att_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        def_obs_full_t = torch.as_tensor(def_obs_full, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            att_logits, _ = self.attacker_net(att_obs_t)
            def_logits, _ = self.defender_net(def_obs_full_t)

        greedy_att_action = int(torch.argmax(att_logits, dim=-1).item())
        greedy_def_action = int(torch.argmax(def_logits, dim=-1).item())

        # 5. Advance Environment
        next_obs_dict, _, terminated, truncated, step_info = self.env.step(
            greedy_att_action, greedy_def_action
        )
        self.step_counter += 1

        # Track trajectory tiles
        cur_pos = (int(round(self.env.attacker_pos[0])), int(round(self.env.attacker_pos[1])))
        if cur_pos not in self.visited_tiles:
            self.visited_tiles.append(cur_pos)

        # Check termination
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

            outcome = self.final_outcome
            print(f"  -> Episode Finished at Step {self.step_counter}: {self.final_outcome}")

        # Record snapshot to history and point view_idx to head
        snapshot = self._create_snapshot(step=self.step_counter, done=is_done, outcome_text=outcome)
        self.history.append(snapshot)
        self.view_idx = len(self.history) - 1

        if not is_done:
            self.att_obs = next_obs_dict["attacker"]
            self.def_obs = next_obs_dict["defender"]

    # ========================================================================
    # RENDERING METHODS (DRAWN FROM SNAPSHOT)
    # ========================================================================

    def render_arena(self, snapshot: dict) -> None:
        """Render the 660x660 Minecraft arena from the given history snapshot."""
        # 1. Base Grid Tiles
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                px, py = self.grid_to_screen(r, c)
                if c == 5:
                    self.screen.blit(self.cobble_tile, (px, py))
                else:
                    tile = self.grass_tile_a if (r + c) % 2 == 0 else self.grass_tile_b
                    self.screen.blit(tile, (px, py))

        # 2. Visited Footprint Trail from snapshot
        trail_offset = (TILE_SIZE - self.trail_sprite.get_width()) // 2
        for r, c in snapshot["visited_tiles"]:
            tx, ty = self.grid_to_screen(r, c)
            self.screen.blit(self.trail_sprite, (tx + trail_offset, ty + trail_offset))

        # 3. Goal Sprites
        sprite_offset = (TILE_SIZE - self.ninja_sprite.get_width()) // 2
        ga_x, ga_y = self.grid_to_screen(10, 0)
        gb_x, gb_y = self.grid_to_screen(10, 10)
        self.screen.blit(self.goal_a_sprite, (ga_x + sprite_offset, ga_y + sprite_offset))
        self.screen.blit(self.goal_b_sprite, (gb_x + sprite_offset, gb_y + sprite_offset))

        # 4. Defender (Knight) from snapshot
        def_r, def_c = snapshot["defender_pos"]
        dx, dy = self.grid_to_screen(def_r, def_c)
        self.screen.blit(self.knight_sprite, (dx + sprite_offset, dy + sprite_offset))

        # 5. Attacker (Ninja) from snapshot
        att_r, att_c = snapshot["attacker_pos"]
        ax, ay = self.grid_to_screen(att_r, att_c)
        self.screen.blit(self.ninja_sprite, (ax + sprite_offset, ay + sprite_offset))

        # 6. Small non-intrusive PAUSED Badge at Top Center (Row 10 Col 5)
        if self.paused:
            total_steps = self.history[-1]["step"]
            cur_step = snapshot["step"]
            if len(self.history) > 1 and cur_step < total_steps:
                pause_label = f"\u23f8 PAUSED [Step {cur_step:02d}/{total_steps:02d}]"
            else:
                pause_label = "\u23f8 PAUSED"

            pause_surf = self.font_body.render(pause_label, True, COLOR_ACCENT_GOLD)
            p_rect = pause_surf.get_rect(center=(ARENA_WIDTH // 2, 22))
            bg_rect = p_rect.inflate(20, 8)
            pygame.draw.rect(self.screen, (18, 20, 26, 235), bg_rect, border_radius=4)
            pygame.draw.rect(self.screen, COLOR_ACCENT_GOLD, bg_rect, width=1, border_radius=4)
            self.screen.blit(pause_surf, p_rect)

    def render_hud(self, snapshot: dict) -> None:
        """Render the 380x660 presentation HUD from the given history snapshot."""
        hud_rect = pygame.Rect(ARENA_WIDTH, 0, HUD_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_HUD_BG, hud_rect)
        pygame.draw.line(self.screen, (40, 42, 52), (ARENA_WIDTH, 0), (ARENA_WIDTH, SCREEN_HEIGHT), 2)

        hud_x = ARENA_WIDTH + 20
        hud_w = HUD_WIDTH - 40

        # --- Section 1: Episode & Step Header ---
        y = 20
        card_rect = pygame.Rect(hud_x, y, hud_w, 48)
        pygame.draw.rect(self.screen, COLOR_HUD_PANEL, card_rect, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_HUD_BORDER, card_rect, width=1, border_radius=6)

        ep_text = self.font_header.render(f"EPISODE #{self.episode_count + 1:02d}", True, COLOR_TEXT_WHITE)
        step_text = self.font_header.render(f"STEP: {snapshot['step']:02d} / 25", True, COLOR_ACCENT_GOLD)
        self.screen.blit(ep_text, (hud_x + 14, y + 14))
        self.screen.blit(step_text, (hud_x + hud_w - step_text.get_width() - 14, y + 14))

        # --- Section 2: Secret Target Card ---
        y = 80
        lbl_target = self.font_small.render("ATTACKER SECRET OBJECTIVE (PRIVATE TYPE \u03b8)", True, COLOR_TEXT_MUTED)
        self.screen.blit(lbl_target, (hud_x, y))
        y += 18

        target_box = pygame.Rect(hud_x, y, hud_w, 42)
        if self.theta == 0:
            target_color = COLOR_ACCENT_BLUE
            target_title = "GOAL A (BLUE) - TOP-LEFT"
            bg_card = (16, 32, 54)
        else:
            target_color = COLOR_ACCENT_RED
            target_title = "GOAL B (RED) - TOP-RIGHT"
            bg_card = (54, 18, 22)

        pygame.draw.rect(self.screen, bg_card, target_box, border_radius=6)
        pygame.draw.rect(self.screen, target_color, target_box, width=2, border_radius=6)
        txt_target = self.font_header.render(target_title, True, target_color)
        txt_rect = txt_target.get_rect(center=target_box.center)
        self.screen.blit(txt_target, txt_rect)

        # --- Section 3: Defender Belief Bar ---
        y = 152
        lbl_belief = self.font_small.render("DEFENDER BELIEF DISTRIBUTION P(\u03b8)", True, COLOR_TEXT_MUTED)
        self.screen.blit(lbl_belief, (hud_x, y))
        y += 18

        p_a = snapshot["p_a"]
        p_b = snapshot["p_b"]

        bar_h = 32
        bar_rect = pygame.Rect(hud_x, y, hud_w, bar_h)
        pygame.draw.rect(self.screen, (15, 15, 20), bar_rect, border_radius=4)

        # Split progress bar
        split_w = int(hud_w * p_a)
        if split_w > 0:
            pygame.draw.rect(self.screen, COLOR_ACCENT_BLUE, (hud_x, y, split_w, bar_h), border_top_left_radius=4, border_bottom_left_radius=4)
        if hud_w - split_w > 0:
            pygame.draw.rect(self.screen, COLOR_ACCENT_RED, (hud_x + split_w, y, hud_w - split_w, bar_h), border_top_right_radius=4, border_bottom_right_radius=4)

        # 50/50 Center indicator line
        pygame.draw.line(self.screen, (255, 255, 255), (hud_x + hud_w // 2, y - 2), (hud_x + hud_w // 2, y + bar_h + 2), 2)
        pygame.draw.rect(self.screen, COLOR_HUD_BORDER, bar_rect, width=1, border_radius=4)

        # Numerical percentage overlays
        pct_a_surf = self.font_body.render(f"P(A): {p_a * 100:.1f}%", True, COLOR_TEXT_WHITE)
        pct_b_surf = self.font_body.render(f"P(B): {p_b * 100:.1f}%", True, COLOR_TEXT_WHITE)
        self.screen.blit(pct_a_surf, (hud_x + 8, y + 8))
        self.screen.blit(pct_b_surf, (hud_x + hud_w - pct_b_surf.get_width() - 8, y + 8))

        # --- Section 4: Shannon Entropy Gauge ---
        y = 216
        lbl_ent = self.font_small.render("DEFENDER SHANNON ENTROPY H(b)", True, COLOR_TEXT_MUTED)
        self.screen.blit(lbl_ent, (hud_x, y))
        y += 18

        ent_card = pygame.Rect(hud_x, y, hud_w, 98)
        pygame.draw.rect(self.screen, COLOR_HUD_PANEL, ent_card, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_HUD_BORDER, ent_card, width=1, border_radius=6)

        ent_val = snapshot["entropy"]
        ent_val_text = self.font_large_stat.render(f"H(b) = {ent_val:.4f} bits", True, COLOR_TEXT_WHITE)
        self.screen.blit(ent_val_text, (hud_x + 14, y + 10))

        # Horizontal entropy meter bar (0 to 1 bit)
        meter_y = y + 42
        meter_w = hud_w - 28
        meter_rect = pygame.Rect(hud_x + 14, meter_y, meter_w, 14)
        pygame.draw.rect(self.screen, (15, 15, 20), meter_rect, border_radius=3)

        fill_w = int(meter_w * min(1.0, max(0.0, ent_val)))
        fill_color = COLOR_ACCENT_GREEN if ent_val >= 0.85 else COLOR_ACCENT_GOLD
        if fill_w > 0:
            pygame.draw.rect(self.screen, fill_color, (hud_x + 14, meter_y, fill_w, 14), border_radius=3)

        # 0.85 Bifurcation Threshold tick mark
        thresh_x = hud_x + 14 + int(meter_w * 0.85)
        pygame.draw.line(self.screen, (255, 255, 255), (thresh_x, meter_y - 2), (thresh_x, meter_y + 16), 2)
        pygame.draw.rect(self.screen, COLOR_HUD_BORDER, meter_rect, width=1, border_radius=3)

        # Status badge underneath
        badge_y = y + 64
        badge_rect = pygame.Rect(hud_x + 14, badge_y, meter_w, 24)
        if ent_val >= 0.85:
            badge_bg = (18, 48, 22)
            badge_border = COLOR_ACCENT_GREEN
            badge_msg = "\u25cf POOLING (MAXIMUM AMBIGUITY)"
        else:
            badge_bg = (56, 30, 10)
            badge_border = COLOR_ACCENT_GOLD
            badge_msg = "\u25cf REVEALED (SEPARATING BIFURCATION)"

        pygame.draw.rect(self.screen, badge_bg, badge_rect, border_radius=4)
        pygame.draw.rect(self.screen, badge_border, badge_rect, width=1, border_radius=4)
        badge_text = self.font_small.render(badge_msg, True, badge_border)
        badge_text_rect = badge_text.get_rect(center=badge_rect.center)
        self.screen.blit(badge_text, badge_text_rect)

        # --- Section 5: Telemetry Card ---
        y = 344
        lbl_coords = self.font_small.render("REAL-TIME TACTICAL TELEMETRY", True, COLOR_TEXT_MUTED)
        self.screen.blit(lbl_coords, (hud_x, y))
        y += 18

        telemetry_card = pygame.Rect(hud_x, y, hud_w, 154)
        pygame.draw.rect(self.screen, COLOR_HUD_PANEL, telemetry_card, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_HUD_BORDER, telemetry_card, width=1, border_radius=6)

        lines = [
            ("Attacker (Ninja):", f"Row {snapshot['att_r']}, Col {snapshot['att_c']}", COLOR_TEXT_WHITE),
            ("Defender (Guard):", f"Row {snapshot['def_r']}, Col {snapshot['def_c']}", COLOR_TEXT_WHITE),
            ("Player Distance:", f"{snapshot['dist']:.2f} tiles", COLOR_TEXT_WHITE),
            ("Tactical Corridor:", f"{snapshot['phase_str']}", snapshot['phase_col']),
            ("Corridor Entropy:", "Zero Information Leakage", (0, 229, 255)),
        ]

        item_y = y + 10
        for label, val, val_col in lines:
            lbl_s = self.font_small.render(label, True, COLOR_TEXT_MUTED)
            val_s = self.font_small.render(val, True, val_col)
            self.screen.blit(lbl_s, (hud_x + 12, item_y))
            self.screen.blit(val_s, (hud_x + hud_w - val_s.get_width() - 12, item_y))
            item_y += 27

        # --- Section 6: Presenter Controls Footer ---
        footer_rect = pygame.Rect(hud_x, 560, hud_w, 82)
        pygame.draw.rect(self.screen, (16, 16, 22), footer_rect, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_HUD_BORDER, footer_rect, width=1, border_radius=6)

        ctrl_1 = self.font_small.render("[P] Pause/Resume  |  [\u2190 / \u2192] Step Back/Fwd", True, COLOR_ACCENT_GOLD)
        ctrl_2 = self.font_small.render("[R] Random  |  [1] Blue  |  [2] Red", True, COLOR_TEXT_WHITE)
        ctrl_3 = self.font_small.render("[SPACE] Rules Screen  |  [ESC] Exit Demo", True, COLOR_TEXT_MUTED)
        self.screen.blit(ctrl_1, (hud_x + (hud_w - ctrl_1.get_width()) // 2, 568))
        self.screen.blit(ctrl_2, (hud_x + (hud_w - ctrl_2.get_width()) // 2, 592))
        self.screen.blit(ctrl_3, (hud_x + (hud_w - ctrl_3.get_width()) // 2, 616))

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

        # Title
        outcome_str = snapshot.get("final_outcome", self.final_outcome)
        title_surf = self.font_title.render(outcome_str, True, status_col)
        self.screen.blit(title_surf, title_surf.get_rect(center=(ARENA_WIDTH // 2, by + 34)))

        # Subtitle metrics
        target_name = "Goal A (Blue)" if self.theta == 0 else "Goal B (Red)"
        sub_str = f"Target: {target_name}  |  Steps: {snapshot['step']}  |  Final H(b): {snapshot['entropy']:.3f} bits"
        sub_surf = self.font_body.render(sub_str, True, COLOR_TEXT_WHITE)
        self.screen.blit(sub_surf, sub_surf.get_rect(center=(ARENA_WIDTH // 2, by + 74)))

        # Action prompts (including Step Back)
        prompt_1 = self.font_body.render("[\u2190] Step Backward to Review Trajectory", True, (0, 229, 255))
        prompt_2 = self.font_body.render("[R] Random  |  [1] Goal A  |  [2] Goal B to Continue", True, COLOR_ACCENT_GOLD)
        self.screen.blit(prompt_1, prompt_1.get_rect(center=(ARENA_WIDTH // 2, by + 112)))
        self.screen.blit(prompt_2, prompt_2.get_rect(center=(ARENA_WIDTH // 2, by + 138)))

    def render_lobby_screen(self) -> None:
        """Render the full-screen Minecraft-style title and rules lobby."""
        self.screen.fill(COLOR_BG_DARK)

        frame_rect = pygame.Rect(40, 30, SCREEN_WIDTH - 80, SCREEN_HEIGHT - 60)
        pygame.draw.rect(self.screen, (26, 28, 38), frame_rect, border_radius=8)
        pygame.draw.rect(self.screen, (60, 65, 85), frame_rect, width=2, border_radius=8)

        title = self.font_title.render("ASYMMETRIC SIGNALING GAME", True, COLOR_ACCENT_GOLD)
        self.screen.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, 75)))

        subtitle = self.font_subtitle.render("Emergent Pooling Equilibrium via Bayesian MARL", True, (0, 229, 255))
        self.screen.blit(subtitle, subtitle.get_rect(center=(SCREEN_WIDTH // 2, 115)))

        pygame.draw.line(self.screen, (50, 55, 75), (80, 140), (SCREEN_WIDTH - 80, 140), 1)

        rules_y = 165
        rules = [
            ("\u25c6 ATTACKER (NINJA)", "Secretly assigned Goal A (Blue) or Goal B (Red). Must reach its target without revealing private intent.", COLOR_ACCENT_BLUE),
            ("\u25c6 DEFENDER (KNIGHT)", "Observes physical coordinates only. Uses a recurrent Belief Head to estimate Bayesian posterior P(\u03b8).", COLOR_ACCENT_RED),
            ("\u25c6 SHANNON ENTROPY REGULARIZATION", "Attacker reward is augmented with Defender ambiguity: R_custom = R_env + \u03bb \u00b7 H(b).", (0, 229, 255)),
            ("\u25c6 EMERGENT POOLING EQUILIBRIUM", "Attacker pools along Column 5 up to Row 7 to maintain H(b) ~ 1.0, then executes a late diagonal bifurcation.", COLOR_ACCENT_GREEN),
        ]

        card_w = SCREEN_WIDTH - 160
        for header, desc, col in rules:
            card_rect = pygame.Rect(80, rules_y, card_w, 58)
            pygame.draw.rect(self.screen, (18, 20, 26), card_rect, border_radius=6)
            pygame.draw.rect(self.screen, (40, 44, 58), card_rect, width=1, border_radius=6)

            h_surf = self.font_header.render(header, True, col)
            d_surf = self.font_body.render(desc, True, COLOR_TEXT_WHITE)
            self.screen.blit(h_surf, (96, rules_y + 8))
            self.screen.blit(d_surf, (96, rules_y + 32))
            rules_y += 70

        legend_rect = pygame.Rect(80, 465, card_w, 52)
        pygame.draw.rect(self.screen, (22, 24, 32), legend_rect, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_ACCENT_GOLD, legend_rect, width=1, border_radius=6)

        legend_text = self.font_body.render(
            "[P] Pause  |  [\u2190 / \u2192] Step  |  [R] Random  |  [1] Goal A  |  [2] Goal B  |  [ESC] Quit",
            True, COLOR_ACCENT_GOLD
        )
        self.screen.blit(legend_text, legend_text.get_rect(center=legend_rect.center))

        current_time = pygame.time.get_ticks()
        if (current_time // 500) % 2 == 0:
            prompt = self.font_title.render("▶  PRESS [SPACE] TO ENTER ARENA  ◀", True, COLOR_TEXT_WHITE)
            self.screen.blit(prompt, prompt.get_rect(center=(SCREEN_WIDTH // 2, 570)))

        pygame.display.flip()

    # ========================================================================
    # MAIN APPLICATION LOOP
    # ========================================================================

    def run(self) -> None:
        """Main event and rendering loop with step-by-step playback controls."""
        print("[LiveDemo] Started live presentation loop.")
        running = True

        while running:
            current_time = pygame.time.get_ticks()

            # Handle user events
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
                            print(f"[LiveDemo] {'Paused' if self.paused else 'Resumed'}")

                    # Playback: Step Backward ([LEFT ARROW])
                    elif event.key == pygame.K_LEFT:
                        if self.state in (STATE_PLAYING, STATE_DONE):
                            self.paused = True
                            if self.state == STATE_DONE:
                                self.state = STATE_PLAYING
                            self.view_idx = max(0, self.view_idx - 1)
                            print(f"[LiveDemo] Step Back -> View Step {self.history[self.view_idx]['step']}")

                    # Playback: Step Forward ([RIGHT ARROW])
                    elif event.key == pygame.K_RIGHT:
                        if self.state == STATE_PLAYING:
                            self.paused = True
                            # If viewing past history, step forward toward head
                            if self.view_idx < len(self.history) - 1:
                                self.view_idx += 1
                                if self.history[self.view_idx]["done"]:
                                    self.state = STATE_DONE
                                print(f"[LiveDemo] Step Fwd -> View Step {self.history[self.view_idx]['step']}")
                            else:
                                # At the head of uncompleted episode -> advance 1 step
                                if not self.history[-1]["done"]:
                                    self.advance_environment_step()
                                    self.last_step_time = current_time
                                    print(f"[LiveDemo] Step Fwd (New) -> Step {self.step_counter}")

                    # Rules / Screen toggle ([SPACE])
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

            # State-specific logic and rendering
            if self.state == STATE_LOBBY:
                self.render_lobby_screen()

            elif self.state == STATE_PLAYING:
                # Timer auto-progression when not paused
                if not self.paused:
                    if current_time - self.last_step_time >= STEP_DELAY_MS:
                        if self.view_idx < len(self.history) - 1:
                            # Catch up through recorded history first
                            self.view_idx += 1
                            self.last_step_time = current_time
                            if self.history[self.view_idx]["done"]:
                                self.state = STATE_DONE
                        else:
                            # Advance live environment step
                            self.advance_environment_step()
                            self.last_step_time = current_time

                snapshot = self.history[self.view_idx]
                self.render_arena(snapshot)
                self.render_hud(snapshot)
                pygame.display.flip()

            elif self.state == STATE_DONE:
                # Frozen final frame: arena + HUD + overlay banner
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
