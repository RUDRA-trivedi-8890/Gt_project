"""Scripted Trick Attacker Suite for Defender Training and Stress Testing.

Provides parameterized adversaries:
1. StraightToKAttacker(k): Moves straight up median corridor to row k, then bifurcates.
2. FeintAttacker: Feints toward the opposite (wrong) goal for 3-4 steps, then cuts sharply back.
3. ZigzagAttacker: Alternates lateral oscillations while progressing forward.
4. MidPoolDashAttacker: Pools up to midfield (row 5) then dashes diagonally to target.
5. IdleRushAttacker: Lingers at spawn for several steps to test temporal patience, then rushes.
6. NoiseJitterAttacker: Direct shortest-path policy corrupted by random lateral jitter actions.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class BaseTrickAttacker:
    """Base class for scripted attackers."""

    name: str = "BaseTrick"

    def reset(self, theta: int) -> None:
        self.theta = theta
        self.step_idx = 0
        self.revealed = False
        self.reveal_step: Optional[int] = None

    def get_action(self, att_pos: np.ndarray, defender_pos: np.ndarray) -> int:
        raise NotImplementedError


class StraightToKAttacker(BaseTrickAttacker):
    """Walks straight up Column 5 to row k, then commits diagonally to active goal."""

    def __init__(self, k: int = 7) -> None:
        self.k = k
        self.name = f"StraightTo{k}"

    def reset(self, theta: int) -> None:
        super().reset(theta)

    def get_action(self, att_pos: np.ndarray, defender_pos: np.ndarray) -> int:
        row = int(round(att_pos[0]))
        col = int(round(att_pos[1]))
        self.step_idx += 1

        if row < self.k:
            # Move straight Down (Forward along x)
            return 1  # Action 1: [1, 0]

        # Row >= k: Bifurcation event
        if not self.revealed:
            self.revealed = True
            self.reveal_step = self.step_idx

        if self.theta == 0:  # Goal A (10, 0)
            if col > 0:
                return 4  # Diagonal Down-Left [1, -1]
            return 1      # Straight Down [1, 0]
        else:  # Goal B (10, 10)
            if col < 10:
                return 5  # Diagonal Down-Right [1, 1]
            return 1      # Straight Down [1, 0]


class FeintAttacker(BaseTrickAttacker):
    """Feints toward wrong goal for 4 steps, then cuts sharply toward true goal."""

    name = "FeintAndCut"

    def __init__(self, feint_steps: int = 4) -> None:
        self.feint_steps = feint_steps

    def reset(self, theta: int) -> None:
        super().reset(theta)

    def get_action(self, att_pos: np.ndarray, defender_pos: np.ndarray) -> int:
        col = int(round(att_pos[1]))
        step = self.step_idx
        self.step_idx += 1

        if step < self.feint_steps:
            # Feint toward wrong goal
            if self.theta == 0:
                return 5  # Feint Down-Right toward B [1, 1]
            else:
                return 4  # Feint Down-Left toward A [1, -1]

        # Cut phase: turn sharply toward true goal
        if not self.revealed:
            self.revealed = True
            self.reveal_step = self.step_idx

        if self.theta == 0:
            if col > 0:
                return 4  # Down-Left toward A [1, -1]
            return 1
        else:
            if col < 10:
                return 5  # Down-Right toward B [1, 1]
            return 1


class ZigzagAttacker(BaseTrickAttacker):
    """Alternates lateral oscillations to induce defender jitter, then commits at row 7."""

    name = "Zigzag"

    def reset(self, theta: int) -> None:
        super().reset(theta)

    def get_action(self, att_pos: np.ndarray, defender_pos: np.ndarray) -> int:
        row = int(round(att_pos[0]))
        col = int(round(att_pos[1]))
        step = self.step_idx
        self.step_idx += 1

        if row < 7:
            # Alternate left and right diagonal forward moves (4: Down-Left, 5: Down-Right)
            return 4 if (step % 2 == 0) else 5

        if not self.revealed:
            self.revealed = True
            self.reveal_step = self.step_idx

        if self.theta == 0:
            return 4 if col > 0 else 1
        else:
            return 5 if col < 10 else 1


class MidPoolDashAttacker(BaseTrickAttacker):
    """Pools on median up to row 5 (midfield), then dashes to target."""

    name = "MidPoolDash"

    def reset(self, theta: int) -> None:
        super().reset(theta)

    def get_action(self, att_pos: np.ndarray, defender_pos: np.ndarray) -> int:
        row = int(round(att_pos[0]))
        col = int(round(att_pos[1]))
        self.step_idx += 1

        if row < 5:
            return 1  # straight down Col 5

        if not self.revealed:
            self.revealed = True
            self.reveal_step = self.step_idx

        if self.theta == 0:
            return 4 if col > 0 else 1
        else:
            return 5 if col < 10 else 1


class IdleRushAttacker(BaseTrickAttacker):
    """Idles at spawn for 3 steps, then rushes toward goal."""

    name = "IdleRush"

    def __init__(self, idle_steps: int = 3) -> None:
        self.idle_steps = idle_steps

    def reset(self, theta: int) -> None:
        super().reset(theta)

    def get_action(self, att_pos: np.ndarray, defender_pos: np.ndarray) -> int:
        step = self.step_idx
        self.step_idx += 1

        if step < self.idle_steps:
            return 8  # Idle [0, 0]

        if not self.revealed:
            self.revealed = True
            self.reveal_step = self.step_idx

        col = int(round(att_pos[1]))
        if self.theta == 0:
            return 4 if col > 0 else 1
        else:
            return 5 if col < 10 else 1


class NoiseJitterAttacker(BaseTrickAttacker):
    """Direct shortest path corrupted with random lateral perturbation actions."""

    name = "NoiseJitter"

    def __init__(self, noise_prob: float = 0.25, seed: int = 42) -> None:
        self.noise_prob = noise_prob
        self.rng = np.random.RandomState(seed)

    def reset(self, theta: int) -> None:
        super().reset(theta)

    def get_action(self, att_pos: np.ndarray, defender_pos: np.ndarray) -> int:
        col = int(round(att_pos[1]))
        self.step_idx += 1

        if not self.revealed and col != 5:
            self.revealed = True
            self.reveal_step = self.step_idx

        if self.rng.rand() < self.noise_prob:
            # Jitter action
            return int(self.rng.choice([1, 2, 3, 4, 5]))

        if self.theta == 0:
            return 4 if col > 0 else 1
        else:
            return 5 if col < 10 else 1


def get_trick_suite() -> List[BaseTrickAttacker]:
    """Instantiate the standard scripted trick attacker curriculum suite."""
    return [
        StraightToKAttacker(k=3),
        StraightToKAttacker(k=5),
        StraightToKAttacker(k=7),
        StraightToKAttacker(k=8),
        StraightToKAttacker(k=9),
        FeintAttacker(feint_steps=4),
        ZigzagAttacker(),
        MidPoolDashAttacker(),
        IdleRushAttacker(idle_steps=3),
        NoiseJitterAttacker(noise_prob=0.25),
    ]
