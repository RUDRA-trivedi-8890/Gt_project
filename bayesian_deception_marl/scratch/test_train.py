import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = "Gt_project/bayesian_deception_marl"
sys.path.insert(0, CURRENT_DIR)

from config import EnvConfig
from envs.grid_world import DeceptionGridWorld
from models.recurrent_defender import RecurrentDefender
from models.networks import get_attacker_network
from algos.exact_bayes import ExactBayesFilter
from envs.trick_attackers import get_trick_suite, StraightToKAttacker

ACTION_DELTAS = np.array([
    [-1, 0],   # 0: Up
    [1, 0],    # 1: Down
    [0, -1],   # 2: Left
    [0, 1],    # 3: Right
    [1, -1],   # 4: Down-Left
    [1, 1],    # 5: Down-Right
    [-1, -1],  # 6: Up-Left
    [-1, 1],   # 7: Up-Right
    [0, 0],    # 8: Idle
], dtype=np.float32)

def compute_expert_action(dr: float, dc: float, ar: float, ac: float, p_A: float, p_B: float) -> int:
    """Compute optimal tactical defender action to intercept or defend goal."""
    # Never try to move Down if already at Row 10 (bottom wall)
    dist_att = np.hypot(ar - dr, ac - dc)
    
    # 1. Close-range interception priority
    if dist_att <= 1.8:
        tr, tc = ar, ac
    # 2. Defend Goal A
    elif p_A >= 0.55:
        if dr >= 9.0 and dc <= 1.0:
            tr, tc = 10.0, 0.0  # Guard Goal A
        elif dr >= 9.5:
            # On bottom row, move left towards Goal A
            return 2  # Left
        else:
            # Cut off attacker trajectory towards Goal A
            tr = min(10.0, (ar + 10.0) / 2.0 + 1.0)
            tc = max(0.0, (ac + 0.0) / 2.0)
    # 3. Defend Goal B
    elif p_B >= 0.55:
        if dr >= 9.0 and dc >= 9.0:
            tr, tc = 10.0, 10.0  # Guard Goal B
        elif dr >= 9.5:
            # On bottom row, move right towards Goal B
            return 3  # Right
        else:
            # Cut off attacker trajectory towards Goal B
            tr = min(10.0, (ar + 10.0) / 2.0 + 1.0)
            tc = min(10.0, (ac + 10.0) / 2.0)
    # 4. Ambiguous (Center Pooling)
    else:
        if dr >= 9.5:
            # On bottom row, center to col 5
            if dc < 4.5:
                return 3  # Right
            elif dc > 5.5:
                return 2  # Left
            else:
                return 0  # Up to row 9
        tr, tc = 9.0, 5.0

    # Pick best action that doesn't walk into walls
    best_a = 8
    min_dist = 999.0
    for a, (delta_r, delta_c) in enumerate(ACTION_DELTAS):
        nr = dr + delta_r
        nc = dc + delta_c
        # Disallow actions that hit boundary walls
        if nr < 0 or nr > 10 or nc < 0 or nc > 10:
            continue
        d = (nr - tr) ** 2 + (nc - tc) ** 2
        if d < min_dist:
            min_dist = d
            best_a = a
    return best_a

print("Expert action logic defined successfully.")
