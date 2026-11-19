import numpy as np

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
    # 1. Goal Defense Priority
    if p_A >= 0.55:
        # If near Goal A (10, 0)
        if dr >= 9.0 and dc <= 1.0:
            if dr < 10.0:
                return 1  # Down into Goal A
            elif dc > 0.0:
                return 2  # Left into Goal A
            else:
                return 8  # Occupying Goal A
        elif dr >= 9.5:
            # On bottom row, move left towards Goal A
            return 2  # Left
    elif p_B >= 0.55:
        # If near Goal B (10, 10)
        if dr >= 9.0 and dc >= 9.0:
            if dr < 10.0:
                return 1  # Down into Goal B
            elif dc < 10.0:
                return 3  # Right into Goal B
            else:
                return 8  # Occupying Goal B
        elif dr >= 9.5:
            # On bottom row, move right towards Goal B
            return 3  # Right

    # 2. Direct Interception if attacker is close
    dist_att = np.hypot(ar - dr, ac - dc)
    if dist_att <= 1.8:
        tr, tc = ar, ac
    elif p_A >= 0.55:
        # Cut off attacker towards Goal A
        tr = min(10.0, max(dr, (ar + 10.0) / 2.0))
        tc = max(0.0, min(dc, (ac + 0.0) / 2.0))
    elif p_B >= 0.55:
        # Cut off attacker towards Goal B
        tr = min(10.0, max(dr, (ar + 10.0) / 2.0))
        tc = min(10.0, max(dc, (ac + 10.0) / 2.0))
    else:
        # Center corridor pooling: defend (9.0, 5.0)
        if dr >= 9.5:
            if dc < 4.5:
                return 3  # Right
            elif dc > 5.5:
                return 2  # Left
            else:
                return 0  # Up to row 9
        tr, tc = 9.0, 5.0

    # Best valid move to (tr, tc)
    best_a = 8
    min_dist = 999.0
    for a, (delta_r, delta_c) in enumerate(ACTION_DELTAS):
        nr = dr + delta_r
        nc = dc + delta_c
        if nr < 0 or nr > 10 or nc < 0 or nc > 10:
            continue
        d = (nr - tr) ** 2 + (nc - tc) ** 2
        if d < min_dist:
            min_dist = d
            best_a = a
    return best_a

print("Def at (9, 0), Att at (10, 1), p_A=0.8 -> Action:", compute_expert_action(9.0, 0.0, 10.0, 1.0, 0.8, 0.2))
print("Def at (10, 4), Att at (9, 3), p_A=0.8 -> Action:", compute_expert_action(10.0, 4.0, 9.0, 3.0, 0.8, 0.2))
print("Def at (9, 5), Att at (0, 5), p_A=0.5 -> Action:", compute_expert_action(9.0, 5.0, 0.0, 5.0, 0.5, 0.5))
print("Def at (9, 10), Att at (10, 9), p_B=0.8 -> Action:", compute_expert_action(9.0, 10.0, 10.0, 9.0, 0.2, 0.8))
print("Def at (10, 6), Att at (9, 7), p_B=0.8 -> Action:", compute_expert_action(10.0, 6.0, 9.0, 7.0, 0.2, 0.8))
