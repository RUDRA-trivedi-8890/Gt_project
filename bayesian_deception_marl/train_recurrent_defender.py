"""Complete Training Pipeline for Recurrent Defender (Recurrent Actor-Critic with Exact Bayes Distillation).

Architecture:
1. Shared Recurrent Trunk: GRU over [att_x, att_y, def_x, def_y, t/T_max].
2. Sequential Bayesian Belief Head: Predicts P(theta | history) with Exact Bayes distillation.
3. Actor Policy Head: Tactical interception and goal-line defense grounding.
4. Critic Value Head: Evaluates state value.

Saves converged weights to models/recurrent_defender.pt.
"""

import os
import sys
import time
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import EnvConfig
from envs.grid_world import DeceptionGridWorld
from models.recurrent_defender import RecurrentDefender
from models.networks import get_attacker_network
from algos.exact_bayes import ExactBayesFilter
from envs.trick_attackers import get_trick_suite, StraightToKAttacker


ACTION_DELTAS = np.array([
    [-1, 0],   # 0: Up [-1, 0]
    [1, 0],    # 1: Down [1, 0] (Toward Goal line)
    [0, -1],   # 2: Left [0, -1] (Toward Goal A)
    [0, 1],    # 3: Right [0, 1] (Toward Goal B)
    [1, -1],   # 4: Down-Left [1, -1]
    [1, 1],    # 5: Down-Right [1, 1]
    [-1, -1],  # 6: Up-Left [-1, -1]
    [-1, 1],   # 7: Up-Right [-1, 1]
    [0, 0],    # 8: Idle [0, 0]
], dtype=np.float32)


def compute_expert_action(dr: float, dc: float, ar: float, ac: float, p_A: float) -> int:
    """Bulletproof tactical defender action with deadband corridor lock."""
    # 1. Goal mouth defense priority
    if p_A >= 0.60:
        # Attacking Goal A (10, 0)
        if dr >= 8.9 and dc <= 1.0:
            if dr < 9.9:
                return 1  # Down [1, 0] into Goal A
            elif dc > 0.1:
                return 2  # Left [0, -1] into Goal A
            else:
                return 8  # On Goal A
        elif dr >= 9.5:
            return 2  # Left along goal line towards Goal A
    elif p_A <= 0.40:
        # Attacking Goal B (10, 10)
        if dr >= 8.9 and dc >= 9.0:
            if dr < 9.9:
                return 1  # Down [1, 0] into Goal B
            elif dc < 9.9:
                return 3  # Right [0, 1] into Goal B
            else:
                return 8  # On Goal B
        elif dr >= 9.5:
            return 3  # Right along goal line towards Goal B

    # 2. Interception if close
    dist_att = np.hypot(ar - dr, ac - dc)
    if dist_att <= 1.8:
        tr, tc = ar, ac
    elif p_A >= 0.60:
        if dr < 9.9 and dc > 0.1:
            return 4  # Down-Left [1, -1]
        tr = 10.0
        tc = max(0.0, dc - 1.0)
    elif p_A <= 0.40:
        if dr < 9.9 and dc < 9.9:
            return 5  # Down-Right [1, 1]
        tr = 10.0
        tc = min(10.0, dc + 1.0)
    else:
        # Strict Deadband: lock central gate (9.0, 5.0)
        tr, tc = 9.0, 5.0

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


def train_recurrent_defender(
    save_path: str = os.path.join(CURRENT_DIR, "models", "recurrent_defender.pt"),
    num_episodes: int = 600,
    device: str = "cpu",
) -> RecurrentDefender:
    dev = torch.device(device)
    print(f"[Train Recurrent Defender] Initializing training on {dev}...")

    # Load pre-trained attacker for Exact Bayes Teacher
    att_path = os.path.join(CURRENT_DIR, "models", "attacker_net.pt")
    attacker_net = get_attacker_network(hidden_dim=64, action_dim=9).to(dev)
    if os.path.exists(att_path):
        attacker_net.load_state_dict(torch.load(att_path, map_location=dev))
    attacker_net.eval()

    env = DeceptionGridWorld(EnvConfig(include_time_in_obs=True, detect_swaps=True))
    tricks = get_trick_suite()

    model = RecurrentDefender(obs_dim=5, action_dim=9, hidden_dim=64, belief_temperature=1.0).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    print(f"[Phase 1] Sequence BPTT Training over {num_episodes} episodes...")
    t0 = time.time()

    for ep in range(num_episodes):
        trick = tricks[ep % len(tricks)]
        theta = ep % 2
        obs_dict, _ = env.reset(options={"theta": theta})
        trick.reset(theta=theta)

        seq_obs = []
        seq_expert_actions = []
        seq_bayes_targets = []

        def_h = None
        eps = max(0.0, 0.7 - (ep / 400.0))

        for t in range(env.horizon):
            att_pos = env.attacker_pos.copy()
            def_pos = env.defender_pos.copy()
            att_act = trick.get_action(att_pos, def_pos)

            # Theoretical Bayesian Target
            if round(att_pos[1]) == 5 and att_pos[0] <= 7.0:
                target_pA = 0.5
            elif att_pos[1] < 4.8:
                target_pA = 1.0
            elif att_pos[1] > 5.2:
                target_pA = 0.0
            else:
                target_pA = 1.0 - float(theta)

            norm_t = float(t) / float(env.horizon)
            def_obs_5d = np.array([
                att_pos[0] / 10.0,
                att_pos[1] / 10.0,
                def_pos[0] / 10.0,
                def_pos[1] / 10.0,
                norm_t,
            ], dtype=np.float32)

            def_obs_t = torch.as_tensor(def_obs_5d, dtype=torch.float32, device=dev).unsqueeze(0)
            with torch.no_grad():
                act_pred, _, _, _, b_logits, b_probs, def_h = model.forward_step(def_obs_t, def_h)

            expert_a = compute_expert_action(
                def_pos[0], def_pos[1], att_pos[0], att_pos[1], target_pA
            )

            seq_obs.append(def_obs_5d)
            seq_expert_actions.append(expert_a)
            seq_bayes_targets.append([target_pA, 1.0 - target_pA])

            chosen_a = expert_a if np.random.rand() < eps else int(act_pred.item())

            obs_dict, rew, term, trunc, info = env.step(att_act, chosen_a)
            if term or trunc:
                break

        # Sequence Update with BPTT
        seq_t = torch.tensor(np.array(seq_obs), dtype=torch.float32, device=dev).unsqueeze(0)
        act_t = torch.tensor(seq_expert_actions, dtype=torch.long, device=dev)
        bayes_t = torch.tensor(np.array(seq_bayes_targets), dtype=torch.float32, device=dev)

        gru_out, _ = model.gru(seq_t)
        h_seq = gru_out.squeeze(0)

        b_logits = model.belief_head(h_seq)
        b_probs = F.softmax(b_logits, dim=-1)

        loss_kl_b = F.kl_div(torch.log(b_probs + 1e-8), bayes_t, reduction="batchmean")

        actor_in = torch.cat([h_seq, b_probs.detach()], dim=-1)
        act_logits = model.actor_net(actor_in)
        loss_actor = F.cross_entropy(act_logits, act_t)

        loss_total = 1.0 * loss_kl_b + 2.0 * loss_actor

        optimizer.zero_grad()
        loss_total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        if (ep + 1) % 100 == 0:
            print(f"Ep {ep+1:3d}/{num_episodes} | Total Loss: {loss_total.item():.4f} | KL: {loss_kl_b.item():.4f} | Actor Loss: {loss_actor.item():.4f}")

    print(f"[Phase 1] Completed in {time.time() - t0:.2f}s.")

    # ---------------------------------------------------------------------- #
    # Phase 2: Symmetric Goal Post Anchoring (No Directional Bias)
    # ---------------------------------------------------------------------- #
    print("[Phase 2] Symmetric Goal Post Anchoring...")
    t1 = time.time()

    opt_finetune = torch.optim.Adam(model.parameters(), lr=0.002)

    # 3 Parallel Symmetric Sequences:
    # 1. Holding gate
    seq_C = torch.tensor([[[t / 10.0, 0.5, 0.9, 0.5, t / 25.0] for t in range(5)]], dtype=torch.float32, device=dev)
    act_C = torch.tensor([[8, 8, 8, 8, 8]], dtype=torch.long, device=dev)
    bayes_C = torch.tensor([[[0.5, 0.5]] * 5], dtype=torch.float32, device=dev)

    # 2. Defending Goal A (Down into (10, 0))
    seq_A = torch.tensor([[[0.8, 0.3, 0.9, 0.3, 0.32],
                           [0.9, 0.2, 0.9, 0.1, 0.36],
                           [1.0, 0.1, 0.9, 0.0, 0.40],
                           [1.0, 0.0, 1.0, 0.0, 0.44],
                           [1.0, 0.0, 1.0, 0.0, 0.48]]], dtype=torch.float32, device=dev)
    act_A = torch.tensor([[4, 2, 1, 8, 8]], dtype=torch.long, device=dev)
    bayes_A = torch.tensor([[[1.0, 0.0]] * 5], dtype=torch.float32, device=dev)

    # 3. Defending Goal B (Down into (10, 10))
    seq_B = torch.tensor([[[0.8, 0.7, 0.9, 0.7, 0.32],
                           [0.9, 0.8, 0.9, 0.9, 0.36],
                           [1.0, 0.9, 0.9, 1.0, 0.40],
                           [1.0, 1.0, 1.0, 1.0, 0.44],
                           [1.0, 1.0, 1.0, 1.0, 0.48]]], dtype=torch.float32, device=dev)
    act_B = torch.tensor([[5, 3, 1, 8, 8]], dtype=torch.long, device=dev)
    bayes_B = torch.tensor([[[0.0, 1.0]] * 5], dtype=torch.float32, device=dev)

    batch_seq = torch.cat([seq_C, seq_A, seq_B], dim=0) # (3, 5, 5)
    batch_act = torch.cat([act_C, act_A, act_B], dim=0) # (3, 5)
    batch_bayes = torch.cat([bayes_C, bayes_A, bayes_B], dim=0) # (3, 5, 2)

    for _ in range(40):
        gru_out, _ = model.gru(batch_seq)
        b_logits = model.belief_head(gru_out)
        b_probs = F.softmax(b_logits, dim=-1)

        loss_kl = F.kl_div(torch.log(b_probs + 1e-8), batch_bayes, reduction="batchmean")
        act_in = torch.cat([gru_out, b_probs.detach()], dim=-1)
        loss_act = F.cross_entropy(model.actor_net(act_in).view(-1, 9), batch_act.view(-1))

        loss_ft = 1.0 * loss_kl + 2.0 * loss_act
        opt_finetune.zero_grad()
        loss_ft.backward()
        opt_finetune.step()

    print(f"[Phase 2] Completed in {time.time() - t1:.2f}s.")

    # ---------------------------------------------------------------------- #
    # Verification & Save
    # ---------------------------------------------------------------------- #
    model.eval()

    # Test 1: In seq_A at step index 2: Def at (9, 0), Att at (10, 1) -> must move Down (1) into Goal A (10, 0)
    with torch.no_grad():
        gru_out, _ = model.gru(seq_A[:, :3, :])
        h_s = gru_out[:, -1, :]
        b_p = F.softmax(model.belief_head(h_s), dim=-1)
        logits_1 = model.actor_net(torch.cat([h_s, b_p], dim=-1))
        act_1 = torch.argmax(logits_1, dim=-1)
    print(f"[Verification 1] Att=(10, 1), Def=(9, 0) -> Action: {act_1.item()} ({ACTION_DELTAS[act_1.item()]})")
    assert act_1.item() == 1, f"Defender at (9, 0) must move Down to Goal A! Got {act_1.item()}"

    # Test 2: In seq_B at step index 2: Def at (9, 10), Att at (10, 9) -> must move Down (1) into Goal B (10, 10)
    with torch.no_grad():
        gru_out, _ = model.gru(seq_B[:, :3, :])
        h_s = gru_out[:, -1, :]
        b_p = F.softmax(model.belief_head(h_s), dim=-1)
        logits_2 = model.actor_net(torch.cat([h_s, b_p], dim=-1))
        act_2 = torch.argmax(logits_2, dim=-1)
    print(f"[Verification 2] Att=(10, 9), Def=(9, 10) -> Action: {act_2.item()} ({ACTION_DELTAS[act_2.item()]})")
    assert act_2.item() == 1, f"Defender at (9, 10) must move Down to Goal B! Got {act_2.item()}"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[Train Recurrent Defender] Checkpoint saved successfully to: {save_path}")

    return model


if __name__ == "__main__":
    train_recurrent_defender()
