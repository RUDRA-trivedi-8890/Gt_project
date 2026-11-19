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
    """Bulletproof tactical defender action."""
    # 1. Goal Defense Priority
    if p_A >= 0.58:
        if dr >= 8.9 and dc <= 1.0:
            if dr < 9.9:
                return 1  # Down [1, 0] into Goal A (10, 0)
            elif dc > 0.1:
                return 2  # Left [0, -1] into Goal A
            else:
                return 8  # Holding Goal A
        elif dr >= 9.5:
            return 2  # Left towards Goal A
    elif p_B >= 0.58:
        if dr >= 8.9 and dc >= 9.0:
            if dr < 9.9:
                return 1  # Down [1, 0] into Goal B (10, 10)
            elif dc < 9.9:
                return 3  # Right [0, 1] into Goal B
            else:
                return 8  # Holding Goal B
        elif dr >= 9.5:
            return 3  # Right towards Goal B

    # 2. Close range interception
    dist_att = np.hypot(ar - dr, ac - dc)
    if dist_att <= 1.8:
        tr, tc = ar, ac
    elif p_A >= 0.58:
        # Move directly to cut off or occupy Goal A
        tr = 10.0
        tc = max(0.0, min(dc - 1.0, (ac + 0.0) / 2.0))
    elif p_B >= 0.58:
        # Move directly to cut off or occupy Goal B
        tr = 10.0
        tc = min(10.0, max(dc + 1.0, (ac + 10.0) / 2.0))
    else:
        # Central bottleneck: HOLD (9.0, 5.0)
        tr, tc = 9.0, 5.0

    # Best valid action
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

dev = torch.device("cpu")
model = RecurrentDefender(obs_dim=5, action_dim=9, hidden_dim=64, belief_temperature=1.0).to(dev)

att_path = os.path.join(CURRENT_DIR, "models", "attacker_net.pt")
attacker_net = get_attacker_network(hidden_dim=64, action_dim=9).to(dev)
if os.path.exists(att_path):
    attacker_net.load_state_dict(torch.load(att_path, map_location=dev))
attacker_net.eval()

bayes_teacher = ExactBayesFilter(attacker_net)
env = DeceptionGridWorld(EnvConfig(include_time_in_obs=True, detect_swaps=True))
tricks = get_trick_suite()

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

print("[Training] Sequence BPTT Training with Gate-Holding & Goal-Line Grounding...")
t0 = time.time()

num_episodes = 480
for ep in range(num_episodes):
    trick = tricks[ep % len(tricks)]
    theta = ep % 2
    obs_dict, _ = env.reset(options={"theta": theta})
    trick.reset(theta=theta)
    bayes_teacher.reset()

    seq_obs = []
    seq_expert_actions = []
    seq_bayes_beliefs = []
    seq_thetas = []

    def_h = None
    eps = max(0.0, 0.8 - (ep / 300.0))

    for t in range(env.horizon):
        att_pos = env.attacker_pos.copy()
        def_pos = env.defender_pos.copy()
        att_act = trick.get_action(att_pos, def_pos)
        bayes_p = bayes_teacher.update(att_pos, att_act)

        norm_t = float(t) / float(env.horizon)
        def_obs_5d = np.array([
            att_pos[0] / 10.0,
            att_pos[1] / 10.0,
            def_pos[0] / 10.0,
            def_pos[1] / 10.0,
            norm_t
        ], dtype=np.float32)

        def_obs_t = torch.as_tensor(def_obs_5d, dtype=torch.float32, device=dev).unsqueeze(0)
        with torch.no_grad():
            act_pred, _, _, _, b_logits, b_probs, def_h = model.forward_step(def_obs_t, def_h)

        expert_a = compute_expert_action(
            def_pos[0], def_pos[1], att_pos[0], att_pos[1], bayes_p[0], bayes_p[1]
        )

        seq_obs.append(def_obs_5d)
        seq_expert_actions.append(expert_a)
        seq_bayes_beliefs.append(bayes_p)
        seq_thetas.append(theta)

        chosen_a = expert_a if np.random.rand() < eps else int(act_pred.item())

        obs_dict, rew, term, trunc, info = env.step(att_act, chosen_a)
        if term or trunc:
            break

    # Sequence Update
    seq_t = torch.tensor(np.array(seq_obs), dtype=torch.float32, device=dev).unsqueeze(0)
    act_t = torch.tensor(seq_expert_actions, dtype=torch.long, device=dev)
    bayes_t = torch.tensor(np.array(seq_bayes_beliefs), dtype=torch.float32, device=dev)
    theta_t = torch.tensor(seq_thetas, dtype=torch.long, device=dev)

    gru_out, _ = model.gru(seq_t)
    h_seq = gru_out.squeeze(0)

    b_logits = model.belief_head(h_seq)
    b_probs = F.softmax(b_logits, dim=-1)

    loss_ce_b = F.cross_entropy(b_logits, theta_t)
    loss_kl_b = F.kl_div(torch.log(b_probs + 1e-8), bayes_t, reduction="batchmean")
    loss_belief = loss_ce_b + 0.6 * loss_kl_b

    actor_in = torch.cat([h_seq, b_probs.detach()], dim=-1)
    act_logits = model.actor_net(actor_in)
    loss_actor = F.cross_entropy(act_logits, act_t)

    loss_total = loss_belief + 1.5 * loss_actor

    optimizer.zero_grad()
    loss_total.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()

    if (ep + 1) % 60 == 0:
        print(f"Ep {ep+1:3d}/{num_episodes} | Total Loss: {loss_total.item():.4f} | Belief Loss: {loss_belief.item():.4f} | Actor Loss: {loss_actor.item():.4f}")

# Static Goal Defense Grounding Batch
print("\n[Grounding] Fine-tuning critical goal mouth and gate-holding anchors...")
X_ground = []
Y_ground = []
for _ in range(300):
    # Def at (9, 0), Att at (10, 1), p_A=1.0 -> Action 1 (Down into (10, 0))
    X_ground.append([1.0, 0.1, 0.9, 0.0, 0.44])
    Y_ground.append(1)
    # Def at (9, 10), Att at (10, 9), p_B=1.0 -> Action 1 (Down into (10, 10))
    X_ground.append([1.0, 0.9, 0.9, 1.0, 0.44])
    Y_ground.append(1)
    # Def at (9, 5), Att at (0, 5), p_A=0.5 -> Action 8 (Hold gate)
    X_ground.append([0.0, 0.5, 0.9, 0.5, 0.04])
    Y_ground.append(8)
    # Def at (9, 5), Att at (3, 5), p_A=0.5 -> Action 8 (Hold gate)
    X_ground.append([0.3, 0.5, 0.9, 0.5, 0.12])
    Y_ground.append(8)
    # Def at (10, 4), Att at (9, 2), p_A=1.0 -> Action 2 (Left)
    X_ground.append([0.9, 0.2, 1.0, 0.4, 0.40])
    Y_ground.append(2)
    # Def at (10, 6), Att at (9, 8), p_B=1.0 -> Action 3 (Right)
    X_ground.append([0.9, 0.8, 1.0, 0.6, 0.40])
    Y_ground.append(3)

X_gt = torch.tensor(X_ground, dtype=torch.float32, device=dev)
Y_gt = torch.tensor(Y_ground, dtype=torch.long, device=dev)
opt_actor = torch.optim.Adam(model.actor_net.parameters(), lr=0.003)

for _ in range(60):
    with torch.no_grad():
        g_out, _ = model.gru(X_gt.unsqueeze(1))
        h_g = g_out.squeeze(1)
        b_l = model.belief_head(h_g)
        b_p = F.softmax(b_l, dim=-1)
        a_in = torch.cat([h_g, b_p], dim=-1)
    l_act = F.cross_entropy(model.actor_net(a_in), Y_gt)
    opt_actor.zero_grad()
    l_act.backward()
    opt_actor.step()

print(f"[Training Complete in {time.time() - t0:.2f}s]")

# Test against StraightToKAttacker(k=7)
print("\n--- TEST: StraightToKAttacker(k=7) vs Upgraded Recurrent Defender ---")
model.eval()
env.reset(options={"theta": 0})
trick = StraightToKAttacker(k=7)
trick.reset(theta=0)
def_h = None

for step in range(1, 26):
    att_pos = env.attacker_pos.copy()
    def_pos = env.defender_pos.copy()
    att_act = trick.get_action(att_pos, def_pos)
    norm_t = float(step) / 25.0
    def_obs_5d = np.array([
        att_pos[0] / 10.0,
        att_pos[1] / 10.0,
        def_pos[0] / 10.0,
        def_pos[1] / 10.0,
        norm_t
    ], dtype=np.float32)
    def_obs_t = torch.as_tensor(def_obs_5d, dtype=torch.float32, device=dev).unsqueeze(0)

    with torch.no_grad():
        act_t, _, _, _, b_logits, probs_t, def_h = model.forward_step(def_obs_t, def_h, deterministic=True)
    def_act = int(act_t.item())
    probs = probs_t.squeeze(0).numpy()

    print(f"Step {step:2d} | Att: ({att_pos[0]}, {att_pos[1]}) | Def: ({def_pos[0]}, {def_pos[1]}) | P(A): {probs[0]*100:5.1f}% | Def Action: {def_act} ({ACTION_DELTAS[def_act]})")

    obs_dict, rewards, term, trunc, info = env.step(att_act, def_act)
    if term or trunc:
        print(f"Game Ended at Step {step}: Goal Captured={info.get('goal_reached')}, INTERCEPTED={info.get('intercepted')}")
        print(f"Final Positions: Attacker=({env.attacker_pos[0]}, {env.attacker_pos[1]}), Defender=({env.defender_pos[0]}, {env.defender_pos[1]})")
        break
