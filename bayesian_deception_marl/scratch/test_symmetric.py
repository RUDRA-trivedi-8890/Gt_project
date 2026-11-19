import sys, os
sys.path.insert(0, 'Gt_project/bayesian_deception_marl')
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.recurrent_defender import RecurrentDefender

dev = torch.device('cpu')
model = RecurrentDefender(obs_dim=5, action_dim=9, hidden_dim=64, belief_temperature=1.0).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=0.003)

# 3 Symmetric Batch Sequences:
# Seq 0: Center Corridor Pooling (steps 0 to 7)
# Attacker walks down column 5, Defender holds (9, 5)
seq_center = [
    [t / 10.0, 0.5, 0.9, 0.5, t / 25.0] for t in range(8)
]
act_center = [8] * 8
bayes_center = [[0.5, 0.5]] * 8

# Seq 1: Left Branch towards Goal A (steps 7 to 11)
# Attacker turns left, Defender intercepts at (10, 0)
seq_A = [
    [0.7, 0.5, 0.9, 0.5, 7 / 25.0],
    [0.8, 0.4, 0.9, 0.4, 8 / 25.0],
    [0.9, 0.3, 0.9, 0.2, 9 / 25.0],
    [1.0, 0.2, 0.9, 0.1, 10 / 25.0],
    [1.0, 0.1, 0.9, 0.0, 11 / 25.0], # Def at (9, 0), Att at (10, 1) -> Action 1 (Down into (10, 0))
    [1.0, 0.0, 1.0, 0.0, 12 / 25.0], # Def at (10, 0), Att at (10, 0) -> Action 8 (Hold goal)
    [1.0, 0.0, 1.0, 0.0, 13 / 25.0],
    [1.0, 0.0, 1.0, 0.0, 14 / 25.0],
]
act_A = [8, 4, 4, 2, 1, 8, 8, 8]
bayes_A = [[0.5, 0.5]] + [[1.0, 0.0]] * 7

# Seq 2: Right Branch towards Goal B (steps 7 to 11)
# Attacker turns right, Defender intercepts at (10, 10)
seq_B = [
    [0.7, 0.5, 0.9, 0.5, 7 / 25.0],
    [0.8, 0.6, 0.9, 0.6, 8 / 25.0],
    [0.9, 0.7, 0.9, 0.8, 9 / 25.0],
    [1.0, 0.8, 0.9, 0.9, 10 / 25.0],
    [1.0, 0.9, 0.9, 1.0, 11 / 25.0], # Def at (9, 10), Att at (10, 9) -> Action 1 (Down into (10, 10))
    [1.0, 1.0, 1.0, 1.0, 12 / 25.0], # Def at (10, 10), Att at (10, 10) -> Action 8 (Hold goal)
    [1.0, 1.0, 1.0, 1.0, 13 / 25.0],
    [1.0, 1.0, 1.0, 1.0, 14 / 25.0],
]
act_B = [8, 5, 5, 3, 1, 8, 8, 8]
bayes_B = [[0.5, 0.5]] + [[0.0, 1.0]] * 7

# Batch tensors of shape (3, 8, obs_dim)
batch_seq = torch.tensor([seq_center, seq_A, seq_B], dtype=torch.float32, device=dev)
batch_act = torch.tensor([act_center, act_A, act_B], dtype=torch.long, device=dev)
batch_bayes = torch.tensor([bayes_center, bayes_A, bayes_B], dtype=torch.float32, device=dev)

print("[Symmetric Training] Training batch of symmetric trajectories...")
for epoch in range(120):
    gru_out, _ = model.gru(batch_seq) # (3, 8, 64)
    b_logits = model.belief_head(gru_out) # (3, 8, 2)
    b_probs = F.softmax(b_logits, dim=-1)

    loss_kl = F.kl_div(torch.log(b_probs + 1e-8), batch_bayes, reduction="batchmean")
    actor_in = torch.cat([gru_out, b_probs.detach()], dim=-1)
    act_logits = model.actor_net(actor_in)
    loss_act = F.cross_entropy(act_logits.view(-1, 9), batch_act.view(-1))

    loss = 2.0 * loss_kl + 3.0 * loss_act
    opt.zero_grad()
    loss.backward()
    opt.step()

# Test symmetry
test_0 = torch.tensor([[0.0, 0.5, 0.9, 0.5, 0.0]], dtype=torch.float32)
act_0, _, _, _, _, probs_0, _ = model.forward_step(test_0, None)
print("Initial state b_probs:", probs_0.detach().numpy())
print("Initial state Action:", act_0.item())
assert abs(probs_0[0, 0].item() - 0.5) < 0.05, f"Prior must be ~0.5! Got {probs_0[0, 0].item()}"
assert act_0.item() == 8, f"Initial action must be 8 (Hold gate)! Got {act_0.item()}"
print("PASSED SYMMETRIC VERIFICATION!")

print("\n--- SIMULATION: StraightToKAttacker(k=7) vs Symmetrically Trained Recurrent Defender ---")
from envs.grid_world import DeceptionGridWorld
from config import EnvConfig
from envs.trick_attackers import StraightToKAttacker

model.eval()
env = DeceptionGridWorld(EnvConfig(detect_swaps=True))
trick = StraightToKAttacker(k=7)
env.reset(options={"theta": 0})
trick.reset(theta=0)
ACTION_DELTAS = [[-1, 0], [1, 0], [0, -1], [0, 1], [1, -1], [1, 1], [-1, -1], [-1, 1], [0, 0]]

def_h = None
for step in range(1, 26):
    att_pos = env.attacker_pos.copy()
    def_pos = env.defender_pos.copy()
    att_act = trick.get_action(att_pos, def_pos)
    norm_t = float(step) / 25.0
    def_obs_5d = [att_pos[0]/10.0, att_pos[1]/10.0, def_pos[0]/10.0, def_pos[1]/10.0, norm_t]
    def_obs_t = torch.tensor([def_obs_5d], dtype=torch.float32, device=dev)
    with torch.no_grad():
        act_t, _, _, _, _, probs_t, def_h = model.forward_step(def_obs_t, def_h, deterministic=True)
    def_act = int(act_t.item())
    pA = probs_t[0, 0].item()
    print(f"Step {step:2d} | Att: ({att_pos[0]}, {att_pos[1]}) | Def: ({def_pos[0]}, {def_pos[1]}) | P(A): {pA*100:5.1f}% | Def Action: {def_act} ({ACTION_DELTAS[def_act]})")
    obs, r, term, trunc, info = env.step(att_act, def_act)
    if term or trunc:
        print(f"Ended at step {step}: Goal Captured={info.get('goal_reached')}, Intercepted={info.get('intercepted')}")
        print(f"Final: Att=({env.attacker_pos[0]}, {env.attacker_pos[1]}), Def=({env.defender_pos[0]}, {env.defender_pos[1]})")
        break
