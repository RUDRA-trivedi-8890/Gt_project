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
from scratch.test_expert import compute_expert_action

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

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

print("[Training] Starting Sequence BPTT Training over 300 episodes...")
t0 = time.time()

num_episodes = 320
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
    eps = max(0.05, 1.0 - (ep / 200.0))

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

        # Rollout action: mixture of expert and model
        chosen_a = expert_a if np.random.rand() < eps else int(act_pred.item())

        obs_dict, rew, term, trunc, info = env.step(att_act, chosen_a)
        if term or trunc:
            break

    # Sequence Update with BPTT
    seq_t = torch.tensor(np.array(seq_obs), dtype=torch.float32, device=dev).unsqueeze(0)
    act_t = torch.tensor(seq_expert_actions, dtype=torch.long, device=dev)
    bayes_t = torch.tensor(np.array(seq_bayes_beliefs), dtype=torch.float32, device=dev)
    theta_t = torch.tensor(seq_thetas, dtype=torch.long, device=dev)

    # Forward evaluate sequence
    gru_out, _ = model.gru(seq_t) # (1, seq_len, 64)
    h_seq = gru_out.squeeze(0)    # (seq_len, 64)

    # Belief head
    b_logits = model.belief_head(h_seq) # (seq_len, 2)
    b_probs = F.softmax(b_logits, dim=-1)

    loss_ce_b = F.cross_entropy(b_logits, theta_t)
    loss_kl_b = F.kl_div(torch.log(b_probs + 1e-8), bayes_t, reduction="batchmean")
    loss_belief = loss_ce_b + 0.5 * loss_kl_b

    # Actor head
    actor_in = torch.cat([h_seq, b_probs.detach()], dim=-1)
    act_logits = model.actor_net(actor_in)
    loss_actor = F.cross_entropy(act_logits, act_t)

    loss_total = loss_belief + 1.2 * loss_actor

    optimizer.zero_grad()
    loss_total.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()

    if (ep + 1) % 40 == 0:
        print(f"Ep {ep+1:3d}/{num_episodes} | Total Loss: {loss_total.item():.4f} | Belief Loss: {loss_belief.item():.4f} | Actor Loss: {loss_actor.item():.4f}")

print(f"[Training] Completed in {time.time() - t0:.2f}s.")

# Now test against StraightToKAttacker(k=7)
print("\n--- EVALUATION: StraightToKAttacker(k=7) vs Newly Trained Recurrent Defender ---")
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

    print(f"Step {step:2d} | Att: ({att_pos[0]}, {att_pos[1]}) | Def: ({def_pos[0]}, {def_pos[1]}) | P(A): {probs[0]*100:5.1f}% | Def Action: {def_act}")

    obs_dict, rewards, term, trunc, info = env.step(att_act, def_act)
    if term or trunc:
        print(f"Ended at step {step}: goal={info.get('goal_reached')}, intercepted={info.get('intercepted')}")
        print(f"Final Att: ({env.attacker_pos[0]}, {env.attacker_pos[1]}), Final Def: ({env.defender_pos[0]}, {env.defender_pos[1]})")
        break
