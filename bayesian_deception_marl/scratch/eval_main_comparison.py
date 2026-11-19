import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.abspath("bayesian_deception_marl")
sys.path.insert(0, CURRENT_DIR)

from config import EnvConfig
from envs.grid_world import DeceptionGridWorld
from models.networks import get_attacker_network, get_defender_network
from models.belief_head import BeliefHead
from models.recurrent_defender import RecurrentDefender
from utils.metrics import shannon_entropy

device = torch.device("cpu")
models_dir = os.path.join(CURRENT_DIR, "models")

# Load trained models
att_net = get_attacker_network(hidden_dim=64, action_dim=9).to(device)
def_net = get_defender_network(hidden_dim=64, action_dim=9).to(device)
bh_net = BeliefHead(input_dim=4, hidden_dim=64).to(device)
rec_def = RecurrentDefender(obs_dim=5, action_dim=9, hidden_dim=64).to(device)

att_net.load_state_dict(torch.load(os.path.join(models_dir, "attacker_net.pt"), map_location=device))
def_net.load_state_dict(torch.load(os.path.join(models_dir, "defender_net.pt"), map_location=device))
bh_net.load_state_dict(torch.load(os.path.join(models_dir, "belief_head.pt"), map_location=device))
rec_def.load_state_dict(torch.load(os.path.join(models_dir, "recurrent_defender.pt"), map_location=device))

att_net.eval()
def_net.eval()
bh_net.eval()
rec_def.eval()

SEEDS = [42, 101, 777, 1337, 2024]
NUM_EPISODES = 100

print("="*80)
print("RUNNING MULTI-SEED EVALUATION FOR TABLE VII")
print("="*80)

# ---------------------------------------------------------
# Row 2: Entropy bonus (lambda_0 = 0.7): attacker_net vs defender_net + belief_head
# ---------------------------------------------------------
print("\n--- Row 2: Entropy bonus (lambda_0 = 0.7) [attacker_net vs defender_net + belief_head] ---")
row2_gsr, row2_ir, row2_hmid, row2_steps = [], [], [], []

env = DeceptionGridWorld(EnvConfig(include_time_in_obs=False, detect_swaps=True))

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    ep_gsr, ep_ir, ep_hmid, ep_steps = [], [], [], []
    for ep in range(NUM_EPISODES):
        theta = ep % 2
        obs_dict, info = env.reset(seed=seed * 1000 + ep, options={"theta": theta})
        att_obs = obs_dict["attacker"]
        def_obs = obs_dict["defender"]
        def_hidden = None
        
        midfield_entropies = []
        step_count = 0
        
        for step in range(env.cfg.horizon):
            step_count += 1
            att_pos = (float(att_obs[0] * 10.0), float(att_obs[1] * 10.0))
            
            # Defender belief
            def_obs_t = torch.as_tensor(def_obs, dtype=torch.float32, device=device).unsqueeze(0)
            b_logits, def_hidden = bh_net.step(def_obs_t, def_hidden)
            probs = F.softmax(b_logits, dim=-1).squeeze(0).detach().cpu().numpy()
            ent = shannon_entropy(probs)
            
            if 4.0 <= att_pos[0] <= 6.0:
                midfield_entropies.append(ent)
                
            def_obs_full = np.concatenate([def_obs, probs], axis=-1).astype(np.float32)
            def_obs_full_t = torch.as_tensor(def_obs_full, dtype=torch.float32, device=device).unsqueeze(0)
            att_obs_t = torch.as_tensor(att_obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                att_logits, _ = att_net(att_obs_t)
                def_logits, _ = def_net(def_obs_full_t)
                att_act = int(torch.argmax(att_logits, dim=-1).item())
                def_act = int(torch.argmax(def_logits, dim=-1).item())
                
            next_obs, rew, term, trunc, sinfo = env.step(att_act, def_act)
            att_obs = next_obs["attacker"]
            def_obs = next_obs["defender"]
            
            if term or trunc:
                is_goal = bool(sinfo.get("goal_reached", False))
                is_intercept = bool(sinfo.get("intercepted", False))
                # Following evaluate.py logic:
                ep_gsr.append(1.0 if is_goal else 0.0)
                ep_ir.append(1.0 if is_intercept else 0.0)
                ep_steps.append(step_count)
                ep_hmid.append(np.mean(midfield_entropies) if midfield_entropies else 0.0)
                break
                
    row2_gsr.append(np.mean(ep_gsr) * 100.0)
    row2_ir.append(np.mean(ep_ir) * 100.0)
    row2_hmid.append(np.mean(ep_hmid))
    row2_steps.append(np.mean(ep_steps))
    print(f"  Seed {seed:4d} | GSR: {row2_gsr[-1]:5.1f}% | IR: {row2_ir[-1]:5.1f}% | H_mid: {row2_hmid[-1]:.4f} | Steps: {row2_steps[-1]:.2f}")

print(f"Row 2 Mean +- Std: GSR={np.mean(row2_gsr):.2f} +- {np.std(row2_gsr):.2f}% | IR={np.mean(row2_ir):.2f} +- {np.std(row2_ir):.2f}% | H_mid={np.mean(row2_hmid):.4f} +- {np.std(row2_hmid):.4f} | Steps={np.mean(row2_steps):.2f} +- {np.std(row2_steps):.2f}")

# ---------------------------------------------------------
# Row 3: + recurrent defender: attacker_net vs recurrent_defender
# ---------------------------------------------------------
print("\n--- Row 3: + recurrent defender [attacker_net vs recurrent_defender.pt] ---")
row3_gsr, row3_ir, row3_hmid, row3_steps = [], [], [], []

env_rec = DeceptionGridWorld(EnvConfig(include_time_in_obs=True, detect_swaps=True))

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    ep_gsr, ep_ir, ep_hmid, ep_steps = [], [], [], []
    for ep in range(NUM_EPISODES):
        theta = ep % 2
        obs_dict, info = env_rec.reset(seed=seed * 1000 + ep, options={"theta": theta})
        att_obs = obs_dict["attacker"]
        def_obs = obs_dict["defender"]
        def_h = None
        
        midfield_entropies = []
        step_count = 0
        
        for step in range(env_rec.cfg.horizon):
            step_count += 1
            att_pos = (float(att_obs[0] * 10.0), float(att_obs[1] * 10.0))
            
            # Recurrent defender step
            def_obs_t = torch.as_tensor(def_obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                act_t, _, _, _, b_logits, probs_t, def_h = rec_def.forward_step(def_obs_t, def_h, deterministic=True)
            def_act = int(act_t.item())
            probs = probs_t.squeeze(0).cpu().numpy()
            ent = shannon_entropy(probs)
            
            if 4.0 <= att_pos[0] <= 6.0:
                midfield_entropies.append(ent)
                
            # Attacker network step (att_obs has dim 3: [att_x, att_y, theta])
            att_obs_t = torch.as_tensor(att_obs[:3], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                att_logits, _ = att_net(att_obs_t)
                att_act = int(torch.argmax(att_logits, dim=-1).item())
                
            next_obs, rew, term, trunc, sinfo = env_rec.step(att_act, def_act)
            att_obs = next_obs["attacker"]
            def_obs = next_obs["defender"]
            
            if term or trunc:
                is_goal = bool(sinfo.get("goal_reached", False))
                is_intercept = bool(sinfo.get("intercepted", False))
                ep_gsr.append(1.0 if is_goal else 0.0)
                ep_ir.append(1.0 if is_intercept else 0.0)
                ep_steps.append(step_count)
                ep_hmid.append(np.mean(midfield_entropies) if midfield_entropies else 0.0)
                break
                
    row3_gsr.append(np.mean(ep_gsr) * 100.0)
    row3_ir.append(np.mean(ep_ir) * 100.0)
    row3_hmid.append(np.mean(ep_hmid))
    row3_steps.append(np.mean(ep_steps))
    print(f"  Seed {seed:4d} | GSR: {row3_gsr[-1]:5.1f}% | IR: {row3_ir[-1]:5.1f}% | H_mid: {row3_hmid[-1]:.4f} | Steps: {row3_steps[-1]:.2f}")

print(f"Row 3 Mean +- Std: GSR={np.mean(row3_gsr):.2f} +- {np.std(row3_gsr):.2f}% | IR={np.mean(row3_ir):.2f} +- {np.std(row3_ir):.2f}% | H_mid={np.mean(row3_hmid):.4f} +- {np.std(row3_hmid):.4f} | Steps={np.mean(row3_steps):.2f} +- {np.std(row3_steps):.2f}")

# ---------------------------------------------------------
# Row 1: Naive (lambda_0 = 0)
# ---------------------------------------------------------
print("\n--- Row 1: Naive (lambda_0 = 0) ---")
print("From logs/benchmark_results.csv:")
print("  GSR: 0.00% | IR: 100.00% | H_mid: 0.0000 bits | Steps: 6.00")
