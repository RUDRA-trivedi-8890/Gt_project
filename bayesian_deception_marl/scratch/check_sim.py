import sys, os
sys.path.insert(0, "Gt_project/bayesian_deception_marl")
import torch, numpy as np
from config import EnvConfig
from envs.grid_world import DeceptionGridWorld
from envs.trick_attackers import StraightToKAttacker
from models.recurrent_defender import RecurrentDefender

dev = torch.device("cpu")
model = RecurrentDefender(obs_dim=5, action_dim=9, hidden_dim=64, belief_temperature=1.0).to(dev)
model.load_state_dict(torch.load("Gt_project/bayesian_deception_marl/models/recurrent_defender.pt", map_location=dev))
model.eval()

env = DeceptionGridWorld(EnvConfig(include_time_in_obs=True, detect_swaps=True))
trick = StraightToKAttacker(k=7)
theta = 0

env.reset(options={"theta": theta})
trick.reset(theta=theta)

def_h = None
print("--- SIMULATING StraightToKAttacker(k=7) vs Upgraded Recurrent Defender ---")
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
