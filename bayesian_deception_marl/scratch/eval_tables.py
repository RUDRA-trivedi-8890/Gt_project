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
from envs.trick_attackers import (
    StraightToKAttacker,
    FeintAttacker,
    ZigzagAttacker,
    MidPoolDashAttacker,
)
from utils.metrics import compute_t_reveal, shannon_entropy

device = torch.device("cpu")

# Load models
models_dir = os.path.join(CURRENT_DIR, "models")
att_net = get_attacker_network(hidden_dim=64, action_dim=9).to(device)
def_net = get_defender_network(hidden_dim=64, action_dim=9).to(device)
bh_net = BeliefHead(input_dim=4, hidden_dim=64).to(device)

att_net.load_state_dict(torch.load(os.path.join(models_dir, "attacker_net.pt"), map_location=device))
def_net.load_state_dict(torch.load(os.path.join(models_dir, "defender_net.pt"), map_location=device))
bh_net.load_state_dict(torch.load(os.path.join(models_dir, "belief_head.pt"), map_location=device))

att_net.eval()
def_net.eval()
bh_net.eval()

rec_def = RecurrentDefender(obs_dim=5, action_dim=9, hidden_dim=64).to(device)
rec_def.load_state_dict(torch.load(os.path.join(models_dir, "recurrent_defender.pt"), map_location=device))
rec_def.eval()

print("Models loaded successfully.")

def eval_trick_suite_baseline_defender(num_episodes=100):
    tricks = [
        ("Straight to row 3 then turn", StraightToKAttacker(k=3)),
        ("Straight to row 7 then turn", StraightToKAttacker(k=7)),
        ("Feint then cut", FeintAttacker(feint_steps=4)),
        ("Zigzag", ZigzagAttacker()),
        ("Pool then dash", MidPoolDashAttacker()),
    ]
    
    env_cfg = EnvConfig(include_time_in_obs=False, detect_swaps=True)
    env = DeceptionGridWorld(env_cfg)
    
    print("\n" + "="*80)
    print("EVALUATING BASELINE DEFENDER (defender_net.pt + belief_head.pt) AGAINST TRICK SUITE")
    print("="*80)
    
    results = []
    for trick_name, trick_policy in tricks:
        intercepted_count = 0
        goal_count = 0
        false_alarm_count = 0
        false_alarm_anytime_count = 0
        t_reveal_list = []
        t_reveal_after_branch_list = []
        
        for ep in range(num_episodes):
            theta = ep % 2
            obs_dict, info = env.reset(seed=ep + 1000, options={"theta": theta})
            trick_policy.reset(theta=theta)
            
            def_obs = obs_dict["defender"]
            def_hidden = None
            beliefs = []
            
            intercepted = False
            goal_reached = False
            first_step_above_09 = None
            had_wrong_side = False
            
            for step in range(env_cfg.horizon):
                att_pos = env.attacker_pos.copy()
                def_pos = env.defender_pos.copy()
                att_act = trick_policy.get_action(att_pos, def_pos)
                
                # Defender belief step
                def_obs_t = torch.as_tensor(def_obs, dtype=torch.float32, device=device).unsqueeze(0)
                belief_logits, def_hidden = bh_net.step(def_obs_t, def_hidden)
                probs = F.softmax(belief_logits, dim=-1).squeeze(0).detach().cpu().numpy()
                beliefs.append(probs)
                
                if probs[theta] >= 0.90 and first_step_above_09 is None:
                    first_step_above_09 = step
                
                # Check commitment
                if env.defender_committed_side != 0:
                    if (env.defender_committed_side == -1 and theta == 1) or (env.defender_committed_side == 1 and theta == 0):
                        had_wrong_side = True
                
                # Defender action step
                def_obs_full = np.concatenate([def_obs, probs], axis=-1).astype(np.float32)
                def_obs_full_t = torch.as_tensor(def_obs_full, dtype=torch.float32, device=device).unsqueeze(0)
                def_logits, _ = def_net(def_obs_full_t)
                def_act = int(torch.argmax(def_logits, dim=-1).item())
                
                next_obs, rew, term, trunc, sinfo = env.step(att_act, def_act)
                def_obs = next_obs["defender"]
                
                if sinfo.get("intercepted", False):
                    intercepted = True
                if sinfo.get("goal_reached", False):
                    goal_reached = True
                
                if term or trunc:
                    break
            
            if intercepted:
                intercepted_count += 1
            if goal_reached:
                goal_count += 1
            
            # False alarm at goal reach
            if goal_reached and env.defender_committed_side != 0:
                if (env.defender_committed_side == -1 and theta == 1) or (env.defender_committed_side == 1 and theta == 0):
                    false_alarm_count += 1
            
            if had_wrong_side:
                false_alarm_anytime_count += 1
                
            t_rev = compute_t_reveal(beliefs, theta, threshold=0.9, t_max=env_cfg.horizon)
            t_reveal_list.append(t_rev)
            
            # If after branch:
            if getattr(trick_policy, "reveal_step", None) is not None:
                rev_step = trick_policy.reveal_step
                t_after = max(0, t_rev - rev_step) if first_step_above_09 is not None else 5.0
                t_reveal_after_branch_list.append(t_after)
            else:
                t_reveal_after_branch_list.append(5.0)

        ir = (intercepted_count / num_episodes) * 100.0
        fa = (false_alarm_count / num_episodes) * 100.0
        fa_any = (false_alarm_anytime_count / num_episodes) * 100.0
        mean_t_rev = float(np.mean(t_reveal_list))
        std_t_rev = float(np.std(t_reveal_list))
        mean_t_after = float(np.mean(t_reveal_after_branch_list))
        
        print(f"Trick: {trick_name:<28} | IR: {ir:5.1f}% | Time to belief>0.9: {mean_t_rev:4.2f} (std {std_t_rev:4.2f}) | FA (term): {fa:5.1f}% | FA (anytime): {fa_any:5.1f}%")
        results.append({
            "trick": trick_name,
            "ir": ir,
            "mean_t_rev": mean_t_rev,
            "std_t_rev": std_t_rev,
            "fa": fa,
            "fa_any": fa_any,
        })
    return results

eval_trick_suite_baseline_defender()
