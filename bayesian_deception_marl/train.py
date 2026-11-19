"""Main Training Script: Asymmetric Markov Signaling Game with Bayesian Deception.

Game Theory Context:
--------------------
- Attacker (Agent 1): Has private information theta in {0, 1} designating Goal A or Goal B.
  A naive RL policy takes the shortest path, inadvertently leaking theta to the Defender early.
- Defender (Agent 2): Observes physical coordinates but cannot observe theta directly.
  Maintains a GRU-based Belief Head predicting P(theta | trajectory).
- Mechanism: We compute the Shannon Entropy H(b) of the Defender's belief:
      H(b) = - sum_i b_i * log2(b_i + 1e-8)
  When the Defender is completely uncertain (50/50 split), H(b) = 1.0 (maximum deception).
  When the Defender knows the target, H(b) = 0.0.
  By augmenting the Attacker's reward with lambda * H(b), the Attacker is incentivized
  to generate deceptive trajectories that delay belief revelation.
"""

import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Ensure current package can be resolved regardless of execution path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import EnvConfig
from envs.grid_world import DeceptionGridWorld as AsymmetricGridEnv
from models.networks import get_attacker_network, get_defender_network, ActorCritic
from models.belief_head import BeliefHead


# ============================================================================
# HELPER: ADVANTAGE ESTIMATION (GAE-Lambda)
# ============================================================================
def compute_gae_and_returns(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE) and discounted returns."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_non_terminal = 1.0 - float(dones[t])
        next_val = 0.0 if (t == T - 1 or dones[t]) else values[t + 1]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


# ============================================================================
# HELPER: PPO ACTOR-CRITIC UPDATE
# ============================================================================
def ppo_update(
    policy: ActorCritic,
    optimizer: optim.Optimizer,
    obs_batch: torch.Tensor,
    actions_batch: torch.Tensor,
    old_logps_batch: torch.Tensor,
    old_values_batch: torch.Tensor,
    advantages_batch: torch.Tensor,
    returns_batch: torch.Tensor,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.03,
    ppo_epochs: int = 4,
    batch_size: int = 64,
    max_grad_norm: float = 0.5,
) -> Dict[str, float]:
    """Perform standard clipped PPO updates for Actor-Critic."""
    # Normalize advantages for variance reduction
    adv_mean = advantages_batch.mean()
    adv_std = advantages_batch.std() + 1e-8
    norm_advantages = (advantages_batch - adv_mean) / adv_std

    dataset_size = obs_batch.shape[0]
    indices = np.arange(dataset_size)

    total_pi_loss = 0.0
    total_vf_loss = 0.0
    total_ent = 0.0
    updates_count = 0

    for _ in range(ppo_epochs):
        np.random.shuffle(indices)
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            mb_idx = indices[start_idx:end_idx]

            mb_obs = obs_batch[mb_idx]
            mb_actions = actions_batch[mb_idx]
            mb_old_logps = old_logps_batch[mb_idx]
            mb_old_values = old_values_batch[mb_idx]
            mb_adv = norm_advantages[mb_idx]
            mb_ret = returns_batch[mb_idx]

            # Re-evaluate policy and value under current parameters
            _, new_logps, entropy, values = policy.get_action_and_value(mb_obs, mb_actions)

            # 1. Clipped Surrogate Policy Objective
            ratio = torch.exp(new_logps - mb_old_logps)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # 2. Value Function Loss (with clipping for stability)
            val_clipped = mb_old_values + (values - mb_old_values).clamp(-clip_eps, clip_eps)
            vf_loss_unclipped = (values - mb_ret).pow(2)
            vf_loss_clipped = (val_clipped - mb_ret).pow(2)
            value_loss = 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped).mean()

            # 3. Policy Entropy Bonus (encourages exploration)
            entropy_loss = -entropy.mean()

            # Combined loss
            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            total_pi_loss += policy_loss.item()
            total_vf_loss += value_loss.item()
            total_ent += entropy.mean().item()
            updates_count += 1

    return {
        "policy_loss": total_pi_loss / max(1, updates_count),
        "value_loss": total_vf_loss / max(1, updates_count),
        "entropy": total_ent / max(1, updates_count),
    }


# ============================================================================
# MAIN TRAINING ENGINE
# ============================================================================
def train() -> None:
    # ------------------------------------------------------------------------
    # 1. Configurations & Hardware Setup
    # ------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Initializing Training on Device: {device}")

    env_config = EnvConfig()
    env = AsymmetricGridEnv(env_config)

    # Hyperparameters
    NUM_EPOCHS = 160            # Training iterations
    EPISODES_PER_EPOCH = 32     # Rollout episodes per epoch (16 theta=0, 16 theta=1)
    LR_ACTOR = 3e-4             # Learning rate for Attacker & Defender policies
    LR_BELIEF = 1e-3            # Learning rate for Defender's auxiliary Belief Head
    LAMBDA_0 = 1.0              # Entropy bonus in pooling corridor (makes Col 5 > diagonal early on)
    CUTOFF_ROW = 7.0            # Exact row threshold: above row 7, lambda drops to 0 immediately
    GAMMA = 0.99                # Discount factor
    GAE_LAMBDA = 0.95           # GAE smoothing factor
    CLIP_EPS = 0.2              # PPO surrogate clip ratio
    PPO_EPOCHS = 4              # Epochs per PPO update
    BATCH_SIZE = 64             # Mini-batch size for PPO updates

    # Strengthen goal incentive vs stalling
    env_config.goal_reward = 35.0
    env_config.c_step = 0.2

    # ------------------------------------------------------------------------
    # 2. Network Instantiations & Optimizers
    # ------------------------------------------------------------------------
    # Attacker AC: obs_dim = 3 [att_x, att_y, theta], action_dim = 9 (includes diagonals)
    attacker_net = get_attacker_network(hidden_dim=64, action_dim=9).to(device)

    # Defender AC: obs_dim = 6 [att_x, att_y, def_x, def_y, belief_A, belief_B], action_dim = 9
    defender_net = get_defender_network(hidden_dim=64, action_dim=9).to(device)

    # Defender Belief Head: Recurrent GRU predicting theta from sequence of physical observations
    # Ingests Defender's 4D observation: [att_x, att_y, def_x, def_y]
    belief_head = BeliefHead(input_dim=4, hidden_dim=64).to(device)

    # Optimizers
    optimizer_attacker = optim.Adam(attacker_net.parameters(), lr=LR_ACTOR)
    optimizer_defender = optim.Adam(defender_net.parameters(), lr=LR_ACTOR)
    optimizer_belief = optim.Adam(belief_head.parameters(), lr=LR_BELIEF)

    bce_loss_fn = nn.BCEWithLogitsLoss()

    # Destination directory for model weights
    save_dir = os.path.join(CURRENT_DIR, "models")
    os.makedirs(save_dir, exist_ok=True)

    print("[Train] Networks and optimizers successfully initialized.")
    print(f"[Train] Starting self-play for {NUM_EPOCHS} epochs ({EPISODES_PER_EPOCH} episodes/epoch)...\n")

    # ------------------------------------------------------------------------
    # 3. Epoch Loop
    # ------------------------------------------------------------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        # Storage buffers for Attacker rollout
        att_obs_list: List[np.ndarray] = []
        att_act_list: List[int] = []
        att_logp_list: List[float] = []
        att_val_list: List[float] = []
        att_rew_list: List[float] = []
        att_done_list: List[bool] = []

        # Storage buffers for Defender rollout
        def_obs_list: List[np.ndarray] = []
        def_act_list: List[int] = []
        def_logp_list: List[float] = []
        def_val_list: List[float] = []
        def_rew_list: List[float] = []
        def_done_list: List[bool] = []

        # Episode-level buffers for the Belief Head (supervised update)
        episode_trajectories: List[List[np.ndarray]] = []
        episode_thetas: List[int] = []

        # Diagnostic trackers for this epoch
        epoch_raw_att_rewards: List[float] = []
        epoch_custom_att_rewards: List[float] = []
        epoch_def_rewards: List[float] = []
        epoch_entropies: List[float] = []
        epoch_goal_reached: int = 0
        epoch_intercepted: int = 0

        # --------------------------------------------------------------------
        # Rollout Collection (Custom Game Theory Logic)
        # --------------------------------------------------------------------
        for ep_i in range(EPISODES_PER_EPOCH):
            obs_dict, info = env.reset(options={"theta": ep_i % 2})
            theta = info["theta"]

            att_obs = obs_dict["attacker"]   # shape: (3,) -> [att_x, att_y, theta]
            def_obs = obs_dict["defender"]   # shape: (4,) -> [att_x, att_y, def_x, def_y]

            # Hidden state for the Defender's recurrent Belief Head
            def_hidden: torch.Tensor | None = None
            def_trajectory_raw: List[np.ndarray] = []

            ep_raw_att_rew = 0.0
            ep_custom_att_rew = 0.0
            ep_def_rew = 0.0
            step_idx = 0

            while True:
                def_trajectory_raw.append(def_obs.copy())

                # EXACT POOLING-TO-BIFURCATION SCHEDULE:
                # Corridor (Row < 6.5): High lambda (1.4) makes pooling on Col 5 (+2.4) beat early diagonal (+1.41)
                # Sector (Row >= 6.5): Lambda drops strictly to 0. Proximity penalty near (9,5) plus diagonal
                # progress shaping (+1.41 vs +0.4) forces smooth, symmetric diagonal bifurcation at row 6.5-7.
                att_row = float(att_obs[0] * 10.0)
                if att_row >= CUTOFF_ROW:
                    current_lambda = 0.0
                else:
                    current_lambda = LAMBDA_0

                # Step 1: Predict Defender's belief over theta using Belief Head
                def_obs_t = torch.as_tensor(def_obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    belief_logits, def_hidden = belief_head.step(def_obs_t, def_hidden)
                    probs = F.softmax(belief_logits, dim=-1)  # shape: (1, 2) -> [P(theta=0), P(theta=1)]

                # Step 2: Calculate Shannon Entropy H(b) of Defender's belief
                # Formula: H(b) = - sum(p * log2(p + eps))
                # Bounds: H in [0, 1]. H=1 when belief is uniform (50/50), H=0 when fully known.
                belief_entropy = -torch.sum(probs * torch.log2(probs + 1e-8), dim=-1).item()
                epoch_entropies.append(belief_entropy)

                # Step 3: Construct Defender AC input: [att_x, att_y, def_x, def_y, belief_0, belief_1] (dim=6)
                probs_np = probs.squeeze(0).cpu().numpy()
                def_obs_full = np.concatenate([def_obs, probs_np], axis=-1).astype(np.float32)

                # Step 4: Sample actions from Attacker and Defender policies
                att_obs_t = torch.as_tensor(att_obs, dtype=torch.float32, device=device).unsqueeze(0)
                def_obs_full_t = torch.as_tensor(def_obs_full, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    att_act, att_logp, _, att_val = attacker_net.get_action_and_value(att_obs_t)
                    def_act, def_logp, _, def_val = defender_net.get_action_and_value(def_obs_full_t)

                action_att = int(att_act.item())
                action_def = int(def_act.item())

                # Step 5: Execute actions in the asymmetric gridworld
                next_obs_dict, env_rewards, terminated, truncated, step_info = env.step(action_att, action_def)
                done = terminated or truncated

                env_att_rew = env_rewards["attacker"]
                env_def_rew = env_rewards["defender"]

                # ============================================================
                # GAME THEORY REWARD SHAPING (ANTI-HACKING & TIME-DECAYING):
                # 1. Gated on forward progress (anti-stalling / anti-idling)
                # 2. Scaled by time-decaying lambda(t) so entropy farming stops
                #    before the endgame, guaranteeing bifurcation toward the goal.
                # ============================================================
                if step_info.get("forward_progress", False) and current_lambda > 0.0 and int(round(att_obs[1] * 10.0)) == 5:
                    custom_att_rew = env_att_rew + (current_lambda * belief_entropy)
                else:
                    custom_att_rew = env_att_rew

                step_idx += 1

                # Store transitions for Attacker PPO
                att_obs_list.append(att_obs)
                att_act_list.append(action_att)
                att_logp_list.append(float(att_logp.item()))
                att_val_list.append(float(att_val.item()))
                att_rew_list.append(custom_att_rew)
                att_done_list.append(done)

                # Store transitions for Defender PPO (Defender receives standard env reward)
                def_obs_list.append(def_obs_full)
                def_act_list.append(action_def)
                def_logp_list.append(float(def_logp.item()))
                def_val_list.append(float(def_val.item()))
                def_rew_list.append(env_def_rew)
                def_done_list.append(done)

                ep_raw_att_rew += env_att_rew
                ep_custom_att_rew += custom_att_rew
                ep_def_rew += env_def_rew

                if done:
                    if step_info.get("goal_reached", False):
                        epoch_goal_reached += 1
                    if step_info.get("intercepted", False):
                        epoch_intercepted += 1
                    break

                att_obs = next_obs_dict["attacker"]
                def_obs = next_obs_dict["defender"]

            episode_trajectories.append(def_trajectory_raw)
            episode_thetas.append(theta)

            epoch_raw_att_rewards.append(ep_raw_att_rew)
            epoch_custom_att_rewards.append(ep_custom_att_rew)
            epoch_def_rewards.append(ep_def_rew)

        # --------------------------------------------------------------------
        # 4. Advantage & Return Computation (GAE)
        # --------------------------------------------------------------------
        att_adv, att_ret = compute_gae_and_returns(
            att_rew_list, att_val_list, att_done_list, gamma=GAMMA, gae_lambda=GAE_LAMBDA
        )
        def_adv, def_ret = compute_gae_and_returns(
            def_rew_list, def_val_list, def_done_list, gamma=GAMMA, gae_lambda=GAE_LAMBDA
        )

        # --------------------------------------------------------------------
        # 5. PPO Policy & Value Updates (Attacker & Defender)
        # --------------------------------------------------------------------
        # Attacker update
        att_metrics = ppo_update(
            policy=attacker_net,
            optimizer=optimizer_attacker,
            obs_batch=torch.as_tensor(np.stack(att_obs_list), dtype=torch.float32, device=device),
            actions_batch=torch.as_tensor(np.array(att_act_list), dtype=torch.int64, device=device),
            old_logps_batch=torch.as_tensor(np.array(att_logp_list), dtype=torch.float32, device=device),
            old_values_batch=torch.as_tensor(np.array(att_val_list), dtype=torch.float32, device=device),
            advantages_batch=torch.as_tensor(att_adv, dtype=torch.float32, device=device),
            returns_batch=torch.as_tensor(att_ret, dtype=torch.float32, device=device),
            clip_eps=CLIP_EPS,
            ent_coef=0.05,
            ppo_epochs=PPO_EPOCHS,
            batch_size=BATCH_SIZE,
        )

        # Defender update
        def_metrics = ppo_update(
            policy=defender_net,
            optimizer=optimizer_defender,
            obs_batch=torch.as_tensor(np.stack(def_obs_list), dtype=torch.float32, device=device),
            actions_batch=torch.as_tensor(np.array(def_act_list), dtype=torch.int64, device=device),
            old_logps_batch=torch.as_tensor(np.array(def_logp_list), dtype=torch.float32, device=device),
            old_values_batch=torch.as_tensor(np.array(def_val_list), dtype=torch.float32, device=device),
            advantages_batch=torch.as_tensor(def_adv, dtype=torch.float32, device=device),
            returns_batch=torch.as_tensor(def_ret, dtype=torch.float32, device=device),
            clip_eps=CLIP_EPS,
            ent_coef=0.05,
            ppo_epochs=PPO_EPOCHS,
            batch_size=BATCH_SIZE,
        )

        # --------------------------------------------------------------------
        # 6. Defender Belief Head Update (Supervised BCE on True Theta)
        # --------------------------------------------------------------------
        # The Defender acts as a Bayesian observer. Given the physical trajectory
        # of the Attacker, the GRU predicts log-odds: logit(theta=1) - logit(theta=0).
        # We optimize using BCEWithLogitsLoss against the ground-truth secret theta.
        belief_head.train()
        total_belief_loss = 0.0

        optimizer_belief.zero_grad()
        for traj, true_theta in zip(episode_trajectories, episode_thetas):
            seq_t = torch.as_tensor(np.stack(traj), dtype=torch.float32, device=device).unsqueeze(0)
            seq_len = seq_t.shape[1]

            logits_seq, _ = belief_head(seq_t)  # shape: (1, seq_len, 2)
            # Binary logit is the difference between class 1 and class 0 logits (log-odds)
            binary_logits = (logits_seq[:, :, 1] - logits_seq[:, :, 0]).squeeze(0)  # shape: (seq_len,)
            # Game-Theoretic Bayesian Target:
            # In the pooling corridor (Column 5, Row <= 7), trajectories are indistinguishable.
            # Information leakage is zero, so the theoretical Bayesian posterior is prior (0.5).
            # Once branched (Column != 5 or Row > 7), physical position separates theta (0 or 1).
            targets = []
            for obs_t in traj:
                att_row = float(obs_t[0] * 10.0)
                att_col = float(obs_t[1] * 10.0)
                if round(att_col) == 5 and att_row <= 7.0:
                    targets.append(0.5)
                else:
                    targets.append(float(true_theta))
            targets_tensor = torch.as_tensor(targets, dtype=torch.float32, device=device)

            bce_loss = bce_loss_fn(binary_logits, targets_tensor)
            bce_loss.backward()
            total_belief_loss += bce_loss.item()

        nn.utils.clip_grad_norm_(belief_head.parameters(), max_norm=0.5)
        optimizer_belief.step()
        avg_belief_loss = total_belief_loss / max(1, len(episode_trajectories))

        # --------------------------------------------------------------------
        # 7. Logging & Diagnostics
        # --------------------------------------------------------------------
        avg_custom_att_rew = float(np.mean(epoch_custom_att_rewards))
        avg_raw_att_rew = float(np.mean(epoch_raw_att_rewards))
        avg_def_rew = float(np.mean(epoch_def_rewards))
        avg_entropy = float(np.mean(epoch_entropies))
        goal_rate = epoch_goal_reached / EPISODES_PER_EPOCH
        intercept_rate = epoch_intercepted / EPISODES_PER_EPOCH

        print(
            f"Epoch [{epoch:3d}/{NUM_EPOCHS:3d}] | "
            f"Att Rew (Shaped): {avg_custom_att_rew:6.2f} (Raw: {avg_raw_att_rew:5.2f}) | "
            f"Def Rew: {avg_def_rew:6.2f} | "
            f"Belief Entropy: {avg_entropy:5.3f} bits | "
            f"Belief BCE: {avg_belief_loss:5.4f} | "
            f"Goal: {goal_rate*100:4.1f}% | "
            f"Intercept: {intercept_rate*100:4.1f}%"
        )

    # ------------------------------------------------------------------------
    # 8. Model Checkpointing
    # ------------------------------------------------------------------------
    att_path = os.path.join(save_dir, "attacker_net.pt")
    def_path = os.path.join(save_dir, "defender_net.pt")
    belief_path = os.path.join(save_dir, "belief_head.pt")
    rec_def_path = os.path.join(save_dir, "recurrent_defender.pt")

    torch.save(attacker_net.state_dict(), att_path)
    torch.save(defender_net.state_dict(), def_path)
    torch.save(belief_head.state_dict(), belief_path)

    print("\n[Train] Baseline training complete!")
    print(f"[Train] Model weights saved successfully:")
    print(f"  - Attacker:    {att_path}")
    print(f"  - Defender:    {def_path}")
    print(f"  - Belief Head: {belief_path}")

    # ------------------------------------------------------------------------
    # 9. Train Upgraded Recurrent Defender with Sequential Belief Distillation
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Train] Training Upgraded Recurrent Defender with Exact Bayes Distillation...")
    print("=" * 70)
    try:
        from train_recurrent_defender import train_recurrent_defender
        train_recurrent_defender(save_path=rec_def_path, num_episodes=600, device=str(device))
        print(f"  - Recurrent Defender: {rec_def_path}")
    except Exception as exc:
        print(f"[Train] Warning: Could not train recurrent defender: {exc}")


if __name__ == "__main__":
    train()
