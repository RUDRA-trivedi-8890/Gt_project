"""Modular Proximal Policy Optimization (PPO) implementation.

Includes a GAE-Lambda rollout storage buffer and the PPO update step with
surrogate clipping and value-function clipping.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.networks import ActorCritic


class RolloutBuffer:
    """Storage for a single agent's on-policy rollout transitions."""

    def __init__(self) -> None:
        self.obs: list = []
        self.actions: list = []
        self.rewards: list = []
        self.dones: list = []
        self.logps: list = []
        self.values: list = []  # old value estimates (for VF clipping)
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.logps.clear()
        self.values.clear()
        self.advantages = None
        self.returns = None

    def __len__(self) -> int:
        return len(self.rewards)


class PPOAgent:
    """Proximal Policy Optimization agent wrapping an ActorCritic network."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        ppo_cfg,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.cfg = ppo_cfg
        self.policy = ActorCritic(obs_dim, action_dim, ppo_cfg.hidden_dim).to(
            self.device
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=ppo_cfg.lr)
        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def select_action(self, obs) -> dict:
        """Select an action given a raw observation (numpy or scalar)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        action, logp, _, value = self.policy.get_action_and_value(obs_t)
        return {
            "action": int(action.cpu().numpy()[0]),
            "logp": float(logp.cpu().numpy()[0]),
            "value": float(value.cpu().numpy()[0]),
        }

    # ------------------------------------------------------------------ #
    # Rollout storage
    # ------------------------------------------------------------------ #
    def store(
        self,
        obs,
        action: int,
        reward: float,
        done: bool,
        logp: float,
        value: float,
    ) -> None:
        self.buffer.obs.append(np.asarray(obs, dtype=np.float32))
        self.buffer.actions.append(int(action))
        self.buffer.rewards.append(float(reward))
        self.buffer.dones.append(bool(done))
        self.buffer.logps.append(float(logp))
        self.buffer.values.append(float(value))

    # ------------------------------------------------------------------ #
    # GAE-Lambda advantage estimation
    # ------------------------------------------------------------------ #
    def compute_gae_and_returns(self, next_value: float = 0.0) -> None:
        """Compute Generalised Advantage Estimation and returns.

        Handles episode terminations via the done-mask so that bootstrapping
        does not cross episode boundaries.
        """
        rewards = np.asarray(self.buffer.rewards, dtype=np.float32)
        dones = np.asarray(self.buffer.dones, dtype=np.float32)
        values = np.asarray(self.buffer.values, dtype=np.float32)
        gamma = self.cfg.gamma
        lam = self.cfg.gae_lambda
        T = len(rewards)

        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_val = values[t + 1]
            delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.buffer.advantages = advantages
        self.buffer.returns = advantages + values

    # ------------------------------------------------------------------ #
    # PPO update
    # ------------------------------------------------------------------ #
    def update(self, next_value: float = 0.0) -> dict:
        """Perform a PPO update over the collected rollout.

        Returns a dict of training diagnostics (averaged across epochs).
        """
        self.compute_gae_and_returns(next_value)

        obs = torch.as_tensor(
            np.stack(self.buffer.obs), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            np.asarray(self.buffer.actions, dtype=np.int64), device=self.device
        )
        old_logps = torch.as_tensor(
            np.asarray(self.buffer.logps, dtype=np.float32), device=self.device
        )
        old_values = torch.as_tensor(
            np.asarray(self.buffer.values, dtype=np.float32), device=self.device
        )
        advantages = torch.as_tensor(
            np.asarray(self.buffer.advantages, dtype=np.float32),
            device=self.device,
        )
        returns = torch.as_tensor(
            np.asarray(self.buffer.returns, dtype=np.float32), device=self.device
        )

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        clip_coef = self.cfg.clip_coef
        ent_coef = self.cfg.ent_coef
        vf_coef = self.cfg.vf_coef
        max_grad_norm = self.cfg.max_grad_norm

        n = len(obs)
        idx = np.arange(n)

        # Running diagnostics
        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        num_updates = 0

        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.cfg.batch_size):
                end = min(start + self.cfg.batch_size, n)
                mb = idx[start:end]

                mb_obs = obs[mb]
                mb_actions = actions[mb]
                mb_old_logps = old_logps[mb]
                mb_old_values = old_values[mb]
                mb_adv = advantages[mb]
                mb_ret = returns[mb]

                _, logp, entropy, value = self.policy.get_action_and_value(
                    mb_obs, mb_actions
                )

                # Surrogate objective with clipping
                ratio = torch.exp(logp - mb_old_logps)
                surr1 = ratio * mb_adv
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_adv
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping
                v_pred_clipped = mb_old_values + (value - mb_old_values).clamp(
                    -clip_coef, clip_coef
                )
                v_loss_unclipped = (value - mb_ret).pow(2)
                v_loss_clipped = (v_pred_clipped - mb_ret).pow(2)
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = policy_loss + vf_coef * v_loss - ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.optimizer.step()

                last_policy_loss = float(policy_loss.detach())
                last_value_loss = float(v_loss.detach())
                last_entropy = float(entropy.mean().detach())
                num_updates += 1

        self.buffer.clear()

        return {
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "entropy": last_entropy,
            "num_updates": num_updates,
        }
