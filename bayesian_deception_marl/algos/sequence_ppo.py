"""Sequence-Aware PPO Engine for the Recurrent Defender.

Implements episode-level recurrent PPO updates without scrambling temporal sequences:
- Rollout buffers preserve strict sequential ordering per episode.
- Computes episode-level Generalized Advantage Estimation (GAE).
- Mini-batches process full padded episodes with boolean sequence masks and h_0 hidden state.
- Jointly updates policy, value function, and sequential belief head (with optional distillation from exact Bayes).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.recurrent_defender import RecurrentDefender


class SequenceRolloutBuffer:
    """Stores full-episode trajectory sequences for recurrent policy optimization."""

    def __init__(self, obs_dim: int = 5, device: str = "cpu") -> None:
        self.obs_dim = obs_dim
        self.device = torch.device(device)
        self.episodes: List[Dict[str, np.ndarray]] = []

    def add_episode(
        self,
        observations: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        values: List[float],
        logps: List[float],
        true_theta: int,
        exact_bayes_posteriors: Optional[List[np.ndarray]] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Process an episode, compute GAE advantages, and store."""
        T = len(actions)
        obs_arr = np.asarray(observations[:T], dtype=np.float32)
        act_arr = np.asarray(actions, dtype=np.int64)
        rew_arr = np.asarray(rewards, dtype=np.float32)
        val_arr = np.asarray(values, dtype=np.float32)
        logp_arr = np.asarray(logps, dtype=np.float32)

        # GAE-Lambda
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_val = 0.0 if t == T - 1 else val_arr[t + 1]
            delta = rew_arr[t] + gamma * next_val - val_arr[t]
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages[t] = last_gae

        returns = advantages + val_arr

        bayes_arr = (
            np.asarray(exact_bayes_posteriors[:T], dtype=np.float32)
            if exact_bayes_posteriors is not None
            else np.full((T, 2), 0.5, dtype=np.float32)
        )

        self.episodes.append({
            "obs": obs_arr,
            "actions": act_arr,
            "advantages": advantages,
            "returns": returns,
            "old_values": val_arr,
            "old_logps": logp_arr,
            "theta": int(true_theta),
            "bayes": bayes_arr,
            "length": T,
        })

    def get_padded_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collate all episodes into a padded batch with a boolean sequence mask."""
        batch_size = len(self.episodes)
        max_len = max(ep["length"] for ep in self.episodes)

        obs_pad = np.zeros((batch_size, max_len, self.obs_dim), dtype=np.float32)
        act_pad = np.zeros((batch_size, max_len), dtype=np.int64)
        adv_pad = np.zeros((batch_size, max_len), dtype=np.float32)
        ret_pad = np.zeros((batch_size, max_len), dtype=np.float32)
        val_pad = np.zeros((batch_size, max_len), dtype=np.float32)
        logp_pad = np.zeros((batch_size, max_len), dtype=np.float32)
        theta_vec = np.zeros(batch_size, dtype=np.int64)
        bayes_pad = np.zeros((batch_size, max_len, 2), dtype=np.float32)
        mask_pad = np.zeros((batch_size, max_len), dtype=np.bool_)

        for i, ep in enumerate(self.episodes):
            L = ep["length"]
            obs_pad[i, :L] = ep["obs"]
            act_pad[i, :L] = ep["actions"]
            adv_pad[i, :L] = ep["advantages"]
            ret_pad[i, :L] = ep["returns"]
            val_pad[i, :L] = ep["old_values"]
            logp_pad[i, :L] = ep["old_logps"]
            theta_vec[i] = ep["theta"]
            bayes_pad[i, :L] = ep["bayes"]
            mask_pad[i, :L] = True

        # Advantage normalization across unmasked elements
        valid_adv = adv_pad[mask_pad]
        if len(valid_adv) > 1 and valid_adv.std() > 1e-8:
            adv_pad[mask_pad] = (valid_adv - valid_adv.mean()) / (valid_adv.std() + 1e-8)

        t_obs = torch.as_tensor(obs_pad, device=self.device)
        t_act = torch.as_tensor(act_pad, device=self.device)
        t_adv = torch.as_tensor(adv_pad, device=self.device)
        t_ret = torch.as_tensor(ret_pad, device=self.device)
        t_val = torch.as_tensor(val_pad, device=self.device)
        t_logp = torch.as_tensor(logp_pad, device=self.device)
        t_theta = torch.as_tensor(theta_vec, device=self.device)
        t_bayes = torch.as_tensor(bayes_pad, device=self.device)
        t_mask = torch.as_tensor(mask_pad, device=self.device)

        return t_obs, t_act, t_adv, t_ret, t_val, t_logp, t_theta, t_bayes, t_mask

    def clear(self) -> None:
        self.episodes.clear()


class RecurrentPPOUpdater:
    """Updates RecurrentDefender parameters using sequence-aware PPO."""

    def __init__(
        self,
        model: RecurrentDefender,
        optimizer: optim.Optimizer,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        belief_coef: float = 1.0,
        distill_coef: float = 0.1,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.belief_coef = belief_coef
        self.distill_coef = distill_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs

    def update(
        self,
        buffer: SequenceRolloutBuffer,
        use_distill: bool = False,
    ) -> Dict[str, float]:
        """Perform sequence-aware PPO optimization across collected full episodes."""
        if len(buffer.episodes) == 0:
            return {}

        (
            obs_pad,
            act_pad,
            adv_pad,
            ret_pad,
            val_pad,
            logp_pad,
            theta_vec,
            bayes_pad,
            mask_pad,
        ) = buffer.get_padded_batch()

        batch_size, max_len, _ = obs_pad.shape
        num_valid = float(mask_pad.sum().item())

        total_pi_loss = 0.0
        total_vf_loss = 0.0
        total_ent_loss = 0.0
        total_belief_loss = 0.0
        total_distill_loss = 0.0

        for _ in range(self.ppo_epochs):
            logp, entropy, values, belief_logits, belief_probs = self.model.evaluate_sequence(obs_pad, act_pad)

            # 1. PPO Clipped Policy Surrogate
            ratio = torch.exp(logp - logp_pad)
            surr1 = ratio * adv_pad
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_pad
            pi_loss = -torch.min(surr1, surr2)
            pi_loss = (pi_loss * mask_pad).sum() / max(1.0, num_valid)

            # 2. Value Function Loss (with clipping)
            val_clipped = val_pad + (values - val_pad).clamp(-self.clip_eps, self.clip_eps)
            vf_loss_raw = (values - ret_pad).pow(2)
            vf_loss_clip = (val_clipped - ret_pad).pow(2)
            vf_loss = 0.5 * torch.max(vf_loss_raw, vf_loss_clip)
            vf_loss = (vf_loss * mask_pad).sum() / max(1.0, num_valid)

            # 3. Policy Entropy
            ent_loss = -(entropy * mask_pad).sum() / max(1.0, num_valid)

            # 4. Supervised Belief Loss (BCE against true theta across all steps)
            target_expanded = theta_vec.unsqueeze(1).expand(-1, max_len)
            targets_bce = target_expanded.float()
            binary_logits = belief_logits[:, :, 1] - belief_logits[:, :, 0]
            bce_loss_raw = F.binary_cross_entropy_with_logits(binary_logits, targets_bce, reduction="none")
            belief_loss = (bce_loss_raw * mask_pad).sum() / max(1.0, num_valid)

            # 5. Distillation Loss (KL to exact Bayes)
            distill_loss = torch.tensor(0.0, device=obs_pad.device)
            if use_distill and self.distill_coef > 0:
                p_bayes = bayes_pad.clamp(min=1e-6)
                log_p_bayes = torch.log(p_bayes)
                log_p_model = F.log_softmax(belief_logits, dim=-1)
                kl_pt = F.kl_div(log_p_model, p_bayes, reduction="none").sum(dim=-1)
                distill_loss = (kl_pt * mask_pad).sum() / max(1.0, num_valid)

            loss = (
                pi_loss
                + self.vf_coef * vf_loss
                + self.ent_coef * ent_loss
                + self.belief_coef * belief_loss
                + (self.distill_coef * distill_loss if use_distill else 0.0)
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_pi_loss += pi_loss.item()
            total_vf_loss += vf_loss.item()
            total_ent_loss += (-ent_loss).item()
            total_belief_loss += belief_loss.item()
            total_distill_loss += distill_loss.item()

        return {
            "pi_loss": total_pi_loss / self.ppo_epochs,
            "vf_loss": total_vf_loss / self.ppo_epochs,
            "entropy": total_ent_loss / self.ppo_epochs,
            "belief_loss": total_belief_loss / self.ppo_epochs,
            "distill_loss": total_distill_loss / self.ppo_epochs,
        }
