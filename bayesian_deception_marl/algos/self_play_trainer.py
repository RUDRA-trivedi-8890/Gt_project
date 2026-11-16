"""Dual-agent co-evolutionary training engine with CTDE.

Centralised Training with Decentralised Execution (CTDE):

* The Defender's GRU belief head trains on rollout batches with supervised
  Binary Cross-Entropy (with label smoothing) against the true private type theta.
* The Attacker policy learns via PPO with an intrinsic Shannon-entropy
  bonus estimated through its internal Theory-of-Mind (ToM) module. The
  entropy bonus is gated on forward progress so the Attacker cannot farm
  deception reward by idling at spawn.
* Neither agent reads the other's internal neural activations; the Attacker
  simulates the Defender's belief locally via its ToM.

Fixes vs. reward hacking & asymmetric reasoning collapse:
  * Entropy bonus only awarded on steps with forward progress (x_pos > running max).
  * Label-smoothed BCE targets prevent the belief head from snapping to 0/1.
  * Temperature-scaled softmax keeps early screening beliefs diffuse.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs.grid_world import DeceptionGridWorld
from models.belief_head import BeliefHead
from algos.ppo_agent import PPOAgent
from config import EnvConfig, PPOConfig, DeceptionConfig
from utils.metrics import shannon_entropy


class SelfPlayTrainer:
    """Co-evolutionary trainer for the asymmetric deception game."""

    def __init__(
        self,
        env_cfg: EnvConfig,
        ppo_cfg: PPOConfig,
        dec_cfg: DeceptionConfig,
        device: str = "cpu",
    ) -> None:
        self.env_cfg = env_cfg
        self.ppo_cfg = ppo_cfg
        self.dec_cfg = dec_cfg
        self.device = torch.device(device)

        self.env = DeceptionGridWorld(env_cfg)

        # ------------------------------------------------------------------ #
        # Agents (Actor-Critic PPO)
        # ------------------------------------------------------------------ #
        self.attacker_agent = PPOAgent(
            obs_dim=3, action_dim=5, ppo_cfg=ppo_cfg, device=device
        )
        self.defender_agent = PPOAgent(
            obs_dim=4, action_dim=5, ppo_cfg=ppo_cfg, device=device
        )

        # ------------------------------------------------------------------ #
        # Belief heads
        #   Defender BeliefHead  : input_dim=4 (full defender observation)
        #   Attacker ToM         : input_dim=2 (attacker position only;
        #                          simulates an observer without theta)
        # ------------------------------------------------------------------ #
        self.defender_belief = BeliefHead(
            input_dim=4, hidden_dim=ppo_cfg.hidden_dim
        ).to(self.device)
        self.belief_optimizer = optim.Adam(
            self.defender_belief.parameters(), lr=dec_cfg.belief_lr
        )

        self.attacker_tom = BeliefHead(
            input_dim=2, hidden_dim=ppo_cfg.hidden_dim
        ).to(self.device)
        self.tom_optimizer = optim.Adam(
            self.attacker_tom.parameters(), lr=dec_cfg.tom_lr
        )

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.global_step: int = 0

    # ------------------------------------------------------------------ #
    # Entropy weight schedule
    # ------------------------------------------------------------------ #
    def _lambda_t(self, t: int) -> float:
        """Linearly decaying entropy weight: lambda_0 * max(0, 1 - t / T_max)."""
        return self.dec_cfg.lambda_0 * max(0.0, 1.0 - t / self.dec_cfg.t_max)

    # ------------------------------------------------------------------ #
    # Belief-head supervised training (BCE against true theta)
    # ------------------------------------------------------------------ #
    def _train_belief_on_sequence(
        self,
        belief_net: BeliefHead,
        optimizer: optim.Optimizer,
        sequence: List[np.ndarray],
        theta: int,
    ) -> float:
        """Train a belief network to predict theta from an observation sequence.

        The binary logit is (logit[class=1] - logit[class=0]), which under a
        2-class softmax equals the log-odds of P(theta=1). BCEWithLogitsLoss
        on this logit is exactly
            -E[ theta*log(b(1)) + (1-theta)*log(b(0)) ].

        Label smoothing (dec_cfg.label_smoothing) softens targets from
        {0,1} to {smooth, 1-smooth}, preventing overconfident snap beliefs.
        """
        seq_t = torch.as_tensor(
            np.stack(sequence), dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, seq_len, input_dim)
        seq_len = seq_t.shape[1]

        # Label smoothing: hard targets {0,1} -> softened {smooth, 1-smooth}
        smooth = self.dec_cfg.label_smoothing
        target_value = 1.0 - smooth if theta == 1 else smooth
        targets = torch.full((seq_len,), target_value, device=self.device)

        logits, _ = belief_net.forward(seq_t)  # (1, seq_len, 2)
        binary_logits = logits[:, :, 1] - logits[:, :, 0]  # (1, seq_len)
        binary_logits = binary_logits.squeeze(0)  # (seq_len,)

        loss = self.bce_loss(binary_logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu().numpy())

    # ------------------------------------------------------------------ #
    # Single training episode (with network updates)
    # ------------------------------------------------------------------ #
    def train_episode(self) -> Dict[str, float]:
        """Run one episode, collect rollouts, update all networks.

        Returns a dict of episode diagnostics.
        """
        obs_dict, info = self.env.reset()
        att_obs = obs_dict["attacker"]
        def_obs = obs_dict["defender"]
        theta = info["theta"]

        # Recurrent hidden states (reset each episode)
        def_hidden: Optional[torch.Tensor] = None
        tom_hidden: Optional[torch.Tensor] = None

        # Observation sequences for belief-head training
        def_seq: List[np.ndarray] = [def_obs.copy()]
        att_pos_seq: List[np.ndarray] = [att_obs[:2].copy()]

        # Per-step logging
        step_entropies: List[float] = []
        step_dists: List[float] = []
        total_att_reward = 0.0

        terminated = False
        truncated = False
        final_att_obs = att_obs
        final_def_obs = def_obs

        for t in range(self.env_cfg.horizon):
            lam = self._lambda_t(t)

            # ---------------- Attacker ---------------- #
            att_out = self.attacker_agent.select_action(att_obs)
            att_action = att_out["action"]
            att_logp = att_out["logp"]
            att_value = att_out["value"]

            # Attacker ToM belief -> entropy bonus (gated on forward progress)
            att_pos = att_obs[:2].copy()
            att_pos_t = torch.as_tensor(
                att_pos, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            tom_logits, tom_hidden = self.attacker_tom.step(att_pos_t, tom_hidden)
            temperature = self.dec_cfg.belief_temperature
            tom_probs = torch.softmax(
                tom_logits / temperature, dim=-1
            ).squeeze(0)
            tom_entropy = shannon_entropy(tom_probs.detach().cpu().numpy())

            # ---------------- Defender ---------------- #
            def_out = self.defender_agent.select_action(def_obs)
            def_action = def_out["action"]
            def_logp = def_out["logp"]
            def_value = def_out["value"]

            # Defender belief (for supervised training)
            def_obs_t = torch.as_tensor(
                def_obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            _, def_hidden = self.defender_belief.step(def_obs_t, def_hidden)

            # ---------------- Environment ---------------- #
            next_obs_dict, rewards, terminated, truncated, info = self.env.step(
                att_action, def_action
            )
            next_att_obs = next_obs_dict["attacker"]
            next_def_obs = next_obs_dict["defender"]

            # Attacker reward: env reward (includes c_step + progress shaping)
            # + entropy bonus ONLY on forward-progress steps (anti-stalling)
            att_reward = rewards["attacker"]
            if info.get("forward_progress", False):
                att_reward += lam * tom_entropy
            def_reward = rewards["defender"]

            # Store PPO transitions
            self.attacker_agent.store(
                att_obs, att_action, att_reward, terminated, att_logp, att_value
            )
            self.defender_agent.store(
                def_obs, def_action, def_reward, terminated, def_logp, def_value
            )

            total_att_reward += att_reward

            # Logging
            dist_to_goal = float(info.get("dist_to_goal", 0.0))
            step_entropies.append(tom_entropy)
            step_dists.append(dist_to_goal)

            # Belief sequences
            def_seq.append(next_def_obs.copy())
            att_pos_seq.append(next_att_obs[:2].copy())

            # Advance
            att_obs = next_att_obs
            def_obs = next_def_obs
            final_att_obs = att_obs
            final_def_obs = def_obs
            self.global_step += 1

            if terminated or truncated:
                break

        # ---- Bootstrap value for GAE (0 if terminated) ----
        if terminated:
            att_next_value = 0.0
            def_next_value = 0.0
        else:
            with torch.no_grad():
                _, _, _, att_v = self.attacker_agent.policy.get_action_and_value(
                    torch.as_tensor(
                        final_att_obs, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                )
                _, _, _, def_v = self.defender_agent.policy.get_action_and_value(
                    torch.as_tensor(
                        final_def_obs, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                )
            att_next_value = float(att_v.cpu().numpy()[0])
            def_next_value = float(def_v.cpu().numpy()[0])

        # ---- PPO updates ----
        att_update = self.attacker_agent.update(att_next_value)
        def_update = self.defender_agent.update(def_next_value)

        # ---- Belief-head supervised updates (label-smoothed BCE) ----
        belief_loss = self._train_belief_on_sequence(
            self.defender_belief, self.belief_optimizer, def_seq, theta
        )
        tom_loss = self._train_belief_on_sequence(
            self.attacker_tom, self.tom_optimizer, att_pos_seq, theta
        )

        # ---- Episode diagnostics ----
        max_dist = float(
            np.linalg.norm(
                np.array(self.env_cfg.goal_a) - np.array(self.env_cfg.attacker_spawn)
            )
        )
        norm_dists = [d / max_dist for d in step_dists]
        entropy_at_mid = self._entropy_at_midfield(step_entropies, norm_dists)

        episode_info: Dict[str, float] = {
            "theta": float(theta),
            "steps": float(len(step_entropies)),
            "goal_reached": float(info.get("goal_reached", False)),
            "intercepted": float(info.get("intercepted", False)),
            "total_attacker_reward": float(total_att_reward),
            "entropy_at_midfield": float(entropy_at_mid),
            "mean_entropy": float(np.mean(step_entropies)) if step_entropies else 0.0,
            "final_dist_to_goal": float(step_dists[-1]) if step_dists else 0.0,
            "att_policy_loss": float(att_update["policy_loss"]),
            "att_value_loss": float(att_update["value_loss"]),
            "att_entropy": float(att_update["entropy"]),
            "def_policy_loss": float(def_update["policy_loss"]),
            "def_value_loss": float(def_update["value_loss"]),
            "def_entropy": float(def_update["entropy"]),
            "belief_loss": float(belief_loss),
            "tom_loss": float(tom_loss),
        }
        return episode_info

    # ------------------------------------------------------------------ #
    # Evaluation episode (no updates, no entropy bonus)
    # ------------------------------------------------------------------ #
    def evaluate_episode(self) -> Dict[str, float]:
        """Run one evaluation episode with greedy-ish policy sampling.

        No network updates; no intrinsic entropy bonus added to rewards so
        that reported metrics reflect pure task performance.
        """
        obs_dict, info = self.env.reset()
        att_obs = obs_dict["attacker"]
        def_obs = obs_dict["defender"]
        theta = info["theta"]

        def_hidden: Optional[torch.Tensor] = None
        tom_hidden: Optional[torch.Tensor] = None

        step_entropies: List[float] = []
        step_dists: List[float] = []

        terminated = False
        truncated = False
        final_att_obs = att_obs
        final_def_obs = def_obs

        for t in range(self.env_cfg.horizon):
            # Greedy action selection (argmax) for evaluation
            att_action, _, _, _ = self._greedy_action(
                self.attacker_agent, att_obs
            )
            def_action, _, _, _ = self._greedy_action(
                self.defender_agent, def_obs
            )

            # ToM entropy (for logging only, not added to reward)
            att_pos_t = torch.as_tensor(
                att_obs[:2], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            tom_logits, tom_hidden = self.attacker_tom.step(att_pos_t, tom_hidden)
            temperature = self.dec_cfg.belief_temperature
            tom_probs = torch.softmax(
                tom_logits / temperature, dim=-1
            ).squeeze(0)
            tom_entropy = shannon_entropy(tom_probs.detach().cpu().numpy())

            _, def_hidden = self.defender_belief.step(
                torch.as_tensor(
                    def_obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0),
                def_hidden,
            )

            next_obs_dict, rewards, terminated, truncated, info = self.env.step(
                att_action, def_action
            )
            next_att_obs = next_obs_dict["attacker"]
            next_def_obs = next_obs_dict["defender"]

            dist_to_goal = float(info.get("dist_to_goal", 0.0))
            step_entropies.append(tom_entropy)
            step_dists.append(dist_to_goal)

            att_obs = next_att_obs
            def_obs = next_def_obs
            final_att_obs = att_obs
            final_def_obs = def_obs

            if terminated or truncated:
                break

        max_dist = float(
            np.linalg.norm(
                np.array(self.env_cfg.goal_a) - np.array(self.env_cfg.attacker_spawn)
            )
        )
        norm_dists = [d / max_dist for d in step_dists]

        return {
            "theta": float(theta),
            "steps": float(len(step_entropies)),
            "goal_reached": float(info.get("goal_reached", False)),
            "intercepted": float(info.get("intercepted", False)),
            "entropy_at_midfield": float(
                self._entropy_at_midfield(step_entropies, norm_dists)
            ),
            "mean_entropy": float(np.mean(step_entropies)) if step_entropies else 0.0,
            "final_dist_to_goal": float(step_dists[-1]) if step_dists else 0.0,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _greedy_action(agent: PPOAgent, obs) -> Tuple[int, float, float, float]:
        """Return the greedy (argmax) action and its logp/value/entropy."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            logits, value = agent.policy.forward(obs_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = torch.argmax(probs, dim=-1)
            logp = dist.log_prob(action)
            entropy = dist.entropy()
        return (
            int(action.cpu().numpy()[0]),
            float(logp.cpu().numpy()[0]),
            float(value.cpu().numpy()[0]),
            float(entropy.cpu().numpy()[0]),
        )

    @staticmethod
    def _entropy_at_midfield(
        entropies: List[float],
        norm_dists: List[float],
        threshold: float = 0.5,
    ) -> float:
        """Mean entropy while the Attacker is at/near midfield."""
        mid = [e for e, d in zip(entropies, norm_dists) if d <= threshold]
        return float(np.mean(mid)) if mid else 0.0

    def evaluate(
        self, num_episodes: int
    ) -> Dict[str, float]:
        """Run multiple evaluation episodes and aggregate metrics."""
        metrics: List[Dict[str, float]] = [
            self.evaluate_episode() for _ in range(num_episodes)
        ]
        keys = metrics[0].keys()
        aggregated = {
            key: float(np.mean([m[key] for m in metrics])) for key in keys
        }
        aggregated["num_episodes"] = float(num_episodes)
        # Derived rates
        aggregated["goal_success_rate"] = aggregated["goal_reached"]
        aggregated["interception_rate"] = aggregated["intercepted"]
        return aggregated

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Save all network weights and optimizers to a checkpoint."""
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "global_step": self.global_step,
            "attacker_policy": self.attacker_agent.policy.state_dict(),
            "defender_policy": self.defender_agent.policy.state_dict(),
            "defender_belief": self.defender_belief.state_dict(),
            "attacker_tom": self.attacker_tom.state_dict(),
            "attacker_optimizer": self.attacker_agent.optimizer.state_dict(),
            "defender_optimizer": self.defender_agent.optimizer.state_dict(),
            "belief_optimizer": self.belief_optimizer.state_dict(),
            "tom_optimizer": self.tom_optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load network weights and optimizers from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.global_step = checkpoint.get("global_step", 0)
        self.attacker_agent.policy.load_state_dict(checkpoint["attacker_policy"])
        self.defender_agent.policy.load_state_dict(checkpoint["defender_policy"])
        self.defender_belief.load_state_dict(checkpoint["defender_belief"])
        self.attacker_tom.load_state_dict(checkpoint["attacker_tom"])
        self.attacker_agent.optimizer.load_state_dict(checkpoint["attacker_optimizer"])
        self.defender_agent.optimizer.load_state_dict(checkpoint["defender_optimizer"])
        self.belief_optimizer.load_state_dict(checkpoint["belief_optimizer"])
        self.tom_optimizer.load_state_dict(checkpoint["tom_optimizer"])