"""Dual-agent co-evolutionary training engine with CTDE and Adaptive Arms Race.

Centralised Training with Decentralised Execution (CTDE):
* The Defender's GRU belief head trains on rollout batches with supervised
  Binary Cross-Entropy (with label smoothing) against the true private type theta.
* The Attacker policy learns via PPO with an intrinsic Shannon-entropy
  bonus estimated through its internal Theory-of-Mind (ToM) module. The
  entropy bonus is gated on forward progress so the Attacker cannot farm
  deception reward by idling at spawn.
* Adaptive Arms Race (Phases 1 -> 2 -> 3):
  - Phase 1: Baseline training (mu_e = 0).
  - Phase 2: Interception rate > 0.7 for 3 consecutive epochs triggers Phase 2.
  - Phase 3: Linear ramp of caught-route penalty mu_e from 0 to mu_max over R epochs.
  - Caught-route archive C_theta(x, y) penalizes visited cells from failed/leaked episodes.
  - Terminal reveal-delay bonus rewards delaying true intent revelation beyond previous mean.
  - Total added shaping capped strictly at 30% of goal_reward.
  - League training: checkpoints saved every K epochs; 70% current / 30% past pool sampling.
  - Cross-episode history vector input (optional flag).
"""

from collections import deque
import copy
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs.grid_world import DeceptionGridWorld
from models.belief_head import BeliefHead
from algos.ppo_agent import PPOAgent
from config import EnvConfig, PPOConfig, DeceptionConfig, ArmsRaceConfig
from utils.metrics import (
    shannon_entropy,
    compute_t_reveal,
    route_diversity_entropy,
    mean_pairwise_trajectory_distance,
)
from algos.phase_scheduler import PhaseScheduler
from algos.caught_route_archive import CaughtRouteArchive


class SelfPlayTrainer:
    """Co-evolutionary trainer for the asymmetric deception game with adaptive arms race."""

    def __init__(
        self,
        env_cfg: EnvConfig,
        ppo_cfg: PPOConfig,
        dec_cfg: DeceptionConfig,
        arms_cfg: Optional[ArmsRaceConfig] = None,
        device: str = "cpu",
        action_dim: Optional[int] = None,
    ) -> None:
        self.env_cfg = env_cfg
        self.ppo_cfg = ppo_cfg
        self.dec_cfg = dec_cfg
        self.arms_cfg = arms_cfg if arms_cfg is not None else ArmsRaceConfig(enabled=False)
        self.device = torch.device(device)

        self.env = DeceptionGridWorld(env_cfg)
        act_dim = action_dim if action_dim is not None else self.env.action_space.n

        # ------------------------------------------------------------------ #
        # Attacker observation dimension (optional cross-episode history input)
        # ------------------------------------------------------------------ #
        self.history_k = self.arms_cfg.history_k
        self.use_history_input = self.arms_cfg.enabled and self.arms_cfg.use_history_input
        self.att_history_buffer: deque = deque(maxlen=self.history_k)

        att_obs_dim = 3 + (3 * self.history_k if self.use_history_input else 0)
        def_obs_dim = self.env.observation_space["defender"].shape[0]

        # ------------------------------------------------------------------ #
        # Agents (Actor-Critic PPO)
        # ------------------------------------------------------------------ #
        self.attacker_agent = PPOAgent(
            obs_dim=att_obs_dim, action_dim=act_dim, ppo_cfg=ppo_cfg, device=device
        )
        self.defender_agent = PPOAgent(
            obs_dim=def_obs_dim, action_dim=act_dim, ppo_cfg=ppo_cfg, device=device
        )

        # ------------------------------------------------------------------ #
        # Belief heads
        #   Defender BeliefHead  : input_dim=def_obs_dim (full defender observation)
        #   Attacker ToM         : input_dim=2 (attacker position only)
        # ------------------------------------------------------------------ #
        self.defender_belief = BeliefHead(
            input_dim=def_obs_dim, hidden_dim=ppo_cfg.hidden_dim
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
        self.current_epoch: int = 0

        # ------------------------------------------------------------------ #
        # Adaptive Arms Race Modules
        # ------------------------------------------------------------------ #
        self.phase_scheduler = PhaseScheduler(self.arms_cfg)
        self.caught_archive = CaughtRouteArchive(
            grid_size=self.env_cfg.grid_size,
            window_epochs=self.arms_cfg.window_epochs,
            midfield_row=self.arms_cfg.midfield_row,
            belief_threshold=self.arms_cfg.belief_reveal_threshold,
        )
        self.current_mu_e: float = 0.0
        self.current_phase: int = 1
        self.t_reveal_prev_mean: float = float(self.env_cfg.horizon) / 2.0

        # League checkpoints
        self.attacker_league_pool: List[dict] = []
        self.defender_league_pool: List[dict] = []

    # ------------------------------------------------------------------ #
    # Observation building (incorporating history vector if enabled)
    # ------------------------------------------------------------------ #
    def _build_attacker_obs(self, raw_att_obs: np.ndarray) -> np.ndarray:
        """Construct attacker observation vector, appending history if enabled."""
        if not self.use_history_input:
            return raw_att_obs

        # Build history vector of shape (3 * K,)
        # Each entry: [outcome (1.0 goal, -1.0 caught, 0.0 timeout), step_caught / horizon, theta]
        hist_vec = np.zeros(3 * self.history_k, dtype=np.float32)
        for idx, item in enumerate(self.att_history_buffer):
            offset = idx * 3
            hist_vec[offset] = item.get("outcome", 0.0)
            hist_vec[offset + 1] = item.get("step_caught", 0.0) / float(self.env_cfg.horizon)
            hist_vec[offset + 2] = float(item.get("theta", 0.0))
        return np.concatenate([raw_att_obs, hist_vec]).astype(np.float32)

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
        """Train a belief network to predict theta from an observation sequence."""
        seq_t = torch.as_tensor(
            np.stack(sequence), dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, seq_len, input_dim)
        seq_len = seq_t.shape[1]

        smooth = self.dec_cfg.label_smoothing
        target_value = 1.0 - smooth if theta == 1 else smooth
        targets = torch.full((seq_len,), target_value, device=self.device)

        logits, _ = belief_net.forward(seq_t)
        binary_logits = logits[:, :, 1] - logits[:, :, 0]
        binary_logits = binary_logits.squeeze(0)

        loss = self.bce_loss(binary_logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu().numpy())

    # ------------------------------------------------------------------ #
    # League opponent selection
    # ------------------------------------------------------------------ #
    def _get_active_defender_policy(self) -> nn.Module:
        """Sample defender policy from current or past league pool."""
        if (
            self.arms_cfg.enabled
            and self.arms_cfg.use_league
            and len(self.defender_league_pool) > 0
            and np.random.rand() < self.arms_cfg.league_sampling_prob
        ):
            # Select random past checkpoint
            past_weights = self.defender_league_pool[np.random.randint(len(self.defender_league_pool))]
            temp_policy = copy.deepcopy(self.defender_agent.policy)
            temp_policy.load_state_dict(past_weights)
            temp_policy.eval()
            return temp_policy
        return self.defender_agent.policy

    # ------------------------------------------------------------------ #
    # Single training episode (with network updates)
    # ------------------------------------------------------------------ #
    def train_episode(self, seed: Optional[int] = None) -> Dict[str, float]:
        """Run one episode, collect rollouts, update all networks."""
        if seed is not None:
            obs_dict, info = self.env.reset(seed=seed)
        else:
            obs_dict, info = self.env.reset()
        raw_att_obs = obs_dict["attacker"]
        def_obs = obs_dict["defender"]
        theta = info["theta"]

        att_obs = self._build_attacker_obs(raw_att_obs)

        # Recurrent hidden states
        def_hidden: Optional[torch.Tensor] = None
        tom_hidden: Optional[torch.Tensor] = None

        def_seq: List[np.ndarray] = [def_obs.copy()]
        att_pos_seq: List[np.ndarray] = [raw_att_obs[:2].copy()]
        trajectory_pts: List[Tuple[int, int]] = [(int(round(raw_att_obs[0] * 10.0)), int(round(raw_att_obs[1] * 10.0)))]
        beliefs_list: List[np.ndarray] = []

        step_entropies: List[float] = []
        step_dists: List[float] = []
        total_att_reward = 0.0

        # Arms race shaping tracking
        total_step_penalties = 0.0
        shaping_cap = self.arms_cfg.shaping_cap_ratio * self.env_cfg.goal_reward

        terminated = False
        truncated = False
        final_att_obs = att_obs
        final_def_obs = def_obs

        # Active policy for defender (may be past checkpoint under league)
        def_policy = self._get_active_defender_policy()

        # Step-by-step rollout storage for post-hoc reward capping
        att_transitions: List[dict] = []

        for t in range(self.env_cfg.horizon):
            lam = self._lambda_t(t)

            # ---------------- Attacker ---------------- #
            att_out = self.attacker_agent.select_action(att_obs)
            att_action = att_out["action"]
            att_logp = att_out["logp"]
            att_value = att_out["value"]

            # Attacker ToM belief
            att_pos = raw_att_obs[:2].copy()
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
            if def_policy is self.defender_agent.policy:
                def_out = self.defender_agent.select_action(def_obs)
                def_action = def_out["action"]
                def_logp = def_out["logp"]
                def_value = def_out["value"]
            else:
                # Past checkpoint used for inference only
                with torch.no_grad():
                    obs_t = torch.as_tensor(def_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    logits, val = def_policy(obs_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    act = dist.sample()
                    def_action = int(act.cpu().numpy()[0])
                    def_logp = float(dist.log_prob(act).cpu().numpy()[0])
                    def_value = float(val.cpu().numpy()[0])

            # Defender belief update
            def_obs_t = torch.as_tensor(
                def_obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            def_logits, def_hidden = self.defender_belief.step(def_obs_t, def_hidden)
            def_probs = torch.softmax(def_logits / temperature, dim=-1).squeeze(0).detach().cpu().numpy()
            beliefs_list.append(def_probs)

            # ---------------- Environment ---------------- #
            next_obs_dict, rewards, terminated, truncated, info = self.env.step(
                att_action, def_action
            )
            next_raw_att_obs = next_obs_dict["attacker"]
            next_def_obs = next_obs_dict["defender"]

            # Base reward
            att_reward = rewards["attacker"]
            is_forward = info.get("forward_progress", False)
            step_pen = 0.0

            if is_forward:
                att_reward += lam * tom_entropy

                # Arms race: caught-route penalty (gated on forward progress)
                if self.arms_cfg.enabled and self.current_mu_e > 0.0:
                    curr_x = raw_att_obs[0] * 10.0
                    curr_y = raw_att_obs[1] * 10.0
                    c_val = self.caught_archive.get_penalty(theta, curr_x, curr_y)
                    step_pen = self.current_mu_e * c_val
                    total_step_penalties += step_pen
                    att_reward -= step_pen

            def_reward = rewards["defender"]

            att_transitions.append({
                "obs": att_obs,
                "action": att_action,
                "reward": att_reward,
                "terminated": terminated,
                "logp": att_logp,
                "value": att_value,
            })

            # Store defender transitions
            self.defender_agent.store(
                def_obs, def_action, def_reward, terminated, def_logp, def_value
            )

            total_att_reward += att_reward

            # Logging & Trajectory tracking
            dist_to_goal = float(info.get("dist_to_goal", 0.0))
            step_entropies.append(tom_entropy)
            step_dists.append(dist_to_goal)

            pt_x = int(np.clip(round(next_raw_att_obs[0] * 10.0), 0, self.env_cfg.grid_size - 1))
            pt_y = int(np.clip(round(next_raw_att_obs[1] * 10.0), 0, self.env_cfg.grid_size - 1))
            trajectory_pts.append((pt_x, pt_y))

            def_seq.append(next_def_obs.copy())
            att_pos_seq.append(next_raw_att_obs[:2].copy())

            # Advance
            raw_att_obs = next_raw_att_obs
            att_obs = self._build_attacker_obs(raw_att_obs)
            def_obs = next_def_obs
            final_att_obs = att_obs
            final_def_obs = def_obs
            self.global_step += 1

            if terminated or truncated:
                break

        # ------------------------------------------------------------------ #
        # Terminal Arms Race Reward & Shaping Cap
        # ------------------------------------------------------------------ #
        t_rev = compute_t_reveal(
            beliefs_list, theta, threshold=self.arms_cfg.belief_reveal_threshold, t_max=self.env_cfg.horizon
        )

        terminal_reveal_bonus = 0.0
        if self.arms_cfg.enabled:
            # Terminal reward: + kappa * max(0, t_reveal - t_reveal_prev_mean) / T_max
            delay_diff = max(0.0, float(t_rev) - self.t_reveal_prev_mean)
            terminal_reveal_bonus = self.arms_cfg.kappa * (delay_diff / float(self.env_cfg.horizon))

            # Total added shaping per episode: terminal bonus - step penalties
            total_added_shaping = terminal_reveal_bonus - total_step_penalties
            capped_shaping = float(np.clip(total_added_shaping, -shaping_cap, shaping_cap))

            # Adjust terminal step reward so sum of additions equals capped_shaping
            terminal_adjustment = capped_shaping - (-total_step_penalties)
            att_transitions[-1]["reward"] += terminal_adjustment
            total_att_reward += (capped_shaping + total_step_penalties)

        # Store all attacker transitions in buffer
        for trans in att_transitions:
            self.attacker_agent.store(
                trans["obs"], trans["action"], trans["reward"], trans["terminated"], trans["logp"], trans["value"]
            )

        # Record in caught archive if intercepted or leaked before midfield
        intercepted = bool(info.get("intercepted", False))
        goal_reached = bool(info.get("goal_reached", False))

        if self.arms_cfg.enabled:
            self.caught_archive.record_episode(theta, trajectory_pts, intercepted, beliefs_list)

        # Record history for optional cross-episode adaptation
        if self.use_history_input:
            outcome = 1.0 if goal_reached else (-1.0 if intercepted else 0.0)
            self.att_history_buffer.append({
                "outcome": outcome,
                "step_caught": float(len(step_entropies)) if intercepted else 0.0,
                "theta": float(theta),
            })

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

        # ---- Belief-head supervised updates ----
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
            "goal_reached": float(goal_reached),
            "intercepted": float(intercepted),
            "total_attacker_reward": float(total_att_reward),
            "entropy_at_midfield": float(entropy_at_mid),
            "mean_entropy": float(np.mean(step_entropies)) if step_entropies else 0.0,
            "final_dist_to_goal": float(step_dists[-1]) if step_dists else 0.0,
            "t_reveal": float(t_rev),
            "phase": float(self.current_phase),
            "mu_e": float(self.current_mu_e),
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
    # Full Epoch Training with Metrics & Phase Progression
    # ------------------------------------------------------------------ #
    def train_epoch(self, num_episodes: int, epoch: Optional[int] = None) -> Dict[str, float]:
        """Train across an epoch of multiple episodes, updating phases and archives."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        ep_infos: List[Dict[str, float]] = []
        trajectories_theta0: List[List[Tuple[float, float]]] = []
        trajectories_theta1: List[List[Tuple[float, float]]] = []
        all_trajectories: List[List[Tuple[float, float]]] = []
        t_reveals: List[float] = []

        for _ in range(num_episodes):
            info = self.train_episode()
            ep_infos.append(info)
            t_reveals.append(info["t_reveal"])

        # Epoch aggregate metrics
        interception_rate = float(np.mean([m["intercepted"] for m in ep_infos]))
        goal_rate = float(np.mean([m["goal_reached"] for m in ep_infos]))
        mean_t_rev = float(np.mean(t_reveals)) if t_reveals else float(self.env_cfg.horizon)

        # Update running reveal baseline
        self.t_reveal_prev_mean = 0.8 * self.t_reveal_prev_mean + 0.2 * mean_t_rev

        # Finalize epoch in caught archive
        if self.arms_cfg.enabled:
            self.caught_archive.end_epoch()

        # Step Phase Scheduler
        self.current_phase, self.current_mu_e = self.phase_scheduler.step(
            interception_rate, self.current_epoch
        )

        # League checkpointing
        if self.arms_cfg.enabled and self.arms_cfg.use_league:
            if self.current_epoch % self.arms_cfg.league_checkpoint_freq == 0:
                self.attacker_league_pool.append(copy.deepcopy(self.attacker_agent.policy.state_dict()))
                self.defender_league_pool.append(copy.deepcopy(self.defender_agent.policy.state_dict()))

        # Evaluate route diversity for this epoch
        eval_metrics = self.evaluate(num_episodes=min(20, num_episodes))

        epoch_summary: Dict[str, float] = {
            "epoch": float(self.current_epoch),
            "phase": float(self.current_phase),
            "mu_e": float(self.current_mu_e),
            "interception_rate": interception_rate,
            "goal_rate": goal_rate,
            "mean_t_reveal": mean_t_rev,
            "route_diversity_entropy": eval_metrics.get("route_diversity_entropy", 0.0),
            "mean_pairwise_distance": eval_metrics.get("mean_pairwise_distance", 0.0),
            "mean_attacker_reward": float(np.mean([m["total_attacker_reward"] for m in ep_infos])),
            "mean_entropy": float(np.mean([m["mean_entropy"] for m in ep_infos])),
        }
        return epoch_summary

    # ------------------------------------------------------------------ #
    # Evaluation episode (no updates, no shaping)
    # ------------------------------------------------------------------ #
    def evaluate_episode(self, seed: Optional[int] = None) -> Dict[str, any]:
        """Run one evaluation episode with greedy policy sampling."""
        if seed is not None:
            obs_dict, info = self.env.reset(seed=seed)
        else:
            obs_dict, info = self.env.reset()
        raw_att_obs = obs_dict["attacker"]
        def_obs = obs_dict["defender"]
        theta = info["theta"]

        att_obs = self._build_attacker_obs(raw_att_obs)

        def_hidden: Optional[torch.Tensor] = None
        tom_hidden: Optional[torch.Tensor] = None

        step_entropies: List[float] = []
        step_dists: List[float] = []
        trajectory: List[Tuple[float, float]] = [(float(raw_att_obs[0] * 10.0), float(raw_att_obs[1] * 10.0))]
        beliefs_list: List[np.ndarray] = []

        terminated = False
        truncated = False

        for t in range(self.env_cfg.horizon):
            att_action, _, _, _ = self._greedy_action(self.attacker_agent, att_obs)
            def_action, _, _, _ = self._greedy_action(self.defender_agent, def_obs)

            att_pos_t = torch.as_tensor(
                raw_att_obs[:2], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            tom_logits, tom_hidden = self.attacker_tom.step(att_pos_t, tom_hidden)
            temperature = self.dec_cfg.belief_temperature
            tom_probs = torch.softmax(
                tom_logits / temperature, dim=-1
            ).squeeze(0)
            tom_entropy = shannon_entropy(tom_probs.detach().cpu().numpy())

            def_logits, def_hidden = self.defender_belief.step(
                torch.as_tensor(
                    def_obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0),
                def_hidden,
            )
            def_probs = torch.softmax(def_logits / temperature, dim=-1).squeeze(0).detach().cpu().numpy()
            beliefs_list.append(def_probs)

            next_obs_dict, rewards, terminated, truncated, info = self.env.step(
                att_action, def_action
            )
            next_raw_att_obs = next_obs_dict["attacker"]
            next_def_obs = next_obs_dict["defender"]

            dist_to_goal = float(info.get("dist_to_goal", 0.0))
            step_entropies.append(tom_entropy)
            step_dists.append(dist_to_goal)

            trajectory.append((float(next_raw_att_obs[0] * 10.0), float(next_raw_att_obs[1] * 10.0)))

            raw_att_obs = next_raw_att_obs
            att_obs = self._build_attacker_obs(raw_att_obs)
            def_obs = next_def_obs

            if terminated or truncated:
                break

        max_dist = float(
            np.linalg.norm(
                np.array(self.env_cfg.goal_a) - np.array(self.env_cfg.attacker_spawn)
            )
        )
        norm_dists = [d / max_dist for d in step_dists]
        t_rev = compute_t_reveal(
            beliefs_list, theta, threshold=self.arms_cfg.belief_reveal_threshold, t_max=self.env_cfg.horizon
        )

        return {
            "theta": int(theta),
            "steps": float(len(step_entropies)),
            "goal_reached": float(info.get("goal_reached", False)),
            "intercepted": float(info.get("intercepted", False)),
            "entropy_at_midfield": float(
                self._entropy_at_midfield(step_entropies, norm_dists)
            ),
            "mean_entropy": float(np.mean(step_entropies)) if step_entropies else 0.0,
            "final_dist_to_goal": float(step_dists[-1]) if step_dists else 0.0,
            "t_reveal": float(t_rev),
            "trajectory": trajectory,
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
        results = [self.evaluate_episode() for _ in range(num_episodes)]

        goal_rate = float(np.mean([r["goal_reached"] for r in results]))
        intercept_rate = float(np.mean([r["intercepted"] for r in results]))
        mean_t_rev = float(np.mean([r["t_reveal"] for r in results]))
        mean_steps = float(np.mean([r["steps"] for r in results]))
        mean_entropy = float(np.mean([r["mean_entropy"] for r in results]))

        # Route diversity computation
        trajectories = [r["trajectory"] for r in results]
        div_entropy = route_diversity_entropy(trajectories, grid_size=self.env_cfg.grid_size)
        pair_dist = mean_pairwise_trajectory_distance(trajectories)

        return {
            "num_episodes": float(num_episodes),
            "goal_reached": goal_rate,
            "goal_success_rate": goal_rate,
            "intercepted": intercept_rate,
            "interception_rate": intercept_rate,
            "mean_t_reveal": mean_t_rev,
            "mean_steps": mean_steps,
            "mean_entropy": mean_entropy,
            "route_diversity_entropy": div_entropy,
            "mean_pairwise_distance": pair_dist,
        }

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Save all network weights and optimizers to a checkpoint."""
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "current_phase": self.current_phase,
            "current_mu_e": self.current_mu_e,
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
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.current_phase = checkpoint.get("current_phase", 1)
        self.current_mu_e = checkpoint.get("current_mu_e", 0.0)
        self.attacker_agent.policy.load_state_dict(checkpoint["attacker_policy"])
        self.defender_agent.policy.load_state_dict(checkpoint["defender_policy"])
        self.defender_belief.load_state_dict(checkpoint["defender_belief"])
        self.attacker_tom.load_state_dict(checkpoint["attacker_tom"])
        self.attacker_agent.optimizer.load_state_dict(checkpoint["attacker_optimizer"])
        self.defender_agent.optimizer.load_state_dict(checkpoint["defender_optimizer"])
        self.belief_optimizer.load_state_dict(checkpoint["belief_optimizer"])
        self.tom_optimizer.load_state_dict(checkpoint["tom_optimizer"])
