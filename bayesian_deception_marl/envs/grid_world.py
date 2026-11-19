"""Custom Gymnasium environment: asymmetric Bayesian signaling gridworld.

Implements a 11x11 tactical gridworld in which an Attacker with a private
objective (Goal A or Goal B) must reach its target while a Defender, unable
to observe the private type, tries to intercept. The Attacker's physical
trajectory acts as a costly signal; the Defender maintains a Bayesian belief
over the Attacker's type.

Observation masking is asymmetric:
  * Attacker obs:  [pos_x / 10.0, pos_y / 10.0, float(theta)]
  * Defender obs:  [att_x / 10.0, att_y / 10.0, def_x / 10.0, def_y / 10.0]
    (theta strictly masked for the Defender).

Fixes incorporated vs. the reward-hacking pathology:
  * Forward-progress gating info flag (x_pos strictly increasing vs. running max).
  * Defender commitment tracking (left / center / right of column 5).
  * False-alarm penalty: defender overcommits to one side but attacker reaches the other.
  * Progress shaping: potential-based reward for reducing distance to the active goal.
  * Milestone penalties: attacker penalized if it lags behind required row by deadline steps.
"""

from typing import Dict, Optional, Tuple, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import EnvConfig


class DeceptionGridWorld(gym.Env):
    """Asymmetric Bayesian signaling gridworld with anti-stalling and defender commitment mechanics."""

    metadata = {"render_modes": ["ascii"], "render_fps": 4}

    def __init__(self, config: EnvConfig, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config
        self.grid_size = config.grid_size
        self.horizon = config.horizon
        self.c_step = config.c_step

        self.attacker_spawn = np.array(config.attacker_spawn, dtype=np.float32)
        self.defender_spawn = np.array(config.defender_spawn, dtype=np.float32)
        self.goal_a = np.array(config.goal_a, dtype=np.float32)
        self.goal_b = np.array(config.goal_b, dtype=np.float32)

        self.action_space = spaces.Discrete(9)
        def_obs_dim = 5 if getattr(config, "include_time_in_obs", False) else 4
        self.observation_space = spaces.Dict({
            "attacker": spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            ),
            "defender": spaces.Box(
                low=-1.0, high=1.0, shape=(def_obs_dim,), dtype=np.float32
            ),
        })

        # Action deltas: 9 discrete actions (4 cardinal + 4 diagonal + 1 idle)
        # Allows smooth diagonal approaches instead of robotic 90-degree turns
        self._action_delta = np.array([
            [-1, 0],   # 0: Up (Backward)
            [1, 0],    # 1: Down (Forward toward row 10)
            [0, -1],   # 2: Left (toward Goal A col 0)
            [0, 1],    # 3: Right (toward Goal B col 10)
            [1, -1],   # 4: Forward-Left (Diagonal toward Goal A)
            [1, 1],    # 5: Forward-Right (Diagonal toward Goal B)
            [-1, -1],  # 6: Backward-Left
            [-1, 1],   # 7: Backward-Right
            [0, 0],    # 8: Idle
        ], dtype=np.float32)

        self.render_mode = render_mode

        # Episode state (initialised by reset)
        self.attacker_pos: Optional[np.ndarray] = None
        self.defender_pos: Optional[np.ndarray] = None
        self.theta: Optional[int] = None
        self.timestep: int = 0
        self.goal_reached: bool = False
        self.intercepted: bool = False

        # Anti-stalling / commitment tracking
        self.attacker_max_x: int = 0
        self.defender_committed_side: int = 0   # -1 = left, 0 = center, 1 = right
        self.milestones_penalized: set = set()

        self.reset()

    # ------------------------------------------------------------------ #
    # Reset / observation helpers
    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        """Reset the environment, sample a new private type, return obs + info."""
        super().reset(seed=seed)
        if options and "theta" in options:
            self.theta = int(options["theta"])
        else:
            self.theta = int(self.np_random.integers(0, 2))
        self.attacker_pos = self.attacker_spawn.copy()
        self.defender_pos = self.defender_spawn.copy()
        self.timestep = 0
        self.goal_reached = False
        self.intercepted = False

        # Reset tracking state
        self.attacker_max_x = 0
        self.defender_committed_side = 0
        self.milestones_penalized = set()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Build asymmetric observations.

        Attacker sees its own normalised position and its private type.
        Defender sees attacker + defender normalised positions; theta masked.
        """
        att_obs = np.array([
            self.attacker_pos[0] / 10.0,
            self.attacker_pos[1] / 10.0,
            float(self.theta),
        ], dtype=np.float32)

        if getattr(self.cfg, "include_time_in_obs", False):
            norm_t = float(self.timestep) / float(max(1, self.horizon))
            def_obs = np.array([
                self.attacker_pos[0] / 10.0,
                self.attacker_pos[1] / 10.0,
                self.defender_pos[0] / 10.0,
                self.defender_pos[1] / 10.0,
                norm_t,
            ], dtype=np.float32)
        else:
            def_obs = np.array([
                self.attacker_pos[0] / 10.0,
                self.attacker_pos[1] / 10.0,
                self.defender_pos[0] / 10.0,
                self.defender_pos[1] / 10.0,
            ], dtype=np.float32)

        return {"attacker": att_obs, "defender": def_obs}

    def _get_info(self) -> dict:
        return {
            "theta": self.theta,
            "timestep": self.timestep,
            "attacker_pos": self.attacker_pos.copy(),
            "defender_pos": self.defender_pos.copy(),
            "goal_reached": self.goal_reached,
            "intercepted": self.intercepted,
        }

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(
        self,
        attacker_action: int,
        defender_action: int,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, bool, dict]:
        """Advance one timestep for both agents.

        Returns (obs_dict, reward_dict, terminated, truncated, info).
        """
        self.timestep += 1

        # Clamp actions to valid range (9 discrete actions)
        att_a = int(np.clip(attacker_action, 0, 8))
        def_a = int(np.clip(defender_action, 0, 8))

        # Remember pre-step positions for progress shaping, commitment, and swap checks
        prev_att_pos = self.attacker_pos.copy()
        prev_def_pos = self.defender_pos.copy()
        prev_def_col = int(self.defender_pos[1])

        # Move agents with fully vectorised bounds clamping
        att_delta = self._action_delta[att_a].copy()

        self.attacker_pos = self.attacker_pos + att_delta
        self.attacker_pos = np.clip(self.attacker_pos, 0, self.grid_size - 1)

        self.defender_pos = self.defender_pos + self._action_delta[def_a]
        self.defender_pos = np.clip(self.defender_pos, 0, self.grid_size - 1)

        # Select the Attacker's active goal based on private type
        goal = self.goal_a if self.theta == 0 else self.goal_b

        prev_dist_to_goal = float(np.linalg.norm(prev_att_pos - goal))
        curr_dist_to_goal = float(np.linalg.norm(self.attacker_pos - goal))
        dist_att_def = float(np.linalg.norm(self.attacker_pos - self.defender_pos))

        # ---- Forward progress tracking (x = row, increasing toward goals) ----
        forward_progress = bool(self.attacker_pos[0] > self.attacker_max_x)
        if forward_progress:
            self.attacker_max_x = int(self.attacker_pos[0])

        # ---- Defender commitment tracking (column relative to center 5) ----
        if self.defender_pos[1] <= 4:
            self.defender_committed_side = -1   # committed to Goal-A side
        elif self.defender_pos[1] >= 6:
            self.defender_committed_side = 1    # committed to Goal-B side
        else:
            self.defender_committed_side = 0    # still center / uncommitted

        # ---- Rewards and termination ----
        reward_attacker = 0.0
        reward_defender = 0.0
        terminated = False

        # Active Goal Reached (Attacker within threshold of its target goal)
        wrong_goal = self.goal_b if self.theta == 0 else self.goal_a
        if curr_dist_to_goal <= 0.85:
            self.goal_reached = True
            reward_attacker += self.cfg.goal_reward          # +35.0
            reward_defender += -self.cfg.goal_reward         # -35.0
            terminated = True

            # False-alarm penalty: defender overcommitted to wrong side
            if self.defender_committed_side == -1 and self.theta == 1:
                reward_defender -= self.cfg.false_alarm_penalty
            elif self.defender_committed_side == 1 and self.theta == 0:
                reward_defender -= self.cfg.false_alarm_penalty

        # Wrong goal deterrence (prevents mode collapse where agent always picks one goal)
        elif np.linalg.norm(self.attacker_pos - wrong_goal) <= 1.0:
            reward_attacker -= 15.0
            terminated = True

        # Interception Detection (including continuous swap / pass-through check)
        is_intercept = False
        if dist_att_def < self.cfg.intercept_threshold:
            is_intercept = True
        elif getattr(self.cfg, "detect_swaps", True):
            # Check discrete cell swap: attacker moved to def's old tile and def moved to att's old tile
            discrete_swap = (
                np.array_equal(np.round(self.attacker_pos), np.round(prev_def_pos))
                and np.array_equal(np.round(self.defender_pos), np.round(prev_att_pos))
            )
            if discrete_swap:
                is_intercept = True
            else:
                # Continuous-time swept-sphere collision between moving agents
                u = prev_att_pos - prev_def_pos
                v = (self.attacker_pos - prev_att_pos) - (self.defender_pos - prev_def_pos)
                v_sq = float(np.dot(v, v))
                if v_sq > 1e-8:
                    t_star = -float(np.dot(u, v)) / v_sq
                    t_min = float(np.clip(t_star, 0.0, 1.0))
                else:
                    t_min = 0.0
                min_step_dist = float(np.linalg.norm(u + t_min * v))
                if min_step_dist < self.cfg.intercept_threshold:
                    is_intercept = True

        if is_intercept:
            self.intercepted = True
            reward_attacker += self.cfg.intercept_reward     # -15.0
            reward_defender += -self.cfg.intercept_reward    # +15.0
            terminated = True

        # Step penalty (Attacker only)
        reward_attacker -= self.c_step

        # Potential-based progress shaping directly toward active goal
        progress = prev_dist_to_goal - curr_dist_to_goal
        reward_attacker += self.cfg.progress_shaping_coef * progress

        # Game-Theoretic Pooling Constraint:
        # Before Row 7, the Attacker is in the open surveillance zone. Any premature lateral deviation
        # (col != 5) triggers severe exposure and is penalized (-10.0), enforcing pooling along Column 5.
        # At Row 7 (the final tactical sector), the constraint lifts, compelling the Attacker
        # to bifurcate diagonally toward its private target.
        if self.attacker_pos[0] < 7.0 and int(self.attacker_pos[1]) != 5:
            reward_attacker -= 10.0

        # Harsh Defender Proximity Penalty: Teaches Attacker to actively dodge Defender at (9, 5)
        # As Attacker approaches within 1.5 units of Defender, steep repulsive penalty is applied
        if not self.intercepted and dist_att_def <= 1.5:
            reward_attacker -= 4.0 * (1.5 - dist_att_def)

        # Defender distance shaping
        reward_defender += 0.01 * (1.0 / (1.0 + dist_att_def))

        # ---- Milestone penalties (anti-stalling) ----
        for i, (milestone_step, min_row) in enumerate(self.cfg.milestones):
            if i in self.milestones_penalized:
                continue
            if self.timestep >= milestone_step and int(self.attacker_pos[0]) < min_row:
                reward_attacker -= 2.0
                self.milestones_penalized.add(i)

        # Truncation at horizon
        truncated = self.timestep >= self.horizon

        obs = self._get_obs()
        rewards = {"attacker": reward_attacker, "defender": reward_defender}
        info = self._get_info()
        info.update({
            "dist_to_goal": curr_dist_to_goal,
            "dist_att_def": dist_att_def,
            "forward_progress": forward_progress,
            "attacker_max_x": float(self.attacker_max_x),
            "defender_committed_side": float(self.defender_committed_side),
            "prev_dist_to_goal": prev_dist_to_goal,
            "curr_dist_to_goal": curr_dist_to_goal,
        })

        return obs, rewards, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self) -> str:
        """ASCII render of the current grid state."""
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        goal = self.goal_a if self.theta == 0 else self.goal_b
        grid[int(goal[0]), int(goal[1])] = "G"
        grid[int(self.attacker_pos[0]), int(self.attacker_pos[1])] = "A"
        grid[int(self.defender_pos[0]), int(self.defender_pos[1])] = "D"
        return "\n".join(" ".join(row) for row in grid)

    def close(self) -> None:
        pass
