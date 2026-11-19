"""Configuration dataclasses for the Bayesian Deception MARL repository.

Contains EnvConfig, PPOConfig, and DeceptionConfig with updated defaults
designed to eliminate reward hacking (entropy farming at spawn) and
asymmetric reasoning collapse (defender twitching / overcommitment).
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class EnvConfig:
    """Gridworld environment configuration.

    Arena: 11x11 Gridworld.
    * Attacker Spawn: (0, 5) [Bottom Center]
    * Defender Spawn: (9, 5) [Top Center]
    * Goal A: (10, 0) [Top-Left]
    * Goal B: (10, 10) [Top-Right]
    * Action Space: Discrete(5) -> {0: Up, 1: Down, 2: Left, 3: Right, 4: Idle}
    """

    grid_size: int = 11
    horizon: int = 25
    c_step: float = 0.15                      # increased to penalize stalling
    goal_reward: float = 30.0                 # increased: reaching the goal must dominate entropy farming
    intercept_reward: float = -15.0           # interception penalty for attacker
    goal_threshold: float = 0.5
    intercept_threshold: float = 0.8
    progress_shaping_coef: float = 1.0        # potential-based shaping for goal progress
    false_alarm_penalty: float = 10.0         # defender penalty for overcommitting to the wrong side
    attacker_spawn: Tuple[int, int] = (0, 5)
    defender_spawn: Tuple[int, int] = (9, 5)
    goal_a: Tuple[int, int] = (10, 0)
    goal_b: Tuple[int, int] = (10, 10)
    # Milestone requirements: (step, min_row). Penalize attacker if row not reached by step.
    milestones: Optional[List[Tuple[int, int]]] = None
    detect_swaps: bool = True                 # Fix pass-through / cell-swap collision bug
    include_time_in_obs: bool = False         # Include t / T_max in defender observation (dim 5 when True)

    def __post_init__(self) -> None:
        if self.milestones is None:
            self.milestones = [(5, 3), (10, 6), (15, 9)]


@dataclass
class PPOConfig:
    """Proximal Policy Optimization hyperparameters."""

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ppo_epochs: int = 4
    batch_size: int = 64
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 64


@dataclass
class DeceptionConfig:
    """Deception / belief-regularization configuration.

    The entropy weight lambda_0 is scaled down (0.7) so deception is a
    means to reach the goal, not the goal itself. Label smoothing and
    belief temperature prevent the screening head from snapping to 0/1
    on the first lateral move.
    """

    lambda_0: float = 0.7                     # reduced from 2.5
    t_max: int = 25
    entropy_eps: float = 1e-8
    tom_lr: float = 1e-3
    belief_lr: float = 1e-3
    label_smoothing: float = 0.1              # soften BCE targets to avoid overconfidence
    belief_temperature: float = 1.5           # soften softmax so early beliefs stay diffuse


@dataclass
class ArmsRaceConfig:
    """Configuration for the adaptive arms-race mechanism (Phases 1 -> 2 -> 3).

    All mechanisms are gated behind `enabled=False` by default to ensure
    exact backward compatibility and reproducibility.
    """

    enabled: bool = False                     # Master flag for arms race mechanism
    window_epochs: int = 5                    # Sliding window W of past epochs for caught-route archive
    mu_max: float = 1.0                       # Maximum caught-route penalty weight
    kappa: float = 5.0                        # Scaling for terminal reveal-delay reward
    shaping_cap_ratio: float = 0.30           # Cap: total added shaping per episode <= 30% of goal_reward
    p2_interception_threshold: float = 0.70   # Phase 2 trigger: interception rate threshold
    p2_consecutive_epochs: int = 3            # Phase 2 trigger: required consecutive epochs above threshold
    p3_ramp_epochs: int = 10                  # Phase 3: epochs R over which mu_e ramps linearly from 0 to mu_max
    midfield_row: int = 5                     # Row <= 5 considered before midfield
    belief_reveal_threshold: float = 0.90     # Defender belief threshold defining intent reveal
    use_league: bool = False                  # League training toggle
    league_checkpoint_freq: int = 5           # Keep past policy checkpoints every K epochs
    league_sampling_prob: float = 0.30        # Opponent sampled as 70% current, 30% from past pool
    use_history_input: bool = False           # Attacker input vector with last K episode outcomes
    history_k: int = 5                        # Number of past episode outcomes tracked


@dataclass
class DefenderUpgradeConfig:
    """Configuration for upgrading the Defender into a sequential Bayesian receiver."""

    enabled: bool = True                      # Master toggle for recurrent receiver upgrade
    use_recurrent_defender: bool = True       # GRU trunk over obs history with belief, policy, value heads
    hidden_dim: int = 64                      # GRU hidden dimensionality
    belief_stop_gradient: bool = False        # Detach belief features before feeding to policy head
    belief_temperature: float = 1.0           # Removed 1.5 handicap: default 1.0 for calibrated probabilities
    label_smoothing: float = 0.0              # Removed 0.1 handicap: default 0.0 for confident classification
    use_exact_bayes_distill: bool = False     # Distillation auxiliary loss against exact Bayes teacher
    distill_coef: float = 0.1                 # Weight for distillation loss
    sync_tom_from_defender: bool = True       # Attacker ToM synced from frozen copy of defender belief head
    tom_sync_freq: int = 5                    # Epoch sync frequency for frozen ToM
    use_fictitious_play: bool = True          # Mix {current, past checkpoints, trick suite} in defender training
    trick_sampling_prob: float = 0.30         # Fraction of episodes sampling from scripted trick attacker suite

