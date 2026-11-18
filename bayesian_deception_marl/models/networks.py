"""Actor-Critic networks with orthogonal weight initialization.

Discrete Actor-Critic with separate policy and value heads. Orthogonal weight
initialisation with standard gain scalings (tanh gain for hidden layers,
small gain for the actor head, unit gain for the critic head).
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonal_init(layer: nn.Linear, gain: float = 1.0) -> None:
    """Apply orthogonal weight initialisation and zero bias to a linear layer."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)


class ActorCritic(nn.Module):
    """Feedforward Actor-Critic with separate policy and value MLPs.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the observation vector.
    action_dim : int
        Number of discrete actions.
    hidden_dim : int
        Width of the hidden layers.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Policy network (shared trunk + action head)
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Value network (shared trunk + scalar head)
        self.critic_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.critic_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialisation with gain scalings per head."""
        tanh_gain = nn.init.calculate_gain("tanh")
        for module in self.actor_net:
            if isinstance(module, nn.Linear):
                orthogonal_init(module, gain=tanh_gain)
        # Small gain for the policy head to keep initial updates stable
        orthogonal_init(self.actor_head, gain=0.01)
        for module in self.critic_net:
            if isinstance(module, nn.Linear):
                orthogonal_init(module, gain=tanh_gain)
        orthogonal_init(self.critic_head, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        logits = self.actor_head(self.actor_net(obs))
        value = self.critic_head(self.critic_net(obs)).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action (or evaluate a given one) and return log-prob, entropy, and value."""
        logits, value = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        if action is None:
            action = dist.sample()
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logp, entropy, value


# ============================================================================
# UTILITY BUILDERS FOR THE TRAINING LOOP
# ============================================================================

def get_attacker_network(hidden_dim: int = 64, action_dim: int = 9) -> ActorCritic:
    """
    Attacker Observation: [x, y, theta] -> obs_dim = 3
    Actions: 9 discrete actions (4 cardinal + 4 diagonal + 1 idle) -> action_dim = 9
    """
    return ActorCritic(obs_dim=3, action_dim=action_dim, hidden_dim=hidden_dim)

def get_defender_network(hidden_dim: int = 64, action_dim: int = 9) -> ActorCritic:
    """
    Defender Observation: [att_x, att_y, def_x, def_y, belief_A, belief_B] -> obs_dim = 6
    Actions: 9 discrete actions (4 cardinal + 4 diagonal + 1 idle) -> action_dim = 9
    """
    return ActorCritic(obs_dim=6, action_dim=action_dim, hidden_dim=hidden_dim)