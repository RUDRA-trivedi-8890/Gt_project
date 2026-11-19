"""Recurrent Defender Actor-Critic with Integrated Bayesian Belief Head.

Architecture:
- Shared Recurrent Trunk: GRU over observation sequence history (att_x, att_y, def_x, def_y, t/T_max).
- Belief Head: Predicts P(theta | history) directly from GRU hidden state.
- Policy Head (Actor): Evaluates action logits conditioned on [hidden_state, belief_probs]
  (with optional stop_gradient on belief to prevent policy gradients from corrupting belief estimation).
- Value Head (Critic): Evaluates state-value V(s_t) from hidden_state.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def orthogonal_init(layer: nn.Linear, gain: float = 1.0) -> None:
    """Initialize linear layers with orthogonal weights and zero bias."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)


class RecurrentDefender(nn.Module):
    """Unified recurrent actor-critic and sequential belief screening network for the Defender."""

    def __init__(
        self,
        obs_dim: int = 5,
        action_dim: int = 9,
        hidden_dim: int = 64,
        belief_stop_gradient: bool = False,
        belief_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.belief_stop_gradient = belief_stop_gradient
        self.belief_temperature = belief_temperature

        # 1. Shared Recurrent Trunk
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)

        # 2. Sequential Belief Head (hidden_dim -> 2 classes)
        self.belief_head = nn.Linear(hidden_dim, 2)

        # 3. Policy Head (Actor): takes [hidden_state, belief_probs]
        self.actor_net = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        # 4. Value Head (Critic): takes hidden_state
        self.critic_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply orthogonal initialization across all network heads."""
        tanh_gain = nn.init.calculate_gain("tanh")
        orthogonal_init(self.belief_head, gain=1.0)

        # Actor net
        orthogonal_init(self.actor_net[0], gain=tanh_gain)
        orthogonal_init(self.actor_net[2], gain=0.01)

        # Critic net
        orthogonal_init(self.critic_net[0], gain=tanh_gain)
        orthogonal_init(self.critic_net[2], gain=1.0)

    def forward_step(
        self,
        obs: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step environment transition.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor of shape (batch_size, obs_dim).
        h_prev : torch.Tensor, optional
            Previous GRU hidden state of shape (1, batch_size, hidden_dim).
        deterministic : bool
            If True, takes greedy argmax action; else samples categorical.

        Returns
        -------
        action : torch.Tensor (batch_size,)
        logp : torch.Tensor (batch_size,)
        entropy : torch.Tensor (batch_size,)
        value : torch.Tensor (batch_size,)
        belief_logits : torch.Tensor (batch_size, 2)
        belief_probs : torch.Tensor (batch_size, 2)
        h_next : torch.Tensor (1, batch_size, hidden_dim)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        batch_size = obs.shape[0]

        if h_prev is None:
            h_prev = torch.zeros(1, batch_size, self.hidden_dim, device=obs.device)

        # Pass through GRU: input shape (batch_size, 1, obs_dim)
        gru_out, h_next = self.gru(obs.unsqueeze(1), h_prev)
        h_feat = gru_out.squeeze(1)  # (batch_size, hidden_dim)

        # 1. Belief Head
        belief_logits = self.belief_head(h_feat)
        temp = max(1e-4, self.belief_temperature)
        belief_probs = F.softmax(belief_logits / temp, dim=-1)

        # 2. Stop Gradient Option
        policy_belief = belief_probs.detach() if self.belief_stop_gradient else belief_probs
        actor_input = torch.cat([h_feat, policy_belief], dim=-1)

        # 3. Policy Head
        action_logits = self.actor_net(actor_input)
        dist = Categorical(logits=action_logits)

        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = dist.sample()

        logp = dist.log_prob(action)
        entropy = dist.entropy()

        # 4. Value Head
        value = self.critic_net(h_feat).squeeze(-1)

        return action, logp, entropy, value, belief_logits, belief_probs, h_next

    def evaluate_sequence(
        self,
        obs_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequence-aware forward evaluation for full episodes or chunks.

        Parameters
        ----------
        obs_seq : torch.Tensor
            Padded observation sequence of shape (batch_size, seq_len, obs_dim).
        actions_seq : torch.Tensor
            Action sequence of shape (batch_size, seq_len).
        h_0 : torch.Tensor, optional
            Initial hidden state of shape (1, batch_size, hidden_dim).

        Returns
        -------
        logp : torch.Tensor (batch_size, seq_len)
        entropy : torch.Tensor (batch_size, seq_len)
        values : torch.Tensor (batch_size, seq_len)
        belief_logits : torch.Tensor (batch_size, seq_len, 2)
        belief_probs : torch.Tensor (batch_size, seq_len, 2)
        """
        batch_size, seq_len, _ = obs_seq.shape
        if h_0 is None:
            h_0 = torch.zeros(1, batch_size, self.hidden_dim, device=obs_seq.device)

        gru_out, _ = self.gru(obs_seq, h_0)  # (batch_size, seq_len, hidden_dim)

        # Belief head across full sequence
        belief_logits = self.belief_head(gru_out)  # (batch_size, seq_len, 2)
        temp = max(1e-4, self.belief_temperature)
        belief_probs = F.softmax(belief_logits / temp, dim=-1)

        policy_belief = belief_probs.detach() if self.belief_stop_gradient else belief_probs
        actor_input = torch.cat([gru_out, policy_belief], dim=-1)

        action_logits = self.actor_net(actor_input)  # (batch_size, seq_len, action_dim)
        dist = Categorical(logits=action_logits)

        logp = dist.log_prob(actions_seq)  # (batch_size, seq_len)
        entropy = dist.entropy()           # (batch_size, seq_len)
        values = self.critic_net(gru_out).squeeze(-1)  # (batch_size, seq_len)

        return logp, entropy, values, belief_logits, belief_probs
