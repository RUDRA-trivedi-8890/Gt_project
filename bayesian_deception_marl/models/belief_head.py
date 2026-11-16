"""Recurrent GRU screening network for belief estimation.

The Defender's BeliefHead ingests a trajectory of observations through a GRU
and projects to a 2D softmax posterior over the Attacker's private type.
The same architecture is reused by the Attacker's internal Theory-of-Mind
(ToM) module, which simulates how an observer would interpret its own moves.

Fixes vs. asymmetric reasoning collapse:
  * Temperature-scaled softmax keeps early beliefs diffuse so the screening
    head does not snap to 0/1 on the first lateral move (which would make
    every feint look like a commitment).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BeliefHead(nn.Module):
    """Recurrent GRU belief network.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each observation in the sequence.
    hidden_dim : int
        GRU hidden size.
    num_layers : int
        Number of GRU layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialisation for GRU and projection weights."""
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        nn.init.orthogonal_(self.fc.weight, gain=1.0)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(
        self,
        sequences: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a full sequence through the GRU.

        Parameters
        ----------
        sequences : torch.Tensor
            Observation sequences of shape (batch, seq_len, input_dim).
        hidden : torch.Tensor, optional
            Initial hidden state of shape (num_layers, batch, hidden_dim).

        Returns
        -------
        logits : torch.Tensor
            Unnormalised log-probabilities of shape (batch, seq_len, 2).
        hidden : torch.Tensor
            Final hidden state of shape (num_layers, batch, hidden_dim).
        """
        out, hidden = self.gru(sequences, hidden)
        logits = self.fc(out)
        return logits, hidden

    def step(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single recurrent step.

        Parameters
        ----------
        obs : torch.Tensor
            Single observation of shape (batch, input_dim).
        hidden : torch.Tensor, optional
            Previous hidden state of shape (num_layers, batch, hidden_dim).

        Returns
        -------
        logits : torch.Tensor
            Unnormalised log-probabilities of shape (batch, 2).
        hidden : torch.Tensor
            Updated hidden state.
        """
        obs_seq = obs.unsqueeze(1)  # (batch, 1, input_dim)
        out, hidden = self.gru(obs_seq, hidden)
        logits = self.fc(out.squeeze(1))
        return logits, hidden

    def get_belief(
        self,
        sequences: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return numerically-clamped softmax belief probabilities.

        Temperature scaling (>1.0) softens the distribution so early beliefs
        do not collapse to confident 0/1 predictions on a single lateral move.

        Parameters
        ----------
        sequences : torch.Tensor
            Observation sequences of shape (batch, seq_len, input_dim).
        hidden : torch.Tensor, optional
            Initial hidden state.
        eps : float
            Clamping epsilon for numerical stability.
        temperature : float
            Softmax temperature. Values > 1.0 produce softer (more uniform)
            beliefs; values < 1.0 produce sharper beliefs.

        Returns
        -------
        probs : torch.Tensor
            Belief probabilities of shape (batch, seq_len, 2), clamped to
            [eps, 1 - eps].
        hidden : torch.Tensor
            Final hidden state.
        """
        logits, hidden = self.forward(sequences, hidden)
        probs = F.softmax(logits / temperature, dim=-1)
        probs = torch.clamp(probs, min=eps, max=1.0 - eps)
        return probs, hidden
